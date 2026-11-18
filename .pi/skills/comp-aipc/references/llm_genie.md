# LLM & Transformer Porting (Custom AR Loop & Genie Contract)

Covers exporting and running PyTorch/HuggingFace transformer models on Qualcomm HTP/NPU via either:
- **Path A — Custom Host AR Loop**: Python + `qai_appbuilder` (`scripts/qai`) driving fixed-shape Prefill/Decode graphs.
- **Path B — Genie Runtime (`genie-t2t-run` / `genie-app`)**: Handing the autoregressive loop directly to Qualcomm's native C++ Genie engine.

---

## 1. Bring-Up Progression & Graph Topologies

1. **Fastest Sanity Bring-Up (Decoder-Only)**: Start with `use_cache=False` and a single fixed input `input_ids [1, SEQ_LEN]` (host-side sliding window). Once validated on HTP, upgrade to explicit KV-cache Prefill + Decode graphs.
2. **Decoder-Only Production (LLaMA, Qwen, GPT)**:
   - **Prefill Graph**: `input_ids [1, prefill_seq]` (+ `position_ids`) $\rightarrow$ `logits` + initial `present_key_values` (use fixed max length + padding, or buckets like `32 / 64 / 128 / 256`).
   - **Decode Graph**: `input_ids [1, 1]` + `position_ids [1, 1]` + `past_key_values [1, kv_heads, past_seq, head_dim]` $\rightarrow$ `logits [1, vocab]` + `present_key_values`.
3. **Encoder-Decoder (Whisper, T5, BART) — 2-Graph Pipeline**:
   - **Graph 1 (`Encoder + Cross-KV`)**: Run **once** per input (`mel [1, 80, 3000]` $\rightarrow$ `ca_key.{0..N-1}, ca_value.{0..N-1}`). Projecting `k_proj`/`v_proj` inside Graph 1 avoids recomputing cross-attention KV every step.
   - **Graph 2 (`Decoder`)**: Run per token (`decoder_input_ids [1, 1]` + `sa_past_kv` + `ca_kv` $\rightarrow$ `logits` + `sa_present_kv`). Feed `ca_kv` unchanged every step; apply sliding window only to `sa_kv`.

---

## 2. HuggingFace Export Blockers & ONNX Patches (Qwen3 / LLaMA)

Always load models with `attn_implementation="eager"` and `torch_dtype=torch.float32`, and export with **`dynamo=False`** (the PyTorch $\ge 2.9$ `torch.export` exporter fails on `*args` KV-cache signatures).

| Blocker / Op | Symptom | Fix |
| :--- | :--- | :--- |
| **`aten::diff`** (`masking_utils.py`) | `UnsupportedOperatorError: Exporting aten::diff` | Monkey-patch before model load: `transformers.masking_utils.find_packed_sequence_indices = lambda *a, **kw: None` |
| **`Cast(BOOL_8 -> FLOAT_16)`** | `qnn-onnx-converter` passes, but `qnn-context-binary-generator` fails on HTP with `validateOpConfig failed 3110` | Pass **`attention_mask=None`** in export wrapper so the model builds a pure float causal mask |
| **Dynamic `Gather` leak** | `logits[:, -1, :]` leaks an unbound graph input (`onnx::Gather_N`) | Replace with fixed slice + squeeze: `outputs.logits[:, -1:, :].squeeze(1)` |
| **`Reciprocal` / `ReduceMean` / `Reshape`** | Schema/op warnings on older QAIRT | Rewrite `Reciprocal(x)` $\rightarrow$ `Div(1.0, x)`; move `ReduceMean` axes from tensor input to `axes` attribute; strip `allowzero` from `Reshape` |

---

## 3. Path A: Host AR Loop & KV-Cache Rules (`scripts/qai`)

- **Left-Pad Initial Window (Critical)**: Always place prompt tokens at the **right end** of the fixed window and pad on the left (`[PAD_ID] * (SEQ_LEN - len(ids)) + ids`). Right-padding puts prompt tokens at `[0..len-1]`, causing them to be discarded first as the window slides and degrading generation into repetition loops.
- **Sliding-Window Truncation**: When `present_seq = past_seq + 1`, slice on host CPU between decode steps: `next_past = present[:, :, -past_seq:, :]`.
- **`past_seq` Sizing**: Use `past_seq >= max_expected_output_tokens * 1.5` (minimum `64`; `32` is only for initial bring-up).
- **Output Selection & Target Dependencies**:
  - Never assume `outputs[0]` is `logits`—QAIRT reorders outputs. Look up `"logits"` by name via `session.get_output_names()` and deploy `<model>.yaml` alongside `<model>.onnx`.
  - On ARM Linux targets without `transformers`, tokenize on host or use Rust-backed `tokenizers.Tokenizer.from_file("tokenizer.json")` directly.

---

## 4. Path B: Genie Runtime Graph Contract (`genie-t2t-run` / `genie-app`)

When targeting Qualcomm's C++ Genie engine directly, the exported graph and quantization encodings must satisfy these strict contracts (`AR` = tokens/step, `CL` = context length):

### A. Tensor Naming & Shapes
- **KV Cache Naming**: `past_key_<L>_in`, `past_value_<L>_in`, `past_key_<L>_out`, `past_value_<L>_out`. Put **exactly one integer** in each KV tensor name (`layer3_past_key_0_in` breaks layer/head parsing).
- **Required Shapes (Transposed Keys & Delta-Only Outputs)**:
  - `attention_mask`: `[1, AR, CL]`
  - `position_ids_cos` / `position_ids_sin`: `[1, AR, rope_dim]` where `rope_dim = head_dim / 2`
  - `past_key_<L>_in`: `[n_kv, 1, kv_dim, CL - AR]` *(transposed)*
  - `past_value_<L>_in`: `[n_kv, 1, CL - AR, kv_dim]`
  - `past_key_<L>_out`: `[n_kv, 1, kv_dim, AR]` *(new delta tokens only, transposed)*
  - `past_value_<L>_out`: `[n_kv, 1, AR, kv_dim]` *(new delta tokens only)*
  - *(Note: Genie owns the ring buffer host-side and aliases `_in` and `_out` onto one memory block—`_in` and `_out` quantization scale/offset MUST be identical per layer.)*

### B. Two Silent-Failure Rules (No Error Logged, Produces Garbage)
1. **Attention Mask Quantization Encoding**:
   - Genie does **not** quantize `attention_mask` using scale/offset—it writes raw integer `0xFFFF` (`uint16`) for "attend" and `0x0000` for "masked".
   - Therefore, the `UFIXED_POINT_16` override for `attention_mask` **must map `0xFFFF` to `0.0`**:
     `offset = -65535`, `scale = |floor| / 65535` (e.g., for floor `-100`, `scale = 100 / 65535`). A symmetric `[-100, +100]` range turns "attend" into `+100` and destroys attention silently.
   - Mask layout splits `CL` into `[0, CL-AR)` (left-aligned history KV) and `[CL-AR, CL)` (current `AR` tokens).
2. **RoPE Pairing & Shared Encoding**:
   - Genie computes `inv_freq[j] = theta^(-j/rope_dim)` for `j = 0..rope_dim-1` (width `head_dim / 2`). For HuggingFace LLaMA/Qwen weights, broadcast to `head_dim` inside the graph via **split-half** `concat([t, t], -1)` (not interleaved).
   - `position_ids_cos` and `position_ids_sin` **must share the exact same quantization encoding** (Genie reads `cos` quant params and reuses them for `sin`).

### C. Graph Naming, Variants, Sharding & Multimodal
- **Graph Name**: Name graphs `ar<N>_cl<M>_1_of_1`.
- **AR Variants**: Use `{ar128, ar1}` with weight sharing. Genie picks one AR variant per prompt and pads non-multiple remainders on that same graph (e.g., 300 tokens runs as `128 + 128 + 44(padded on ar128)`), so intermediate `ar32` variants do not speed up long prompts.
- **Multi-Shard Contexts (`ctx-bins`)**: Genie orders shards by **lexicographic sort** of graph names. Always **zero-pad** shard indices (`_01_of_16`, `_02_of_16`) when $\ge 10$ shards exist so `_10` does not sort before `_2`.
- **Config Keys**: Always set `"version": 1` in every config block, set `"kv-dim": <head_dim>` explicitly, keep `"use-mmap": true`, and omit `"spill-fill-bufsize"` when using a single `.bin` file.
- **Multimodal (`genie-app`) & LUT (`-t`)**:
  - `genie-app` connects `image-encoder` (input name must be `pixel_values`) + `lut-encoder` $\rightarrow$ `text-generator`. Image/text interleaving is controlled purely by the execution order of `node set text` / `node set image` commands, and Genie automatically requantizes vision embeddings to the LLM's input encoding.
  - Embedding LUT files (`-t`) are headerless row-major `[n_vocab][embed_dim]` binary dumps.