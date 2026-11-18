# Multi-Component Pipelines & Large Model Splitting

Covers:
1. **Multi-Component Pipelines** (Stable Diffusion, Whisper, CLIP) where sub-models are exported and converted separately.
2. **Oversized Single Models** (> ~800 MB) that hit DSP SMMU mapping limits and require model splitting.

---

## Part 1: Multi-Component Pipelines (Diffusion / Whisper / CLIP)

### 1. Sub-Model Extraction & Export
- **Config Pattern**: Track `COMPONENTS` (e.g., `text_encoder, unet, vae_decoder`) and export each to `{ONNX_DIR}/{COMP}.onnx`.
- **`diffusers` Rules**:
  - Pipeline objects are not `nn.Module`—call `.eval()` on each sub-model (`pipe.unet.eval()`, `pipe.vae.eval()`).
  - Export VAE decoder via a wrapper that divides `latent_sample / 0.18215` before `vae.decode()`.
  - Always pass **`dynamo=False`** to `torch.onnx.export()` for `diffusers` sub-models.
- **Per-Component Conversion & Context Binary**: Run `qai_convert_fp.py`/`qai_convert_int.py` and `qnn-context-binary-generator` in a loop per component, matching `graph_names` in each `.conf` to `{COMP}`.

### 2. Target Disk Optimization (0-Byte ONNX Stubs)
`onnxwrapper` only needs `<comp>.onnx` on the target board to resolve filenames—weights are loaded from `<comp>.onnx.so.bin`. To save GBs of storage on ARM Linux boards, deploy only:
- `<comp>.onnx` (created via `touch <comp>.onnx` as a 0-byte stub)
- `<comp>.onnx.so.bin` (compiled context binary)
- `<comp>.yaml` (required for output tensor reordering)

### 3. Pipeline Inference & Diffusion Pitfalls
- **Sequential Batch=1 Passes**: For Classifier-Free Guidance (CFG) on a `batch=1` UNet, run unconditional and conditional passes sequentially (`noise_neg + scale * (noise_pos - noise_neg)`) rather than concatenating to `batch=2`.
- **Scheduler Execution**: Either run the `diffusers` scheduler on host CPU, or serialize scheduler arrays (`alphas_cumprod`, `timesteps`) to `.npy` and validate the NumPy scheduler step-by-step (`diff < 1e-5` across **all** steps, as small errors compound).
- **Doubled Output Channels**: For models like PixArt-Sigma / DiT where `out_channels == 2 * in_channels`, slice the first half (`noise_pred.chunk(2, dim=1)[0]`).
- **Latent Sanity Check**: After denoising, `latents.std()` should be `~0.5–1.0` (`abs(max) < 5`). If `std > 5` or output looks like static, verify scheduler class, integer timesteps from `sched.set_timesteps(N)`, and CFG formula.
- **Linux ARM Safe-Exit Guard**: AppBuilder teardown can segfault after inference finishes on ARM Linux. Flush stdout and save all outputs first, then exit cleanly with `os._exit(0)` (e.g., guarded by `AIPC_SAFE_EXIT=1`).

---

## Part 2: Splitting Oversized Models (DSP SMMU Limit)

### 1. Diagnosis: SMMU Map Failure vs. VTCM Error
When a single context binary exceeds the DSP SMMU contiguous mapping window (typically ~512 MB–1 GB on embedded QCS/SA boards), target loading fails with:
```text
fastrpc memory map for fd: N with length: ... failed with error: 0x1
Failed to initialize graph ... with err 1002
Context create from binary failed ..., err 1002
```
- **VTCM sweep (`vtcm_mb`) does NOT fix `err 1002` / SMMU failures**.
- *(Note: On some SoCs such as QCS9075, SMMU warnings may be logged at load time yet inference still succeeds—always verify whether inference actually fails before splitting.)*

### 2. Resolution Options (In Order of Effort)
1. **Immediate Bring-Up Fallback — JIT `.so`**: Remove `.so.bin` and let `onnxwrapper` load the ARM `.so` directly. On-device JIT compiles incrementally in small chunks without a single giant `fastrpc_mmap` (slower cold start, identical numerics).
2. **INT8 Quantization**: Reduces binary size by ~4× vs FP32, often fitting under the SMMU limit without splitting.
3. **Graph Splitting (Production Fixed-SoC)**: Partition the model into sequential `PartA` and `PartB` modules (< 800 MB each).

### 3. Safe Model Splitting Rules & FP16 Boundary Trap
- **Choose a Clean, Balanced Cut**: Inspect cumulative parameter counts per top-level child module and cut near the ~50% mark at a clean intermediate boundary. Assign **only** needed sub-modules to `PartA` and `PartB` (never hold a reference to the full model inside either wrapper).
- **CRITICAL — Boundary Tensor Magnitude (`abs(max) < 50`)**:
  - When `PartA` and `PartB` run in FP16, large activations crossing the split boundary (`std > 5` or range outside `[-50, 50]`, such as UNet `up_blocks` skip residuals) cause severe FP16 accumulation error and output `std` collapse in `PartB`.
  - **Rule**: Always check `out_a.std()` and `[min, max]` before exporting. For UNets, split after `down_blocks + mid_block` (bottleneck) rather than inside `up_blocks`.
  - **Advanced Fix (`--custom_io`)**: Force boundary tensors to `float32` via `--custom_io custom_io.yaml` (must explicitly declare `Layout: Model: NCHW / Custom: NCHW` to avoid silent NHWC transpose).
- **Trace & Validate**: Run `PartA` with `torch.no_grad()` to get exact boundary tensor shapes for `PartB` dummy inputs. Deploy `<part>.onnx`, `<part>.onnx.so.bin`, and `<part>.yaml` for every part, and verify chained HTP output Cosine $\ge 0.99$ (FP16) / $\ge 0.95$ (INT8) against the unsplit ONNX CPU baseline.