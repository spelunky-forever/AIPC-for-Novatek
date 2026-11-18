# Transformer Decoder PyTorch Modification for NPU Prefill + Decode

Use this guide to adapt a PyTorch/HuggingFace transformer decoder into static-shape **Prefill** and **Decode** wrappers with explicit KV-cache tensors before exporting to ONNX (`references/llm_genie.md`).

---

## 1. Bring-Up Workflow

1. **Discover Model Config**: Read `num_hidden_layers`, `num_attention_heads`, `num_key_value_heads` (defaults to `num_attention_heads` for MHA), `hidden_size`, `head_dim` (`hidden_size // num_attention_heads`), and `vocab_size` from `model.config`.
2. **Start with Fixed-Shape Single-Token Decode**:
   - Bring up `batch = 1`, `decode_seq = 1`, `past_seq = 32` (small fixed window) first.
3. **Implement & Validate `DecodeWrapper` in PyTorch**:
   - Verify logits shape, present KV tensor count (`2 * num_hidden_layers`), shapes, dtypes, finite values, and sliding-window truncation.
4. **Implement & Validate `PrefillWrapper` + Handoff**:
   - Use fixed max prompt length (padded) or fixed buckets (e.g., `32 / 64 / 128 / 256`).
   - Run `PrefillWrapper` once, feed its `present_key_values` into `DecodeWrapper` as `past_key_values`, and verify parity against standard `model.generate()`.

---

## 2. Wrapper Templates (`eval()` + `use_cache=True`)

Never export a full-sequence `use_cache=False` forward pass as the decode model, and never hide `DynamicCache` objects inside the forward boundary.

```python
import torch
from transformers.cache_utils import DynamicCache

def flatten_cache(cache):
    flat = []
    for k, v in cache:
        flat.extend([k, v])
    return tuple(flat)

class PrefillWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model.eval()

    def forward(self, input_ids, position_ids, attention_mask=None):
        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,  # Pass None if BOOL->FP16 Cast triggers HTP 3110
            use_cache=True,
        )
        present = _extract_legacy_cache(outputs.past_key_values, self.model.config.num_hidden_layers)
        return outputs.logits[:, -1, :], *flatten_cache(present)

class DecodeWrapper(torch.nn.Module):
    def __init__(self, model, num_layers):
        super().__init__()
        self.model = model.eval()
        self.num_layers = num_layers

    def forward(self, input_ids, position_ids, *flat_past):
        legacy_cache = tuple(
            (flat_past[2 * i], flat_past[2 * i + 1]) for i in range(self.num_layers)
        )
        cache = DynamicCache.from_legacy_cache(legacy_cache)
        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=cache,
            use_cache=True,
        )
        present = _extract_legacy_cache(outputs.past_key_values, self.num_layers)
        return outputs.logits[:, -1, :], *flatten_cache(present)
```

---

## 3. `DynamicCache` Cross-Version Helper & Fixed-Cache Rules

- **`transformers` Version Compatibility**:
  ```python
  def _extract_legacy_cache(past_kv, num_layers):
      if hasattr(past_kv, "layers"):          # transformers >= 4.47
          return [(past_kv.layers[i].keys, past_kv.layers[i].values) for i in range(num_layers)]
      elif hasattr(past_kv, "key_cache"):     # transformers 4.40 - 4.46
          return [(past_kv.key_cache[i], past_kv.value_cache[i]) for i in range(num_layers)]
      return [past_kv[i] for i in range(num_layers)]  # < 4.40 or legacy tuple
  ```
- **Sliding-Window KV Truncation**:
  - When `decode_seq = 1` appends to `past_seq`, `present` has length `past_seq + 1`. Truncate before feeding the next decode step:
    ```python
    next_past = present[:, :, -past_seq:, :]
    ```
- **Tensor Ordering**: Maintain `layer0.key, layer0.value, layer1.key, layer1.value, ...` identically across export, runtime feeding, and validation.