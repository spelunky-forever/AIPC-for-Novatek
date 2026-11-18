# In-Memory Operator Patching

Replace unsupported operators (`Einsum`, `Mod`, `Floor`, `Cast`, `ScatterND`, `GridSample`) with HTP-compatible equivalents **without modifying installed library source code**.

> **Hard Rule**: Never fall back to CPU (`libQnnCpu.so`) as a workaround for unsupported operators. Patch until all ops compile on HTP, or escalate if no semantic-preserving replacement exists.

---

## 1. Approach Selection Decision Tree

1. **Export-Time / Custom Symbolic Handler (Highest Success)**:
   - Register `torch.onnx.register_custom_op_symbolic()` or adjust export wrapper inputs before `torch.onnx.export()`.
   - *Best for*: HuggingFace decoder causal masks (`Cast(BOOL_8 -> FLOAT_16)`), `torch.einsum`, `torch.mod`.
2. **In-Memory `nn.Module` Patching (High Success)**:
   - Override `module.forward = types.MethodType(patched_forward, module)` on model instances before export. Never edit `site-packages`.
3. **ONNX Graph Surgery (Fallback)**:
   - Modify nodes directly via `onnx.helper` when PyTorch source is unavailable.

---

## 2. Type-First Pattern Matrix

Always inspect producer tensor dtypes first (`INT64` vs `FLOAT32`)—keeping integer ops in `INT` avoids `Cast` type-chain bugs.

| Operator | Input Types | Replacement Pattern | Priority | Notes / Error Signatures |
| :--- | :--- | :--- | :---: | :--- |
| **Mod** | `INT64/32, INT64/32` | `Sub(a, Mul(b, Div(a, b)))` | ★★★★★ | Stays in INT domain; no `Cast` needed. |
| **Mod** | `FLOAT, FLOAT` | `Sub(a, Mul(b, Floor(Div(a, b))))` | ★★ | `Floor` on float may also fail. |
| **Mod** | `FLOAT, CONST(int)` | `Sub(a, Mul(b, Cast(Cast(Div(a,b), INT32), FLOAT)))` | ★★ | Add `Add(0.0)` after final `Cast` if `Tensor mismatch 0x32 != 0x216` occurs. |
| **Floor** | `INT64/32` | `Identity(x)` (or rewire directly) | ★★★★★ | Floor of integer is a no-op. |
| **Floor** | `FLOAT` | `Cast(Cast(x, INT32), FLOAT)` | ★★ | May introduce downstream type-inference issues. |
| **Ceil / Round** | `FLOAT` | `Neg(Floor(Neg(x)))` / `Floor(Add(x, 0.5))` | ★★ | Inherits `Floor` float caveats. |
| **Cast** | `BOOL_8 -> FLOAT_16` | **Fix at export**: pass `attention_mask=None` in HF decoders | ★★★★★ | HTP hard-rejects with `validateOpConfig failed 3110` during context-bin compile. |
| **Einsum** | `FLOAT` | Decompose to `Permute` + `Reshape` + `MatMul` | ★★★★ | See decomposition templates below. |
| **ScatterND** | `FLOAT` (non-overlap) | `Gather` $\rightarrow$ `Where` mask $\rightarrow$ `Add` | ★★★★ | Overlapping indices require escalation. |
| **GridSample** | `FLOAT` (bilinear) | `AffineGrid` generator + `Resize(bilinear)` | ★★ | Nearest/bicubic modes usually require model change. |
| **MaxPool** | `dilation=[1,1]` | **Do NOT patch** | ★★★★★ | `dilations: unsupported in Converter` is a **benign warning**; conversion exits 0. |

---

## 3. Concrete Patch Templates

### A. HuggingFace Decoder `Cast(BOOL_8 -> FLOAT_16)` (HTP Error `3110`)
`qnn-onnx-converter` accepts `Cast(BOOL -> FLOAT)`, but `qnn-context-binary-generator` fails on HTP with `validateOpConfig failed 3110`. Fix at export time by passing `attention_mask=None` so the model builds a pure float causal mask internally:
```python
outputs = model(input_ids=input_ids, attention_mask=None, position_ids=position_ids, use_cache=False)
```

### B. Einsum $\rightarrow$ MatMul Decompositions
- **`bij,bjk->bik`**: `torch.matmul(A, B)`
- **`bhij,bhjk->bhik`**: `torch.matmul(A.reshape(b*h, i, j), B.reshape(b*h, j, k)).reshape(b, h, i, k)`
- **`bmchw,bnmc->bmhwn`** (5D Attention):
  ```python
  embed_r = embed.permute(0, 1, 3, 4, 2).reshape(bs * nh, h * w, hc)
  guide_r = guide.permute(0, 2, 3, 1).reshape(bs * nh, hc, n)
  aw = torch.matmul(embed_r, guide_r).view(bs, nh, h, w, n)
  ```
- **`bchw,bkc->bkhw`** (4D Contrastive Head):
  ```python
  x_r = x.permute(0, 2, 3, 1).reshape(bs, h * w, c)
  out = torch.matmul(x_r, w.transpose(1, 2)).permute(0, 2, 1).reshape(bs, k, h, w)
  ```

### C. In-Memory Module Replacement & ONNX INT64 `Mod` Surgery
```python
# In-memory module replacement
import types
for name, module in model.named_modules():
    if isinstance(module, TargetLayerClass):
        module.forward = types.MethodType(patched_forward, module)

# ONNX Surgery for INT64 Mod: output = a - b * (a / b)
from onnx import helper
div_n = helper.make_node("Div", [in_a, in_b], ["mod_div"])
mul_n = helper.make_node("Mul", [in_b, "mod_div"], ["mod_mul"])
sub_n = helper.make_node("Sub", [in_a, "mod_mul"], [out_name])
```

---

## 4. Mandatory Validation Gates & Escalation

Run all 4 gates after each patch iteration:
1. **Gate 1 (Structure)**: `python -c "import onnx; onnx.checker.check_model('patched.onnx')"`
2. **Gate 2 (Dry-Run)**: `$QAIRT_SDK_ROOT/bin/x86_64-linux-clang/qnn-onnx-converter --input_network patched.onnx --dry_run`
3. **Gate 3 (Numerical Parity)**: Compare ONNX CPU outputs before vs. after patch across **all output channels** (Cosine $\ge 0.99$ for FP, $\ge 0.95$ for INT).
4. **Gate 4 (Full Compile)**: Run `qai_convert_fp.py` + `qnn-context-binary-generator`.

**Stop & Escalate to User when**:
- No replacement pattern exists for an unsupported operator.
- The only viable replacement alters model semantics.
- 7+ patch iterations fail to make progress on the same operator.