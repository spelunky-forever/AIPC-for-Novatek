# Model Export, Conversion & Quantization (Linux x86)

Covers ONNX export/validation, QNN/SNPE conversion, and standard QAIRT quantization (`FP16/FP32/INT8/A16W8/INT4`). For advanced AIMET PTQ, see `references/model_quantization_aimet.md`.

- **Default Unified Entrypoint**: Use `python scripts/aipc.py convert --model model.onnx ...` (wraps `run_pipeline.py` to run ONNX $\rightarrow$ DLC $\rightarrow$ Quantized DLC $\rightarrow$ Context Binary `.bin` end-to-end).
- **Direct Wrapper Scripts**: Use `qai_convert_fp.py` / `qai_convert_int.py` only when you specifically need the `.cpp`/`.bin`/`.so` shared-library flow, or `qai_convert_snpe.py` for standalone `.dlc` export.

---

## 1. Linux x86 Preflight

1. **Activate Environment**: Run your container's `source /home/spelunky-forever/workplace/AIPC-for-Novatek/env_setup.sh` and verify `$QAIRT_SDK_ROOT` and `cmake` (`which cmake`).
2. **Disk & Temp Space (`QAIRT_TMP_DIR`)**:
   - Reserve **≥ 3× FP16 model size** in your workspace (`.bin` + `.so` + `.so.bin` coexist during build).
   - `qnn-onnx-converter` copies external-data weights to `/tmp` before processing. If `/tmp` is a small `tmpfs`, set `QAIRT_TMP_DIR` to a volume with **≥ 2× model weight size** free:
     ```bash
     df -h /tmp .
     export QAIRT_TMP_DIR=/home/spelunky-forever/workplace/AIPC-for-Novatek/qairt_tmp
     mkdir -p "$QAIRT_TMP_DIR"
     ```
3. **ARM Linux Cross-Compile Toolchain** *(only needed when `--target-arch aarch64-ubuntu-gcc9.4`)*:
   ```bash
   # Requires: sudo apt install g++-aarch64-linux-gnu
   export QNN_AARCH64_UBUNTU_GCC_94=/   # literal "/" is required; resolves to //usr/bin/aarch64-linux-gnu-g++
   which aarch64-linux-gnu-g++
   ```

---

## 2. Export to ONNX & Validation

> **Transformer Decoders / Multi-Component Pipelines**: Use `references/llm_genie.md` (prefill/decode KV-cache split) or `references/complex_models.md` (`diffusers` sub-models) instead of a single generic export.

### Export Rules
- **Always export in FP32**: PyTorch CPU lacks FP16 `Conv2d`; let QAIRT apply FP16/INT quantization later. (BF16 ONNX is experimental—validate end-to-end if used.)
- **Opset 18 Recommended**: Best QNN compatibility on PyTorch 2.x. Avoid opset 17 (`Resize` conversion bugs). Benign warnings like `WARNING_OP_VERSION_NOT_SUPPORTED` on `LeakyRelu` can be ignored.
- **Large Model Speed & Memory**:
  - Set `do_constant_folding=False` (saves 15–30 min and peak RAM; QAIRT runs its own graph optimizations).
  - Call `model.eval()`, disable training-only auxiliary branches (`model.aux_logits = False`, `model.AuxLogits = None`), and `del ckpt, weights` after `load_state_dict()`.
- **PyTorch ≥ 2.9 Exporter (`dynamo=False`)**: PyTorch 2.9+ defaults to the `torch.export` exporter, which splits large weights into `<model>.onnx` + `<model>.onnx.data` and fails on `*args` forward signatures (e.g., KV-cache decoders: `treespec.unflatten` error) or `diffusers` modules. Pass `dynamo=False` to use the legacy single-file TorchScript exporter when needed.

```python
import warnings, torch

del ckpt, weights
model.eval()
dummy = torch.zeros(1, 3, H, W, dtype=torch.float32)

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="The shape inference of prim::Constant type is missing")
    torch.onnx.export(
        model, dummy, "model.onnx",
        opset_version=18,
        input_names=["input"],
        output_names=["output"],
        do_constant_folding=False,
        # dynamo=False,  # Uncomment for *args models, diffusers, or single-file .onnx output
    )
```

### Validation & External Data Caveats
1. **Large Models (> 2 GB)**: Always pass the **file path string** to `onnx.checker.check_model("model.onnx")`. Calling `onnx.load()` first raises `ValueError: protobuf ... is too large (>2GB)`.
2. **External Data Sibling (`.onnx` + `.onnx.data`)**: Keep both files in the exact same directory.
3. **`onnxsim` Safety Wrapper**: `onnxsim.simplify()` can fail on external-data models (`ir_version` validation error). Wrap in `try/except` and never block conversion if `onnx.checker` passes:
   ```python
   import onnx, onnxsim
   onnx.checker.check_model("model.onnx")
   try:
       m_sim, ok = onnxsim.simplify(onnx.load("model.onnx"))
       if ok:
           onnx.save(m_sim, "model.onnx")
   except Exception as e:
       print(f"onnxsim skipped: {e}")
   ```
4. **Numerical Parity**: Compare `ort.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])` against PyTorch baseline before converting.

---

## 3. Flow A: QNN Conversion (`qai_convert_fp.py` / `qai_convert_int.py`)

Prefer the skill wrapper scripts over raw `qnn-onnx-converter` + `qnn-model-lib-generator`.

### Target Architecture (`--target-arch`)
- `x86_64-linux-clang` *(default on Linux x86)*: Use for x86 host testing **and** when generating a host-side HTP Context Binary (`.so.bin`) for an ARM target.
- `aarch64-ubuntu-gcc9.4`: Use only when cross-compiling a `.so` to run directly on ARM Linux without a context binary.

### FP16 / FP32 Conversion
```bash
python scripts/qai_convert_fp.py \
  --onnx model.onnx \
  --output-root qnn_output \
  --precision 16 \
  --preserve-io-mode datatype \
  --target-arch x86_64-linux-clang
```
- **Preserve-IO Policy**: `--preserve-io-mode datatype` emits `--preserve_io` (preserves both layout and datatype). If target HTP rejects FP16 I/O, retry with `--preserve-io-mode layout`. Never use `none` in initial bring-up.
- **Dynamic Inputs**: NPU requires static shapes. Pass `--input-dim input_name,1,3,224,224` (repeat flag for multiple inputs). Note: direct `qnn-onnx-converter` uses space-separated `--input_dim input_name 1,3,224,224`, while the wrapper uses comma-separated `--input-dim input_name,1,3,224,224`.
- **Outputs**: `model.cpp`, `model.bin`, `model_net.json`, and `libmodel.so`.

### Quantized Conversion (`INT8` / `A16W8` / `INT4`)
1. **Prepare Calibration `.raw` & List**:
   - Convert 50–200 diverse, real samples using the **exact** inference preprocessing into float32 `.raw` files (`arr.astype(np.float32).tofile(...)`).
   - **`qai_convert_int.py` requires the `input:=` prefix** in `calibration_list.txt`:
     ```text
     input:=calibration_raw/input_0000.raw
     input:=calibration_raw/input_0001.raw
     ```
2. **Run Quantization**:
   ```bash
   python scripts/qai_convert_int.py \
     --input_network model.onnx \
     --input_list calibration_list.txt \
     --output-root qnn_output \
     --act_bw 16 --weight_bw 8 \
     --preserve-io-mode datatype \
     --target-arch x86_64-linux-clang
   ```
   - Common presets: **A16W8** (`--act_bw 16 --weight_bw 8`, recommended for vision), **A8W8** (`--act_bw 8 --weight_bw 8`), **A16W4** (`--act_bw 16 --weight_bw 4`), **A8W4** (`--act_bw 8 --weight_bw 4`).

---

## 4. Flow B: SNPE DLC Conversion & Quantization

`qairt-converter` does **not** support inline quantization—quantizing a DLC is always a two-step process.

### Step 1: ONNX to FP DLC
```bash
python scripts/qai_convert_snpe.py \
  --onnx model.onnx \
  --output snpe_output/model.dlc \
  --precision 16
# Add --source-model-input-shape images 1,3,640,640 if ONNX has dynamic inputs
```
*(Note: `qairt-converter` stores FP32 weights inside the DLC; FP16 is applied on HTP at runtime/context-binary generation.)*

### Step 2: Quantize DLC (`qairt-quantizer`)
> Use `qairt-quantizer` (do not use legacy `snpe-dlc-quant` unless on an old SDK lacking `qairt-quantizer`).

1. **Calibration List Format (differs from `qai_convert_int.py`)**:
   - **Single-input**: Plain file paths, one per line (`calib/s0.raw`).
   - **Multi-input**: All named inputs for one sample on the **same line** (`name_a:=calib/s0_a.raw name_b:=calib/s0_b.raw`). Never rely on positional order for >3 inputs.
2. **Run Quantizer**:
   ```bash
   qairt-quantizer \
     --input_dlc snpe_output/model.dlc \
     --output_dlc snpe_output/model_w8a8.dlc \
     --input_list calib_list_plain.txt \
     --param_quantizer tf \
     --act_quantizer tf \
     --act_bitwidth 8 \
     --weights_bitwidth 8
   # Optional: add --algorithms cle (Cross-Layer Equalization) if accuracy drops
   ```

---

## 5. Tool Flag Mapping Cheat Sheet

| Intent | `run_pipeline.py` | `qai_convert_int.py` (QNN) | `qairt-quantizer` (SNPE/DLC) |
| :--- | :--- | :--- | :--- |
| **Activation BW** | `--act_bw <4\|8\|16>` | `--act_bw <8\|16>` | `--act_bitwidth <8\|16>` |
| **Weight BW** | `--weight_bw <4\|8\|16>` | `--weight_bw <4\|8>` | `--weights_bitwidth <4\|8>` *(plural `s`)* |
| **Bias BW** | `--bias_bw <8\|32>` | `--bias_bw <8>` | `--bias_bitwidth <8\|32>` |
| **Calib List** | `--calib_list <file>` | `--input_list <file>` (`input:=` prefix) | `--input_list <file>` (plain or `name:=`) |
| **CLE** | `--cle` | — *(use AIMET)* | `--algorithms cle` |

---

## 6. Quantization Pitfalls & Troubleshooting

- **Large-Dynamic-Range Output Trap (YOLO / Detection)**:
  - *Symptom*: Overall cosine similarity is > 0.999, yet class score channels are all zeros after W8A8.
  - *Cause*: QNN assigns one shared quantization scale per output tensor; large bbox coordinates (0–640) dominate the scale and crush small class scores (0–1) to zero.
  - *Fix*: Always inspect min/max and cosine **per output channel/sub-task**. Resolve by (1) removing the final `Concat` so boxes and scores get separate tensors/scales, (2) switching to **A16W8** (`--act_bw 16 --weight_bw 8`), or (3) using ≥ 50 diverse multi-class calibration images.
- **Calibration Diversity**: Cropping/flipping a single image does **not** provide diverse activation ranges. When searching for local calibration images, scan only explicit project/dataset directories—never run recursive whole-disk `**` globs.
- **Disk Full During `libgen`**: `qnn-model-lib-generator` runs `objcopy` to embed `.bin` into `.so`, requiring ~2× `.bin` size free in `QAIRT_TMP_DIR`. Clean up stale `.cpp` and `.bin` intermediates once `.so.bin` is validated.