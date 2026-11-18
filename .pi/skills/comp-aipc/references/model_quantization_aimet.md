# AIMET-ONNX PTQ Quantization (Linux Only)

Use AIMET when standard QAIRT quantization (`qai_convert_int.py` / `qairt-quantizer`) falls below accuracy thresholds at the same bitwidth. AIMET exports a pre-calibrated `.onnx` + `.encodings` file, which is then handed off to QAIRT via `--quantization_overrides`.

---

## Stage 1: Python PTQ Calibration (`aimet_onnx`)

- **Namespace Guard**: If both `aimet_onnx` and `aimet_torch` are installed, import strictly from `aimet_onnx.*` (avoid `aimet_common`).
- **Sample Count**: 50–200 representative samples matching exact inference preprocessing.

```python
import numpy as np
import onnx
from aimet_onnx.batch_norm_fold import fold_all_batch_norms_to_weight
from aimet_onnx.cross_layer_equalization import equalize_model
from aimet_onnx.common.defs import QuantScheme
from aimet_onnx.quantsim import QuantizationSimModel

model = onnx.load("model.onnx")

# 1. Data-free optimization: BN folding + Cross-Layer Equalization (CLE)
fold_all_batch_norms_to_weight(model)
equalize_model(model)

# 2. Configure QuantSim (example: A16W8 -> param_type="int8", activation_type="int16")
sim = QuantizationSimModel(
    model=model,
    quant_scheme=QuantScheme.post_training_tf_enhanced,
    param_type="int8",
    activation_type="int16",
)

# 3. Compute range encodings
def get_calib_inputs(num_samples=100):
    for _ in range(num_samples):
        yield {"images": np.random.rand(1, 3, 640, 640).astype(np.float32)}  # Replace with real preprocessed inputs

sim.compute_encodings(list(get_calib_inputs(100)))

# 4. Export pre-calibrated ONNX + encodings JSON
sim.export(
    path="runs/aimet_artifacts",
    filename_prefix="model_ptq_calibrated",
    export_model=True,
    encoding_version="1.0.0",
)
```

---

## Stage 2: Compile to QNN `.so` or SNPE `.dlc`

> **Critical**: Do **not** pass `--input_list` or run `qairt-quantizer` after AIMET export—quantization scales are already embedded in `.encodings`.

### Option A: QNN Shared Library
```bash
python scripts/qai_convert_aimet.py \
  --input_network runs/aimet_artifacts/model_ptq_calibrated.onnx \
  --quantization_overrides runs/aimet_artifacts/model_ptq_calibrated.encodings \
  --output-root ./qairt_output \
  --act_bw 16 --weight_bw 8 \
  --flow QNN \
  --target-arch x86_64-linux-clang   # Use x86_64-linux-clang for host context-binary gen, or aarch64-oe-linux-gcc11.2 for direct ARM target
```

### Option B: SNPE DLC
```bash
python scripts/qai_convert_aimet.py \
  --input_network runs/aimet_artifacts/model_ptq_calibrated.onnx \
  --quantization_overrides runs/aimet_artifacts/model_ptq_calibrated.encodings \
  --output-root ./qairt_output \
  --act_bw 8 --weight_bw 8 \
  --flow SNPE
```

### Direct CLI Fallback (if wrapper fails)
- **QNN**: `qnn-onnx-converter --input_network <ptq.onnx> --quantization_overrides <ptq.encodings> -d <input> <shape_csv> --output_path <out.cpp> --act_bw 8 --weight_bw 8 --bias_bw 32 --use_per_channel_quantization` $\rightarrow$ `qnn-model-lib-generator`.
- **SNPE**: `snpe-onnx-to-dlc --input_network <ptq.onnx> --quantization_overrides <ptq.encodings> -d <input> <shape_csv> -o <out.dlc>`.