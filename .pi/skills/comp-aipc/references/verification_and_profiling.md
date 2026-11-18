# Validation, Profiling & Layout Optimization (Linux)

Covers numerical/task accuracy validation, HTP/SNPE per-layer profiling, and optional QNN `--preserve-io-mode none` layout optimization.

---

## 1. Accuracy Validation & Pass Thresholds

Reuse already-deployed artifacts (`.onnx.so.bin`, `.so`, or `.dlc`) via `python scripts/qai`—never proactively reconvert just to check accuracy.

### Acceptance Thresholds
| Precision / Model Type | Raw Tensor Cosine | SNR | Decoded / Task Requirement |
| :--- | :---: | :---: | :--- |
| **FP32 / FP16 / BF16** | $\ge 0.99$ | $\ge 30\text{ dB}$ | Decoded labels/boxes/transcripts match baseline |
| **INT8 / A16W8 / INT4** | $\ge 0.95$ | $\ge 20\text{ dB}$ | Task metric drop $\le 1\%$ vs FP32 baseline |
| **Embedding** | $\ge 0.999$ | $\ge 30\text{ dB}$ | Retrieval/similarity rankings remain stable |
| **LLM / Decoder** | Check NaN/Inf first | — | First $\ge 10$ greedy tokens match baseline; coherent text |

### Quick Cosine & SNR Comparison Script
Run ONNX CPU baseline (`CPUExecutionProvider`) in **Process A** and `python scripts/qai` in **Process B**, saving `.npy` tensors by output name:
```python
import json
from pathlib import Path
import numpy as np

metrics = {}
for ref_p in sorted(Path("accuracy_check/baseline_outputs").glob("*.npy")):
    out_p = Path("accuracy_check/target_outputs") / ref_p.name
    ref = np.load(ref_p).astype(np.float64).reshape(-1)
    out = np.load(out_p).astype(np.float64).reshape(-1)
    diff = out - ref
    metrics[ref_p.stem] = {
        "cosine": float(np.dot(ref, out) / (np.linalg.norm(ref) * np.linalg.norm(out) + 1e-12)),
        "snr_db": float(20 * np.log10((np.linalg.norm(ref) + 1e-12) / (np.linalg.norm(diff) + 1e-12))),
        "max_abs_diff": float(np.max(np.abs(diff))),
        "nan_or_inf": bool((~np.isfinite(out)).any()),
    }
print(json.dumps(metrics, indent=2))
```

### Accuracy Drop Triage & Mitigation
1. **Inspect Raw Logits/Tensors First**: All `NaN`/`-Inf` = FP16 overflow (`Softmax`, `LayerNorm`/`RMSNorm`, `MatMul`); normal floats but scrambled text = output tensor order (missing `<model>.yaml`) or NCHW/NHWC mismatch.
2. **QAIRT Accuracy Debugger Ladder** *(use an isolated venv if extra packages are needed)*:
   - `qairt-accuracy-debugger verification --reference_tensor <dir1> --inference_tensor <dir2> --comparators cosine snr mse mae`
   - `qairt-accuracy-debugger snooping` (`oneshot` / `layerwise` / `cumulative_layerwise`) to pinpoint the exact layer where cosine drops off a cliff.
   - For AIMET `.encodings`: run `validate_encoding` and `compare_encodings` to verify scale handoff.
3. **Mitigation Strategies (Best to Fallback)**:
   - **FP32 Accumulation (Preferred, ~0 latency penalty)**: Set `"accum_precision": "fp32"` on sensitive `MatMul` nodes in HTP/converter config.
   - **Mixed Precision / AIMET**: Keep sensitive norms/heads in FP16/INT16 while quantizing backbone to INT8.
   - **Power-of-Two Input Scaling**: Scale overflow-prone `Y = X @ W` by $S = 2^n \approx \max(\vert{}X\vert{}) / 10000$ (`X/S` and absorb `W*S` offline).
   - **Local CPU Fallback**: Force only 1–2 extreme outlier ops to FP32 CPU (avoid frequent HTP$\leftrightarrow$CPU hops).

---

## 2. Per-Layer Profiling (QNN HTP & SNPE)

> **Directory Isolation Rule**: Save all profiling artifacts into a dedicated folder named `qairt_profile_{layout}/` (e.g., `qairt_profile_datatype/` or `qairt_profile_none/`). **Never** use profiling-enabled binaries or `enable_profiling=True` when measuring production FPS/latency.

### QNN Profiling (`.so` / `.dlc` / `.bin`)
1. **Pre-step (Context Binary `.bin` only)**: Regenerate the context binary with `--profiling_level detailed --profiling_option optrace` (keep the converter's schematic `model.bin` alongside the compiled context `.bin`).
2. **Collect Trace**: Enable `sess_options.enable_profiling = True` in your script and run via `python scripts/qai` $\rightarrow$ writes `qairt_profile_output/qnn-profiling-data_0.log`.
3. **Parse Logs (`qnn-profile-viewer`)**:
   - **HTP Cycle & Per-Layer Stats (Most Reliable)**:
     ```bash
     $QAIRT_SDK_ROOT/bin/x86_64-linux-clang/qnn-profile-viewer \
       --reader $QAIRT_SDK_ROOT/lib/x86_64-linux-clang/libQnnHtpProfilingReader.so \
       --input_log qairt_profile_output/qnn-profiling-data_0.log \
       --output htp_stats.json | tee htp_stats.txt
     ```
   - **QHAS Interactive HTML + Chrome Trace** (requires converter schematic `model.bin`):
     ```bash
     $QAIRT_SDK_ROOT/bin/x86_64-linux-clang/qnn-profile-viewer \
       --reader $QAIRT_SDK_ROOT/lib/x86_64-linux-clang/libQnnHtpOptraceProfilingReader.so \
       --input_log qairt_profile_output/qnn-profiling-data_0.log \
       --schematic model.bin \
       --output qairt_profile_output/chromeTrace_optrace.json
     # Generates chromeTrace_optrace_qnn_htp_analysis_summary.html (view traces in chrome://tracing)
     ```
   - *(If `libQnnChrometraceProfilingReader.so` fails with `Error printing stats.`, use `libQnnJsonProfilingReader.so` or default CSV output instead.)*

### SNPE Profiling (`.dlc`)
```bash
# Use QAI_SNPE_PROFILING_LEVEL=detailed for per-layer text stats, or =linting for --chrometrace
QAI_SNPE_PROFILING_LEVEL=detailed python scripts/qai infer_model.py
$QAIRT_SDK_ROOT/bin/x86_64-linux-clang/snpe-diagview \
  --input_log snpe_output/SNPEDiag_0.log \
  --output snpe_profile.csv | tee snpe-diagview_stdout.txt
```

### Key Bottlenecks to Check
- Any op executing on **CPU fallback** instead of HTP/DSP.
- High DDR $\leftrightarrow$ VTCM spill traffic (solve by sweeping `vtcm_mb` up to 8 MB).
- Expensive edge layout/dtype conversions (solve with Section 3 below).

---

## 3. QNN Layout Optimization (`--preserve-io-mode none`)

Only apply **after** baseline `--preserve-io-mode datatype` validation passes (`FLOW=QNN` only).
- **Benefit**: Removes edge `Transpose` / `Cast` nodes injected to preserve ONNX NCHW/FP32 I/O, typically reducing HTP latency by ~10–15%.
- **Trade-off**: Model inputs/outputs shift to HTP native layout (e.g., `NHWC`) or native quantized dtypes.
- **Integration**: Re-convert into `qairt_profile_none/` with `--preserve-io-mode none`, inspect `_net.json`, and configure explicit permutations in your IO YAML (loaded via `QAI_IO_CONFIG`):
  ```yaml
  layout_qairt_none_convert:
    inputs:
      images:
        onnx_to_qnn: [0, 2, 3, 1]
    outputs:
      output0:
        qnn_to_onnx: [0, 1, 2]
  ```
- Measure end-to-end latency with **profiling OFF** (`enable_profiling = False`, non-optrace `.bin`) to verify net speedup over baseline.