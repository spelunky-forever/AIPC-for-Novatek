# ACCURACY_REPORT — MobileNetV2 QNN Bring-up (x86 Linux, CPU)

| Item | Value |
|---|---|
| Report path | `ACCURACY_REPORT.md` (this file) |
| Validation status | ✅ **PASS** |
| Main raw tensor metric | cosine = **1.00000000**, SNR = **115.25 dB** (QNN vs PyTorch golden, CPU path) |
| Task metric | Top-1 index match (92 == 92), Top-5 sets identical → **0% accuracy drop** |
| HTP-context path (2026-09-14) | cosine = 1.00000000 vs golden via `mobilenet_v2.onnx.so.bin` + `libQnnHtp.so` on x86 HTP simulator (see REPORT §9) |
| Profiling summary | n/a — CPU-sim bring-up; latency: mean 62.4 ms / 16.0 FPS (informational) |
| Debugger backup used | **none** (AIPC wrapper acceptance only, per Phase R.4) |
| Debugger artifacts | none |
| Direct model-library fallback | N/A (no context binary; `.so` used directly on x86 Linux CPU) |

---

## 1. Source of Truth

- Model: `torchvision.models.mobilenet_v2(weights=DEFAULT)` — ImageNet-1k pretrained, FP32, eval mode.
- Bring-up reference: deterministic seed-42 input `input.npy` `[1,3,224,224]` fp32;
  PyTorch golden `golden_output.npy` `(1,1000)`, finite, 1000 nonzero.
- Flow: QNN (Flow A), precision FP32, target **x86 Linux** (`x86_64-linux-clang`), runtime **QNN CPU** (`libQnnCpu.so`).

## 2. Selected Deployed Artifact

| Artifact | Path |
|---|---|
| QNN model library | `qairt_output/test_libs_mobilenet_v2_fp32_x86_64-linux-clang/x86_64-linux-clang/libmobilenet_v2.so` (ELF x86-64, 14,322,248 B) |
| Wrapper flavor | `onnxwrapper.py` **= skill `onnxwrapper_x86.py`** (x86 Linux CPU simulation; no `qai_appbuilder`) |
| Inference launcher | `python aipc infer_mobilenet_v2.py` (skill `aipc` hot-patch) |

## 3. Raw Tensor Metrics (QNN vs references)

| Comparison | Cosine | SNR (dB) | MAE | Max abs diff |
|---|---|---|---|---|
| QNN vs PyTorch golden | 1.00000000 | 115.25 | 1.00e-06 | 5.84e-06 |
| QNN vs ONNX CPU baseline | 1.00000000 | 118.14 | 7.26e-07 | 3.58e-06 |
| ONNX CPU baseline vs golden | 1.00000000 | 113.00 | 1.37e-06 | 6.23e-06 |

Thresholds: user acceptance cosine ≥ 0.95 ✅; AGENTS FP floor cosine ≥ 0.99 ✅;
Phase 7 criteria cosine > 0.995 ✅ and SNR > 30 dB ✅.

## 4. Decoded / Task-specific Comparison

| Check | Golden (PyTorch) | QNN FP32 | Result |
|---|---|---|---|
| Top-1 argmax index | 92 | 92 | ✅ match |
| Top-5 index set | {…} | identical set | ✅ match |
| Task accuracy drop | — | — | ✅ 0% |

(Classification head; no external labeled dataset — reference is the PyTorch golden on the deterministic bring-up input.)

## 5. Performance (informational, not HTP)

| Metric | Value |
|---|---|
| Mean latency (batch 1) | 62.37 ms |
| p50 / p95 | 42.94 / 216.87 ms |
| Throughput | 16.03 FPS |

> x86 `onnxwrapper_x86` spawns `qnn-net-run` per run — subprocess overhead dominates; not HTP acceptance.

## 6. Pass/Fail Conclusion

**✅ PASS.** All raw-tensor and decoded-output checks meet or exceed thresholds. Quantization not
applied (FP32); context binary not applicable (x86 Linux CPU, `CONTEXT_BINARY_GEN = NO`).

## 7. Known Limitations

- Acceptance is on the **x86 Linux CPU / HTP-simulator** paths, not a physical Qualcomm HTP device.
- Latency/FPS are informational: CPU path is subprocess-spawn dominated (62 ms/16 FPS); HTP-sim path
  uses the QEMU simulator (684 ms/1.46 FPS) — neither is hardware HTP performance.
- Accuracy is verified against the PyTorch golden on one deterministic bring-up input
  (no full ImageNet evaluation); task metric = Top-1/Top-5 index match.

## 8. QAIRT Accuracy Debugger

Not used (Phase R.4). Default acceptance remains the AIPC wrapper output from Phase 6/7
(`real_inference_output.txt`, `qnn_output.npy`). No reconversion/debugger ladder was required.