# REPORT — MobileNetV2 QNN Bring-up (x86 Linux, CPU)

| Field | Value |
|---|---|
| **Project** | `mobilenet_v2_bringup` |
| **Model** | MobileNetV2 (`torchvision.models.mobilenet_v2`, ImageNet-1k pretrained, FP32) |
| **Flow** | QNN (Flow A) |
| **Precision** | FP32 |
| **Target Device** | x86 Linux (`x86_64-linux-clang`) — local acceptance (no `RETMOE_DEVICE_INFO`) |
| **Runtime** | QNN CPU (`libQnnCpu.so`) via skill `aipc` launcher + `onnxwrapper_x86.py` |
| **START_TIME** | `2026-09-05 17:29` |
| **END_TIME** | `2026-09-14 17:24` (incl. HTP context-binary/simulator follow-up) |
| **WORK_TIME** | `8d 23h 55m` |
| **Final Status** | ✅ **PASS** |

---

## 1. Executive Summary

The MobileNetV2 (FP32) model was brought up end-to-end on the **QNN** flow targeting
**x86 Linux CPU**: PyTorch baseline → ONNX export → inspection → QNN conversion → QNN CPU
inference through the skill `aipc` wrapper. All validation thresholds were met:

| Metric | Result | Threshold | Verdict |
|---|---|---|---|
| Cosine similarity (QNN vs PyTorch golden) | **1.00000000** | ≥ 0.95 (user) / ≥ 0.99 (AGENTS FP) | ✅ PASS |
| Cosine similarity (QNN vs ONNX CPU baseline) | **1.00000000** | ≥ 0.99 | ✅ PASS |
| SNR (QNN vs golden) | **115.25 dB** | > 30 dB (Phase 7 criteria) | ✅ PASS |
| MAE / max abs diff (QNN vs golden) | 1.00e-06 / 5.84e-06 | — | ✅ PASS |
| Top-1 class index (QNN vs golden) | 92 == 92 (match) | match | ✅ PASS |
| Top-5 set (QNN vs golden) | identical | match | ✅ PASS |
| Latency (QNN CPU sim, batch 1) | mean **62.4 ms** / p50 42.9 ms / p95 216.9 ms / **16.0 FPS** | recorded (informational) | ✅ |

No quantization was applied (`PRECISION = FP32`); `CONTEXT_BINARY_GEN = NO` and an x86 Linux
CPU target make the context binary path not applicable — the `.so` model library is used
directly (context binary optional on Linux per skill rules).

---

## 2. Phase Completion

| Phase | Description | Status |
|---|---|---|
| 0 | Environment & Prerequisites | ✅ Done |
| 1 | NPU Model Adaptation | ✅ Done (`PYTORCH_ADAPTATION_NEEDED = No` — generic CNN) |
| 2 | Model Export to ONNX | ✅ Done (`mobilenet_v2.onnx`, opset 13, fixed I/O) |
| 3 | Model Inspection | ✅ Done (I/O confirmed; 100% op compatibility verified) |
| QNN-4A | FP32 Conversion | ✅ Done (`libmobilenet_v2.so`, ELF x86-64) |
| QNN-4B/4C | Quantization | ⏭️ Skipped (FP32 — not applicable) |
| QNN-5 | Context Binary Generation | ✅ Done (HTP sim: `SOC_ID=0`, `DSP_ARCH=v73` → `libmobilenet_v2.so.bin` 14.2 MB — §9) |
| QNN-6 | Inference (aipc wrapper) | ✅ Done (CPU + HTP-context paths both PASS — §4.4/§9) |
| 7 | Validation & Testing | ✅ Done (this report) |
| 8 | Profiling | ✅ Done (skipped — CPU sim bring-up; latency benchmark recorded instead, §4.3) |
| R | Accuracy Report | ✅ Done (`ACCURACY_REPORT.md`, linked below) |
| E | Skill Evolution | ⏭️ Skipped (`EVOLVE = NO`) |
---

## 3. Evidence per Phase

### Phase 1 — Baseline (PyTorch)
- Script `baseline_mobilenet_v2.py`: `torchvision.models.mobilenet_v2(weights=DEFAULT)`, eval, seed 42.
- Input: `input.npy` `[1,3,224,224]` float32 (deterministic randn).
- Golden: `golden_output.npy` `(1,1000)` float32 — finite, 1000 nonzero entries.
- Decision: `MODEL_STRUCTURE = generic`, `PYTORCH_ADAPTATION_NEEDED = No`.

### Phase 2 — ONNX Export + Validation
- `export_onnx.py`: opset 13, `do_constant_folding=True`, fixed shapes (`dynamic_axes=None`).
- `onnx.checker.check_model()` ✅; `onnxsim` simplified ✅.
- ONNX Runtime CPU inference vs golden: cosine = **1.00000012** (≥ 0.95 ✅).

### Phase 3 — Inspection
- `aipc_inspect_onnxio.py` → `mobilenet_v2.yaml`; I/O: `input [1,3,224,224]` → `output [1,1000]`.
- Dry-run flagged 35× `Clip: unsupported version` (ReLU6 at opset 13) — **verified benign** by a full
  converter run (`Conversion complete!`): warnings only (`WARNING_OP_VERSION_NOT_SUPPORTED`).
- Ops inventory: Conv 52, Clip 35, Add 10, GlobalAveragePool 1, Flatten 1, Gemm 1 — all QNN-IR compatible.
  → `PATCH_NEEDED = No` confirmed, **100% operator compatibility**.

### QNN-4A — FP32 Conversion
- Skill wrapper: `aipc_convert_fp.py --onnx mobilenet_v2.onnx --output-root qairt_output --precision 32 --preserve-io-mode datatype --target-arch x86_64-linux-clang`
- Result: `Converted: 1, Failed: 0` → `qairt_output/test_libs_mobilenet_v2_fp32_x86_64-linux-clang/x86_64-linux-clang/libmobilenet_v2.so` (14,322,248 B, **ELF x86-64** ✅).

### QNN-5 — Context Binary
- Skipped by config (`CONTEXT_BINARY_GEN = NO`); x86 Linux target → `.so` direct path (optional on Linux).
- **2026-09-14**: `CONTEXT_BINARY_GEN` set to `YES` for HTP simulator validation — see §9.

### QNN-6 — Inference (aipc wrapper)
- Deployed skill `aipc` + `onnxwrapper.py` (= `onnxwrapper_x86.py`, no `qai_appbuilder`; passes launcher platform check).
- `python aipc infer_mobilenet_v2.py` (QNN CPU backend via `libQnnCpu.so`).
- 3/3 acceptance runs: cosine = **1.00000000**, Top-1 = 92 == golden, Top-5 sets identical → PASS.

### Phase 7 — Validation
- §4 below. Full metrics in `metrics_phase7.json` and `qairt_output/latency_benchmark.json`.

---

## 4. Validation Results

### 4.1 Accuracy comparison (Phase 7 / 6.1)
Baseline scripts: `report_metrics_phase7.py` (plain onnxruntime CPU), QNN via `aipc` wrapper.

| Comparison | Cosine | SNR (dB) | MAE | Max abs diff |
|---|---|---|---|---|
| ONNX CPU vs PyTorch golden | 1.00000000 | 113.00 | 1.37e-06 | 6.23e-06 |
| **QNN vs PyTorch golden** | **1.00000000** | **115.25** | **1.00e-06** | **5.84e-06** |
| **QNN vs ONNX CPU** | **1.00000000** | **118.14** | **7.26e-07** | **3.58e-06** |

- Cosine thresholds: **≥ 0.99 (FP)** ✅ and **≥ 0.95 (user acceptance)** ✅
- Phase 7 criteria: cosine > 0.995 ✅, SNR > 30 dB ✅

### 4.2 Task-specific accuracy (Phase 7 / 6.2)
- Metric: Top-1 accuracy (classification, compared against PyTorch golden reference on the bring-up input).
- Baseline (PyTorch golden): argmax index = **92**
- QNN FP32: argmax index = **92** → **exact match, 0% drop** (≤ 1% ✅)
- Top-5 index sets: identical ✅

### 4.3 Latency / throughput (Phase 7 / 6.3) — x86 Linux QNN CPU (simulation)
`python aipc bench_mobilenet_v2.py` — batch 1, 3 warmup + 20 timed runs.

| Metric | Value |
|---|---|
| Mean latency | **62.37 ms** |
| p50 | 42.94 ms |
| p95 | 216.87 ms |
| Min / Max | 38.56 ms / 246.53 ms |
| Throughput | **16.03 FPS** |

> ⚠️ Informational only: the x86 `onnxwrapper_x86` path spawns `qnn-net-run` per run (subprocess
> overhead dominates). This is a CPU **simulation** bring-up, not HTP acceptance.

### 4.4 Regression (Phase 7 / 6.4)
| Test case | Runs | Pass / Fail |
|---|---|---|
| Acceptance `infer_mobilenet_v2.py` (known-good `input.npy`) — CPU backend | 3 | 3 / 0 |
| Benchmark `bench_mobilenet_v2.py` (23 more inferences) — CPU backend | 23 | 23 / 0 |
| ONNX CPU baseline `validate_onnx.py` | 1 | 1 / 0 |
| **HTP-context `infer_mobilenet_v2.py` (simulator, `libQnnHtp.so`)** | 1 | 1 / 0 |
| Diagnostic `qnn-net-run --retrieve_context` (HTP sim) | 1 | 1 / 0 |

---

## 5. Issues, Decisions & Resolutions (summary from aipc_plan.md Issue Log)

| # | Issue | Resolution |
|---|---|---|
| 1 | `QAIRT_ROOT` pointed to non-existent nested path | User fixed to `.../toolchains/qairt/2.45.0.260326/` |
| 2 | Converter C-extension crash: `libpython3.10.so.1.0` missing + `numpy.dtype size changed` (numpy 2.2.6 vs pandas 2.0.1 ABI) | **B2 (user-approved)** `pip install "numpy==1.26.4"` + `LD_LIBRARY_PATH=$CONDA_PREFIX/lib:...` baked into `env_setup.sh` (durable) |
| 3 | Dry-run `Clip: unsupported version` (35× ReLU6) | Verified benign via full conversion — `WARNING_OP_VERSION_NOT_SUPPORTED` only; no patch needed |
| 4 | Missing CMake for `qnn-model-lib-generator` | User installed CMake 3.28.3 |
| 5 | Wrapper CLI uses `--output-root` (not `--output_dir`) | Used the wrapper's documented flag |
| 6 | `CONTEXT_BINARY_GEN = NO` (QNN-5 skip, CPU path) | `.so` direct inference path on x86 Linux — CPU acceptance passed (later superseded for HTP by Issue #8) |
| 7 | QNN-5/QNN-6 HTP context workflow (2026-09-14) | ✔️ Completed — see §9 |

---

## 6. User Prompts & Interventions

**Total user interventions: 9** (8 prompts + 1 interactive decision).

| # | Prompt (abridged) |
|---|---|
| 1 | Inspect `aipc_plan.md` Config, confirm variables, summarize Phase 1–5 plan (no code changes) |
| 2 | [TASK RESUMPTION] Continue where you left off |
| 3 | QAIRT root fixed → Execute Phase 1 & 2 (baseline, export ONNX opset 13, ORT validation, cosine ≥ 0.95) |
| 4 | Execute Phase 3 (inspection script, QAIRT dry-run, record I/O + compatibility) |
| 5 | CMake installed → Execute Phase 4 (skill wrapper FP conversion, verify `.so` arch) |
| 6 | `CONTEXT_BINARY_GEN = NO` → Execute Phase 6 (preflight wrappers, CPU inference, cosine threshold, record, close Phase 5) |
| 7 | Execute Phase 7 (REPORT.md, completion metrics, close bring-up) |
| 8 | Enable HTP context binary generation + validate `.bin` on x86 QNN HTP simulator (config update, QNN-5/6/7 re-run, REPORT update) |
| — | **Decision (ask_question, B2)**: approved `pip install numpy==1.26.4` in conda env `aipc` |
---

## 7. Artifacts

| Artifact | Path |
|---|---|
| Baseline input / golden | `input.npy` / `golden_output.npy` |
| ONNX model | `mobilenet_v2.onnx` (opset 13, simplified) |
| I/O config YAML | `mobilenet_v2.yaml` |
| QNN model library | `qairt_output/test_libs_mobilenet_v2_fp32_x86_64-linux-clang/x86_64-linux-clang/libmobilenet_v2.so` |
| HTP context binary | `qairt_output/libmobilenet_v2.so.bin` + deployed `mobilenet_v2.onnx.so.bin` (ONNX-matching rule) |
| HTP config files | `/tmp/soc0_v73.conf`, `/tmp/soc0_v73.json` |
| HTP-sim output / result | `qnn_output_htp.npy` / `real_inference_output_htp.txt` |
| HTP-sim latency | `qairt_output/latency_benchmark_htp.json` |
| Converter IR evidence | `_dryrun_test/mobilenet_v2.cpp` / `.bin` / `_net.json` + repo-root `mobilenet_v2.cpp/.bin/_net.json` |
| Inference stack | `aipc`, `onnxwrapper.py` (= skill `onnxwrapper_x86.py`), `libmobilenet_v2.so`, `infer_mobilenet_v2.py` |
| QNN output / baseline output | `qnn_output.npy` / `onnx_cpu_output.npy` |
| Acceptance result | `real_inference_output.txt` |
| Metrics | `metrics_phase7.json`, `qairt_output/latency_benchmark.json` |
| Env snapshot | `qairt_output/acceptance_env_snapshot.txt` |
| Accuracy report | [`ACCURACY_REPORT.md`](./ACCURACY_REPORT.md) |

---

## 8. Acceptance Verdict

**✅ PASS** — MobileNetV2 FP32 on QNN (x86 Linux CPU) satisfies all acceptance criteria:
- QNN output matches the PyTorch golden: cosine = 1.00000000 (≥ 0.95 user / ≥ 0.99 FP floor).
- Decoded classification matches: Top-1 = 92, Top-5 sets identical → 0% accuracy drop.
- SNR = 115.25 dB (> 30 dB); regression 3/3 passes.
- HTP context-binary path (x86 HTP simulator, `libQnnHtp.so`): cosine = 1.00000000, Top-1 = 92 — PASS (§9)

---

## 9. HTP Context Binary Workflow (QNN-5 + HTP-Simulator Validation) — 2026-09-14

> Follow-up to the CPU bring-up: `CONTEXT_BINARY_GEN` set to `YES`, `SOC_ID = 0`, `DSP_ARCH = v73`
> (emulation defaults matching the shipped `lib/hexagon-v73/unsigned` simulator).

### 9.1 Context binary generation (QNN-5)

| Step | Command / Artifact | Result |
|---|---|---|
| SoC config | `/tmp/soc0_v73.conf` (graph `mobilenet_v2`, `vtcm_mb=0`, `soc_id=0`, `dsp_arch=v73`) | ✅ created |
| Backend ext | `/tmp/soc0_v73.json` (→ `libQnnHtpNetRunExtensions.so`) | ✅ created |
| Generator | `qnn-context-binary-generator --backend $QAIRT_SDK_ROOT/lib/x86_64-linux-clang/libQnnHtp.so --model .../libmobilenet_v2.so --binary_file libmobilenet_v2.so --output_dir qairt_output --config_file /tmp/soc0_v73.json` | ✅ `Completed stage: Completion` — no errors |
| Output | `qairt_output/libmobilenet_v2.so.bin` (14,217,216 B) | ✅ non-zero |

### 9.2 Validation on the x86 QNN HTP simulator (QNN-6 + Phase 7)

- **Diagnostic** `qnn-net-run --retrieve_context qairt_output/libmobilenet_v2.so.bin --backend .../libQnnHtp.so --input_list ...` → output `Result_0/output.raw`; cosine vs golden = **1.0**, Top-1 = 92 ✅
- **Deployment** (ONNX-matching rule): `cp qairt_output/libmobilenet_v2.so.bin ./mobilenet_v2.onnx.so.bin`
- **Wrapper fix (local, documented)**: deployed `onnxwrapper.py` (skill `onnxwrapper_x86.py` copy) extended with
  (a) `QAI_QNN_RUNTIME`-driven backend selection, (b) `<model>.onnx.so.bin` ONNX-match resolution,
  (c) **runtime-aware resolution** (CPU→`lib<model>.so`/libQnnCpu.so; HTP→`.so.bin`/libQnnHtp.so),
  (d) **artifact-aware backend** (`force_htp=True` for any `.bin` context), and
  (e) auto `ADSP_LIBRARY_PATH=lib/hexagon-v73/unsigned` for HTP contexts.
  Skill repo copy untouched. Reason: stock x86 wrapper forces CPU and direct `qnn-net-run` is forbidden for acceptance.
  Fix verified on real-image test: CPU path (libQnnCpu.so) and HTP path (libQnnHtp.so + retrieve_context) both classify `test/image.jpg` → **Golden Retriever (Top-1 18.67%)**, reproducible.
- **Run**: `source env_setup.sh && export ADSP_LIBRARY_PATH=$QAIRT_SDK_ROOT/lib/hexagon-v73/unsigned && QAI_QNN_RUNTIME=HTP python aipc infer_mobilenet_v2.py`
- **Verified resolution**: artifact = `mobilenet_v2.onnx.so.bin`, backend = `libQnnHtp.so`, flag = `--retrieve_context`

| Metric (HTP sim) | Value | Threshold | Verdict |
|---|---|---|---|
| Cosine similarity vs golden | **1.00000000** | ≥ 0.95 / ≥ 0.99 | ✅ PASS |
| Top-1 / Top-5 | 92 / identical | match | ✅ PASS |
| Latency (sim, informational) | mean 684.4 ms, p50 667.7 ms, 1.46 FPS | — | recorded |

**Conclusion**: the HTP context binary is generated, deployed per the ONNX-naming rule, and executed
through the `aipc` wrapper via `libQnnHtp.so` on the x86 HTP simulator with perfect numerical parity.

---

## Accuracy Report
See [`ACCURACY_REPORT.md`](./ACCURACY_REPORT.md) (Phase R — consolidated accuracy evidence).