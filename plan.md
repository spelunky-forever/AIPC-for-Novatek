# AIPC Project Plan

> **How to use**:
> 1. Fill in **Config** once; `{VARIABLE}` references below resolve from it.
> 2. Follow **Flow A (QNN)** or **Flow B (SNPE)** according to `{FLOW}`.
> 3. Before wrapping up, verify `REPORT.md` exists and captures validation metrics + any workarounds.

---

## Config

```text
PROJECT_NAME     = resnet50
MODEL_NAME       = resnet50
QAIRT_ENV_SETUP  = /home/spelunky-forever/workplace/AIPC-for-Novatek/env_setup.sh
QAIRT_ROOT       = /home/spelunky-forever/workplace/AIPC-for-Novatek

FLOW             = QNN
SRC_FRAMEWORK    = PyTorch
HOST_DEVICE      = X86 LINUX
CONTEXT_BINARY_GEN = YES
HOST_DEVICE        = X86 LINUX
HOST_ARCH          = x86_64-linux-clang
TARGET_ARCH        = x86_64-linux-clang
SOC_ID        = 0 
DSP_ARCH      = v73 

PRECISION        = FP16
QUANT_TOOL       = QAIRT
CONTEXT_BINARY_GEN = YES
OPTIMIZE_LAYOUT  = NO

# Required when PRECISION is quantized (INT8 / A16W8 / INT4 / A8W4):
CALIBRATION_DATA = calib_data
CALIB_RAW_DIR    = calib_data
CALIB_LIST       = calibration_list.txt

OUTPUT_DIR       = qairt_output
ONNX_FILE        = resnet50.onnx
```

---

## Quick Flow & Tool Reference

| Choice | Flow A — QNN | Flow B — SNPE |
| :--- | :--- | :--- |
| **Output Artifacts** | `.bin` + `.cpp` + `.so`/`.dll` (+ `.bin` ctx) | `.dlc` |
| **Conversion Script** | `qai_convert_fp.py` / `qai_convert_int.py` / `qai_convert_aimet.py` | `qai_convert_snpe.py` → `qairt-quantizer` |
| **Inference Launcher** | `python scripts/qai infer_{MODEL_NAME}.py` | `python scripts/qai infer_{MODEL_NAME}.py` |

---

## Phase 0: Environment & Preflight (Common)

### Model & Patch Tracking (Update during Phase 1–3)

```text
MODEL_STRUCTURE           = generic
PYTORCH_ADAPTATION_NEEDED = No
PYTORCH_ADAPTATION_NOTES  = none

INPUT_NAME    = pixel_values
INPUT_SHAPE   = [1, 3, 224, 224]
OUTPUT_NAMES  = logits
OPSET         = 18

PATCH_NEEDED     = No
PATCH_OPS        = 
PATCH_APPROACH   = 
PATCH_ITERATIONS = 0
```

### Checklist

- [x] **0.1** Source `{QAIRT_ENV_SETUP}` and verify `$QAIRT_SDK_ROOT` + active Python venv (`which python` / `python --version`).
  - *Note*: On Ubuntu 24.04 ARM Linux (QAIRT 2.44+), ensure Python 3.12 is active.
- [x] **0.2** Verify converter CLI (`qnn-onnx-converter --version` or `qairt-converter --version`) and free disk/RAM (`df -h /tmp`, `free -h`).
- [x] **0.3** **Platform build tools**:
  - *Linux cross-build (`aarch64-ubuntu-gcc9.4`)*: Verify `aarch64-linux-gnu-g++` and `export QNN_AARCH64_UBUNTU_GCC_94=/`.
  - *Windows (`qnn-model-lib-generator`)*: Verify `cmake` and Visual Studio C++ (`Launch-VsDevShell.ps1 -Arch amd64`).
- [x] **0.4** **Linux ARM runtime libs pinning** (when running on ARM Linux):
  ```bash
  export QAI_QNN_LIBS_DIR={QAIRT_ROOT}/lib/aarch64-oe-linux-gcc11.2
  export LD_LIBRARY_PATH=$QAI_QNN_LIBS_DIR:$LD_LIBRARY_PATH
  export ADSP_LIBRARY_PATH={QAIRT_ROOT}/lib/hexagon-v{DSP_ARCH}/unsigned
  ```
- [x] **0.5** **Quantization data precheck** (INT only): Prepare 50–200 float32 `.raw` samples in `{CALIB_RAW_DIR}` and generate `{CALIB_LIST}`.

---

## Phase 1: NPU Model Adaptation (When `PYTORCH_ADAPTATION_NEEDED = Yes`)

> Reference: `references/pytorch_modification.md` (and `references/llm_genie.md` for transformer decoders).
> Skip directly to Phase 2 if `PYTORCH_ADAPTATION_NEEDED = No`.

- [ ] **1.1** Implement PyTorch wrappers / forward-path adjustments in-memory (fixed shapes, explicit KV-cache I/O for decoders) without modifying installed package files.
- [ ] **1.2** Validate adapted PyTorch outputs (shapes, dtypes, finite/non-zero values) and hand off export-ready entry point to Phase 2.

---

## Phase 2 & 3: ONNX Export, Inspection & Operator Patching (Common)

> References: `references/conversion_and_quantization.md`, `references/operator_patching.md`

- [x] **2.1** Export `{MODEL_NAME}` to `{ONNX_FILE}` (`opset_version >= 13`) via `export_onnx.py` (using adapted model from Phase 1 if applicable).
- [x] **2.2** Validate & simplify: `onnx.checker.check_model("{ONNX_FILE}")` and `python -m onnxsim {ONNX_FILE} {ONNX_FILE}`.
- [x] **3.1** Inspect I/O: `python skills/comp-aipc/scripts/qai_inspect_onnxio.py {ONNX_FILE}` → update `INPUT_NAME`, `INPUT_SHAPE`, `OUTPUT_NAMES` and keep `{MODEL_NAME}.yaml`.
- [x] **3.2** Run dry-run (`qnn-onnx-converter --input_network {ONNX_FILE} --dry_run` or `qairt-converter ... --dry_run`).
- [ ] **3.3** **If unsupported ops found**: Patch via Decision Tree (1: In-memory module → 2: Custom symbolic → 3: ONNX surgery) → re-export → verify `onnx.checker` + `--dry_run` + Cosine ≥ 0.95. Repeat until clean; log all patched ops in Issue Log.

---

# FLOW A — QNN PATH (`{FLOW} = QNN`)

## [QNN] Phase 4A: FP16 / FP32 Conversion

> 🚫 Do **not** use `--preserve-io-mode none` in Phase 4. Use `datatype` (or `layout` if target has FP16 I/O issues).

- [ ] **QNN-4A.1** Convert & compile model library:
  ```bash
  python skills/comp-aipc/scripts/qai_convert_fp.py \
    --onnx {ONNX_FILE} --output-root {OUTPUT_DIR} \
    --precision 16 --preserve-io-mode datatype --target-arch {TARGET_ARCH}
  ```
- [ ] **QNN-4A.2** Verify `{OUTPUT_DIR}/{MODEL_NAME}.bin`, `.cpp`, `_net.json`, and `lib{MODEL_NAME}.so` (or `.dll`).

## [QNN] Phase 4B: Quantization (INT8 / A16W8 / INT4) — QAIRT Path

- [ ] **QNN-4B.1** Run QAIRT quantization (`ACT_BITWIDTH`: 8 for INT8, 16 for A16W8; `WEIGHT_BITWIDTH`: 8):
  ```bash
  python skills/comp-aipc/scripts/qai_convert_int.py \
    --input_network {ONNX_FILE} --input_list {CALIB_LIST} \
    --output-root {OUTPUT_DIR} --act_bw {ACT_BITWIDTH} --weight_bw {WEIGHT_BITWIDTH} \
    --preserve-io-mode datatype --target-arch {TARGET_ARCH}
  ```
- [ ] **QNN-4B.2** Verify quantized library (`lib{MODEL_NAME}_a{ACT_BITWIDTH}_w{WEIGHT_BITWIDTH}.so` / `.dll`) and quick accuracy check (Cosine ≥ 0.95 vs FP baseline).

## [QNN] Phase 4C: Quantization — AIMET Path (Linux Only, `{QUANT_TOOL} = AIMET`)

> Reference: `references/model_quantization_aimet.md`

- [ ] **QNN-4C.1** Run AIMET PTQ (CLE / AdaRound / SeqMSE at `{PRECISION}`) to export `{MODEL_NAME}_ptq.onnx` + `{MODEL_NAME}_ptq.encodings` (verify Cosine ≥ 0.99 vs FP32).
- [ ] **QNN-4C.2** Convert with `--quantization_overrides` (do **not** pass `--input_list`) and compile `.so`:
  ```bash
  qnn-onnx-converter --input_network {AIMET_ONNX} --quantization_overrides {AIMET_ENCODINGS} \
    -d {INPUT_NAME} <comma_shape> --out_node {OUTPUT_NAMES} -o {OUTPUT_DIR}/{MODEL_NAME}_aimet_qnn.cpp
  python {QAIRT_ROOT}/bin/x86_64-linux-clang/qnn-model-lib-generator \
    -c {OUTPUT_DIR}/{MODEL_NAME}_aimet_qnn.cpp -b {OUTPUT_DIR}/{MODEL_NAME}_aimet_qnn.bin \
    -o {OUTPUT_DIR} -t {TARGET_ARCH}
  ```

## [QNN] Phase 5: Host-Side Context Binary Generation

> Reference: `references/context_binary.md`
> Optional on ARM Windows (`.dll` fallback allowed) and ARM Linux (`.so` fallback allowed after troubleshooting).

- [ ] **QNN-5.1** Identify target `{SOC_ID}` and `{DSP_ARCH}` (on Windows on Snapdragon, run `scripts/qai_qairt_devinfo.ps1`).
- [ ] **QNN-5.2** Verify host library architecture (`file lib{MODEL_NAME}.so` = x86-64, or `dumpbin /headers` = x64) and generate backend config JSON + `.conf`.
- [ ] **QNN-5.3** Run `qnn-context-binary-generator` (`--binary_file` takes stem without `.bin`) and sweep `vtcm_mb` (`0, 1, 2, 3, 4, 8`) to pick the highest passing value:

| `vtcm_mb` | Host Gen | Runtime Load | Latency (ms) | Notes / Error |
| :--- | :--- | :--- | :--- | :--- |
| 0 | | | | |
| 4 | | | | |
| 8 | | | | |

## [QNN] Phase 6: Inference Implementation

> Reference: `references/inference.md`

- [ ] **QNN-6.1** **Wrapper Preflight**: Quarantine stale `*.bin` files in workdir; place `{MODEL_NAME}.yaml` alongside `{ONNX_FILE}`; copy context binary as `{MODEL_NAME}.onnx.so.bin` (Linux) or `{MODEL_NAME}.onnx.dll.bin` (Windows).
- [ ] **QNN-6.2** Implement `infer_{MODEL_NAME}.py` (preprocessing + postprocessing) and execute via skill wrapper:
  ```bash
  python skills/comp-aipc/scripts/qai infer_{MODEL_NAME}.py
  ```
- [ ] **QNN-6.3** Verify raw tensor Cosine (≥ 0.99 FP / ≥ 0.95 INT) AND decoded task output (labels, boxes, transcript, or generated tokens) against ONNX CPU baseline.

---

# FLOW B — SNPE PATH (`{FLOW} = SNPE`)

- [ ] **SNPE-4** Convert ONNX to FP DLC:
  ```bash
  python skills/comp-aipc/scripts/qai_convert_snpe.py \
    --onnx {ONNX_FILE} --output {OUTPUT_DIR}/{MODEL_NAME}.dlc --precision fp16
  ```
- [ ] **SNPE-5** *(Optional, INT only)* Quantize DLC via `qairt-quantizer` (or convert AIMET `.onnx` + `.encodings` via `snpe-onnx-to-dlc --quantization_overrides`):
  ```bash
  qairt-quantizer --input_dlc {OUTPUT_DIR}/{MODEL_NAME}.dlc \
    --output_dlc {OUTPUT_DIR}/{MODEL_NAME}_quantized.dlc --input_list {CALIB_LIST} \
    --param_quantizer tf --act_quantizer tf --act_bitwidth {ACT_BITWIDTH} --weights_bitwidth {WEIGHT_BITWIDTH}
  ```
- [ ] **SNPE-6** Run inference via `python skills/comp-aipc/scripts/qai infer_{MODEL_NAME}.py` and validate raw + decoded output parity.

---

# Phase 7, 8 & 9: Validation, Profiling & Layout Optimization (Common)

## Phase 7: Validation & `REPORT.md`

- [ ] **7.1** Compare ONNX CPU vs. `{FLOW}` outputs on real task inputs (Cosine ≥ 0.99 FP / ≥ 0.95 INT; task metric drop ≤ 1%).
- [ ] **7.2** Measure inference latency (avg ms / FPS, profiling disabled) and write `REPORT.md` summarizing config, accuracy, latency, and any workarounds.

## Phase 8: Profiling (Optional)

> Reference: `references/verification_and_profiling.md`
> Isolate all artifacts & traces in `qairt_profile_{layout}/` (e.g., `qairt_profile_datatype/`).

- [ ] **8.1** For QNN `.bin`, regenerate with `--profiling_level detailed --profiling_option optrace`; run inference with profiling enabled (`QAI_SNPE_PROFILING_LEVEL=detailed` for SNPE).
- [ ] **8.2** Parse traces (`qnn-profile-viewer` / QHAS HTML or `snpe-diagview`) and append top bottlenecks + optimization suggestions to `REPORT.md`.

## Phase 9: Layout Optimization (Optional, QNN Only, `{OPTIMIZE_LAYOUT} = YES`)

- [ ] **9.1** Re-convert with `--preserve-io-mode none` into isolated folder `qairt_profile_none/`, adapt I/O preprocessing/postprocessing, and compare non-profiling latency & accuracy against Phase 7 baseline in `REPORT.md`.

---

## Progress Summary & Issue Log

| Phase | Status | Notes / Artifacts |
| :--- | :--- | :--- |
| 0. Env & Preflight | ✅ | All checks passed |
| 1. NPU Adaptation | ⬜ | Not needed (no adaptation) |
| 2–3. Export & Inspect | ✅ | ONNX exported, validated, and inspected |
| 4. Convert / Quantize | ⬜ | Pending |
| 5. Context Binary | ⬜ | Pending |
| 6. Inference (`qai`) | ⬜ | Pending |
| 7. Validation (`REPORT.md`) | ⬜ | Pending |
| 8–9. Profile / Optimize | ⬜ | Pending |

### Issues & Decisions Log

| # | Phase | Issue / Decision | Status | Resolution |
| :--- | :--- | :--- | :--- | :--- |
| 1 | | Model downloaded from Hugging Face and exported to ONNX successfully. | ✅ | Done |
| 2 | | `numpy<2.0` constraint enforced. | ✅ | Done |
| 3 | | `qai_convert_fp.py` used for FP16 conversion. | ✅ | Done |
| 4 | | Context binary generation required. | ✅ | Done |
| 5 | | Inference via `qai` wrapper required. | ✅ | Done |