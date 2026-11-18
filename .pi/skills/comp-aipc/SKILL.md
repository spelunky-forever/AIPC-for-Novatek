---
name: aipc-toolkit
description: AIPC (AI Porting Conversion). Convert, quantize, compile context binaries, patch unsupported operators, and run inference for PyTorch/ONNX models on Qualcomm platforms (QNN / SNPE / QAIRT).
---

# AIPC Toolkit (Qualcomm QAIRT — Linux x86 & ARM Linux)

Use the scripts bundled inside this skill directory (`scripts/`). Resolve paths relative to this skill root, not the working directory. Prefer the unified entrypoint `python scripts/aipc.py <subcommand>` (or bundled `qai_*.py` wrappers) over raw SDK CLIs unless explicitly required.

## 1. Choose The Route First

- **New Project Setup**: Bootstrap first via `python scripts/aipc.py init <project_dir>` (`qai_project_setup.py`). Never manually write `AGENTS.md`, `CLAUDE.md`, or `plan.md`.
- **LLM / Transformer Decoder (Prefill + Decode + KV-Cache / Genie)**: Read `references/pytorch_modification.md` and `references/llm_genie.md` before exporting ONNX.
- **Multi-Component / Oversized Models (SD, Whisper, CLIP, >800 MB models)**: Read `references/complex_models.md` for sub-graph export and SMMU model splitting.
- **Standard Single-Model QNN / SNPE Bring-up**: Follow the Core Workflow below and open `references/` files only on demand.

## 2. Hard Guardrails (Mandatory)

1. **Environment Initialization**:
   - Always activate the Conda + QAIRT SDK environment before running any tool:
     `source /home/spelunky-forever/workplace/AIPC-for-Novatek/env_setup.sh`
   - Never run `pip install` without explicit user permission (Blocking Condition B2).
2. **Disk & Memory Check Before Conversion**:
   - Check `df -h /tmp` and `free -h` before calling converters. If `/tmp` space is less than 2× the model weight size, set `export QAIRT_TMP_DIR="$PWD/qairt_tmp" && mkdir -p "$QAIRT_TMP_DIR"`.
   - Warn the user if available RAM is less than 2× the model file size.
3. **Architecture & Context Binary Rules**:
   - **Model library (`.so`)** must match the **host/target Python process architecture** (`x86_64-linux-clang` on x86 host; `aarch64-ubuntu-gcc9.4` only if cross-compiling `.so` for direct ARM execution without a context binary).
   - **Context binary (`.so.bin` / `.bin`)** is compiled offline on the **Linux x86 host** for the target SoC using mandatory `{SOC_ID}` and `{DSP_ARCH}` from `scripts/_soc_targets.py` (never read Linux kernel `/sys/devices/soc0/soc_id` or `$PRODUCT_SOC`).
   - Context binary is optional on ARM Linux (`.so` fallback allowed only after exhausting `references/context_binary.md` troubleshooting).
4. **Operator Patching**:
   - Patch in-memory or via ONNX surgery; never edit installed library source files (`site-packages`).
   - Never fall back to CPU (`libQnnCpu.so`) as a workaround for unsupported HTP operators. Iterate patching until all ops compile on HTP, or escalate if no replacement pattern exists (B3) or semantics change (B4).
5. **Inference Execution Policy**:
   - **ALWAYS** execute inference via `python scripts/aipc.py infer <infer_script>.py` (invokes `scripts/qai` + `scripts/onnxwrapper.py`). Never use `site-packages/qai_appbuilder/onnxwrapper.py` or raw `qnn-net-run`/`snpe-net-run` for final acceptance.
   - **Process Isolation**: Never import `torch` or run `onnxruntime` CPU baseline in the same Python process as `qai` (C++ heap allocator collision causes silent segfaults).
   - **Wrapper Preflight**: Quarantine stale `*.bin` files in the workdir before running, and keep `<model>.onnx.so.bin` alongside `<model>.yaml` (required to restore ONNX output tensor order).
   - **Linux ARM Runtime Pinning**: Explicitly export `QAI_QNN_LIBS_DIR`, `LD_LIBRARY_PATH`, and `ADSP_LIBRARY_PATH` before running `qai` on ARM Linux.

## 3. Core Workflow

1. **Bootstrap**: `python scripts/aipc.py init <project_dir>` → confirm `plan.md` Config with the user → auto-fill defaults.
2. **Adapt & Export**: Apply PyTorch NPU wrappers if needed (`references/pytorch_modification.md`) → export FP32 ONNX (opset 18, `do_constant_folding=False`) + `onnxsim` (`references/conversion_and_quantization.md`).
3. **Preflight & Inspect**: `python scripts/aipc.py preflight <model>.onnx` and `python scripts/aipc.py inspect <model>.onnx` (generates `<model>.yaml` + converter `--dry_run`).
4. **Patch (if needed)**: Follow Decision Tree (1: Export-time/symbolic handler → 2: In-memory module replacement → 3: ONNX surgery) per `references/operator_patching.md`. Gate: `onnx.checker` + `--dry_run` + Cosine ≥ 0.99 (FP) / ≥ 0.95 (INT).
5. **Convert / Quantize**:
   - **Unified DLC + Context Binary Flow (Default)**: `python scripts/aipc.py convert --model <model>.onnx --precision <fp16|w8a8|w8a16> ...` (wraps `run_pipeline.py`).
   - **QNN Shared Library Flow**: `python scripts/qai_convert_fp.py` or `qai_convert_int.py` (`--preserve-io-mode datatype` or `layout`; **never `none`** in Phase 4).
   - **Advanced PTQ (AIMET)**: See `references/model_quantization_aimet.md`.
6. **Context Binary & VTCM Sweep**: Compile for target `{SOC_ID}`/`{DSP_ARCH}` (`python scripts/aipc.py ctxbin ...`) and sweep `vtcm_mb = 0, 1, 2, 3, 4, 8` on target to pick the highest passing value (`references/context_binary.md`).
7. **Inference & Validation**: Run via `python scripts/aipc.py infer <infer_script>.py` → verify raw tensor cosine (≥ 0.99 FP / ≥ 0.95 INT) AND decoded task output → write `REPORT.md` (`references/verification_and_profiling.md`).

## 4. Reference Map (Load On Demand)

| Topic | File |
| :--- | :--- |
| PyTorch NPU Adaptation (Prefill / Decode / KV-Cache) | `references/pytorch_modification.md` |
| Export, Conversion & Standard Quantization | `references/conversion_and_quantization.md` |
| Advanced PTQ (AIMET) | `references/model_quantization_aimet.md` |
| Operator Patching | `references/operator_patching.md` |
| Context Binary Generation & VTCM Sweep | `references/context_binary.md` |
| LLM Custom AR Loop & Genie Graph Contract | `references/llm_genie.md` |
| Multi-Component Pipelines & SMMU Model Split | `references/complex_models.md` |
| Inference & `qai` Wrapper | `references/inference.md` |
| Accuracy, Profiling & Layout Optimization | `references/verification_and_profiling.md` |
| Troubleshooting & Linux ARM Triage | `references/troubleshooting.md` |