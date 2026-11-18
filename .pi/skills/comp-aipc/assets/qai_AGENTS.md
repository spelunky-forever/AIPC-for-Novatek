# AIPC Project Agents

> **Usage**: Defines conversational workflow rules, blocking conditions, and phase progression.
> Project config, commands, and state live in `plan.md`. Core guardrails live in `SKILL.md`.

**Project**: `{PROJECT_NAME}` | **Model**: `{MODEL_NAME}` | **Flow**: `{FLOW}` | **Precision**: `{PRECISION}`

---

## 1. Conversational Workflow Rules

- **Direct Execution**: Execute clear, non-destructive pipeline steps directly and report key outputs, metrics, and decisions concisely in chat.
- **Ask Before Risky Actions**: Confirm with the user before installing packages (`pip install`), applying patches that might alter model semantics, or choosing between major accuracy/performance trade-offs.
- **Keep `plan.md` Updated**: Update checkboxes in `plan.md` as phases complete, and record errors, workarounds, and patched operators in the `plan.md` Issue Log.

---

## 2. Blocking Conditions (Always Stop & Ask User)

| # | Condition | Required Action |
| :--- | :--- | :--- |
| **B1** | Required `plan.md` Config variable is empty or `<!--...-->` | Stop. List missing variables and ask the user to confirm. |
| **B2** | `pip install` is required | Stop. State package name and reason; ask permission first. |
| **B3** | Unsupported operator has no replacement pattern or patching stalls after 7+ attempts | Stop. List attempted patches and logs; escalate to user. |
| **B4** | Operator patch would alter model semantics | Stop. Explain the behavioral change and ask approval. |
| **B5** | Post-quantization accuracy drops below threshold (Cosine < 0.95) | Stop. Report metrics and ask whether to accept, adjust calibration, or switch to AIMET/mixed precision. |

---

## 3. Pipeline & Phase Progression

```
Phase 0: Environment & Preflight (toolchain, venv, QAI_QNN_LIBS_DIR, calib data check)
  │
  ├─► Phase 1: NPU Model Adaptation (if PYTORCH_ADAPTATION_NEEDED = Yes)
  ├─► Phase 2: ONNX Export & Simplification (`export_onnx.py` + `onnxsim`)
  ├─► Phase 3: Model Inspection (`qai_inspect_onnxio.py` + converter `--dry_run`)
  ├─► Phase 4: Conversion / Quantization (QNN-4A/4B/4C or SNPE-4/5)
  ├─► Phase 5: Context Binary Generation (QNN-5, host-side + VTCM sweep)
  ├─► Phase 6: Inference via `python qai` (`infer_{MODEL_NAME}.py`)
  ├─► Phase 7: Validation & `REPORT.md` (raw tensor cosine + decoded task parity)
  ├─► Phase 8: Profiling (optional, isolated in `qairt_profile_{layout}/`)
  └─► Phase 9: Layout Optimization (optional, QNN only, `--preserve-io-mode none`)
```

---

## 4. Mandatory Engineering Rules

1. **No SDK Internals Browsing**: Do not survey or read source trees inside `$QAIRT_SDK_ROOT` / `$QNN_SDK_ROOT`. Use documented CLIs and skill scripts only.
2. **No Stale Artifact Reuse**: Do not reuse pre-existing `.bin`, `.cpp`, `_net.json`, or `.so` files in the working folder as substitutes for running conversion stages.
3. **Real Inference Gate**: Dummy-tensor smoke runs never satisfy Phase 6/7. Final validation requires real task inputs, decoded output comparison against PyTorch/ONNX CPU baseline, and `REPORT.md` written on disk.