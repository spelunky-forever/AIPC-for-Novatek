# Troubleshooting Guide (Linux x86 Host & ARM Linux Target)

---

## 1. Environment & Conversion Errors

| Error Signature | Cause | Fix |
| :--- | :--- | :--- |
| `QAIRT_SDK_ROOT not set` or unbound `PYTHONPATH` on `source` | Env script not sourced, or shell has `set -u` (`nounset`) enabled | Disable `set -u` temporarily (`set +u; source /home/spelunky-forever/workplace/AIPC-for-Novatek/env_setup.sh; set -u`). |
| `Failed to copy external data to /tmp/... space issue` | `/tmp` `tmpfs` is too small for large ONNX external weights | Set `export QAIRT_TMP_DIR=/path/with/2x_model_space` before converting. |
| `ValueError: protobuf of onnx model is too large (>2GB)` | `onnx.load()` called before `onnx.checker.check_model()` | Pass the file path string directly: `onnx.checker.check_model("model.onnx")`. |
| `Missing command line inputs for dynamic inputs [...]` | ONNX model has symbolic/0 dimensions | Pass `--input-dim name,1,3,224,224` (QNN) or `--source-model-input-shape name 1,3,224,224` (SNPE). |
| `Could not find compiler: .../aarch64-linux-gnu-g++` | Cross-compile sysroot var unset | Install `g++-aarch64-linux-gnu` and `export QNN_AARCH64_UBUNTU_GCC_94=/`. |

---

## 2. Context Binary & ARM Linux HTP Runtime Errors

### A. Wrong Artifact Selected / `Failed to load skel (4000)` / `Transport error 14001` / `1008`
**Causes**: Stale `.bin` in workdir, `onnxwrapper.py` defaulting to bundled `qai_appbuilder/libs/` instead of SDK runtime libs, or missing `.so.2` symlink.
**Fix Sequence**:
1. Quarantine stale binaries and deploy with ONNX-matching name (`<model>.onnx.so.bin`) + `<model>.yaml`:
   ```bash
   mkdir -p _ctx_backup && mv -f *.bin _ctx_backup/ 2>/dev/null || true
   cp <output>.bin <model>.onnx.so.bin
   ```
2. Pin runtime libraries explicitly (do not rely on auto-detection):
   ```bash
   export QAI_QNN_LIBS_DIR="$QAIRT_SDK_ROOT/lib/aarch64-oe-linux-gcc11.2"
   export LD_LIBRARY_PATH="$QAI_QNN_LIBS_DIR:$LD_LIBRARY_PATH"
   export ADSP_LIBRARY_PATH="$QAIRT_SDK_ROOT/lib/hexagon-v${DSP_ARCH}/unsigned"
   export QAI_QNN_RUNTIME=HTP
   ```
3. If FastRPC daemon expects `.so.2` SONAME:
   ```bash
   ln -sf "$ADSP_LIBRARY_PATH/libQnnHtpV${DSP_ARCH}Skel.so" "$ADSP_LIBRARY_PATH/libQnnHtpV${DSP_ARCH}Skel.so.2"
   ```

### B. `Unknown config socModel 0` or Signal 11 Crash on Load
- **Cause**: Passed Linux kernel sysfs ID (`/sys/devices/soc0/soc_id`) or `$PRODUCT_SOC` (e.g., `9075`) instead of QAIRT `soc_id` (e.g., `77` for `qcs9075`), or mismatched `dsp_arch`.
- **Fix**: Look up `/proc/device-tree/compatible` in `scripts/_soc_targets.py`.

### C. `Failed to create context from binary with err 0x138d`
- **Cause**: Unsupported `vtcm_mb` size on target SoC.
- **Fix**: Sweep `vtcm_mb = 0, 1, 2, 3, 4, 8` on the board and pick the **maximum passing** value.

### D. `fastrpc memory map ... failed with error: 0x1` / `err 1002`
- **Cause**: Context binary exceeds DSP SMMU contiguous mapping window (~512 MB–1 GB).
- **Fix**: First check if inference actually completes (non-fatal on some SoCs like QCS9075). Otherwise use JIT `.so` fallback or split the model per `references/complex_models.md`.

---

## 3. Inference Crashes & Accuracy Failures

- **Silent Segfault (`exit 139`) When Importing `torch` + `qai`**:
  - PyTorch and QNN HTP C++ heap allocators conflict in the same process. Always run PyTorch/ORT CPU baseline and `python scripts/qai` in **separate processes** exchanging `.npy` files.
- **Segfault (`exit 139`) at Process Exit After Inference Completes (ARM Linux)**:
  - Known `libappbuilder.so` destructor teardown bug (`ModelDestroy`). Flush stdout and save all outputs first, then exit cleanly via `os._exit(0)` (e.g., `export AIPC_SAFE_EXIT=1`).
- **Segfault (`exit 139`) Immediately on Context Binary Load**:
  - `qai_appbuilder` wheel version mismatches the active QAIRT SDK. Rebuild `qai-appbuilder` wheel from source against `$QAIRT_SDK_ROOT` (`QAI_TOOLCHAINS=aarch64-oe-linux-gcc11.2`).
- **Completely Wrong Predictions Despite High Overall Cosine**:
  - Check (1) `NCHW` vs `NHWC` input layout (`--preserve_io` keeps NCHW), (2) missing `<model>.yaml` causing output tensor reordering, and (3) per-channel quantization scale domination in detection models (boxes `0–640` crushing class scores `0–1` $\rightarrow$ switch to `A16W8` or split final `Concat`).

---

## 4. One-Command ARM Linux Triage Bundle

When HTP initialization fails on the board, run this to capture all diagnostics in one folder:
```bash
OUT=triage_$(date +%Y%m%d_%H%M%S) && mkdir -p "$OUT"
env | grep -E 'QAI|QAIRT|QNN|ADSP|LD_LIBRARY|SOC|DSP' > "$OUT/env.txt"
qnn-platform-validator --backend dsp --coreVersion > "$OUT/validator_core.txt" 2>&1 || true
qnn-platform-validator --backend dsp --testBackend > "$OUT/validator_test.txt" 2>&1 || true
LD_DEBUG=libs timeout 15 python3 -u scripts/qai infer_model.py > "$OUT/lddebug.log" 2>&1 || true
grep -E 'libQnnHtp\.so|libQnnSystem\.so' "$OUT/lddebug.log" > "$OUT/loaded_libs.txt" || true
dmesg -T | grep -iE 'fastrpc|adsprpc|cdsp|qnn|htp' > "$OUT/dmesg.txt" 2>&1 || true
```