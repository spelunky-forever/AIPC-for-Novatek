# Context Binary Generation (Linux x86 Host & ARM Linux Target)

Generates hardware-specific HTP context binaries (`.so.bin` / `.bin`) for Qualcomm NPU deployment.

---

## 1. Workflow Selection: x86 Host vs. ARM Linux Target

| Scenario | Host Machine | Model Input | Key Rules |
| :--- | :--- | :--- | :--- |
| **Cross-SoC Offline Compile (Primary)** | Linux x86 (`x86_64`) | `lib<model>.so` compiled with **`x86_64-linux-clang`** (or `.dlc`) | Uses `libHtpPrepare.so` offline. **Must** specify target `soc_id` + `dsp_arch`. Cannot load `aarch64` `.so` on x86 host. |
| **On-Device Native Compile** | ARM Linux (`aarch64`) | `lib<model>.so` (`aarch64-oe-linux-gcc11.2`) or `.dlc` | **Omit `--soc_model`** (local HTP supplies SoC). Must stage Hexagon `.cat`/`.so` skel files in CWD and use absolute paths. |

> **Linux Fallback Policy**: Context binary is optional on ARM Linux (raw `aarch64` `.so` works directly), **but** do not fall back to `.so` until you have verified `soc_id`/`dsp_arch`, swept `vtcm_mb`, and tested `htp_arch`/no-`soc_id` configs.

---

## 2. Step 1: Identify Target `soc_id` & `dsp_arch` (Critical)

Wrong `soc_id` / `dsp_arch` pairs compile with exit code 0 on x86 host but crash with **signal 11** or `err:14` on the ARM Linux board.

1. **Read Target Board Hardware Identity** (run on the ARM Linux target):
   ```bash
   cat /proc/device-tree/model; echo; tr -d "\0" </proc/device-tree/compatible; echo
   ```
2. **Map Compatible Token to QAIRT Table**:
   - Look up the token (e.g., `qcs9075`, `sa8775p`) in **`scripts/_soc_targets.py`** (single source of truth) or `$QAIRT_SDK_ROOT/docs/QAIRT-Docs/QNN/general/overview.html`.
   - Example verified mapping: `qcs9075` / `sa8775p` $\rightarrow$ `soc_id = 77` (or `52`), `dsp_arch = "v73"`.
3. **Common Traps to Avoid**:
   - **NEVER** use `/sys/devices/soc0/soc_id` or `$PRODUCT_SOC` directly—these are Linux kernel/platform IDs (e.g., `PRODUCT_SOC=9075`), which fail with `Unknown config socModel 0`.
   - **SM8750 (Snapdragon 8 Elite)** is `v79`, **not** `v81` (`v81` is SM8850 / SC8480XP / SA8797).
   - **SM8650** = `57` (not `43`, which is SM8550).
   - If `qnn-context-binary-utility` reports `socModel: 2147483647`, your requested `soc_model` was ignored (happens when running cross-SoC compile on a host that already has local HTP hardware).

---

## 3. Step 2: Generate Context Binary on Linux x86 Host

### Using Wrapper Script (`qai_dev_gen_contextbin_x86.py`)
```bash
# By platform name (auto-fills soc_model & dsp_arch from scripts/_soc_targets.py)
python scripts/qai_dev_gen_contextbin_x86.py \
  --dlc /abs/path/to/model.dlc \
  --output /abs/path/to/model.so.bin \
  --backend htp \
  --target_platform <PLATFORM_NAME> --vtcm_mb 8
```

### Direct CLI (`qnn-context-binary-generator`)
1. **Preflight**: Confirm `--model` `.so` is compiled for x86-64 (`file libmodel.so` $\rightarrow$ `x86-64`) and check exact graph name (`qairt-dlc-info -i <model>.dlc` or converter `--output_path` stem). Mismatched `graph_names` causes `Graph Compose failure`.
2. **Create `.conf` and Backend Extensions JSON**:
   ```bash
   cat > /tmp/htp_soc.conf << 'CONF'
   {"graphs":[{"graph_names":["<GRAPH_NAME>"],"vtcm_mb":8,"O":3}],"devices":[{"soc_id":<SOC_ID>,"dsp_arch":"<DSP_ARCH>","cores":[{"perf_profile":"burst","rpc_control_latency":50}]}]}
   CONF

   cat > /tmp/backend_ext.json << JSONEOF
   {"backend_extensions":{"shared_library_path":"$QAIRT_SDK_ROOT/lib/x86_64-linux-clang/libQnnHtpNetRunExtensions.so","config_file_path":"/tmp/htp_soc.conf"}}
   JSONEOF
   ```
   *(If explicit `soc_id` fails on target, try replacing `"soc_id": <SOC_ID>` with `"htp_arch": "<DSP_ARCH>"`.)*
3. **Run Generator**:
   ```bash
   mkdir -p /abs/output/dir
   $QAIRT_SDK_ROOT/bin/x86_64-linux-clang/qnn-context-binary-generator \
     --backend     $QAIRT_SDK_ROOT/lib/x86_64-linux-clang/libQnnHtp.so \
     --model       /abs/path/to/libmodel_x86.so \
     --binary_file model.onnx.so \
     --output_dir  /abs/output/dir \
     --config_file /tmp/backend_ext.json
   # Output: /abs/output/dir/model.onnx.so.bin
   ```
   - **`--binary_file` Rules**: Pass a **stem without `.bin`** (tool appends `.bin`) and never an absolute path. Use separate `--output_dir` per precision/model to prevent 0-byte placeholder collisions.

---

## 4. Alternative: On-Device Generation (ARM Linux Board)

If generating directly on the ARM Linux target (`aarch64`), follow these three rules:
1. **Stage Hexagon runtime in CWD** (FastRPC resolves skel relative to CWD; note that only `v73` and `v81` ship `.cat` files—its absence on `v68/v69/v75/v79` is normal):
   ```bash
   mkdir -p /abs/output/dir && cd /abs/output/dir
   cp ${QAIRT_SDK_ROOT}/lib/hexagon-v73/unsigned/libqnnhtpv73.cat    .
   cp ${QAIRT_SDK_ROOT}/lib/hexagon-v73/unsigned/libQnnHtpV73Skel.so .
   ```
2. **Use absolute paths** for `--backend` and `--model` (relative paths exit 9 with `Initialization failure`).
3. **Omit `--soc_model`** (local hardware already sets the SoC).
   ```bash
   ${QAIRT_SDK_ROOT}/bin/aarch64-oe-linux-gcc11.2/qnn-context-binary-generator \
     --backend ${QAIRT_SDK_ROOT}/lib/aarch64-oe-linux-gcc11.2/libQnnHtp.so \
     --model   /abs/path/to/libmodel_aarch64.so \
     --binary_file model.onnx.so \
     --output_dir  /abs/output/dir
   ```

---

## 5. VTCM Sweep on Target Board

Host generation succeeds for **any** `vtcm_mb` value; invalid sizes fail only at runtime on the target board (`Failed to create context from binary with err 0x138d` or `Failed to register context`).
- Sweep `vtcm_mb = 0, 1, 2, 3, 4, 8` on the target board.
- **Do not stop at the first passing value**: test the full range and select the **maximum passing `vtcm_mb`** (higher VTCM reduces DDR spills and significantly speeds up large models > 500 MB).