#!/usr/bin/env bash
# QAIRT environment bootstrap
export QAIRT_SDK_ROOT="/home/spelunky-forever/workplace/AIPC-for-Novatek/toolchains/qairt/2.45.0.260326"

# 載入 Python 虛擬環境
source $(conda info --base)/etc/profile.d/conda.sh
conda activate aipc

export PYTHONPATH="${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
source "${QAIRT_SDK_ROOT}/bin/envsetup.sh"

# Workaround (recorded in aipc_plan.md Issue Log #3): QAIRT converter C-extensions
# (libPyIrGraph) need libpython3.10.so.1.0 from the conda env and pandas must see
# numpy 1.26.4 (SDK check-python-dependency matrix). Prepend $CONDA_PREFIX/lib so
# the converted tools find the conda runtime libs.
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

# 依架構選擇工具鏈路徑
if [ "$(uname -m)" = "x86_64" ]; then
  QAIRT_DEVICE_BIN="x86_64-linux-clang"
elif [ "$(uname -m)" = "aarch64" ]; then
  if [ -d "${QAIRT_SDK_ROOT}/bin/aarch64-ubuntu-gcc9.4" ]; then
    QAIRT_DEVICE_BIN="aarch64-ubuntu-gcc9.4"
  elif [ -d "${QAIRT_SDK_ROOT}/bin/aarch64-oe-linux-gcc11.2" ]; then
    QAIRT_DEVICE_BIN="aarch64-oe-linux-gcc11.2"
  elif [ -d "${QAIRT_SDK_ROOT}/bin/aarch64-oe-linux-gcc9.3" ]; then
    QAIRT_DEVICE_BIN="aarch64-oe-linux-gcc9.3"
  else
    QAIRT_DEVICE_BIN="aarch64-oe-linux-gcc8.2"
  fi
else
  QAIRT_DEVICE_BIN="x86_64-linux-clang"
fi

export QAIRT_DEVICE_BIN
export PATH="${QAIRT_SDK_ROOT}/bin/${QAIRT_DEVICE_BIN}:${PATH}"

# Default runtime: HTP context binary (CONTEXT_BINARY_GEN = YES).
# The deployed onnxwrapper resolves/executes mobilenet_v2.onnx.so.bin via libQnnHtp.so.
# Override per-run with: QAI_QNN_RUNTIME=CPU python aipc ...  (uses libmobilenet_v2.so).
export QAI_QNN_RUNTIME="${QAI_QNN_RUNTIME:-HTP}"
echo "QAIRT 2.45.0 環境載入成功！"
