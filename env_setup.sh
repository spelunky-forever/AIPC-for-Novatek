#!/usr/bin/env bash
# QAIRT environment bootstrap
export QAIRT_SDK_ROOT="/home/spelunky-forever/workplace/AIPC-for-Novatek/toolchains/qairt/2.45.0.260326"

# 載入 Python 虛擬環境
source $(conda info --base)/etc/profile.d/conda.sh
conda activate aipc

export PYTHONPATH="${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
source "${QAIRT_SDK_ROOT}/bin/envsetup.sh"

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
echo "QAIRT 2.45.0 環境載入成功！"
