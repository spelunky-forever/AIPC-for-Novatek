# Attempt 1
<!-- If you are agent, please ignore this file-->

## Model Name
MobileNetV2

## Stage 1: Plan & Configuration Synchronization
執行目標：確認 Agent 已讀取並鎖定根目錄 aipc_plan.md 的 ## Config 設定值，並完成內部參數解析。

Prompt
```
Follow aipc-toolkit skill strictly. I have manually configured the `## Config` section in aipc_plan.md for MobileNetV2 CPU bring-up on x86 Linux.
Please inspect aipc_plan.md, confirm that all configuration variables (MODEL_NAME, FLOW, TARGET_DEVICE, PRECISION, QAIRT_ROOT, HOST_ARCH, TARGET_ARCH) are recognized, and summarize the execution plan for Phase 1 to Phase 5. Do not modify any code yet.
```

檢驗方法：

- 觀察 Agent 回應，確認輸出清單中 TARGET_DEVICE 為 x86 Linux、PRECISION 為 FP32、CONTEXT_BINARY_GEN 為 NO。

- 確認 Agent 未擅自重寫或損壞 aipc_plan.md 的既有結構。

## Stage 2: Model Baseline & ONNX Export (Phase 1 & 2)
執行目標：
1. 載入 PyTorch 預訓練 MobileNetV2，以固定尺寸 $[1, 3, 224, 224]$ 執行一次前向傳播，產出並儲存 input.npy 與 golden_output.npy。
2. 將模型導出為靜態圖 mobilenet_v2.onnx（Opset 13）。
3. 使用 ONNX Runtime 載入該 ONNX 執行推理，比對 ONNX 輸出與 PyTorch Golden 輸出的餘弦相似度（Cosine Similarity），驗證導出無損。

Prompt
```
Follow aipc-toolkit skill strictly. Execute Phase 1 and Phase 2:
1. Create and run a baseline script using PyTorch:
   - Load `torchvision.models.mobilenet_v2(weights="DEFAULT")` and switch to eval mode.
   - Generate a deterministic/random input tensor with shape [1, 3, 224, 224] (float32).
   - Save the input array as `input.npy` and the PyTorch golden output as `golden_output.npy`.
2. Export the model to `mobilenet_v2.onnx` with opset_version=13, do_constant_folding=True, and fixed input/output shapes (no dynamic axes).
3. Verify the exported ONNX model using onnx.checker and run inference with ONNX Runtime using `input.npy`.
4. Compare ONNX Runtime output with `golden_output.npy`. Assert that Cosine Similarity is >= 0.95.
Update the progress in aipc_plan.md once validated.
```

檢驗方法：

- 檔案檢查：在專案根目錄執行 ls -lh mobilenet_v2.onnx input.npy golden_output.npy，確認三個檔案皆已生成。

- 維度與精度檢查：在終端機執行快速確認：
```Bash
python -c "import numpy as np; inp = np.load('input.npy'); out = np.load('golden_output.npy'); print('Input shape:', inp.shape, '| Output shape:', out.shape)"
```

- 輸出必須為 Input shape: (1, 3, 224, 224) | Output shape: (1, 1000)。

- 終端機或 Agent 日誌中必須印出 Cosine Similarity >= 0.999 的通過訊息。

## Stage 3: ONNX Inspection & Compatibility Check (Phase 3)
執行目標：使用 Skill 自帶的檢測腳本檢查 ONNX 輸入輸出節點名稱與形狀，並透過 QAIRT 轉譯器進行乾跑（Dry-run），確認無缺失算子。

Prompt
```
Follow aipc-toolkit skill strictly. Execute Phase 3 (Inspection):
1. Run the inspection script:
   python .cline/skills/QAI-Runner-Skill/scripts/aipc_inspect_onnxio.py mobilenet_v2.onnx
2. Perform a dry-run conversion using QAIRT converter tools to ensure 100% operator compatibility on the QNN backend.
3. Record the input tensor name, input dimensions, output tensor name, and dry-run compatibility results into aipc_plan.md.
```

- 檢驗方法：

- 檢查命令輸出中是否明確解析出輸入張量名稱（通常為 input 或 onnx::Conv_0）與維度 [1, 3, 224, 224]。

- 確認乾跑結果顯示 Total unsupported operators: 0。

## Stage 4: QNN Float Conversion & Model Compilation (Phase 4)
執行目標：調用 Skill 專用的 FP 轉譯包裝腳本 aipc_convert_fp.py，將 mobilenet_v2.onnx 轉換為 QNN 中介表達（C++ / Binary），並由 SDK 工具鏈編譯為本機 x86 架構的動態函式庫 .so。

Prompt
```
Follow aipc-toolkit skill strictly. Execute Phase 4 (QNN Float Conversion):
1. Source the environment script if needed, and execute the conversion using the skill wrapper:
   python .cline/skills/QAI-Runner-Skill/scripts/aipc_convert_fp.py --onnx mobilenet_v2.onnx --output_dir qairt_output
2. Verify that the QNN model library (.so) is successfully generated under `qairt_output/` matching the host architecture x86_64-linux-clang.
Do not call raw converter binaries directly; adhere to the wrapper script requirement.
```

檢驗方法：

- 檔案產物檢查：檢查 qairt_output 目錄：
```Bash
ls -lh qairt_output/
必須出現編譯完成的動態庫檔案（如 libmobilenet_v2.so 或 x86_64-linux-clang/libmobilenet_v2.so）以及 QNN 圖結構檔（mobilenet_v2.cpp、mobilenet_v2.bin）。
```

- 檢查日誌中無任何 Segmentation fault 或 Failed to generate model library 錯誤。

## Stage 5: CPU Backend Inference & Accuracy Acceptance (Phase 6)
執行目標：依據防護守則（Guardrails），必須使用 scripts/aipc wrapper 與 Skill 修補過的 scripts/onnxwrapper.py 加載編譯產物，在 CPU 後端運行推理，計算 QNN 推理輸出與 PyTorch Golden 輸出的餘弦相似度是否符合 $\ge 0.95$ 標準。

Prompt
```
Follow aipc-toolkit skill strictly. Because config setting says CONTEXT_BINARY_GEN = NO, so directly execute Phase 6 (Inference Validation & Acceptance):
1. Preflight check: Ensure the inference wrapper uses `scripts/aipc` and the patched `scripts/onnxwrapper.py` provided by the skill. Never fallback to the bundled SDK copy.
2. Execute inference using the CPU backend on x86 Linux with `input.npy`.
3. Load the resulting QNN inference output and compute the Cosine Similarity against `golden_output.npy`.
4. Confirm whether the acceptance threshold (Cosine Similarity >= 0.95) is satisfied.
5. Record the final metrics, mark Phase 5 as completed in aipc_plan.md, and display the final acceptance summary.
```

檢驗方法：
- 數值門檻：終端最後輸出的比對指標：$$\text{Cosine Similarity} \ge 0.95$$在 FP32 精度下，CPU 後端的比對結果通常落在 $0.999 \sim 1.000$ 之間。

- 計畫看板確認：開啟 aipc_plan.md，確認 Phase 1 至 Phase 5 的核取方塊均已被 Agent 標記為完成 [x]，且附帶了數值相似度與耗時記錄。

## Stage 6: Wrap up
Prompt
```
Follow aipc-toolkit skill strictly. Please proceed with Phase 7 to generate REPORT.md, summarize the task completion metrics, and close the project bring-up.
```