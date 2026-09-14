import urllib.request
import json
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import onnxruntime as ort

IMAGE_PATH = "test/image.jpg"

# 1. 取得 ImageNet 1000 類別標籤文字
print("[1/4] 下載 ImageNet 類別標籤字典...")
url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
labels = json.loads(urllib.request.urlopen(url).read().decode())

# 2. 影像前處理 (標準 ImageNet 流程: Resize -> CenterCrop -> Normalize)
print(f"[2/4] 讀取並前處理圖片: {IMAGE_PATH}")
img = Image.open(IMAGE_PATH).convert("RGB")
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(img).unsqueeze(0).numpy().astype(np.float32)

# 3. 執行推論 (透過 AIPC onnxwrapper / QNN CPU 推論)
print("[3/4] 執行 QNN 模型推論...")
session = ort.InferenceSession("mobilenet_v2.onnx")
outputs = session.run(["output"], {"input": input_tensor})[0]

# 4. 後處理：計算機率百分比並抓出 Top-5
print("\n[4/4] 辨識結果 (Top-5):")
print("=" * 45)
# Softmax
exp_scores = np.exp(outputs[0] - np.max(outputs[0]))
probs = exp_scores / np.sum(exp_scores)

top5_idx = np.argsort(probs)[::-1][:5]

for rank, idx in enumerate(top5_idx, 1):
    label_name = labels[idx]
    confidence = probs[idx] * 100
    print(f"Top-{rank}: [{confidence:6.2f}%] (類別 ID {idx:3d}) -> {label_name}")
print("=" * 45)