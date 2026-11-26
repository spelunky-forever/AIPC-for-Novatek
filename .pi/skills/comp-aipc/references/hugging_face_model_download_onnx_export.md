# Hugging Face Model Download & ONNX Export

Use this guide to download pretrained Hugging Face models and export clean FP32 ONNX graphs for QAIRT conversion (`references/conversion_and_quantization.md`).

> **Scope**: Vision, audio, and encoder models (e.g., ResNet, ViT, ConvNeXt, Whisper encoder). For CausalLM / LLM KV-cache splitting, use `references/llm_genie.md` and `references/pytorch_modification.md`.

---

## 1. Environment Guardrails (Read First)

- **DO NOT force-reinstall `torch` / `torchvision`**: Never run `pip install --force-reinstall` on PyTorch packages inside the QAIRT environment. It breaks CPU/CUDA bindings and upgrades `numpy >= 2.0`, which causes QAIRT converter failures.
- **DO NOT use `torch.hub.load()` with old version tags**: Avoid `torch.hub.load('pytorch/vision:v0.x.x', ...)`—it triggers `RuntimeError: operator torchvision::nms does not exist`.
- **Install missing HF packages safely**:
  ```bash
  pip install transformers timm huggingface_hub safetensors "numpy<2.0"
  ```
- **Offline / Local Cache**: If `HF_HOME` or a local snapshot directory already exists, pass `local_files_only=True` to `from_pretrained()`.

---

## 2. Standard Download & Export Templates

Always load models on **CPU** in **FP32** (`torch.float32`) and call `model.eval()` before ONNX export.

### Option A: `transformers` Models (Recommended for Standard HF Repos)
Hugging Face `transformers` models return `ModelOutput` objects by default. Wrap the forward pass to return raw tensors cleanly:

```python
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification

MODEL_ID = "microsoft/resnet-50"  # Replace with target HF repo ID or local path

processor = AutoImageProcessor.from_pretrained(MODEL_ID)
hf_model = AutoModelForImageClassification.from_pretrained(
    MODEL_ID, torch_dtype=torch.float32
).eval()

class OnnxWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values):
        return self.model(pixel_values=pixel_values).logits

# Read expected resolution from processor (defaults to 224x224 if absent)
size = processor.size
H = size.get("height", size.get("shortest_edge", 224))
W = size.get("width", size.get("shortest_edge", 224))
dummy = torch.zeros(1, 3, H, W, dtype=torch.float32)

wrapper = OnnxWrapper(hf_model).eval()
torch.onnx.export(
    wrapper,
    dummy,
    "resnet50.onnx",
    opset_version=18,
    input_names=["pixel_values"],
    output_names=["logits"],
    do_constant_folding=False,
)
```

### Option B: `timm` Vision Models (`hf_hub:*`)
`timm` models accept raw `(N, C, H, W)` tensors and return raw logits directly without a wrapper:

```python
import timm, torch

model = timm.create_model("hf_hub:timm/resnet50.a1_in1k", pretrained=True).eval()
H, W = model.default_cfg["input_size"][1:]
dummy = torch.zeros(1, 3, H, W, dtype=torch.float32)

torch.onnx.export(
    model,
    dummy,
    "resnet50.onnx",
    opset_version=18,
    input_names=["input"],
    output_names=["output"],
    do_constant_folding=False,
)
```

### Option C: Raw Weight File via `huggingface_hub`
When only a `.safetensors` / `.bin` / `.onnx` file is needed from a repo:

```python
from huggingface_hub import hf_hub_download

ckpt_path = hf_hub_download(repo_id="microsoft/resnet-50", filename="model.safetensors")
```

---

## 3. Post-Export Validation Checklist

Run this immediately after export before starting QNN/SNPE conversion:

```python
import onnx, onnxruntime as ort, numpy as np

onnx.checker.check_model("resnet50.onnx")
sess = ort.InferenceSession("resnet50.onnx", providers=["CPUExecutionProvider"])
inp = sess.get_inputs()[0]
out = sess.run(None, {inp.name: np.zeros(inp.shape, dtype=np.float32)})
print(f"Verified ONNX | Input: {inp.name} {inp.shape} | Output: {out[0].shape}")
```

1. **Static Shapes**: Ensure `inp.shape` has concrete integers (e.g., `[1, 3, 224, 224]`), not dynamic strings.
2. **Preprocessing Parity**: Record `image_mean` and `image_std` from `processor` (or `model.default_cfg`) so calibration `.raw` generation uses identical normalization.
3. **Next Step**: Proceed directly to `references/conversion_and_quantization.md` (`python scripts/aipc.py convert ...`).