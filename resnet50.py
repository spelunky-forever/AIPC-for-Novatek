import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification

# Model ID from Hugging Face
MODEL_ID = "microsoft/resnet-50"

# Load processor and model
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

# Get expected input size
size = processor.size
H = size.get("height", size.get("shortest_edge", 224))
W = size.get("width", size.get("shortest_edge", 224))

dummy = torch.zeros(1, 3, H, W, dtype=torch.float32)

# Wrap and export
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

print(f"ONNX model exported to resnet50.onnx with input shape [1, 3, {H}, {W}]")