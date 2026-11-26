import torch
import torchvision
import onnx
from onnxsim import simplify

def export_resnet34_onnx():
    # Load the pretrained ResNet34 model
    model = torch.hub.load('pytorch/vision:v0.19.0', 'resnet34', pretrained=True)
    model.eval()

    # Create a dummy input tensor (batch_size=1, channels=3, height=224, width=224)
    dummy_input = torch.randn(1, 3, 224, 224)

    # Export the model to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        "resnet34.onnx",
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=["input"]
    )

    # Simplify the ONNX model
    onnx_model, check = simplify("resnet34.onnx")
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(onnx_model, "resnet34.onnx")

    print("ONNX model exported and simplified successfully")

if __name__ == "__main__":
    export_resnet34_onnx()