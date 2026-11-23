import torch
import torchvision.models as models
import onnx
import onnxsim

# Load the ResNet18 model with pretrained weights
model = models.resnet18(pretrained=True)
model.eval()  # Set to evaluation mode

# Create a dummy input tensor with the appropriate shape
# ResNet18 expects 3-channel images of size 224x224
input_shape = (1, 3, 224, 224)
dummy_input = torch.randn(input_shape, requires_grad=False)

# Export the model to ONNX format
onnx_file = "resnet18.onnx"
with torch.no_grad():
    torch.onnx.export(
        model,  # model to export
        dummy_input,  # dummy input
        onnx_file,  # output file
        opset_version=18,  # recommended opset version
        input_names=["input"],  # input name
        output_names=["output"],  # output name
        do_constant_folding=False,  # disable constant folding
        verbose=False,  # don't print export details
        # dynamo=False,  # not needed for this model
    )

# Validate the ONNX model
print("Validating ONNX model...")
onnx.checker.check_model(onnx_file)

# Simplify the model with onnxsim
print("Simplifying ONNX model with onnxsim...")
try:
    model_sim, check = onnxsim.simplify(onnx_file)
    if check:
        print("Model simplified successfully")
        onnx.save(model_sim, onnx_file)
    else:
        print("Model simplification failed")
except Exception as e:
    print(f"onnxsim failed: {e}")

print(f"ResNet18 model exported successfully to {onnx_file}")