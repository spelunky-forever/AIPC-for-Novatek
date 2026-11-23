#!/usr/bin/env python
# ResNet18 inference script

import os
import torch
import numpy as np
from PIL import Image

# Load the ResNet18 model
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
model.eval()

# Preprocess the input image
input_image = Image.open('example.jpg')
input_tensor = torch.from_numpy(np.array(input_image))
input_tensor = input_tensor.permute(2, 0, 1).unsqueeze(0).float() / 255.0

# Run inference
with torch.no_grad():
    output = model(input_tensor)

# Print the output
print(output)

# Save the output
np.save('output.npy', output.numpy())
print('Inference completed and output saved to output.npy')