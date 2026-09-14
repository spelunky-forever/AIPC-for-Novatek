#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase 1/2 baseline: load torchvision MobileNetV2 (pretrained, eval mode),
generate a deterministic random input [1, 3, 224, 224] float32,
save `input.npy` and the PyTorch golden output `golden_output.npy`.
"""
import numpy as np
import torch
import torchvision.models as models

torch.manual_seed(42)
np.random.seed(42)

DEVICE = "cpu"

# 1. Load pretrained MobileNetV2 and switch to eval mode
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
model.to(DEVICE).eval()

# 2. Deterministic random input [1, 3, 224, 224] float32
dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32, device=DEVICE)

# 3. Golden output
with torch.inference_mode():
    golden = model(dummy_input)

# 4. Save artifacts
np.save("input.npy", dummy_input.cpu().numpy())
np.save("golden_output.npy", golden.cpu().numpy())

print("Input shape:", tuple(dummy_input.shape), "dtype:", dummy_input.dtype)
print("Golden output shape:", tuple(golden.shape), "dtype:", golden.dtype)
print("Golden output finite:", torch.isfinite(golden).all().item(),
      "| nonzero count:", int((golden != 0).sum().item()))
print("Saved input.npy and golden_output.npy")