#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase 2: export MobileNetV2 to `mobilenet_v2.onnx`.
- opset_version=13 (Safe per aipc-toolkit reference; user-requested)
- do_constant_folding=True
- fixed input/output shapes (no dynamic axes)
"""
import torch
import torchvision.models as models

torch.manual_seed(42)

model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
model.eval()

dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)

ONNX_FILE = "mobilenet_v2.onnx"
INPUT_NAME = "input"
OUTPUT_NAME = "output"

torch.onnx.export(
    model,
    dummy_input,
    ONNX_FILE,
    export_params=True,          # store trained weights inside the model file
    opset_version=13,
    do_constant_folding=True,
    input_names=[INPUT_NAME],
    output_names=[OUTPUT_NAME],
    dynamic_axes=None,           # fixed shapes — no dynamic axes
)

print(f"Exported {ONNX_FILE} (opset 13, fixed I/O shapes, constant folding ON)")