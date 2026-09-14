#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase 2 verification:
1. onnx.checker.check_model()
2. onnxsim simplification (wrapped in try/except per aipc-toolkit reference)
3. ONNX Runtime CPU inference using input.npy
4. Cosine similarity vs golden_output.npy (assert >= 0.95)
"""
import numpy as np
import onnx
import onnxruntime as ort

ONNX_FILE = "mobilenet_v2.onnx"

# 1. onnx checker
model = onnx.load(ONNX_FILE)
onnx.checker.check_model(model)
print("onnx.checker.check_model: OK")

# 2. onnxsim (do not block on failure — model remains valid for conversion)
try:
    from onnxsim import simplify
    model_sim, ok = simplify(model)
    if ok:
        model = model_sim
        onnx.save(model, ONNX_FILE)
        print("onnxsim: simplified and saved")
    else:
        print("onnxsim: no further simplification possible")
except Exception as e:  # noqa: BLE001
    print(f"onnxsim skipped ({e})")

# 3. ONNX Runtime inference with input.npy
sess = ort.InferenceSession(ONNX_FILE, providers=["CPUExecutionProvider"])
input_name = sess.get_inputs()[0].name
output_info = sess.get_outputs()[0]
print(f"ORT I/O: {input_name} ({sess.get_inputs()[0].shape}) -> "
      f"{output_info.name} ({output_info.shape})")

input_np = np.load("input.npy")
golden = np.load("golden_output.npy")
ort_out = sess.run([output_info.name], {input_name: input_np})[0]

# 4. Cosine similarity
def cosine_similarity(a, b):
    a = np.asarray(a, dtype=np.float32).flatten()
    b = np.asarray(b, dtype=np.float32).flatten()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

sim = cosine_similarity(ort_out, golden)
print(f"ORT output shape: {ort_out.shape} | golden shape: {golden.shape}")
print(f"Cosine similarity = {sim:.8f}")

assert ort_out.shape == golden.shape, f"shape mismatch: {ort_out.shape} vs {golden.shape}"
assert sim >= 0.95, f"FAIL: cosine similarity {sim:.6f} < 0.95"
print("PASS: ONNX Runtime output matches PyTorch golden (Cosine Similarity >= 0.95)")