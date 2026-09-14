#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase 7.1 / 6.1 — Accuracy comparison: ONNX CPU baseline vs QNN output vs PyTorch golden.
Computes cosine similarity, SNR, MAE, max abs diff. Saves onnx_cpu_output.npy + metrics JSON.

Run DIRECTLY (plain onnxruntime = baseline; NOT via aipc).
"""
import json
import numpy as np
import onnxruntime as ort

ONNX_FILE = "mobilenet_v2.onnx"
INPUT_NPY = "input.npy"
GOLDEN_NPY = "golden_output.npy"
QNN_NPY = "qnn_output.npy"
ONNX_CPU_NPY = "onnx_cpu_output.npy"
META_JSON = "metrics_phase7.json"

input_np = np.load(INPUT_NPY).astype(np.float32)
golden = np.load(GOLDEN_NPY).astype(np.float32)
qnn = np.load(QNN_NPY).astype(np.float32)

# --- ONNX CPU baseline ---
sess = ort.InferenceSession(ONNX_FILE, providers=["CPUExecutionProvider"])
inp = sess.get_inputs()[0]
onnx_cpu = sess.run([sess.get_outputs()[0].name], {inp.name: input_np})[0]
onnx_cpu = np.asarray(onnx_cpu, dtype=np.float32)
np.save(ONNX_CPU_NPY, onnx_cpu)


def cosine(a, b):
    a = a.flatten().astype(np.float64)
    b = b.flatten().astype(np.float64)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def snr_db(ref, test):
    ref = ref.flatten().astype(np.float64)
    test = test.flatten().astype(np.float64)
    noise = ref - test
    sig_pow = np.mean(ref ** 2)
    noise_pow = np.mean(noise ** 2) + 1e-30
    return float(10.0 * np.log10(sig_pow / noise_pow))


metrics = {
    "onnx_vs_golden": {
        "cosine": cosine(onnx_cpu, golden),
        "snr_db": snr_db(golden, onnx_cpu),
        "mae": float(np.mean(np.abs(onnx_cpu - golden))),
        "max_abs_diff": float(np.max(np.abs(onnx_cpu - golden))),
    },
    "qnn_vs_golden": {
        "cosine": cosine(qnn, golden),
        "snr_db": snr_db(golden, qnn),
        "mae": float(np.mean(np.abs(qnn - golden))),
        "max_abs_diff": float(np.max(np.abs(qnn - golden))),
    },
    "qnn_vs_onnx_cpu": {
        "cosine": cosine(qnn, onnx_cpu),
        "snr_db": snr_db(onnx_cpu, qnn),
        "mae": float(np.mean(np.abs(qnn - onnx_cpu))),
        "max_abs_diff": float(np.max(np.abs(qnn - onnx_cpu))),
    },
}
with open(META_JSON, "w", encoding="utf-8") as fh:
    json.dump(metrics, fh, indent=2)

print(json.dumps(metrics, indent=2))
ok = metrics["qnn_vs_golden"]["cosine"] >= 0.99 and metrics["qnn_vs_golden"]["snr_db"] > 30
print("PASS" if ok else "FAIL", "(cosine>=0.99 & SNR>30 dB vs golden)")