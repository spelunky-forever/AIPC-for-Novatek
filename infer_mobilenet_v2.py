#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase 6 / QNN-6 Inference — MobileNetV2 (x86 Linux, CPU backend).

PREFACE: This script MUST be launched via the skill `aipc` launcher so that
`import onnxruntime` is hot-patched to the skill's `onnxwrapper.py`
(x86 flavor on this host). Never call QNNContext directly.

Usage:
    source env_setup.sh
    python aipc infer_mobilenet_v2.py
"""
import json
import os
import time

import numpy as np
import onnxruntime as ort  # hot-patched by `aipc` to skill onnxwrapper

ONNX_FILE = "mobilenet_v2.onnx"
INPUT_NPY = "input.npy"
GOLDEN_NPY = "golden_output.npy"
QNN_OUT_NPY = "qnn_output.npy"
RESULT_TXT = "real_inference_output.txt"

# 1. Load input exactly as exported (same seed-42 tensor as baseline/ORT)
input_np = np.load(INPUT_NPY).astype(np.float32)
assert input_np.shape == (1, 3, 224, 224), f"unexpected input shape {input_np.shape}"

# 2. Inference via skill wrapper (QNN CPU backend on x86 Linux)
start = time.monotonic()
sess = ort.InferenceSession(ONNX_FILE, providers=["CPUExecutionProvider"])
inp = sess.get_inputs()[0]
out = sess.get_outputs()[0]
print(f"[preflight] input  -> {inp.name} {list(inp.shape)} {inp.type}")
print(f"[preflight] output -> {out.name} {list(out.shape)} {out.type}")

qnn_out = sess.run([out.name], {inp.name: input_np})[0]
elapsed_ms = (time.monotonic() - start) * 1000.0
qnn_out = np.asarray(qnn_out, dtype=np.float32)
np.save(QNN_OUT_NPY, qnn_out)
print(f"[run] QNN output shape: {qnn_out.shape}, dtype: {qnn_out.dtype}")
print(f"[run] elapsed (incl. session init): {elapsed_ms:.2f} ms")

# 3. Load golden (PyTorch baseline) and compare
golden = np.load(GOLDEN_NPY).astype(np.float32)


def cosine_similarity(a, b):
    a = a.flatten()
    b = b.flatten()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


cos = cosine_similarity(qnn_out, golden)
print(f"[metrics] cosine similarity (QNN vs PyTorch golden) = {cos:.8f}")

# 4. Decoded-output check (classification): Top-1 / Top-5 index match
qnn_top1 = int(np.argmax(qnn_out[0]))
ref_top1 = int(np.argmax(golden[0]))
qnn_top5 = set(np.argsort(qnn_out[0])[::-1][:5].tolist())
ref_top5 = set(np.argsort(golden[0])[::-1][:5].tolist())
top1_match = qnn_top1 == ref_top1
top5_match = qnn_top5 == ref_top5
print(f"[decoded] Top-1: QNN={qnn_top1} Golden={ref_top1} -> match={top1_match}")
print(f"[decoded] Top-5 sets equal -> match={top5_match}")

# 5. Acceptance criteria
accept_raw = cos >= 0.95          # user acceptance threshold (INT8 floor)
higher_bar = cos >= 0.99          # AGENTS.md FP validation floor
print(f"[acceptance] cosine >= 0.95 (user threshold): {accept_raw}")
print(f"[acceptance] cosine >= 0.99 (AGENTS FP floor): {higher_bar}")

results = {
    "model": "mobilenet_v2",
    "runtime": os.environ.get("QAI_QNN_RUNTIME", "CPU").upper()
    + " via aipc/onnxwrapper"
    + (" (libQnnHtp.so, context .so.bin)" if os.environ.get("QAI_QNN_RUNTIME", "CPU").upper() == "HTP" else " (libQnnCpu.so)"),
    "input": {"name": inp.name, "shape": list(inp.shape)},
    "output": {"name": out.name, "shape": list(out.shape)},
    "qnn_output_file": QNN_OUT_NPY,
    "cosine_similarity_vs_golden": cos,
    "elapsed_ms": elapsed_ms,
    "top1_index": {"qnn": qnn_top1, "golden": ref_top1, "match": top1_match},
    "top5_match": top5_match,
    "acceptance_cosine_0_95": bool(accept_raw),
    "acceptance_cosine_0_99": bool(higher_bar),
    "status": "PASS" if accept_raw else "FAIL",
}
with open(RESULT_TXT, "w", encoding="utf-8") as fh:
    fh.write(json.dumps(results, indent=2))
print(f"[result] saved -> {RESULT_TXT}")

assert accept_raw, f"FAIL: cosine similarity {cos:.6f} < 0.95"
print("\n=== FINAL: ACCEPTANCE PASS (Cosine Similarity >= 0.95) ===")