#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase 7 / 6.3 — Latency benchmark on x86 Linux QNN CPU (via aipc wrapper).
Warmup runs + timed loop -> mean / p50 / p95 / FPS. Saves qairt_output/latency_benchmark.json.

MUST be launched via: python aipc bench_mobilenet_v2.py  (onnxruntime hot-patched to skill wrapper)
"""
import json
import os
import time

import numpy as np
import onnxruntime as ort  # hot-patched by aipc launcher to skill onnxwrapper (x86)

ONNX_FILE = "mobilenet_v2.onnx"
INPUT_NPY = "input.npy"
WARMUP = 3
ITERS = 20

input_np = np.load(INPUT_NPY).astype(np.float32)
sess = ort.InferenceSession(ONNX_FILE, providers=["CPUExecutionProvider"])
inp = sess.get_inputs()[0]
out = sess.get_outputs()[0]

for _ in range(WARMUP):
    sess.run([out.name], {inp.name: input_np})

lat = []
for _ in range(ITERS):
    t0 = time.perf_counter()
    sess.run([out.name], {inp.name: input_np})
    lat.append((time.perf_counter() - t0) * 1000.0)

lat = np.asarray(lat)
res = {
    "runtime": "QNN-CPU (libQnnCpu.so via aipc/onnxwrapper_x86)",
    "batch": 1,
    "warmup_runs": WARMUP,
    "timed_runs": ITERS,
    "mean_ms": float(lat.mean()),
    "p50_ms": float(np.percentile(lat, 50)),
    "p95_ms": float(np.percentile(lat, 95)),
    "min_ms": float(lat.min()),
    "max_ms": float(lat.max()),
    "fps": float(1000.0 / lat.mean()),
}
out_dir = "qairt_output"
os.makedirs(out_dir, exist_ok=True)
with open(os.path.join(out_dir, "latency_benchmark.json"), "w", encoding="utf-8") as fh:
    json.dump(res, fh, indent=2)

print(json.dumps(res, indent=2))
print("PASS: benchmark recorded -> qairt_output/latency_benchmark.json")