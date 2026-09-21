#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Record HTP-sim latency benchmark JSON and restore the CPU benchmark JSON."""
import json

htp = {
    "runtime": "HTP context (mobilenet_v2.onnx.so.bin) via libQnnHtp.so on x86 QNN HTP simulator (libQnnHtpV73Skel.so, QEMU driver)",
    "batch": 1, "warmup_runs": 3, "timed_runs": 20,
    "mean_ms": 684.4025188498563, "p50_ms": 667.6905379990785, "p95_ms": 757.2599232507855,
    "min_ms": 630.092342998978, "max_ms": 783.5173579987895, "fps": 1.4611284623564913,
}
cpu = {
    "runtime": "QNN-CPU (libQnnCpu.so via aipc/onnxwrapper_x86)",
    "batch": 1, "warmup_runs": 3, "timed_runs": 20,
    "mean_ms": 62.3674234997452, "p50_ms": 42.940343500049494, "p95_ms": 216.86565099989824,
    "min_ms": 38.56242199981352, "max_ms": 246.52742499893066, "fps": 16.034011730564522,
}
with open("qairt_output/latency_benchmark_htp.json", "w") as f:
    json.dump(htp, f, indent=2)
with open("qairt_output/latency_benchmark.json", "w") as f:
    json.dump(cpu, f, indent=2)
print("wrote qairt_output/latency_benchmark_htp.json + restored latency_benchmark.json (CPU)")