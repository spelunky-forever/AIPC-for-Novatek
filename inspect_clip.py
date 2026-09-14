#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Inspect Clip node versions and min/max representation in mobilenet_v2.onnx."""
import onnx
from collections import Counter

m = onnx.load("mobilenet_v2.onnx")
print("opset_imports:", [(o.domain, o.version) for o in m.opset_import])
ops = Counter(n.op_type for n in m.graph.node)
print("op counts:", dict(ops))
clips = [n for n in m.graph.node if n.op_type == "Clip"]
print("num Clip nodes:", len(clips))
if clips:
    n = clips[0]
    print("sample Clip node name:", n.name)
    print("sample Clip node inputs:", list(n.input))
    print("sample Clip node attrs:", [(a.name, a.i, a.f) for a in n.attribute])
    gi = {v.name for v in m.graph.input}
    initializers = {i.name for i in m.graph.initializer}
    for f in n.input:
        print("  input:", f, "| is_graph_input:", f in gi, "| is_initializer:", f in initializers)