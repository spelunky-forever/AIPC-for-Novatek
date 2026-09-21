#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Compute cosine similarity of HTP-sim diagnostic output vs golden."""
import numpy as np

out = np.fromfile("_htp_diag/out/Result_0/output.raw", dtype=np.float32).reshape(1, 1000)
g = np.load("golden_output.npy")
c = float(np.dot(out.flatten(), g.flatten()) / (np.linalg.norm(out) * np.linalg.norm(g) + 1e-12))
print("HTP-sim diagnostic cosine vs golden:", round(c, 8))
print("top1 qnn:", int(np.argmax(out[0])), "| top1 golden:", int(np.argmax(g[0])))
