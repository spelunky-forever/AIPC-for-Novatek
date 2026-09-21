#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Confirm aipc wrapper artifact resolution selects the HTP context (.so.bin) + libQnnHtp.so backend."""
import os

# Mimic deployed local onnxwrapper resolution with QAI_QNN_RUNTIME=HTP
from onnxwrapper import InferenceSession, SessionOptions

os.environ.setdefault("QAI_QNN_RUNTIME", "HTP")
opts = SessionOptions()
print("qnn_runtime:", opts.qnn_runtime)

sess = InferenceSession("mobilenet_v2.onnx", sess_options=opts)
print("selected backend model:", sess.backend_model)
print("is_context (.bin selected):", sess.backend_model.endswith(".bin"))
print("backend lib:", sess._qnn_backend_lib())
print("resolve command flag:", "--retrieve_context" if sess.backend_model.endswith(".bin") else "--model")
