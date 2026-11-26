import os
import ctypes
import numpy as np
from typing import List, Dict, Any, Tuple

# --- Configuration ---
SO_FILE = "qairt_output/test_libs_resnet34_fp16_x86_64-linux-clang/x86_64-linux-clang/libresnet34.so"

# --- Wrapper Class ---
class QnnWrapper:
    """Wrapper class for Qualcomm QNN model library"""
    
    def __init__(self, so_file: str = SO_FILE):
        self.so_file = so_file
        self.lib = None
        self.model = None
        self.run_func = None
        self.unload_func = None
        self.model_instance = None
        
    def load(self) -> bool:
        """Load the QNN model library"""
        try:
            self.lib = ctypes.CDLL(self.so_file)
            # Get model initialization function
            self.model = self.lib.QnnModel_init
            self.model.restype = ctypes.c_void_p
            
            # Set up input/output function signatures
            self.run_func = self.lib.QnnModel_run
            self.run_func.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)]
            self.run_func.restype = ctypes.c_int
            
            # Set up cleanup function
            self.unload_func = self.lib.QnnModel_deinit
            self.unload_func.argtypes = [ctypes.c_void_p]
            self.unload_func.restype = None
            
            self.model_instance = self.model()
            if not self.model_instance:
                raise RuntimeError("Failed to initialize QNN model")
                
            print(f"QNN model loaded successfully from {self.so_file}")
            return True
            
        except Exception as e:
            print(f"Error loading QNN model: {e}")
            return False
    
    def run(self, input_data: np.ndarray) -> np.ndarray:
        """Run inference on input data"""
        if not self.model_instance:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        # Convert input data to float32
        input_data = input_data.astype(np.float32)
        
        # Create output buffer
        output_shape = (1, 1000)
        output_data = np.zeros(output_shape, dtype=np.float32)
        
        # Convert numpy arrays to ctypes pointers
        input_ptr = input_data.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        output_ptr = output_data.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        
        # Run inference
        result = self.run_func(self.model_instance, input_ptr, output_ptr)
        
        if result != 0:
            raise RuntimeError(f"QNN inference failed with code {result}")
        
        return output_data
    
    def unload(self):
        """Unload the QNN model library"""
        if self.model_instance:
            self.unload_func(self.model_instance)
            self.model_instance = None
        
        if self.lib:
            self.lib.close()
            self.lib = None
        
        print("QNN model unloaded")
    
    def __del__(self):
        self.unload()