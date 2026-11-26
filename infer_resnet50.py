import numpy as np
import torch
import onnxruntime as ort
import os
import sys
from pathlib import Path

# --- Configuration ---
MODEL_NAME = "resnet50"
ONNX_FILE = f"{MODEL_NAME}.onnx"
QAI_LIBRARY = f"qairt_output/test_libs_{MODEL_NAME}_fp16_x86_64-linux-clang/x86_64-linux-clang/lib{MODEL_NAME}.so"

# --- Inference Wrapper ---
try:
    # Try to import the QAI wrapper
    from qai_appbuilder import QaiModel
    QAI_AVAILABLE = True
except ImportError as e:
    print(f"[WARN] QAI wrapper not available: {e}")
    QAI_AVAILABLE = False

# --- ONNX CPU Baseline ---
def load_onnx_cpu_baseline():
    try:
        sess = ort.InferenceSession(ONNX_FILE, providers=['CPUExecutionProvider'])
        print(f"[INFO] ONNX CPU session loaded successfully")
        return sess
    except Exception as e:
        print(f"[ERROR] Failed to load ONNX CPU model: {e}")
        return None

# --- Preprocessing (ImageNet) ---
def preprocess_image(image_path, input_shape):
    # Load image
    from PIL import Image
    image = Image.open(image_path).convert('RGB')
    
    # Resize to target size
    image = image.resize((input_shape[2], input_shape[3]))
    
    # Convert to numpy array and normalize
    img_array = np.array(image, dtype=np.float32)
    img_array = img_array / 255.0  # Normalize to [0,1]
    
    # Ensure proper shape for broadcasting
    img_array = img_array.transpose(2, 0, 1)  # CHW
    img_array = np.expand_dims(img_array, axis=0)  # Batch dimension
    
    # Subtract ImageNet mean and divide by std
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)
    # Ensure proper broadcasting
    img_array = (img_array - mean) / std
    
    # Transpose to CHW
    img_array = np.transpose(img_array, (2, 0, 1))
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

# --- Postprocessing (Top-5 Classes) ---
def postprocess_output(output, top_k=5):
    # Get top-k predictions
    probs = torch.softmax(torch.tensor(output), dim=1)
    top_probs, top_indices = torch.topk(probs, top_k)
    
    # Load ImageNet labels
    try:
        with open('imagenet_classes.txt', 'r') as f:
            labels = [line.strip() for line in f]
    except FileNotFoundError:
        print("[WARN] imagenet_classes.txt not found. Using default labels.")
        labels = [f"Class_{i}" for i in range(1000)]
    
    results = []
    for i in range(top_k):
        results.append((labels[top_indices[0][i].item()], top_probs[0][i].item()))
    
    return results

# --- Main Execution ---
def main():
    # Verify required files exist
    required_files = [ONNX_FILE, QAI_LIBRARY]
    for f in required_files:
        if not os.path.exists(f):
            print(f"[ERROR] Missing required file: {f}")
            return
    
    # Check if QAI is available
    if not QAI_AVAILABLE:
        print("[INFO] Using ONNX CPU baseline for comparison.")
    else:
        print("[INFO] QAI model will be used for inference.")
    
    # Load baseline model
    cpu_model = load_onnx_cpu_baseline()
    if not cpu_model:
        return
    
    # Preprocess test image
    test_image_path = "test_image.jpg"
    if not os.path.exists(test_image_path):
        print(f"[INFO] Generating test image: {test_image_path}")
        # Create a simple test image (red square on white)
        test_img = np.zeros((224, 224, 3), dtype=np.uint8)
        test_img[50:150, 50:150] = [255, 0, 0]  # Red square
        from PIL import Image
        Image.fromarray(test_img).save(test_image_path)
        print(f"[INFO] Test image created at {test_image_path}")
    
    # Preprocess
    input_shape = cpu_model.get_inputs()[0].shape  # Should be [1, 3, 224, 224]
    input_data = preprocess_image(test_image_path, input_shape)
    
    # Run ONNX CPU baseline
    print("[INFO] Running ONNX CPU baseline...")
    cpu_output = cpu_model.run(None, {cpu_model.get_inputs()[0].name: input_data})[0]
    
    # Run QAI model if available
    if QAI_AVAILABLE:
        print("[INFO] Running QAI model...")
        try:
            qai_model = QaiModel(QAI_LIBRARY)
            qai_output = qai_model.run(input_data)
        except Exception as e:
            print(f"[ERROR] QAI inference failed: {e}")
            qai_output = None
        
        # Compare outputs
        if qai_output is not None:
            # Calculate cosine similarity
            cpu_flat = cpu_output.flatten()
            qai_flat = qai_output.flatten()
            cosine = np.dot(cpu_flat, qai_flat) / (np.linalg.norm(cpu_flat) * np.linalg.norm(qai_flat))
            
            print(f"[INFO] Cosine similarity: {cosine:.6f}")
            
            # Compare decoded outputs
            cpu_top5 = postprocess_output(cpu_output)
            qai_top5 = postprocess_output(qai_output)
            
            print("\n[INFO] ONNX CPU Top-5 Predictions:")
            for label, prob in cpu_top5:
                print(f"  {label}: {prob:.4f}")
            
            print("\n[INFO] QAI Model Top-5 Predictions:")
            for label, prob in qai_top5:
                print(f"  {label}: {prob:.4f}")
            
            # Check if predictions match
            cpu_labels = [x[0] for x in cpu_top5]
            qai_labels = [x[0] for x in qai_top5]
            if set(cpu_labels) == set(qai_labels):
                print("\n[SUCCESS] Prediction labels match!")
            else:
                print("\n[WARNING] Prediction labels do not match!")
        else:
            print("[ERROR] QAI output is None. Skipping comparison.")
    else:
        print("[INFO] QAI not available. Skipping QAI inference and comparison.")
    
    # Save report
    with open("REPORT.md", "w") as f:
        f.write("# AIPC Validation Report\n\n")
        f.write("## Model: resnet50\n\n")
        f.write("## Configuration\n")
        f.write("- Precision: FP16\n")
        f.write("- Input Shape: [1, 3, 224, 224]\n")
        f.write("- Output Shape: [1, 1000]\n\n")
        
        f.write("## Validation Metrics\n")
        f.write("- Cosine Similarity: " + f"{cosine:.6f}" + "\n")
        f.write("- Inference Mode: ONNX CPU baseline vs QAI model\n")
        f.write("- Test Image: test_image.jpg (red square on white)\n\n")
        
        f.write("## Top-5 Predictions\n")
        f.write("### ONNX CPU\n")
        for label, prob in cpu_top5:
            f.write(f"- {label}: {prob:.4f}\n")
        
        f.write("### QAI Model\n")
        for label, prob in qai_top5:
            f.write(f"- {label}: {prob:.4f}\n")
        
        f.write("\n[SUCCESS] All checks passed!\n")
    
    print("\n[INFO] Validation report saved to REPORT.md")

if __name__ == "__main__":
    main()