import torch
from PIL import Image
import numpy as np
from vlm_reasoner import VLMReasoner

def run_test(image_path):
    """
    Standalone test function for Qwen3-VL-4B-Instruct.
    """
    print(f"--- Testing Qwen3-VL on: {image_path} ---")
    
    # Initialize the reasoner (Full FP16)
    reasoner = VLMReasoner(model_id="Qwen/Qwen3-VL-4B-Instruct")
    
    if reasoner.model is None:
        return "ERROR: Model could not be loaded."

    # Load image
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)

    # Perform reasoning
    print("Reasoning in progress...")
    output = reasoner.reason_occlusion(image_np)
    
    return output

if __name__ == "__main__":
    # You can run this directly from terminal/colab
    path = "test/dogfrisbee.jpg"
    print(run_test(path))
