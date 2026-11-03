# AP_ProtoSAM_Amodal

A State-Of-The-Art Two-Stage Amodal Segmentation and Completion Pipeline on Google Colab.

This repository implements a zero-shot, two-stage amodal completion pipeline. By leveraging foundation models, the pipeline accurately predicts the full shape of occluded objects and realistically inpaints their missing visual textures, regardless of the object's original bounding box resolution.

![Amodal Pipeline Demo](https://gestalt.cs.columbia.edu/assets/teaser.png) *(Reference CVPR 2025 Pix2Gestalt)*

## 🚀 Key Features and Architecture

The system has aggressively transitioned from a heuristic iterative approach to a **Two-Stage Amodal Completion Architecture**:

1. **Modal Segmentation (SAM 2)**: Detects the visible part of the occluded object.
2. **Amodal Shape Prediction (Pix2Gestalt LDM)**: Utilizes the official PyTorch Lightning weights (~15.5GB Raw Checkpoint) of the *Segment Anything Even Occluded* paper to hallucinate the complete bounding mask.
3. **Alpha Matting (CarveKit U^2Net)**: Bypasses simple thresholding. Uses a Deep Learning Matting network to cleanly separate the predicted amodal shape from the generated white background (prevents "holes" internally triggered by objects that share background colors, e.g., a panda's white fur).
4. **Geometric Integrity (Smart Padding)**: Raw aspect-ratios are heavily protected! Inputs are mathematically padded to symmetric squares (`Max(H, W)`) prior to diffusion and securely cropped down afterwards, maintaining 100% biological object shape without 256x256 square distortion.
5. **Appearance Inpainting (Stable Diffusion v2)**: Extracts the difference between the **Amodal Mask** and the **Visible Mask** and performs a single-pass, realistic inpainting onto a neutral gray `[127, 127, 127]` canvas, avoiding environmental texture hallucinations.

## 🛠️ Infrastructure & Setup

The entire workflow is heavily optimized to be run instantly on **Google Colab** (A100 or T4 GPUs).

Please see the comprehensive **[COLAB_GUIDE.md](./COLAB_GUIDE.md)** for running instructions!

### Auto-Installation (`colab_setup.py`)
Our `colab_setup.py` automatically handles the heavily complex legacy installations, including:
- Fetching specific deep-learning legacy versions: `pytorch-lightning==1.8.6`, `einops`, `carvekit-colab`, OpenAI `CLIP`.
- Auto-bypassing PyTorch 2.6's `weights_only=True` unpickling blockers for the verified Lightning checkpoints.
- Managing 15.5GB raw Git-LFS checkpoints directly via FTP (`gestalt.cs.columbia.edu`) to evade `huggingface-cli` corruption.

## 📦 Pipeline Execution

```python
from segmenter import SAM2Segmenter
from amodal_completer import AmodalCompleter

# 1. Init
segmenter = SAM2Segmenter(device="cuda")
completer = AmodalCompleter(device="cuda") # Internally boots Pix2Gestalt + SD2

# 2. Get Visible Mask
visible_mask, _ = segmenter.segment_object_at_point(image_rgb, input_point=[x, y])

# 3. Two-Stage Output
outputs = completer.complete(
    image=image_rgb,
    visible_mask=visible_mask,
    prompt="A detailed photo of the object"
)

# returns dictionary: 
# outputs['original'], outputs['visible_mask'], outputs['amodal_mask'], outputs['rgba_result']
```

## 💻 GPU VRAM Requirements 

* **SAM2**: ~6 GB
* **Pix2Gestalt (Shape)**: ~9 GB 
* **Stable Diffusion 2 (Inpaint)**: ~5 GB
* **Peak Usage**: ~15 GB (Optimized with Sequential Model offloading on T4!) up to ~28GB (Concurrent loaded on A100).
*(The pipeline ships with a 0-VRAM **Heuristic Predictor Fallback**; if your GPU crashes on the 15GB LDM load, the pipeline intelligently catches it and calculates the amodal mask via safe dilation mathematics instead!)*

