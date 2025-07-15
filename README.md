# Harnessing Depth Gradients: A New Framework for Precise
## Abstract
Depth information provides critical geometric cues
for instance segmentation, offering a powerful complement to
methods that rely solely on RGB imagery. However, prevailing
approaches often employ simplistic feature extraction and direct
fusion strategies, which fail to fully leverage the rich structural
information inherent in depth data. To address this limitation, we
propose a novel depth-guided framework for RGB-D instance
segmentation. Our framework features two core innovative
modules: the Depth Gradient Guidance Module (DGGM) and the
Enhanced Depth-Sensitive Attention Module (E-DSAM).
Specifically, the DGGM provides a fine-grained structural prior
by converting the raw depth map into an explicit gradient map of
object boundaries, which is then refined and fused using gating
mechanisms and a specialized processor. Concurrently, our E-
DSAM incorporates a lightweight predictor to dynamically
adjust the attention windows of the DSAM module in a content-
aware manner, yielding a more precise hierarchical depth
context. Extensive experiments on the public NYUv2 dataset
demonstrate that our method achieves a mask mean Average
Precision (mAP) of 26.8, a 4.1-point improvement over a strong
baseline model, while generating more precise and complete
segmentation masks.

## Highlights
- We propose a Depth Gradient Guidance Module (DGGM) that uses depth gradients to focus on geometric object boundaries and mitigate noise.
- We introduce an Enhanced Depth-Sensitive Attention Module (E-DSAM) with an adaptive receptive field to better handle objects at multiple scales.
- Our model achieves a 26.8 mean Average Precision (mAP) on a challenging benchmark, surpassing a strong baseline by 4.1 points.

## Brief Project Structure
- `dataset`: Contains the instance segmentation datasets used for training and evaluation.
- `intelRealSense`: Contains a sample script demonstrating how to capture RGB-D image pairs using an Intel RealSense camera.
- `mask2former`: The core source code of our project. Our implementation is built upon the official Mask2Former framework.
    - `checkpoints`: Default directory for storing trained model checkpoints and training logs.
    - `experiments`: An archive of experimental strategies and prototypes explored during the research and development phase. Note: These do not necessarily correspond to the final results in the paper.
    - `transformers_guide`: Includes a guide for installing the specific version of the Hugging Face Transformers library required by our project.
    - `utils`: A collection of utility scripts and helper functions for data processing, evaluation, and visualization.
    - `finetuning.py`: The main entry point script for training, evaluating, and running inference with our model.

## Installation
```shell
git clone https://github.com/TheoBald200814/RGB-D-Instance-Segmentation.git
cd RGB-D-Instance-Segmentation
pip install -r requirements.txt
```

