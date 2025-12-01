# note_for_GR_ConvNet
My note for Generative Residual Convolutional Neural Network (GR-ConvNet).

## Reference
- Paper : 
  - [Antipodal Robotic Grasping using Generative Residual Convolutional Neural Network](https://arxiv.org/abs/1909.04810)
- Github : 
  - [robotic-grasping](https://github.com/skumra/robotic-grasping)

## Introduction
**Generative Residual Convolutional Neural Network (GR-ConvNet)**
- The model architecture which detects objects in the camera's field of view and **predicts a suitable antipodal grasp** configuration for the objects in the image.

## System setup
### Package installation
```bash
# torch, torchvision
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128
# other
pip install opencv-python, matplotlib, scikit-image, imageio, torchsummary, tensorboardX
```

## Script
### run_offline.py