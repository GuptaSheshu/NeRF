# Tiny NeRF

## Introduction
In the task of novel view synthesis, training set consists of a set of images of a scene where we know the camera parameters (intrinsic and extrinsic) for each image. The goal is to create a model that can synthesize images showing the scene from new viewpoints unseen in the training set. NeRF is one such methods which tries to represent a static scene as a continuous 5D function that outputs a density and the radiance emitted in each direction (θ, φ) at each point (x, y, z) in space. The density at each point acts like a differential opacity controlling how much radiance is accumulated by passing a ray through the point (x, y, z). NeRF paper [Mildenhall et al, ECCV 2020](https://arxiv.org/abs/2003.08934) tries to predict the density and the RGB color at some input point (x, y, z) and a input viewing direction using fully-connected neural network.

## Our Implemenetation
One of the major drawbacks of NeRF is the training time. Reducing training time implies trad-offs with 3D reconstruction quality. Our project experimented with multiple reduced resolutions of input images and varying the width and depth of the Multi-Layer Perceptron(MLP) to find an optimal solution that reduces the training time while maintaining satisfactory 3D reconsturction quality.

* **Note** that this work has been done as part of the course EECS 598 (Deep learning for computer vision).

## Requirements
- torch
- Numpy
- torchsummary
- skimage
- imageio

## File Distribution
- ``NeRF_lego.ipynb`` and ``NeRF_drums.ipynb`` are training and evaluation of NeRF on Lego and Drum datasets. The python notebook contains all the required information about the NeRF.
- ``utils.py`` is the utility file for NeRF rendering function and neural network architecture details.
- **drums_video** and **lego_vide** are the results. 

