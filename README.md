# ImageTo3D-Reconstruction

This project performs monocular depth estimation from RGB images using a ResNet-based encoder-decoder neural network, and reconstructs 3D models by generating point clouds and surface meshes using Open3D.

## Features
- Pretrained ResNet50 encoder for extracting features.
- Decoder for predicting dense depth maps from single images.
- Point cloud generation from depth maps using camera intrinsics.
- Poisson surface reconstruction with mesh smoothing and cleaning.