# ImageTo3D-Reconstruction

An end-to-end deep learning pipeline that turns a single 2D photo into a 3D model. A ResNet50-based encoder-decoder estimates depth from the image, which is then converted into a 3D point cloud and refined into a mesh using Open3D's Poisson surface reconstruction. Built with PyTorch, OpenCV, and NumPy as part of my MSc dissertation.

## Features
- Pretrained ResNet50 encoder for extracting features.
- Decoder for predicting dense depth maps from single images.
- Point cloud generation from depth maps using camera intrinsics.
- Poisson surface reconstruction with mesh smoothing and cleaning.
