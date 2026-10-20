# 🧠 Image-to-3D Reconstruction System (Hybrid Deep Learning Pipeline)

## 📌 Overview

This project explores a **hybrid pipeline that combines traditional computer vision techniques with deep learning-based depth estimation** to reconstruct 3D models from a single monocular RGB image.

The goal is to generate **geometrically meaningful and visually coherent 3D representations** from a single view, addressing key challenges in accessibility and automation within:

- Computer Graphics
- Augmented Reality (AR)
- 3D Content Creation

---

## 🎯 Aim of the Project

This research aims to:

- Reconstruct 3D structures from a single RGB image
- Reduce dependency on multi-view stereo or specialized hardware
- Improve accessibility of 3D modelling using AI-driven pipelines

---

## 🧠 Methodology (Hybrid Approach)

The system integrates **deep learning + classical geometry** in a structured pipeline:

### 1. Depth Estimation (Deep Learning)

- Uses a **ResNet-50 backbone pretrained on ImageNet**
- Fine-tuned on paired **RGB–Depth datasets**
- Predicts dense depth maps from monocular images

### 2. Training Pipeline

- Supervised learning using image-depth pairs
- Custom data loader with preprocessing and augmentation
- Real-time loss monitoring for convergence tracking

### 3. Point Cloud Generation (Geometry)

- Depth maps combined with camera intrinsic parameters
- Conversion of depth → 3D point clouds using Open3D

### 4. Mesh Reconstruction

- Normal estimation from point clouds
- Poisson surface reconstruction for mesh generation
- Post-processing steps:
  - Mesh cropping
  - Noise removal
  - Artifact reduction

## 🧩 System Pipeline

RGB Image -> ResNet50 Depth Estimator -> Predicted Depth Map -> Point Cloud Generation(Open 3D) -> Normal Estimation -> Poisson Surface Reconstruction -> Final 3D Mesh

---

## 🔬 Key Technologies

- PyTorch (Deep Learning Framework)
- ResNet-50 (Transfer Learning Backbone)
- Open3D (3D Processing & Reconstruction)
- NumPy / SciPy (Data Processing)
- Custom Data Pipeline (Preprocessing & Augmentation)

---

## 🌍 Applications

- 3D modelling & animation
- Virtual Reality (VR) and Augmented Reality (AR)
- Robotics perception systems
- Industrial inspection
- Cultural heritage reconstruction

---

## 💡 Motivation

Traditional methods such as Structure from Motion (SfM) and multi-view stereo require:

- Multiple calibrated images
- Controlled environments
- Specialized hardware

This project explores whether **deep learning can reduce these constraints by enabling single-image 3D reconstruction**.

---

## 📊 Contributions

- Hybrid deep learning + geometry-based pipeline
- Transfer learning with ResNet-50 for depth estimation
- Custom training and evaluation pipeline
- End-to-end reconstruction from image → mesh

---

## ⚠️ Limitations

- Reconstruction is limited to visible surfaces in the input image
- Occluded regions cannot be fully inferred
- Performance depends on training dataset quality

---

## 🚀 Future Work

- Real-time inference pipeline
- Improved depth refinement using transformer-based models
- Deployment in AR/VR applications
- Mobile or web-based 3D reconstruction system

## 👨‍💻 Author

**Rikesh Mandal**  
MSc Data Science – Dissertation Project  
Title: _An AI-Driven Framework for Automated 3D Model Generation from 2D Images _

---

## 🧩 System Pipeline
