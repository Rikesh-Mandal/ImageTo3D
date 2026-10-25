# ImageTo3D

### Monocular Image-to-3D Reconstruction Using Deep Learning and Computer Vision

ImageTo3D is a monocular image-to-3D reconstruction system developed from my **MSc Artificial Intelligence dissertation at the University of East London**.

The project investigates how deep learning-based depth estimation can be combined with classical computer vision and geometric reconstruction techniques to generate three-dimensional representations from a single RGB image.

Originally developed as a research proof-of-concept, ImageTo3D has since been refactored into a modular Python application supporting reusable depth inference, automatic foreground segmentation, depth-based triangle-mesh reconstruction, surface smoothing, and 3D model export.

The system focuses on **visible-surface reconstruction**. It does not claim to infer complete, geometrically accurate objects from a single photograph.

---

## 1. Project Overview

Reconstructing three-dimensional geometry from two-dimensional images is a fundamental problem in computer vision.

Traditional methods such as Structure from Motion (SfM) and Multi-View Stereo (MVS) typically use multiple photographs captured from different viewpoints to recover scene geometry.

Monocular reconstruction presents a different challenge: estimating three-dimensional structure using only one RGB image, without additional camera views or a dedicated depth sensor.

A single photograph does not explicitly contain the distances between the camera and individual scene surfaces. These distances must be estimated from visual information such as perspective, object boundaries, textures, and contextual relationships.

ImageTo3D explores a hybrid solution combining:

1. Deep learning for monocular depth estimation.
2. Image preprocessing and numerical depth processing.
3. Camera-based pixel back-projection.
4. Geometric mesh reconstruction.
5. Surface refinement and visualisation.

The objective is to transform a single RGB photograph into a dense depth prediction and a corresponding coloured 3D representation.

### Core Technologies

- Python
- PyTorch
- Torchvision
- ResNet-50
- NumPy
- OpenCV
- Open3D
- Rembg / ISNet
- NYU Depth V2

---

## 2. Original MSc Dissertation

### 2.1 Research Motivation

The original dissertation investigated whether a single RGB image could provide sufficient visual information to reconstruct a useful three-dimensional representation of a scene.

Conventional multi-view reconstruction pipelines rely on multiple images, camera calibration, feature matching, and geometric optimisation.

The research instead explored a monocular approach in which a neural network predicts scene depth from one image, after which conventional geometric techniques transform the depth predictions into three-dimensional coordinates.

The central idea was to combine the ability of deep learning to estimate depth from visual cues with established methods for point-cloud and mesh reconstruction.

### 2.2 Original Research Objective

The dissertation aimed to develop and evaluate a proof-of-concept pipeline capable of:

- Accepting a single RGB photograph.
- Predicting a dense depth map using a convolutional neural network.
- Converting depth predictions into a coloured 3D point cloud.
- Reconstructing a surface mesh from the generated geometry.
- Investigating the practical limitations of single-image reconstruction.

The project was designed as an experimental research system rather than a production-ready application.

### 2.3 Investigated Approaches

During the dissertation, different approaches to image-based 3D reconstruction were investigated.

These included:

**Monocular Depth Estimation**

Using a deep learning model to estimate depth directly from a single RGB image.

**Pretrained Depth Models**

Investigating approaches including MiDaS and Dense Prediction Transformer (DPT) architectures.

**Multi-View Reconstruction**

Exploring conventional Structure from Motion and Multi-View Stereo workflows using tools such as COLMAP and OpenMVS.

The final implementation focused on a custom monocular depth-estimation pipeline built around ResNet-50, reflecting the project's research scope and available computational resources.

The alternative approaches were exploratory investigations and are not components of the current inference pipeline.

---

## 3. Original Deep Learning Architecture

The dissertation model uses a convolutional encoder-decoder architecture.

### 3.1 Encoder: ResNet-50

ResNet-50 was selected as the feature-extraction backbone.

The encoder was initialised using weights pretrained on ImageNet (`IMAGENET1K_V2`), allowing the network to begin with previously learned visual representations.

The original classification layers were removed so that the network retained spatial feature maps rather than producing image-classification scores.

For a 384 × 384 RGB input, the encoder produces a feature representation with dimensions:

```text
[Batch, 2048, 12, 12]
```

ResNet-50 was used because it provides a well-established convolutional architecture, supports transfer learning, and offers a practical balance between representational capability and computational requirements.

### 3.2 Custom Depth Decoder

A custom convolutional decoder transforms the extracted feature maps into a dense depth prediction.

The decoder uses:

- Convolutional layers.
- ReLU activations.
- Bilinear upsampling.
- Progressive reduction of feature channels.
- A final single-channel output convolution.

The decoder progressively increases the spatial resolution:

```text
12 × 12
   ↓
24 × 24
   ↓
48 × 48
   ↓
96 × 96
   ↓
192 × 192
   ↓
384 × 384
```

The final output has dimensions:

```text
[Batch, 1, 384, 384]
```

Each output pixel contains a predicted depth value.

### 3.3 Model Architecture

```text
Input RGB Image
      |
      v
Resize and ImageNet Normalization
      |
      v
ResNet-50 Encoder
      |
      v
Feature Maps [2048 × 12 × 12]
      |
      v
Custom Convolutional Decoder
      |
      v
Progressive Bilinear Upsampling
      |
      v
Predicted Depth Map [384 × 384]
```

The encoder and decoder are implemented in:

```text
src/model.py
```

---

## 4. Dataset and Model Training

### 4.1 NYU Depth V2

The dissertation used the **NYU Depth V2** dataset, containing RGB images and corresponding depth measurements captured from indoor environments.

The dataset includes scenes such as rooms, furniture, walls, and everyday indoor objects.

The paired RGB-depth information enabled supervised training: the model learned to predict depth from RGB images using recorded depth data as the training target.

Training samples were loaded from HDF5 files.

### 4.2 Data Preprocessing

The original preprocessing pipeline included:

- Resizing RGB images to 384 × 384.
- Bilinear interpolation for RGB resizing.
- Normalisation using ImageNet mean and standard deviation.
- Resizing ground-truth depth using nearest-neighbour interpolation.
- Conversion to PyTorch tensors.

The ImageNet normalisation parameters are:

```python
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
```

Consistent preprocessing is important because the model expects images with the same numerical representation used during training.

### 4.3 Training Implementation

The preserved training implementation uses:

- PyTorch for model training.
- Adam optimisation.
- Supervised learning using paired RGB-depth samples.
- Training and validation loops.
- Model checkpointing based on validation loss.

The loss implementation retained in `src/model.py` combines pixel-wise L1 depth error with a weighted depth-gradient consistency term.

The gradient component encourages the model to better reproduce local spatial changes in the target depth maps.

The implementation is preserved as part of the dissertation research codebase.

### 4.4 Evaluation

The research implementation includes the following depth-estimation metrics:

**Root Mean Squared Error (RMSE)**

Measures the magnitude of prediction errors relative to ground-truth depth.

**Absolute Relative Error (Abs Rel)**

Measures prediction error relative to the corresponding ground-truth depth values.

These metrics were used to assess depth-estimation behaviour, rather than treating the visual appearance of reconstructed meshes as the only evaluation criterion.

The relevant code is retained in:

```text
src/eval.py
```

---

## 5. Original Proof-of-Concept Reconstruction

The original dissertation implemented a hybrid reconstruction pipeline:

```text
Single RGB Image
       |
       v
ResNet-50 Depth Estimation
       |
       v
Predicted Depth Map
       |
       v
Point Cloud Generation
       |
       v
Surface Normal Estimation
       |
       v
Poisson Surface Reconstruction
       |
       v
3D Mesh
```

### 5.1 Depth Estimation

A trained encoder-decoder network generated a depth prediction from the input image.

### 5.2 Point Cloud Generation

The depth predictions were combined with RGB information and assumed camera parameters to approximate three-dimensional coordinates.

These coordinates formed a coloured point cloud using Open3D.

### 5.3 Surface Reconstruction

The proof-of-concept investigated Poisson surface reconstruction to generate a mesh from the point cloud.

This involved:

- Surface-normal estimation.
- Poisson reconstruction.
- Bounding-box cropping.
- Geometry cleanup.
- Mesh smoothing.

### 5.4 Proof-of-Concept Outcome

The dissertation demonstrated an end-to-end experimental workflow combining monocular depth prediction and geometric reconstruction.

It provided a foundation for investigating how depth-estimation quality influences the geometry generated from a single photograph.

The original implementation also revealed important practical limitations, including:

- Depth prediction errors.
- Distorted geometry.
- Sensitivity to camera assumptions.
- Limited visibility of occluded surfaces.
- Artefacts introduced during mesh reconstruction.
- Computational and environment-specific constraints.

The dissertation system served as the starting point for the subsequent engineering improvements described below.

---

## 6. Post-Dissertation Development

Following completion of the dissertation, the original research repository was reorganised and extended to create a more reusable reconstruction application.

The main objective of this development was to move from experimental scripts with fixed file paths towards an organised pipeline that could process arbitrary input photographs.

### 6.1 Repository Restructuring

The project was reorganised into dedicated modules for:

- Model definition and checkpoint loading.
- Image preprocessing.
- Depth inference.
- Geometric reconstruction.
- Application execution.
- Training and evaluation.

Legacy experimental scripts, machine-specific paths, unnecessary dependencies, generated outputs, and development artefacts were separated from the current application workflow.

Git configuration was also updated to exclude large model checkpoints, datasets, Python environments, and generated reconstruction files.

### 6.2 Reusable Model Inference

The original model-loading and inference logic was refactored into reusable functions.

The application can load the trained ResNet-50 encoder and depth decoder from the saved checkpoint without requiring the original training dataset.

Inference is executed using PyTorch evaluation mode and inference mode.

The current workflow is configured for CPU execution.

This separates the trained model from the command-line application, allowing the same inference functions to be reused by other application components.

### 6.3 Numerical Depth Preservation

An important improvement was separating numerical depth predictions from visual depth-map representations.

The original experimental workflow used display-oriented depth images in parts of the point-cloud generation process.

The updated pipeline retains floating-point depth values in memory for geometry reconstruction.

A colour-mapped PNG is generated separately for visualisation.

This avoids attempting to recover physical or numerical depth information from RGB display colours.

### 6.4 Direct Depth-Grid Triangulation

The reconstruction approach was changed from Poisson reconstruction to direct triangle-mesh construction using neighbouring depth pixels.

Each valid depth pixel is back-projected into 3D using an approximate pinhole camera model.

Neighbouring pixels are connected into candidate triangles based on their original image-grid relationships.

Triangles are filtered to avoid connections across excessive depth discontinuities.

This approach is particularly suitable for representing visible surfaces reconstructed from a single depth map.

Unlike Poisson reconstruction, it does not attempt to close the geometry or infer unobserved rear surfaces.

### 6.5 Automatic Foreground Segmentation

Automatic foreground isolation was introduced using Rembg and a pretrained ISNet segmentation model.

The segmentation stage estimates which image pixels belong to foreground subjects.

The resulting mask is used during mesh construction to exclude background regions.

Foreground segmentation is enabled by default, with an option to include the full scene instead.

This improves control over reconstructed geometry, particularly when the intended output is an individual foreground subject.

Foreground segmentation is separate from the depth-estimation model. It does not retrain or modify ResNet-50.

### 6.6 Mesh Refinement

The updated application supports:

- Invalid-depth filtering.
- Image-boundary cropping.
- Configurable pixel sampling.
- Relative depth-discontinuity thresholds.
- Optional foreground masking.
- Taubin mesh smoothing.
- Vertex-normal computation.

Taubin smoothing is used to reduce local geometric irregularities while limiting the shrinkage associated with basic iterative smoothing.

Smoothing does not recover missing geometry or correct major errors in the predicted depth map.

### 6.7 Simplified Output Management

The reconstruction pipeline was consolidated into a single command-line entry point.

Intermediate numerical arrays, resized RGB images, foreground masks, and point clouds do not need to be permanently saved for normal operation.

For each reconstruction, the application exports:

```text
depth_map.png
model.ply
```

The numerical depth array remains available internally during reconstruction.

---

## 7. Current Reconstruction Pipeline

The current application follows this sequence:

```text
                  Input RGB Image
                         |
                         v
                RGB Preprocessing
                         |
                         v
               ResNet-50 Encoder
                         |
                         v
                Depth Decoder
                         |
                         v
              Numerical Depth Map
                         |
                         +----------------------+
                         |                      |
                         v                      v
              Foreground Masking        Depth Visualisation
                         |                      |
                         v                      v
              Pixel Back-Projection     depth_map.png
                         |
                         v
              Depth-Grid Triangulation
                         |
                         v
              Discontinuity Filtering
                         |
                         v
                 Taubin Smoothing
                         |
                         v
                    model.ply
```

Automatic segmentation operates on the RGB image and supplies a foreground mask to the reconstruction stage.

The depth model receives the original RGB image rather than a background-removed image, preserving the input representation expected from training.

The reconstruction combines the numerical depth prediction, corresponding RGB colours, and optional foreground mask.

---

## 8. Geometric Reconstruction

### 8.1 Pinhole Camera Model

A predicted depth value is combined with pixel coordinates and camera intrinsics to calculate a 3D position.

The projection equations are:

```text
X = (u - cx) * Z / fx
Y = (v - cy) * Z / fy
Z = predicted depth
```

Where:

- `u, v` are image pixel coordinates.
- `fx, fy` are focal lengths expressed in pixels.
- `cx, cy` represent the camera principal point.
- `Z` is the predicted depth.
- `X, Y, Z` are the reconstructed 3D coordinates.

When camera intrinsics are unavailable, the application estimates focal length from an assumed horizontal field of view.

The default horizontal field of view is 60 degrees.

These parameters are approximate and can affect the geometric appearance of the reconstructed scene.

### 8.2 Triangle Construction

The depth map is treated as a regular image grid.

Neighbouring valid pixels are connected into triangles using their spatial relationships in the image.

Each grid cell can generate up to two triangles.

Both possible cell diagonals are evaluated, and the triangulation attempts to preserve valid surfaces without unnecessarily crossing depth discontinuities.

### 8.3 Depth-Discontinuity Filtering

Large depth differences between neighbouring pixels can indicate object boundaries, occlusions, or unreliable predictions.

The application rejects candidate triangles that exceed a configurable relative depth-difference threshold.

The default threshold is:

```text
max_relative_jump = 0.03
```

This corresponds to a maximum relative depth difference of approximately 3% within a triangle, based on the implementation's depth-filtering rule.

The threshold is a reconstruction heuristic rather than a confidence measure produced by the neural network.

### 8.4 Surface Smoothing

The triangle mesh is optionally refined using Open3D's Taubin smoothing filter.

The current default is:

```text
mesh_smooth_iterations = 5
```

This reduces some local surface irregularities while preserving the overall reconstruction more effectively than unrestricted simple smoothing.

Excessive smoothing may remove meaningful geometric details.

### 8.5 Mesh Representation

The generated mesh contains:

- Three-dimensional vertex coordinates.
- Triangle connectivity.
- RGB vertex colours.
- Computed vertex normals.

The output is stored in Polygon File Format (PLY).

The mesh represents reconstructed visible surfaces and is not guaranteed to be watertight.

---

## 9. Project Structure

```text
ImageTo3D/
│
├── src/
│   ├── __init__.py
│   │
│   ├── model.py
│   ├── preprocessing.py
│   ├── inference.py
│   ├── depth_mesh.py
│   ├── pointcloud.py
│   ├── run_pipeline.py
│   │
│   ├── data_loader.py
│   ├── train.py
│   ├── eval.py
│   ├── main.py
│   └── visualization.py
│
├── saved_models/             # Local checkpoint, Git-ignored
│   └── best_model.pth
│
├── sample_input/             # Local input images, Git-ignored
│
├── outputs/                  # Generated results, Git-ignored
│
├── .gitignore
├── README.md
├── requirements.txt
└── requirements-legacy.txt
```

### Application Modules

| File               | Responsibility                                       |
| ------------------ | ---------------------------------------------------- |
| `model.py`         | ResNet-50 encoder, decoder, checkpoint utilities     |
| `preprocessing.py` | RGB resizing, normalisation and tensor conversion    |
| `inference.py`     | Model initialisation and numerical depth prediction  |
| `depth_mesh.py`    | Depth back-projection and triangle-mesh construction |
| `pointcloud.py`    | Supporting RGB/depth and point-cloud utilities       |
| `run_pipeline.py`  | End-to-end command-line reconstruction pipeline      |

### Preserved Research Modules

| File               | Responsibility                              |
| ------------------ | ------------------------------------------- |
| `data_loader.py`   | NYU Depth V2 data loading and preprocessing |
| `train.py`         | Training and validation steps               |
| `eval.py`          | Depth-estimation evaluation metrics         |
| `main.py`          | Dissertation training entry point           |
| `visualization.py` | Research plots and prediction visualisation |

The preserved research modules document the original training and evaluation implementation.

They may require additional training dependencies, dataset files, and configuration before the original experiments can be reproduced.

---

## 10. Installation and Setup

### Requirements

- Python 3.12
- pip
- A compatible CPU-supported PyTorch installation
- The trained ImageTo3D model checkpoint

An NVIDIA GPU is not required for the current inference workflow.

### Step 1: Clone the Repository

```bash
git clone git@github.com:Rikesh-Mandal/ImageTo3D.git
cd ImageTo3D
```

### Step 2: Create a Virtual Environment

Windows:

```cmd
py -3.12 -m venv .venv
```

Activate using Command Prompt:

```cmd
.venv\Scripts\activate.bat
```

Linux:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The inference dependency list includes PyTorch, Torchvision, NumPy, OpenCV, Open3D, and Rembg with its CPU runtime.

PyTorch CPU and CUDA distributions may require platform-specific installation choices.

### Step 4: Add the Trained Model

Place the trained model at:

```text
saved_models/best_model.pth
```

The checkpoint is not included in GitHub because of its size.

The checkpoint contains the trained weights for the ResNet-50 encoder and custom depth decoder.

### Step 5: Supply an RGB Image

Place a JPG or PNG photograph in the local `sample_input` directory.

For example:

```text
sample_input/table.jpg
```

---

## 11. Running the Application

The main command-line entry point is:

```text
src/run_pipeline.py
```

### Default Reconstruction

From the project root:

```bash
python -m src.run_pipeline --image sample_input/table.jpg --output-dir outputs --view
```

This command:

1. Loads the trained checkpoint on the CPU.
2. Preprocesses the RGB photograph.
3. Predicts the numerical depth map.
4. Automatically estimates the foreground mask.
5. Back-projects valid depth pixels into 3D.
6. Generates triangles between neighbouring pixels.
7. Rejects triangles spanning excessive depth changes.
8. Applies five iterations of Taubin smoothing.
9. Saves the depth visualisation and coloured mesh.
10. Opens the result in the Open3D viewer.

### Reconstruct the Full Visible Scene

To skip foreground removal:

```bash
python -m src.run_pipeline --image sample_input/room.jpg --output-dir outputs --include-background --view
```

This reconstructs the full valid visible scene without foreground segmentation.

### Use a Custom Foreground Mask

```bash
python -m src.run_pipeline --image sample_input/table.jpg --output-dir outputs/table_masked --foreground-mask sample_input/table_mask.png --view
```

The mask must align with the 384 × 384 depth prediction.

### Override Reconstruction Settings

```bash
python -m src.run_pipeline --image sample_input/table.jpg --output-dir outputs/table_custom --max-relative-jump 0.05 --mesh-smooth-iterations 10 --view
```

The defaults can be overridden without modifying the Python source code.

---

## 12. Configuration

| Parameter                    | Default  | Description                                |
| ---------------------------- | -------- | ------------------------------------------ |
| `--fov`                      | `60.0`   | Assumed horizontal camera field of view    |
| `--stride`                   | `1`      | Pixel-grid sampling interval               |
| `--border-crop`              | `12`     | Pixels excluded around image boundaries    |
| `--min-depth`                | `0.1`    | Minimum accepted positive depth prediction |
| `--max-relative-jump`        | `0.03`   | Relative depth-discontinuity threshold     |
| `--mesh-smooth-iterations`   | `5`      | Taubin smoothing iterations                |
| Automatic foreground removal | Enabled  | Estimates likely foreground regions        |
| `--include-background`       | Disabled | Reconstructs the full scene when provided  |
| `--foreground-mask`          | Optional | Uses a user-supplied foreground mask       |
| `--view`                     | Optional | Opens the interactive mesh viewer          |

Depth thresholds operate in model-output units. They should not automatically be interpreted as physical distances in metres.

---

## 13. Generated Outputs

Each reconstruction run produces two final files in a new or empty output directory:

```text
outputs/
└── ├── depth_map.png
    └── model.ply
```

### depth_map.png

A colour-mapped visualisation of the predicted numerical depth.

The colour mapping is used for display and does not preserve the original floating-point values.

### model.ply

A coloured triangle mesh containing the reconstructed visible geometry.

The file can be inspected using compatible 3D software, including Open3D.

Raw numerical depth, aligned RGB data and segmentation masks are maintained in memory during normal execution rather than exported as separate intermediate files.

Existing files in an output directory are not automatically deleted.

---

## 14. Key Improvements Since the Dissertation

| Area                     | Dissertation Proof-of-Concept                   | Updated Implementation                                    |
| ------------------------ | ----------------------------------------------- | --------------------------------------------------------- |
| Application structure    | Experimental research scripts                   | Modular reconstruction pipeline                           |
| File paths               | Machine-specific paths                          | Project-relative paths                                    |
| Model execution          | Research-oriented inference scripts             | Reusable checkpoint-loading and inference functions       |
| Depth representation     | Intermediate display-oriented depth files       | Numerical depth preserved for reconstruction              |
| Geometric representation | Point clouds followed by Poisson reconstruction | Direct depth-grid triangle mesh                           |
| Background handling      | Scene content generally retained                | Automatic foreground segmentation                         |
| Mesh connectivity        | Poisson-based surface generation                | Neighbour-based triangulation and discontinuity filtering |
| Surface refinement       | Mesh cleanup and basic smoothing                | Configurable Taubin smoothing                             |
| Outputs                  | Multiple intermediate and experimental files    | Depth PNG and coloured PLY mesh                           |
| Dependencies             | Dissertation development environment            | Separate inference dependency declaration                 |
| Execution                | Multiple manually executed scripts              | Single command-line entry point                           |

These changes focus on software organisation, reproducibility, and reconstruction workflow design.

They do not imply that the underlying trained neural network has been retrained or that reconstruction accuracy has been quantitatively improved.

---

## 15. Limitations and Considerations

### Single-View Geometry

A monocular photograph only provides one view of a scene.

The system reconstructs estimated visible surfaces, not complete hidden geometry.

The resulting mesh is generally open rather than watertight.

### Domain Generalisation

The depth model was trained using indoor RGB-depth data from NYU Depth V2.

Images outside this training distribution, including outdoor scenes and unfamiliar objects, may produce unreliable depth estimates.

### Camera Calibration

When real camera intrinsics are unavailable, the reconstruction assumes approximate camera parameters.

Errors in these assumptions can affect reconstructed object proportions and perspective.

### Depth Accuracy

Predicted depth values may include noise, invalid values and distorted spatial relationships.

The model output is not guaranteed to represent calibrated real-world distance.

### Foreground Segmentation

Automatic background removal may fail to preserve all relevant subjects or may retain unwanted background regions.

Foreground segmentation accuracy depends on image content and the pretrained segmentation model.

### Mesh Quality

Depth-grid triangulation preserves neighbouring image relationships but cannot recover geometry absent from the input view.

Depth filtering and smoothing can improve visual continuity, but excessive filtering creates holes and excessive smoothing removes geometric detail.

### Reconstruction Interpretation

The exported PLY mesh should be interpreted as an estimated visible-surface representation.

It is not a ground-truth 3D scan or a complete object reconstruction.

---

## 16. Project Significance

ImageTo3D represents the progression of an academic computer vision experiment into a reusable software system.

The original dissertation investigated the feasibility of combining a learned monocular depth estimator with geometric reconstruction techniques.

The subsequent development has focused on improving the engineering surrounding that research: separating inference from training, preserving numerical depth information, modularising reconstruction, introducing foreground segmentation, improving mesh generation, and simplifying output management.

The project combines practical experience in:

- Deep learning model development and evaluation.
- Transfer learning and convolutional neural networks.
- Image preprocessing and depth estimation.
- Camera geometry and 3D back-projection.
- Point-cloud and triangle-mesh processing.
- Model inference and application architecture.
- Python dependency and repository management.
- Research-oriented experimentation and iterative software refinement.

The system remains an experimental reconstruction application whose geometry depends on the quality of its monocular depth predictions.

---

## 17. Author

**Rikesh Mandal**

MSc Artificial Intelligence  
University of East London

Originally developed as an MSc dissertation project and subsequently extended into a modular image-to-3D reconstruction application.
