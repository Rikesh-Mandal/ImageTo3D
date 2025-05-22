import os
import glob
import torch
import cv2
import matplotlib.pyplot as plt

from ImageTo3D.src.model import load_model  # assumes load_model returns (backbone, head)
from ImageTo3D.src.data_loader import preprocess_rgb  # you should move your preprocess_rgb to src/preprocessing.py
from ImageTo3D.src.visualization import denormalize_rgb

# ----- CONFIGURATION -----
model_path = r"E:\Dissertation\Trial\ImageTo3D\ImageTo3D\saved_models\best_model.pth"
image_dir = r"E:\Dissertation\Trial\ImageTo3D\ImageTo3D\sample_input"
output_path = r"E:\Dissertation\Trial\ImageTo3D\ImageTo3D\sample_output\depth_prediction.png"
rgb_path = r"E:\Dissertation\Trial\ImageTo3D\ImageTo3D\rgb_path\rgb.png"
depth_path = r"E:\Dissertation\Trial\ImageTo3D\ImageTo3D\depth_path\depth.png"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using: {device}")

# ----- FIND FIRST IMAGE IN FOLDER -----
image_extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff')
image_files = []
for ext in image_extensions:
    image_files.extend(glob.glob(os.path.join(image_dir, ext)))
if not image_files:
    raise FileNotFoundError(f"No image found in {image_dir}")
image_path = image_files[0]

# ----- LOAD AND PREPROCESS -----
rgb = cv2.imread(image_path)[:, :, ::-1]  # BGR to RGB
rgb_tensor = preprocess_rgb(rgb).unsqueeze(0).to(device)  # add batch dimension

# ----- LOAD MODEL -----
backbone, head = load_model(model_path, device)
backbone.eval()
head.eval()

# ----- INFERENCE -----
with torch.no_grad():
    features = backbone(rgb_tensor)
    pred = head(features)

# ----- RESIZE DEPTH TO MATCH INPUT -----
if pred.shape[-2:] != rgb_tensor.shape[-2:]:
    pred = torch.nn.functional.interpolate(pred, size=rgb_tensor.shape[-2:], mode='bilinear')


def plot_results(rgb, pred, save_path, rgb_path, depth_path):
    """Visualize RGB input, ground truth depth and prediction"""
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    # RGB image
    rgb_img = denormalize_rgb(rgb)
    axes[0].imshow(denormalize_rgb(rgb))
    axes[0].set_title('RGB Input')
    axes[0].axis('off')


    # Predicted depth
    pred_depth = pred.squeeze().cpu().numpy()
    pred_plot = axes[1].imshow(pred_depth, cmap='plasma')
    axes[1].set_title('Predicted Depth')
    plt.colorbar(pred_plot, ax=axes[1], fraction=0.046, pad=0.04)
    axes[1].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=120)

    plt.show()

    if rgb_path:
        plt.imsave(rgb_path, rgb_img)
    if depth_path:
        plt.imsave(depth_path, pred_depth, cmap='plasma')
# ----- VISUALIZE & SAVE -----
plot_results(rgb_tensor[0].cpu(), pred[0].cpu(), output_path, rgb_path, depth_path)

