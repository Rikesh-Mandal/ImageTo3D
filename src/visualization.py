import os.path

import matplotlib.pyplot as plt
import numpy as np


def denormalize_rgb(rgb_tensor):
    """Convert normalized RGB back to original for visualization"""
    rgb = rgb_tensor.cpu().numpy()
    rgb = np.transpose(rgb, (1, 2, 0))  # (H,W,3)  # matplotlib wants (H, W, C)
    # Reverse ImageNet normalization
    rgb = rgb * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    return np.clip(rgb, 0, 1)


def plot_results(rgb, depth, pred, save_path):
    """Visualize RGB input, ground truth depth and prediction"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # RGB image
    axes[0].imshow(denormalize_rgb(rgb))
    axes[0].set_title('RGB Input')
    axes[0].axis('off')

    # Ground truth depth
    gt_depth = depth.squeeze().cpu().numpy()
    gt_plot = axes[1].imshow(gt_depth, cmap='plasma')
    axes[1].set_title('Ground Truth Depth')
    plt.colorbar(gt_plot, ax=axes[1], fraction=0.046, pad=0.04)
    axes[1].axis('off')

    # Predicted depth
    pred_depth = pred.squeeze().cpu().numpy()
    pred_plot = axes[2].imshow(pred_depth, cmap='plasma')
    axes[2].set_title('Predicted Depth')
    plt.colorbar(pred_plot, ax=axes[2], fraction=0.046, pad=0.04)
    axes[2].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=120)

    plt.show()

