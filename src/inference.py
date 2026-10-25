
from pathlib import Path

import cv2
import numpy as np
import torch

from src.model import load_model
from src.preprocessing import preprocess_rgb


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_MODEL_PATH = (
    PROJECT_ROOT / "saved_models" / "best_model.pth"
)

IMAGE_SIZE = (384, 384)


def initialize_model(model_path=DEFAULT_MODEL_PATH):
    """Load the saved depth model on CPU once for reuse."""

    model_path = Path(model_path)

    if not model_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {model_path}"
        )

    device = torch.device("cpu")

    backbone, head = load_model(model_path, device)

    backbone.eval()
    head.eval()

    return backbone, head, device


def predict_depth(image_path, backbone, head, device):
    """Return raw depth and the matching 384x384 RGB image."""

    image_path = Path(image_path)

    if not image_path.is_file():
        raise FileNotFoundError(
            f"Input image not found: {image_path}"
        )

    image_bgr = cv2.imread(
        str(image_path),
        cv2.IMREAD_COLOR
    )

    if image_bgr is None:
        raise ValueError(
            f"Unable to decode input image: {image_path}"
        )

    image_rgb = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB
    )

    image_rgb_float = image_rgb.astype(np.float32)

    # Use the same preprocessing as dissertation training.
    image_tensor = preprocess_rgb(
        image_rgb_float
    ).unsqueeze(0).to(device)

    # Generate predictions without computing gradients.
    with torch.inference_mode():
        features = backbone(image_tensor)
        prediction = head(features)

    # Convert the tensor into a 2D NumPy array.
    depth = (
        prediction[0, 0]
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    if depth.shape != (384, 384):
        raise ValueError(
            f"Unexpected model depth shape: {depth.shape}"
        )

    # Prepare matching RGB coordinates for later 3D work.
    rgb_384 = cv2.resize(
        image_rgb_float,
        IMAGE_SIZE,
        interpolation=cv2.INTER_LINEAR
    )

    rgb_384 = np.clip(
        rgb_384, 0, 255
    ).astype(np.uint8)

    return depth, rgb_384


def save_depth_outputs(depth, rgb_384, output_dir):
    """Save depth values, a preview, and matching RGB image."""

    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    if (
        depth.shape != (384, 384)
        or rgb_384.shape != (384, 384, 3)
    ):
        raise ValueError(
            "Depth and RGB image must both be 384x384"
        )

    # Save numerical depth without modifying predictions.
    raw_path = output_dir / "depth_raw.npy"

    np.save(raw_path, depth)

    # Save the corresponding RGB image.
    rgb_path = output_dir / "input_384.png"

    rgb_bgr = cv2.cvtColor(
        rgb_384,
        cv2.COLOR_RGB2BGR
    )

    if not cv2.imwrite(str(rgb_path), rgb_bgr):
        raise OSError(
            f"Could not save RGB image: {rgb_path}"
        )

    # Identify positive, finite depth predictions.
    valid = np.isfinite(depth) & (depth > 0)

    if not valid.any():
        raise ValueError(
            "No positive finite depth values. "
            "Raw output was saved for debugging."
        )

    # Use percentiles for a readable visualization.
    low, high = np.percentile(
        depth[valid],
        [2, 98]
    )

    gray = np.zeros(
        depth.shape,
        dtype=np.uint8
    )

    if high > low:
        clipped = np.clip(
            depth[valid],
            low,
            high
        )

        gray[valid] = (
            (clipped - low) / (high - low) * 255
        ).astype(np.uint8)

    else:
        gray[valid] = 127

    # Create a colour-mapped PNG for visual inspection.
    preview = cv2.applyColorMap(
        gray,
        cv2.COLORMAP_TURBO
    )

    preview[~valid] = 0

    preview_path = output_dir / "depth_preview.png"

    if not cv2.imwrite(
        str(preview_path),
        preview
    ):
        raise OSError(
            f"Could not save depth preview: {preview_path}"
        )

    return raw_path, preview_path, rgb_path
