
import cv2
import numpy as np
import torch


IMAGE_SIZE = (384, 384)

IMAGENET_MEAN = np.array(
    [0.485, 0.456, 0.406],
    dtype=np.float32
)

IMAGENET_STD = np.array(
    [0.229, 0.224, 0.225],
    dtype=np.float32
)


def preprocess_rgb(rgb_image):
    """Convert an RGB image into a normalized PyTorch tensor."""

    resized = cv2.resize(
        rgb_image,
        IMAGE_SIZE,
        interpolation=cv2.INTER_LINEAR
    )

    normalized = resized.astype(np.float32) / 255.0

    normalized = (normalized - IMAGENET_MEAN) / IMAGENET_STD

    tensor = torch.from_numpy(normalized.transpose(2, 0, 1)).float()

    return tensor
