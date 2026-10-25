
import h5py
import cv2
import numpy as np
import torch
import glob
import os

from sklearn.model_selection import train_test_split
from src.preprocessing import preprocess_rgb


def get_data_paths(dataset_root, test_ratio, seed):
    """Get paths to training, validation, and test H5 files."""

    all_files = glob.glob(
        os.path.join(dataset_root, "train", "**", "*.h5"),
        recursive=True
    )

    train_files, test_files = train_test_split(
        all_files,
        test_size=test_ratio,
        random_state=seed
    )

    val_files = glob.glob(
        os.path.join(dataset_root, "val", "**", "*.h5"),
        recursive=True
    )

    return train_files, val_files, test_files


def load_h5_data(h5_path):
    """Load RGB and depth images from an HDF5 file."""

    with h5py.File(h5_path, "r") as f:

        rgb = np.array(f["rgb"]).transpose(1, 2, 0)
        depth = np.array(f["depth"])

        return (
            rgb.astype(np.float32),
            depth.astype(np.float32)
        )


def preprocess_depth(depth, target_size=(384, 384)):
    """Prepare ground-truth depth for training."""

    depth = cv2.resize(
        depth,
        target_size,
        interpolation=cv2.INTER_NEAREST
    )

    return torch.from_numpy(
        depth
    ).unsqueeze(0).float()


def data_generators(dataset_root, batch_size):
    """Create training, validation, and testing data generators."""

    train_files, val_files, test_files = get_data_paths(
        dataset_root,
        test_ratio=0.2,
        seed=99
    )

    def train_generator():
        """Infinite training generator with shuffling."""

        while True:

            np.random.shuffle(train_files)

            for i in range(0, len(train_files), batch_size):

                batch_files = train_files[i:i + batch_size]

                rgb_batch = []
                depth_batch = []

                for file in batch_files:

                    rgb, depth = load_h5_data(file)

                    rgb_batch.append(
                        preprocess_rgb(rgb)
                    )

                    depth_batch.append(
                        preprocess_depth(depth)
                    )

                yield (
                    torch.stack(rgb_batch),
                    torch.stack(depth_batch)
                )

    def test_generator():
        """Finite generator for test data."""

        for i in range(0, len(test_files), batch_size):

            batch_files = test_files[i:i + batch_size]

            rgb_batch = []
            depth_batch = []

            for file in batch_files:

                rgb, depth = load_h5_data(file)

                rgb_batch.append(
                    preprocess_rgb(rgb)
                )

                depth_batch.append(
                    preprocess_depth(depth)
                )

            yield (
                torch.stack(rgb_batch),
                torch.stack(depth_batch)
            )

    def val_generator():
        """Finite generator for validation data."""

        for i in range(0, len(val_files), batch_size):

            batch_files = val_files[i:i + batch_size]

            rgb_batch = []
            depth_batch = []

            for file in batch_files:

                rgb, depth = load_h5_data(file)

                rgb_batch.append(
                    preprocess_rgb(rgb)
                )

                depth_batch.append(
                    preprocess_depth(depth)
                )

            yield (
                torch.stack(rgb_batch),
                torch.stack(depth_batch)
            )

    return (
        train_generator(),
        val_generator,
        test_generator,
        len(train_files),
        len(val_files),
        len(test_files)
    )
