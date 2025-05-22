import h5py
import cv2
import numpy as np
import torch
import glob
import os
from sklearn.model_selection import train_test_split


def get_data_paths(dataset_root, test_ratio, seed):
    """Get paths to all training and validation H5 files"""
    all_files = glob.glob(os.path.join(dataset_root, "train", "**", "*.h5"), recursive=True)
    train_files, test_files = train_test_split(all_files, test_size=test_ratio, random_state=seed)
    val_files = glob.glob(os.path.join(dataset_root, "val", "**", "*.h5"), recursive=True)
    return train_files, val_files, test_files


def load_h5_data(h5_path):
    """Load RGB and depth from HDF5 file"""
    with h5py.File(h5_path, 'r') as f:
        # changing the original  rgb shape (C, H, W) to (H, W, C) for later to be used by opencv for resizing
        rgb = np.array(f['rgb']).transpose(1, 2, 0)
        depth = np.array(f['depth'])  # (H,W)
        return rgb.astype(np.float32), depth.astype(np.float32)  # Ensure float32


def preprocess_rgb(rgb, target_size=(384, 384)):
    """Prepare RGB image for model input"""
    rgb = cv2.resize(rgb, target_size, interpolation=cv2.INTER_LINEAR)
    rgb = (rgb / 255.0).astype(np.float32)  # normalizing rgb values [0, 255] to values between [0,1]
    ''' ImageNet normalization
    subtracting the mean and dividing by std to normalize the data 
    just as they were trained on original dataset i.e. ImageNet_V2'''
    rgb = (rgb - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    return torch.from_numpy(rgb).permute(2, 0, 1).float()  # (3,H,W) as float32


def preprocess_depth(depth, target_size=(384, 384)):
    """Prepare depth map for training"""
    depth = cv2.resize(depth, target_size, interpolation=cv2.INTER_NEAREST)
    return torch.from_numpy(depth).unsqueeze(0).float()  # (1,H,W) as float32


def data_generators(dataset_root, batch_size):
    """Create training, testing and validation data generators"""
    train_files, val_files, test_files = get_data_paths(dataset_root, 0.2, 99)

    def train_generator():
        """Infinite training data generator with shuffling"""
        while True:
            np.random.shuffle(train_files)  # shuffles the training data on every epoch
            for i in range(0, len(train_files), batch_size):
                batch_files = train_files[i:i + batch_size]
                rgb_batch = []
                depth_batch = []
                for file in batch_files:
                    rgb, depth = load_h5_data(file)
                    rgb_batch.append(preprocess_rgb(rgb))
                    depth_batch.append(preprocess_depth(depth))
                yield torch.stack(rgb_batch), torch.stack(depth_batch)

    def test_generator():
        """Finite test data generator (loops once)"""
        for i in range(0, len(test_files), batch_size):
            batch_files = test_files[i:i + batch_size]
            rgb_batch = []
            depth_batch = []
            for file in batch_files:
                rgb, depth = load_h5_data(file)
                rgb_batch.append(preprocess_rgb(rgb))
                depth_batch.append(preprocess_depth(depth))
            yield torch.stack(rgb_batch), torch.stack(depth_batch)

    def val_generator():
        """Finite validation data generator (loops once)"""

        for i in range(0, len(val_files), batch_size):
            batch_files = val_files[i:i + batch_size]
            rgb_batch = []
            depth_batch = []
            for file in batch_files:
                rgb, depth = load_h5_data(file)
                rgb_batch.append(preprocess_rgb(rgb))
                depth_batch.append(preprocess_depth(depth))
            yield torch.stack(rgb_batch), torch.stack(depth_batch)

    '''
    train_generator() object is returned since it loops indefinitely
    -- val_generator and test_generator fucntions are returned
    to get fresh data every time those functions are called
     '''
    return train_generator(), val_generator, test_generator, len(train_files), len(val_files), len(test_files)
