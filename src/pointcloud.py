
"""Back-project aligned RGB and depth into a 3D point cloud."""

from pathlib import Path

import cv2
import numpy as np
import open3d as o3d


def load_rgb_and_depth(input_dir):
    """Load raw depth and the corresponding RGB image."""

    input_dir = Path(input_dir)

    depth_path = input_dir / "depth_raw.npy"
    rgb_path = input_dir / "input_384.png"

    if not depth_path.is_file():
        raise FileNotFoundError(
            f"Missing depth array: {depth_path}"
        )

    if not rgb_path.is_file():
        raise FileNotFoundError(
            f"Missing RGB image: {rgb_path}"
        )

    depth = np.load(
        depth_path,
        allow_pickle=False
    )

    image_bgr = cv2.imread(
        str(rgb_path),
        cv2.IMREAD_COLOR
    )

    if image_bgr is None:
        raise ValueError(
            f"Cannot decode RGB image: {rgb_path}"
        )

    rgb = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB
    )

    if depth.ndim != 2 or rgb.shape != (*depth.shape, 3):
        raise ValueError(
            f"Depth shape {depth.shape} and "
            f"RGB shape {rgb.shape} do not match."
        )

    if (
        not np.issubdtype(depth.dtype, np.number)
        or np.iscomplexobj(depth)
    ):
        raise TypeError(
            f"Expected real numerical depth, got {depth.dtype}"
        )

    return depth.astype(np.float32), rgb


def generate_point_cloud(
    depth,
    rgb,
    horizontal_fov=60.0,
    stride=2,
    min_depth=0.0,
    max_depth=None,
    border_crop=0
):
    """
    Convert depth pixels to coloured 3D points.

    Camera coordinates:
        X: right
        Y: down
        Z: forward

    Depth remains in model-output units.
    """

    if depth.ndim != 2 or rgb.shape != (*depth.shape, 3):
        raise ValueError(
            "Expected depth [H,W] and aligned RGB [H,W,3]."
        )

    if (
        not np.isfinite(horizontal_fov)
        or not 10.0 <= horizontal_fov <= 170.0
    ):
        raise ValueError(
            "FOV must be between 10 and 170 degrees."
        )

    if (
        not isinstance(stride, int)
        or isinstance(stride, bool)
        or stride < 1
    ):
        raise ValueError(
            "Stride must be a positive integer."
        )

    if not np.isfinite(min_depth) or min_depth < 0:
        raise ValueError(
            "Minimum depth must be finite and nonnegative."
        )

    if max_depth is not None:
        if (
            not np.isfinite(max_depth)
            or max_depth <= min_depth
        ):
            raise ValueError(
                "Maximum depth must exceed minimum depth."
            )

    height, width = depth.shape

    if (
        not isinstance(border_crop, int)
        or isinstance(border_crop, bool)
        or border_crop < 0
        or 2 * border_crop >= min(height, width)
    ):
        raise ValueError(
            "Invalid border crop size."
        )

    # Calculate approximate camera intrinsics.
    fx = (
        (width / 2.0)
        / np.tan(np.deg2rad(horizontal_fov) / 2.0)
    )

    fy = fx

    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0

    # Generate sampled pixel coordinates.
    u, v = np.meshgrid(
        np.arange(0, width, stride),
        np.arange(0, height, stride)
    )

    z = depth[v, u]
    sampled_rgb = rgb[v, u]

    # Keep finite, positive depths.
    finite_positive = np.isfinite(z) & (z > 0)

    valid_depth = (
        finite_positive
        & (z >= min_depth)
    )

    if max_depth is not None:
        valid_depth &= z <= max_depth

    # Mask pixels near image boundaries.
    interior = (
        (u >= border_crop)
        & (u < width - border_crop)
        & (v >= border_crop)
        & (v < height - border_crop)
    )

    valid = valid_depth & interior

    count = int(np.count_nonzero(valid))

    if count < 100:
        raise ValueError(
            f"Only {count} valid points remain. "
            "Reduce filtering thresholds."
        )

    # Retrieve valid depths.
    z_valid = z[valid].astype(np.float64)

    # Back-project into 3D.
    x_valid = (
        (u[valid] - cx) * z_valid / fx
    )

    y_valid = (
        (v[valid] - cy) * z_valid / fy
    )

    xyz = np.column_stack(
        (x_valid, y_valid, z_valid)
    )

    # Normalize RGB colours to [0,1].
    colors = (
        sampled_rgb[valid].astype(np.float64)
        / 255.0
    )

    # Build Open3D point cloud.
    cloud = o3d.geometry.PointCloud()

    cloud.points = o3d.utility.Vector3dVector(xyz)
    cloud.colors = o3d.utility.Vector3dVector(colors)

    stats = {
        "image_width": int(width),
        "image_height": int(height),
        "horizontal_fov_degrees_assumed": float(
            horizontal_fov
        ),
        "fx_pixels_approximate": float(fx),
        "fy_pixels_approximate": float(fy),
        "cx_pixels": float(cx),
        "cy_pixels": float(cy),
        "stride": stride,
        "border_crop_pixels": border_crop,
        "min_depth_filter": float(min_depth),
        "max_depth_filter": (
            float(max_depth)
            if max_depth is not None
            else None
        ),
        "sampled_pixels": int(z.size),
        "positive_finite_sampled_pixels": int(
            np.count_nonzero(finite_positive)
        ),
        "pixels_excluded_by_border": int(
            np.count_nonzero(~interior)
        ),
        "valid_points": count,
        "valid_fraction_of_sampled": float(
            np.mean(valid)
        ),
        "minimum_kept_depth": float(
            z_valid.min()
        ),
        "median_kept_depth": float(
            np.median(z_valid)
        ),
        "maximum_kept_depth": float(
            z_valid.max()
        ),
        "depth_units": (
            "Raw model output; "
            "not calibrated metres"
        ),
        "coordinate_frame": (
            "X right, Y down, Z forward"
        )
    }

    return cloud, stats
