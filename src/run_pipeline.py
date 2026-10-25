
"""Monocular image-to-3D reconstruction pipeline.

RGB image
    -> Depth prediction
    -> Automatic foreground segmentation
    -> Grid-based triangle mesh
    -> Optional mesh smoothing
    -> PNG and PLY export
"""

import argparse
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

from PIL import Image
from rembg import remove, new_session

from src.inference import (
    initialize_model,
    predict_depth
)

from src.depth_mesh import create_depth_mesh


# Default reconstruction settings.
DEFAULT_FOV = 60.0
DEFAULT_STRIDE = 1
DEFAULT_BORDER_CROP = 12
DEFAULT_MIN_DEPTH = 0.1

DEFAULT_MAX_RELATIVE_JUMP = 0.03
DEFAULT_MESH_SMOOTH_ITERATIONS = 5


@lru_cache(maxsize=1)
def get_segmentation_session():
    """Load and reuse the CPU foreground segmentation model."""

    print("Loading foreground segmentation model...")

    return new_session("isnet-general-use")


def generate_foreground_mask(rgb):
    """Automatically estimate foreground subjects from an RGB image."""

    pil_image = Image.fromarray(rgb)

    pil_mask = remove(
        pil_image,
        session=get_segmentation_session(),
        only_mask=True
    )

    mask_array = np.asarray(
        pil_mask.convert("L"),
        dtype=np.uint8
    )

    if mask_array.shape != rgb.shape[:2]:
        raise ValueError(
            "Foreground mask dimensions do not match RGB."
        )

    # Convert the grayscale confidence mask to binary.
    binary = (
        mask_array >= 127
    ).astype(np.uint8)

    # Remove tiny disconnected regions.
    count, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            binary,
            connectivity=8
        )
    )

    min_component_area = max(
        100,
        int(binary.size * 0.001)
    )

    cleaned = np.zeros(
        binary.shape,
        dtype=bool
    )

    # Retain all sufficiently large components,
    # not just the single largest object.
    for label_id in range(1, count):

        area = stats[
            label_id,
            cv2.CC_STAT_AREA
        ]

        if area >= min_component_area:
            cleaned |= labels == label_id

    retained_pixels = int(cleaned.sum())

    if retained_pixels < 100:
        raise ValueError(
            "Automatic segmentation found no usable foreground."
        )

    retained_fraction = cleaned.mean()

    print(
        "Automatic foreground coverage: "
        f"{retained_fraction:.2%}"
    )

    if retained_fraction > 0.95:
        print(
            "Warning: Nearly the entire image was selected. "
            "Background removal may be inaccurate."
        )

    return cleaned


def load_foreground_mask(path, expected_shape):
    """Load a manually supplied mask, if needed."""

    mask_image = cv2.imread(
        str(path),
        cv2.IMREAD_GRAYSCALE
    )

    if mask_image is None:
        raise FileNotFoundError(
            f"Unable to read foreground mask: {path}"
        )

    if mask_image.shape != expected_shape:
        raise ValueError(
            "Foreground mask must match the depth resolution."
        )

    mask = mask_image > 0

    if np.count_nonzero(mask) < 100:
        raise ValueError(
            "Provided foreground mask is too small."
        )

    return mask


def make_depth_preview(depth, mask=None):
    """Create a colour-mapped depth preview."""

    valid = np.isfinite(depth) & (depth > 0)

    if mask is not None:
        valid &= mask

    if not valid.any():
        raise ValueError(
            "No valid depth pixels available for visualization."
        )

    low, high = np.percentile(
        depth[valid],
        [2, 98]
    )

    grayscale = np.zeros(
        depth.shape,
        dtype=np.uint8
    )

    if high > low:

        clipped = np.clip(
            depth[valid],
            low,
            high
        )

        scaled = (
            (clipped - low) / (high - low)
        )

        grayscale[valid] = (
            scaled * 255
        ).astype(np.uint8)

    else:
        grayscale[valid] = 127

    preview = cv2.applyColorMap(
        grayscale,
        cv2.COLORMAP_TURBO
    )

    # Black regions represent excluded/invalid pixels.
    preview[~valid] = 0

    return preview


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Automatically isolate foreground subjects, "
            "predict depth, and reconstruct a 3D mesh."
        )
    )

    parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="Input RGB photograph"
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for final output files"
    )

    foreground_group = (
        parser.add_mutually_exclusive_group()
    )

    foreground_group.add_argument(
        "--include-background",
        action="store_true",
        help="Disable automatic foreground segmentation"
    )

    foreground_group.add_argument(
        "--foreground-mask",
        type=Path,
        help="Use an existing binary foreground mask"
    )

    parser.add_argument(
        "--fov",
        type=float,
        default=DEFAULT_FOV
    )

    parser.add_argument(
        "--stride",
        type=int,
        default=DEFAULT_STRIDE
    )

    parser.add_argument(
        "--border-crop",
        type=int,
        default=DEFAULT_BORDER_CROP
    )

    parser.add_argument(
        "--min-depth",
        type=float,
        default=DEFAULT_MIN_DEPTH
    )

    parser.add_argument(
        "--max-relative-jump",
        type=float,
        default=DEFAULT_MAX_RELATIVE_JUMP
    )

    parser.add_argument(
        "--mesh-smooth-iterations",
        type=int,
        default=DEFAULT_MESH_SMOOTH_ITERATIONS
    )

    parser.add_argument(
        "--view",
        action="store_true",
        help="Open the reconstructed mesh in Open3D"
    )

    args = parser.parse_args()

    if not args.image.is_file():
        parser.error(
            f"Input image not found: {args.image}"
        )

    if not 0 <= args.mesh_smooth_iterations <= 30:
        parser.error(
            "Mesh smoothing iterations must be between 0 and 30."
        )

    # STEP 1: Load your trained depth model.
    print("Loading trained depth model on CPU...")

    backbone, head, device = initialize_model()

    # STEP 2: Generate numerical depth.
    print("Generating depth prediction...")

    depth, rgb = predict_depth(
        args.image,
        backbone,
        head,
        device
    )

    # STEP 3: Determine which pixels to reconstruct.
    if args.include_background:

        print("Background removal disabled.")
        foreground_mask = None

    elif args.foreground_mask is not None:

        print("Loading custom foreground mask...")

        foreground_mask = load_foreground_mask(
            args.foreground_mask,
            depth.shape
        )

    else:

        print("Automatically isolating foreground...")

        foreground_mask = generate_foreground_mask(
            rgb
        )

    # STEP 4: Convert valid depth pixels into triangles.
    print("Constructing triangle mesh...")

    mesh, stats = create_depth_mesh(
        depth,
        rgb,
        horizontal_fov=args.fov,
        stride=args.stride,
        border_crop=args.border_crop,
        min_depth=args.min_depth,
        max_relative_jump=args.max_relative_jump,
        foreground_mask=foreground_mask
    )

    # STEP 5: Apply general-purpose Taubin smoothing.
    if args.mesh_smooth_iterations > 0:

        print(
            "Smoothing mesh: "
            f"{args.mesh_smooth_iterations} iterations"
        )

        mesh = mesh.filter_smooth_taubin(
            number_of_iterations=args.mesh_smooth_iterations,
            filter_scope=o3d.geometry.FilterScope.Vertex
        )

        mesh.compute_vertex_normals()

    # STEP 6: Produce the final depth preview.
    depth_preview = make_depth_preview(
        depth,
        foreground_mask
    )

    # STEP 7: Save only the final outputs.
    args.output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    depth_path = args.output_dir / "depth_map.png"
    mesh_path = args.output_dir / "model.ply"

    if not cv2.imwrite(
        str(depth_path),
        depth_preview
    ):
        raise OSError(
            f"Failed to save depth map: {depth_path}"
        )

    mesh_saved = o3d.io.write_triangle_mesh(
        str(mesh_path),
        mesh,
        write_vertex_colors=True,
        write_vertex_normals=True
    )

    if not mesh_saved:
        raise OSError(
            f"Failed to save mesh: {mesh_path}"
        )

    # STEP 8: Display reconstruction results.
    print("\nReconstruction completed successfully!")

    print(f"Device: {device}")
    print(f"Mesh vertices: {len(mesh.vertices):,}")
    print(f"Mesh triangles: {len(mesh.triangles):,}")

    print(
        "Rejected triangles: "
        f"{stats['triangles_rejected']:,}"
    )

    print(
        "Max relative depth jump: "
        f"{args.max_relative_jump}"
    )

    print(
        "Mesh smoothing iterations: "
        f"{args.mesh_smooth_iterations}"
    )

    print(f"Depth map: {depth_path.resolve()}")
    print(f"3D model: {mesh_path.resolve()}")

    if args.view:

        o3d.visualization.draw_geometries(
            [mesh],
            window_name="ImageTo3D - Reconstructed Mesh",
            mesh_show_back_face=True
        )


if __name__ == "__main__":
    main()
