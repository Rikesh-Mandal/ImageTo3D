"""Build a coloured 2.5D triangle mesh from an aligned depth and RGB image."""

import numpy as np
import open3d as o3d

from src.pointcloud import load_rgb_and_depth  # Reuse Milestone 3's file loader.


def _acceptable_triangles(faces, depths, valid, relative_jump, absolute_jump):
    """Keep faces whose three vertices have valid, similar depths."""
    z = depths[faces]
    z_min = z.min(axis=1)
    z_max = z.max(axis=1)

    good = valid[faces].all(axis=1)
    good &= (z_max - z_min) <= relative_jump * z_min

    if absolute_jump is not None:
        good &= (z_max - z_min) <= absolute_jump

    return good


def create_depth_mesh(
    depth,
    rgb,
    horizontal_fov=60.0,
    stride=1,
    border_crop=12,
    min_depth=0.1,
    max_depth=None,
    max_relative_jump=0.03,
    max_absolute_jump=None,
    foreground_mask=None,
):
    """
    Back-project sampled pixels and connect suitable neighbours.

    Coordinates: X right, Y down, Z forward.
    Depth values remain in uncalibrated model-output units.
    """
    depth = np.asarray(depth)
    rgb = np.asarray(rgb)

    if depth.ndim != 2 or rgb.shape != (*depth.shape, 3):
        raise ValueError("Expected depth [H,W] and aligned RGB [H,W,3].")
    if not np.issubdtype(depth.dtype, np.number) or np.iscomplexobj(depth):
        raise TypeError("Depth must contain real numerical values.")
    if not np.isfinite(horizontal_fov) or not 10 < horizontal_fov < 170:
        raise ValueError("Horizontal FOV must be between 10 and 170 degrees.")
    if not isinstance(stride, int) or isinstance(stride, bool) or stride < 1:
        raise ValueError("Stride must be a positive integer.")
    height, width = depth.shape
    if (not isinstance(border_crop, int) or isinstance(border_crop, bool)
            or border_crop < 0 or 2 * border_crop >= min(height, width)):
        raise ValueError("Invalid border crop size.")
    if not np.isfinite(min_depth) or min_depth < 0:
        raise ValueError("Minimum depth must be finite and nonnegative.")
    if max_depth is not None and (not np.isfinite(max_depth) or max_depth <= min_depth):
        raise ValueError("Maximum depth must exceed minimum depth.")
    if not np.isfinite(max_relative_jump) or max_relative_jump <= 0:
        raise ValueError("Maximum relative jump must be positive and finite.")
    if max_absolute_jump is not None and (
        not np.isfinite(max_absolute_jump) or max_absolute_jump <= 0
    ):
        raise ValueError("Maximum absolute jump must be positive and finite.")

    # Keep pixel positions in the ORIGINAL image coordinate system.
    rows = np.arange(border_crop, height - border_crop, stride)
    cols = np.arange(border_crop, width - border_crop, stride)
    if len(rows) < 2 or len(cols) < 2:
        raise ValueError("Crop/stride leaves fewer than two rows or columns.")
    u, v = np.meshgrid(cols, rows)
    sampled_depth = depth[v, u].astype(np.float64)
    sampled_rgb = rgb[v, u]

    valid = np.isfinite(sampled_depth) & (sampled_depth > 0)
    valid &= sampled_depth >= min_depth
    if max_depth is not None:
        valid &= sampled_depth <= max_depth

    # Optional subject mask: keep only pixels classified as foreground.
    if foreground_mask is not None:
        foreground_mask = np.asarray(foreground_mask)
        if foreground_mask.shape != depth.shape:
            raise ValueError(
                f"Mask shape {foreground_mask.shape} does not match depth shape {depth.shape}."
            )
        valid &= foreground_mask[v, u] > 0

    if valid.sum() < 3:
        raise ValueError("Not enough valid depth pixels to construct a mesh.")

    focal = (width / 2.0) / np.tan(np.deg2rad(horizontal_fov) / 2.0)
    cx, cy = (width - 1) / 2.0, (height - 1) / 2.0

    # Each pixel is initially one vertex. Unused vertices are removed below.
    z = sampled_depth.ravel()
    xyz = np.column_stack((
        (u.ravel() - cx) * z / focal,
        (v.ravel() - cy) * z / focal,
        z,
    ))
    colors = sampled_rgb.reshape(-1, 3).astype(np.float64) / 255.0
    flat_valid = valid.ravel()

    # Every block of four neighbouring pixels can form two triangles.
    indices = np.arange(z.size, dtype=np.int32).reshape(sampled_depth.shape)
    tl = indices[:-1, :-1].ravel()  # top left
    tr = indices[:-1, 1:].ravel()   # top right
    bl = indices[1:, :-1].ravel()   # bottom left
    br = indices[1:, 1:].ravel()    # bottom right

    # Two possible ways to split a grid square. Winding faces the camera.
    a1 = np.column_stack((tl, br, tr))
    a2 = np.column_stack((tl, bl, br))
    b1 = np.column_stack((tl, bl, tr))
    b2 = np.column_stack((tr, bl, br))

    a1_ok = _acceptable_triangles(a1, z, flat_valid, max_relative_jump, max_absolute_jump)
    a2_ok = _acceptable_triangles(a2, z, flat_valid, max_relative_jump, max_absolute_jump)
    b1_ok = _acceptable_triangles(b1, z, flat_valid, max_relative_jump, max_absolute_jump)
    b2_ok = _acceptable_triangles(b2, z, flat_valid, max_relative_jump, max_absolute_jump)

    # Prefer the diagonal that preserves the most safe triangles.
    a_count = a1_ok.astype(np.int8) + a2_ok.astype(np.int8)
    b_count = b1_ok.astype(np.int8) + b2_ok.astype(np.int8)
    a_diagonal = np.nan_to_num(np.abs(z[tl] - z[br]), nan=np.inf, posinf=np.inf, neginf=np.inf)
    b_diagonal = np.nan_to_num(np.abs(z[tr] - z[bl]), nan=np.inf, posinf=np.inf, neginf=np.inf)
    use_a = (a_count > b_count) | (
        (a_count == b_count) & (a_diagonal <= b_diagonal)
    )

    faces = np.concatenate((
        a1[use_a & a1_ok],
        a2[use_a & a2_ok],
        b1[~use_a & b1_ok],
        b2[~use_a & b2_ok],
    ), axis=0)
    if len(faces) == 0:
        raise ValueError("No triangles survived depth-edge filtering; increase --max-relative-jump.")

    # Compact vertex indices so exported meshes contain no unused vertices.
    used = np.unique(faces)
    remap = np.full(z.size, -1, dtype=np.int32)
    remap[used] = np.arange(len(used), dtype=np.int32)
    faces = remap[faces]

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(xyz[used])
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors[used])
    mesh.compute_vertex_normals()

    possible_faces = 2 * (len(rows) - 1) * (len(cols) - 1)
    stats = {
        "sampled_image_size": [int(len(rows)), int(len(cols))],
        "sampled_pixels": int(z.size),
        "valid_sampled_pixels": int(flat_valid.sum()),
        "foreground_mask_used": foreground_mask is not None,
        "mesh_vertices": int(len(used)),
        "mesh_triangles": int(len(faces)),
        "possible_triangles": int(possible_faces),
        "triangles_rejected": int(possible_faces - len(faces)),
        "border_crop_pixels": int(border_crop),
        "stride": int(stride),
        "min_depth_filter": float(min_depth),
        "max_depth_filter": None if max_depth is None else float(max_depth),
        "max_relative_jump": float(max_relative_jump),
        "max_absolute_jump": None if max_absolute_jump is None else float(max_absolute_jump),
        "assumed_horizontal_fov_degrees": float(horizontal_fov),
        "approximate_fx_fy_pixels": float(focal),
        "principal_point_pixels": [float(cx), float(cy)],
        "depth_units": "Uncalibrated model output, not guaranteed metres",
        "mesh_type": "Open single-view visible-surface mesh (not watertight)",
    }
    return mesh, stats
