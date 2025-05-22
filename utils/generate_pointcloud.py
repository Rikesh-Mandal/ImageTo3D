import numpy as np
import cv2
import open3d as o3d
import os

# ----- CONFIG -----
rgb_path = r"E:\Dissertation\Trial\ImageTo3D\ImageTo3D\rgb_path\rgb.png"
depth_path = r"E:\Dissertation\Trial\ImageTo3D\ImageTo3D\depth_path\depth.png"
output_ply_path = r"E:\Dissertation\Trial\ImageTo3D\ImageTo3D\sample_output\pointcloud.ply"
fx = fy = 525.0  # focal length in pixels
max_depth = 3.0  # meters
# ----- LOAD IMAGES -----
rgb = cv2.imread(rgb_path)
rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
depth = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
depth *= max_depth
# ----- OPTIONAL: Smooth Depth -----
depth = cv2.bilateralFilter(depth, d=7, sigmaColor=0.1, sigmaSpace=25)
# ----- CONVERT TO OPEN3D FORMAT -----
rgb_o3d = o3d.geometry.Image(rgb)
depth_o3d = o3d.geometry.Image(depth)
rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
    rgb_o3d,
    depth_o3d,
    convert_rgb_to_intensity=False,
    depth_scale=1.0,
    depth_trunc=max_depth
)
# ----- CAMERA INTRINSICS -----
height, width = depth.shape
cx, cy = width / 2, height / 2
intrinsics = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
# ----- GENERATE POINT CLOUD -----
pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsics)

# Flip for correct orientation to match softwares like Blender or Maya
pcd.transform([[1, 0, 0, 0],
               [0, -1, 0, 0],
               [0, 0, -1, 0],
               [0, 0, 0, 1]])
# ----- SAVE -----
o3d.io.write_point_cloud(output_ply_path, pcd)
print(f"Point cloud saved to: {output_ply_path}")

# Optional visualization
o3d.visualization.draw_geometries([pcd])
