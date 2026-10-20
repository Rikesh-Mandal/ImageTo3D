import open3d as o3d
import numpy as np
# ---- PATHS ----
pointcloud_path = r"E:\Dissertation\Trial\ImageTo3D\ImageTo3D\sample_output\pointcloud.ply"
mesh_output_path = r"E:\Dissertation\Trial\ImageTo3D\ImageTo3D\sample_output\mesh_poisson.ply"
# ---- LOAD POINT CLOUD ----
pcd = o3d.io.read_point_cloud(pointcloud_path)
# pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points) * 0.001)
# ---- ESTIMATE NORMALS ----
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))
pcd.orient_normals_consistent_tangent_plane(100)
# ---- POISSON SURFACE RECONSTRUCTION ----
mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
print("🔧 Poisson surface reconstruction completed")
# ---- CROP LOW-DENSITY REGION ----
bbox = pcd.get_axis_aligned_bounding_box()
mesh_crop = mesh.crop(bbox)
# ---- SMOOTHING FILTER (Optional) ----
mesh_crop = mesh_crop.filter_smooth_simple(number_of_iterations=3)
mesh_crop.remove_degenerate_triangles()
mesh_crop.remove_duplicated_triangles()
mesh_crop.remove_duplicated_vertices()
mesh_crop.remove_non_manifold_edges()
# ---- SAVE MESH ----
o3d.io.write_triangle_mesh(mesh_output_path, mesh_crop)
print(f"Mesh saved to {mesh_output_path}")
o3d.visualization.draw_geometries([pcd, mesh_crop])
