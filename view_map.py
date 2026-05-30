import open3d as o3d


print("Loading the cleaned point cloud map...")
pcd = o3d.io.read_point_cloud("data/cleaned_map.pcd")

print(f"Loaded cloud successfully with {len(pcd.points)} points.")


o3d.visualization.draw_geometries([pcd])