import laspy
import open3d as o3d
import numpy as np
from PIL import Image


las = laspy.read("data/points.laz")
print("Original point count:", las.header.point_count)
print("Dimensions:", list(las.point_format.dimension_names))

point_data = np.stack([las.X, las.Y, las.Z], axis=0).transpose((1, 0)) #lowercase is scaled data

buildings = laspy.create(point_format=las.header.point_format, file_version=las.header.version)
buildings.points = las.points[las.classification == 6]


geom = o3d.geometry.PointCloud()
geom.points = o3d.utility.Vector3dVector(point_data)


voxel_size = 0.2 
downsampled_geom = geom.voxel_down_sample(voxel_size=voxel_size)
print(f"Points remaining after downsampling: {len(downsampled_geom.points)}")


filtered_geom, ind = downsampled_geom.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
print(f"Points remaining after noise removal: {len(filtered_geom.points)}")

points_npy = np.asarray(filtered_geom.points)


resolution = 0.1  

x_coords = points_npy[:, 0]
y_coords = points_npy[:, 1]


x_min, y_min = np.min(x_coords), np.min(y_coords)
x_indices = ((x_coords - x_min) / resolution).astype(int)
y_indices = ((y_coords - y_min) / resolution).astype(int)


grid_width = x_indices.max() + 1
grid_height = y_indices.max() + 1
occupancy_grid = np.zeros((grid_height, grid_width), dtype=np.uint8)


occupancy_grid[y_indices, x_indices] = 255


oriented_grid = np.flipud(occupancy_grid)


o3d.io.write_point_cloud("data/cleaned_map.pcd", filtered_geom)


img = Image.fromarray(oriented_grid)
img.save("data/occupancy_grid_map.png")
print("Saved cleaned 3D point cloud and 2D occupancy grid map image successfully!")


o3d.visualization.draw_geometries([filtered_geom])
