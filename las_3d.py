import laspy
import open3d as o3d
import numpy as np
from PIL import Image

# 1. Read the LAS/LAZ file
las = laspy.read("data/points.laz")
print("Original point count:", las.header.point_count)
print("Dimensions:", list(las.point_format.dimension_names))

# 2. Extract coordinates and create building-only cloud
point_data = np.stack([las.X, las.Y, las.Z], axis=0).transpose((1, 0))

buildings = laspy.create(point_format=las.header.point_format, file_version=las.header.version)
buildings.points = las.points[las.classification == 6]

# Convert full point data to Open3D format
geom = o3d.geometry.PointCloud()
geom.points = o3d.utility.Vector3dVector(point_data)

# 3. Downsample the Point Cloud (Fixed function name with underscores)
voxel_size = 0.2 
downsampled_geom = geom.voxel_down_sample(voxel_size=voxel_size)
print(f"Points remaining after downsampling: {len(downsampled_geom.points)}")

# 4. Filter Noise (Statistical Outlier Removal)
filtered_geom, ind = downsampled_geom.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
print(f"Points remaining after noise removal: {len(filtered_geom.points)}")

# 5. Project into a 2D Occupancy Grid Map
points_npy = np.asarray(filtered_geom.points)

# Define map resolution (0.1 meters per grid cell pixel)
resolution = 0.1  

# Extract horizontal coordinates
x_coords = points_npy[:, 0]
y_coords = points_npy[:, 1]

# Shift coordinates to start at index (0,0)
x_min, y_min = np.min(x_coords), np.min(y_coords)
x_indices = ((x_coords - x_min) / resolution).astype(int)
y_indices = ((y_coords - y_min) / resolution).astype(int)

# Create a blank grid canvas (0 = free space)
grid_width = x_indices.max() + 1
grid_height = y_indices.max() + 1
occupancy_grid = np.zeros((grid_height, grid_width), dtype=np.uint8)

# Mark cells containing LiDAR points as obstacles (255 = occupied)
occupancy_grid[y_indices, x_indices] = 255

# Flip vertically to match real-world coordinate orientation
oriented_grid = np.flipud(occupancy_grid)

# 6. Save Outputs
# Save the cleaned 3D map format
o3d.io.write_point_cloud("data/cleaned_map.pcd", filtered_geom)

# Save the 2D grid map image using PIL
img = Image.fromarray(oriented_grid)
img.save("data/occupancy_grid_map.png")
print("Saved cleaned 3D point cloud and 2D occupancy grid map image successfully!")

# 7. Visualize the Processed 3D Point Cloud
o3d.visualization.draw_geometries([filtered_geom])
