import laspy
import open3d as o3d
import numpy as np

las = laspy.read(“data/lidar.las”)
print(las)

'''
all properties of the las file can be accessed via the header and vlrs attributes of the las object.
For example, you can access the point format and point count as follows:
las.header
las.header.point_format
las.header.point_count
las.vlrs
'''
print(list(las.point_format.dimension_names)))

point_data = np.stack([las.X, las.Y, las.Z], axis=0).transpose((1, 0))

buildings = laspy.create(point_format=las.header.point_format, file_version=las.header.version)
buildings.points = las.points[las.classification == 6]

geom = o3d.geometry.PointCloud()
geom.points = o3d.utility.Vector3dVector(point_data)
o3d.visualization.draw_geometries([geom])