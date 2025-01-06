#IMPORT PACKAGES
import os
import os.path
from pathlib import Path
import numpy as np
import open3d as o3d

from geomapi.utils import geometryutils as gmu
from geomapi.nodes import PointCloudNode

from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt

import context 
import utils as utl
from utils import t7_utils

## PROCESS COLUMNS
# Print visualisation
plot = False

## INPUTS and OUTPUTS
name = '34_Parking_04_F1_small_pred' # Name of the 7 test files
# '08_ShortOffice_01_F1_small_pred'
# '08_ShortOffice_01_F2_small_pred'
# '11_MedOffice_05_F2_small_pred'
# '11_MedOffice_05_F4_small_pred'
# '25_Parking_01_F1_small_pred'
# '25_Parking_01_F2_small_pred'
# '34_Parking_04_F1_small_pred'

# Get the path of the directory
path = Path(os.getcwd()).parents[2]

# Construct the input folder path using 
input_folder_laz = path / 'data' / 'CVPR_Challenge'/'Challenge_laz'
# 'challenge_test_laz'
print('input folder laz:', input_folder_laz)
input_folder_ttl = path / 'data' / 'CVPR_Challenge'/ 'Challenge_ttl'
# 'challenge_test_ttl'
print('input folder ttl:', input_folder_ttl)

# Construct the file paths
file_name = input_folder_laz / f'{name}.laz'
print('input file name:', file_name)
class_file = input_folder_laz / '_classes.json'
graph_path = input_folder_ttl / f'{name}.ttl'
print('ttl:', graph_path)

# Output folder path
output_folder = path / 'data' /'CVPR_Challenge'/'Outputs'

# Output file path
output_file_path = output_folder / f'{name}_columns.json'

laz, _, _, normals = t7_utils.load_point_cloud(input_folder_laz/f'{name}.laz')
_, columns_nodes = t7_utils.load_graph(laz, input_folder_ttl / f'{name}.ttl')
min_z, max_z, avg_z, height = t7_utils.compute_column_height(laz.z)
# reference_levels = t7_utils.load_levels(laz, graph_path, columns_nodes)
floors_z, ceilings_z, floor_z, ceiling_z = t7_utils.load_levels(laz, graph_path)

with open(output_file_path, "w") as output_file:

    columns_points, columns_points_2d = t7_utils.load_columns_data(laz, columns_nodes, avg_z, normals)
    
    # For each column
    for key in columns_points_2d:

        points = np.array(columns_points_2d[key][:, :2])        

        bounding_box, rotation_matrix = t7_utils.compute_bounding_box(points)
        center = np.mean(bounding_box, axis=0)

        # Rotate bounding box vertices
        rotated_box = t7_utils.rotate_points(bounding_box, center, rotation_matrix.T)

        # Rotate all the points
        rotated_points = t7_utils.rotate_points(points, center, rotation_matrix.T)

        if plot:
            # Plotting
            plt.figure(figsize=(8, 6))
            plt.plot(points[:, 0], points[:, 1], 'o', markersize= 0.50, label='Column bounding boxes ')
            plt.plot(rotated_points[:, 0], rotated_points[:, 1], 'o', markersize= 0.50, label='Rotated points ')
            plt.plot(points[ConvexHull(points).vertices,0], points[ConvexHull(points).vertices,1], 'r--', lw=1, label='Convex Hull')
            plt.plot(np.append(bounding_box[:, 0], bounding_box[0, 0]), np.append(bounding_box[:, 1], bounding_box[0, 1]), 'g-', lw=1, label='Bounding Rectangle')
            plt.plot(np.append(rotated_box[:, 0], rotated_box[0, 0]), np.append(rotated_box[:, 1], rotated_box[0, 1]), 'g-', lw=1, label='Bounding Rectangle')
            plt.plot(rotated_box[0][0],rotated_box[0][1], 'ro') 
            plt.plot(rotated_box[1][0],rotated_box[1][1], 'bo') 
            plt.plot(rotated_box[2][0],rotated_box[2][1], 'go') 
            plt.plot(rotated_box[3][0],rotated_box[3][1], 'yo') 
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title(f'Minimum Bounding Rectangle for column {key}')
            plt.legend()
            plt.grid(True)
            plt.axis('equal')
            plt.show()

        # Vertices
        A = rotated_box[0]
        B = rotated_box[1]
        C = rotated_box[2]
        D = rotated_box[3]

        A1, B1, C1, D1 = t7_utils.compute_middle_points({'AB': (A,B), 'BC': (B,C), 'CD': (C,D), 'DA': (D,A)})

        # Direction vectors
        horizontal_segment = (D1[0] - B1[0], D1[1] - B1[1])
        vertical_segment = (C1[0] - A1[0], C1[1] - A1[1])

        # HISTOGRAM
        histogram_step = 0.010
        coordinate = 0

        new_x_coordinate, success = t7_utils.compute_best_coverage_segments(rotated_points, histogram_step, coordinate, key, output_file)

        if not success:
            continue

        B[0] = new_x_coordinate[0] 
        C[0] = new_x_coordinate[1]
        A[0] = new_x_coordinate[2]
        D[0] = new_x_coordinate[3]
        
        coordinate = 1
        new_y_coordinate, success = t7_utils.compute_best_coverage_segments(rotated_points, histogram_step, coordinate, key, output_file)

        if not success:
            continue
        
        B[1] = new_y_coordinate[0] 
        A[1] = new_y_coordinate[1]
        C[1] = new_y_coordinate[2]
        D[1] = new_y_coordinate[3]

        # PROCESSING POINTS_ VERTICAL DIRECTION

        # # New bbox
        new_points = np.array([A,B,C,D])
        print(f'Column {key}, New_points, {new_points}')
        print(rotated_points)

        if plot:
            # Plotting
            plt.figure(figsize=(8, 6))
            plt.plot(rotated_points[:, 0], rotated_points[:, 1], 'o', markersize= 0.50, label='Rotated points')
            plt.plot(np.append(rotated_box[:, 0], rotated_box[0, 0]), np.append(rotated_box[:, 1], rotated_box[0, 1]), 'g-', lw=2, label='Bounding box')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title(f'Bounding Rectangle: vertical and horizontal adjustments for column {key}')
            plt.legend()
            plt.grid(True)
            plt.axis('equal')
            plt.show()

        # Inputs
        print('BBox_:', rotated_box)
        print("Center:", center)

        minimum_oriented_box = t7_utils.rotate_points(rotated_box, center, rotation_matrix)
        print('BBox: ', minimum_oriented_box)
        # points = rotate_points (rotated_points, center, rotation_matrix )

        if plot:
            # Plotting
            plt.figure(figsize=(8, 6))
            plt.plot(points[:, 0], points[:, 1], 'o', markersize= 0.50, label='Rotated points')
            plt.plot(np.append(minimum_oriented_box[:, 0], minimum_oriented_box[0, 0]), np.append(minimum_oriented_box[:, 1], minimum_oriented_box[0, 1]), 'g-', lw = 2, label='Bounding box')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title(f'Bounding Rectangle: vertical and horizontal adjustments for column {key}')
            plt.legend()
            plt.grid(True)
            plt.axis('equal')
            plt.show()

        width, depth, height_column, center, rotation, min_z, max_z = t7_utils.column_features(columns_points[key], minimum_oriented_box, rotation_matrix, floors_z, ceilings_z)
        print ('Minimum oriented bounding box: ', minimum_oriented_box)   
        # print(f'Column: {key}, Width: {width} Depth: {depth} Height: {height} loc: {center} rotation {rotation}')

        # column_features = width, depth, height, center, rotation
        output_file.write(f'Column: {key}\n Width: {width}\n Depth: {depth}\n Height: {height_column}\n loc: {center} rotation {rotation}\n')
        print(f'Column: {key}\n Width: {width}\n Depth: {depth}\n Height: {height_column}\n loc: {center} rotation {rotation}\n')
        json_object_info = t7_utils.json_export(output_folder, name, key, width, depth, height_column, center, rotation)

        t7_utils.column_mesh(columns_points[key], min_z, max_z, height_column, minimum_oriented_box, depth, width, output_folder, f"{name}_{key}")
     


