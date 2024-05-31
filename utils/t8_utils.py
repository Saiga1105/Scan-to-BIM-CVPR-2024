#IMPORT PACKAGES
from rdflib import Graph
import rdflib
import os.path
import importlib
from pathlib import Path
import numpy as np
import xml.etree.ElementTree as ET
import open3d as o3d

import uuid    
import pye57 
import ifcopenshell
import ifcopenshell.geom as geom
import ifcopenshell.util
from ifcopenshell.util.selector import Selector
import multiprocessing
import random as rd
import pandas as pd
# from tabulate import tabulate
import cv2
import laspy
import json
from scipy.spatial.transform import Rotation   
import copy
import geomapi
from geomapi.nodes import *
import geomapi.utils as ut
from geomapi.utils import geometryutils as gmu
import geomapi.tools as tl
from scipy.linalg import orthogonal_procrustes
import matplotlib.pyplot as plt
import geomapi.tools.progresstools as pt

from sklearn.cluster import DBSCAN
from PIL import Image

import geomapi
from geomapi.nodes import *
import geomapi.utils as ut
from geomapi.utils import geometryutils as gmu
import geomapi.tools as tl
import geomapi.tools.progresstools as pt

from typing import Dict, Any, Tuple,List

import torch
from PIL import Image
from torchvision.ops import box_convert

# Grounding DINO
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict
from groundingdino.util.inference import annotate, load_image, predict



import supervision as sv

# segment anything
# from segment_anything import build_sam, SamPredictor 
import numpy as np


# diffusers
import torch

from huggingface_hub import hf_hub_download
import math

def points_on_line(point1, point2, step_size):
    """
    Generate points on a line between two given points with a specified step size.

    Parameters:
    - point1: The starting point of the line.
    - point2: The ending point of the line.
    - step_size: The step size between consecutive points.

    Returns:
    - points: A list of points on the line.
    """
    # Calculate the direction vector
    direction = point2 - point1

    # Calculate the length of the line segment
    length = np.linalg.norm(direction)

    # Normalize the direction vector
    direction /= length

    # Calculate the number of steps needed
    num_steps = int(length / step_size)

    # Generate points along the line
    points = np.array([point1 + i * step_size * direction for i in range(num_steps + 1)])

    return points

def create_wall_ortho(startpoint, endpoint, height, resolution, direction, scene, triangle_colors, offset =1, path = None, show = False, dominant = True, max_distance = 0.5, min_distance = 0.5):
    image_size = (int(np.sqrt(np.sum((endpoint - startpoint)**2)) / resolution)+1, int(height / resolution))
    
    min_z = np.min([endpoint[2], startpoint[2]])
    max_z = min_z + height
    num_z_steps = int(height /resolution)
    z_grid = np.linspace(min_z, max_z, num_z_steps)  # Adjust the number of grid points as needed
    z_grid = z_grid[::-1]
    xyz_grid = []
    for z in z_grid:
        start = startpoint.copy()
        end =  endpoint.copy()
        start[2] = z
        end[2] = z
        if not dominant:
            xyz_grid.append(points_on_line(start, end, resolution)[::-1])
        else: 
            xyz_grid.append(points_on_line(start, end, resolution))

    grid = np.asarray(xyz_grid).reshape((-1, 3), order='C') 
    ray_grid = grid + direction*offset
    
    #create rays for the in side (towards the dominant side
    ori_x = direction[0] * np.ones(len(ray_grid))
    ori_y = direction[1] * np.ones(len(ray_grid))
    ori_z = direction[2] * np.ones(len(ray_grid))
    
    pos_x = ray_grid[:,0]
    pos_y = ray_grid[:,1]
    pos_z = ray_grid[:,2]
    
    # Stack the calculated values along the third axis to create the grid
    rays_values = np.stack((pos_x, pos_y, pos_z, -ori_x, -ori_y, -ori_z), axis=1)
    rays_tensor = o3d.core.Tensor(rays_values, dtype=o3d.core.Dtype.Float32)
    
    ans= scene.cast_rays(rays_tensor) 
    
    
    triangle_ids = ans["primitive_ids"].numpy() # triangles     
    triangle_ids = triangle_ids.flatten()
    np.put(triangle_ids,np.where(triangle_ids==scene.INVALID_ID),triangle_colors.shape[0]-1) # replace invalid id's by last (which is the above added black color)
    
    # Get the hit distances for each ray
    hit_distances = ans["t_hit"].numpy().flatten()
    
    # Filter out hits that are too far or too close
    if max_distance is not None:
        triangle_ids[hit_distances > max_distance+offset] = triangle_colors.shape[0] - 1  # Set to black
        
    if min_distance is not None:
        triangle_ids[hit_distances < min_distance] = triangle_colors.shape[0] - 1  # Set to black
    
    colors = triangle_colors[triangle_ids]
    ortho = np.reshape(colors,(image_size[1],image_size[0],3))
    
    if show:
        plt.imshow(ortho)
        plt.show()
    if path:
        image = Image.fromarray((ortho * 255).astype(np.uint8))
        # Save the image
        image.save(path)
        
    return ortho

def fill_black_pixels(image:np.array,region:int=5)->np.array:
    """Fill in the black pixels in an RGB image given a search distance.\n
 
    Args:
        image (np.array)\n
        region (int, optional): search distance. Defaults to 5.\n
 
    Returns:
        np.array: image
    """
    kernel = np.ones((region,region),np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

def load_model_hf(repo_id, filename, ckpt_config_filename, device='cpu'):
    cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)

    args = SLConfig.fromfile(cache_config_file) 
    model = build_model(args)
    args.device = device

    cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    checkpoint = torch.load(cache_file, map_location=device)
    log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    print("Model loaded from {} \n => {}".format(cache_file, log))
    _ = model.eval()
    return model   

# def extract_box_with_margin(image, box, margin = 1):
#     # Extract center and size from the detection array
#     width = int(np.asarray(box)[2]*image.shape[1])
#     height =   int(np.asarray(box)[3]*image.shape[0])
    
#     center_x = int(np.asarray(box)[0]*image.shape[1])
#     center_y = int(np.asarray(box)[1]*image.shape[0])  
    

#     # Calculate coordinates of the bounding box with extra margin
#     x1 = int(center_x - width/2) - margin
#     y1 = int(center_y - height/2) - margin
#     x2 = int(center_x + width/2) + margin
#     y2 = int(center_y + height/2) + margin

#     # Ensure the box is within the image bounds
#     x1 = max(x1, 0)
#     y1 = max(y1, 0)
#     x2 = min(x2, image.shape[1])
#     y2 = min(y2, image.shape[0])

#     # Create a new image containing pixels within the bounding box
#     extracted_image = np.copy(image[y1:y2, x1:x2])
#     normalized_image = cv2.normalize(extracted_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

#     return normalized_image.astype(np.uint8)

def compute_center(point1, point2):
    center = [(point1[0] + point2[0]) / 2,
              (point1[1] + point2[1]) / 2,
              (point1[2] + point2[2]) / 2]
    return center

def gradient_score_height(height):     
    if height < 1.5 or height > 2.4:         
        return 0
    elif height <= 2.0:         
        return (height - 1.5) / (2.0 - 1.5)     
    else:         
        return (2.4 - height) / (2.4 - 2.0)
def gradient_score_width(width):     
    if width < 0.5 or width > 1.5:         
        return 0
    elif width <= 1.0:         
        return (width - 0.5) / (1.0 - 0.5)     
    else:         
        return (1.5 - width) / (1.5 - 1.0)

def gradient_score_distance_to_ground(distance):     
    if distance < 0 or distance > 0.5:         
        return 0
    else:         
        return 1 - (distance / 0.5)
    
def is_door(probability, width, height, reference_level, prob_weight=0.15, width_weight=0.15, height_weight=0.15, ref_level_weight=0.01, print_overview = False):
    
    width_score = gradient_score_width(width)
    height_score = gradient_score_height(height)

    # Reference level score calculation with additional penalty
    reference_level_score = gradient_score_distance_to_ground(reference_level)
   
    # Normalize weights so that their sum is 1
    total_weight = prob_weight + width_weight + height_weight + ref_level_weight
    normalized_prob_weight = prob_weight / total_weight
    normalized_width_weight = width_weight / total_weight
    normalized_height_weight = height_weight / total_weight
    normalized_ref_level_weight = ref_level_weight / total_weight

    # Weighted scores
    weighted_prob = probability * normalized_prob_weight
    weighted_width = width_score * normalized_width_weight
    weighted_height = height_score * normalized_height_weight
    weighted_ref_level = reference_level_score * normalized_ref_level_weight

    # Combine weighted scores
    combined_score = weighted_prob + weighted_width + weighted_height + weighted_ref_level
    if  0.5 < width > 1.5:
        combined_score = 0
    elif 1.5 > height > 2.5:
        combined_score = 0
        
    if print_overview:
        print("Probability: %s - SCORE: %s" %(probability, weighted_prob))
        print("Width: %s - SCORE: %s" %(width, weighted_width))
        print("height: %s - SCORE: %s" %(height, weighted_height))
        print("Reference Level: %s - SCORE: %s" %(reference_level, weighted_ref_level))
        print("==========================================================================")
        print("Total SCORE: %s" %(combined_score))
        print("")

    return combined_score

# def box_to_corners(box):
#     u_center, v_center, u_size, v_size = box
#     half_u_size = u_size / 2
#     half_v_size = v_size / 2
#     u_min = u_center - half_u_size
#     u_max = u_center + half_u_size
#     v_min = v_center - half_v_size
#     v_max = v_center + half_v_size
#     return u_min, v_min, u_max, v_max

# def compute_iou(box1, box2):
#     # Convert boxes to corners
#     u_min1, v_min1, u_max1, v_max1 = box_to_corners(box1)
#     u_min2, v_min2, u_max2, v_max2 = box_to_corners(box2)
    
#     # Calculate intersection coordinates
#     inter_u_min = max(u_min1, u_min2)
#     inter_v_min = max(v_min1, v_min2)
#     inter_u_max = min(u_max1, u_max2)
#     inter_v_max = min(v_max1, v_max2)
    
#     # Calculate intersection area
#     inter_area = max(0, inter_u_max - inter_u_min) * max(0, inter_v_max - inter_v_min)
    
#     # Calculate areas of each box
#     area1 = (u_max1 - u_min1) * (v_max1 - v_min1)
#     area2 = (u_max2 - u_min2) * (v_max2 - v_min2)
    
#     # Calculate union area
#     union_area = area1 + area2 - inter_area
    
#     # Calculate IoU
#     iou = inter_area / union_area if union_area > 0 else 0
    
#     return iou

# def compute_parameter_similarity(params1, params2):
#     # Example: use Euclidean distance and convert it to a similarity score
#     distance = np.linalg.norm(np.array(params1) - np.array(params2))
#     similarity = 1 / (1 + distance)  # similarity decreases with distance
#     return similarity

# def compute_doorness_similarity(score1, score2):
#     # Assuming score1 and score2 are probabilities or confidence scores
#     # Convert them into numpy arrays for easier computation
#     score1 = np.array(score1)
#     score2 = np.array(score2)
    
#     # Compute Euclidean distance between the scores
#     distance = np.linalg.norm(score1 - score2)
    
#     # Similarity is inversely proportional to distance
#     similarity = 1 / (1 + distance)
    
#     return similarity

# def find_best_matches(boxes1, boxes2, params1, params2, iou_weight=0.33, param_weight=0.33, doorness_weight=0.33, size_diff_threshold = 0.2):
#     matches = []
#     used_boxes1 = set()  # Track used boxes from boxes1
#     used_boxes2 = set()  # Track used boxes from boxes2

#     for i, box1 in enumerate(boxes1):
#         max_score = -1
#         best_match = -1
#         area1 = compute_area(box1)

#         for j, box2 in enumerate(boxes2):
#             if j in used_boxes2:
#                 continue
            
#             iou = compute_iou(box1, box2)
#             area2 = compute_area(box2)
                        
#             if iou > 0.2: #or max(area1, area2) / min(area1, area2) < 0.9:
#                 used_boxes2.add(j)
#             # else:
#             param_similarity = compute_parameter_similarity(params1[i][0:2], params2[j][0:2])
#             doorness_similarity = compute_doorness_similarity(params1[i][4:5], params2[j][4:5])
            
#             # Combine IoU, parameter similarity, and doorness score into a single score
#             combined_score = iou_weight * iou + param_weight * param_similarity + doorness_weight * doorness_similarity
            
#             if combined_score > max_score:
#                 max_score = combined_score
#                 best_match = j
            
        
#         if best_match != -1:
#             matches.append((i, best_match, max_score))
#             used_boxes1.add(i)
#             used_boxes2.add(best_match)

#     # Identify unmatched elements in boxes1 and boxes2
#     unmatched_boxes1 = [i for i in range(len(boxes1)) if i not in used_boxes1]
#     unmatched_boxes2 = [j for j in range(len(boxes2)) if j not in used_boxes2]

#     return matches, unmatched_boxes1, unmatched_boxes2

# def combine_boxes(box1, box2):
#     # # print(box1)
#     # # print(box2)
#     # box2 = np.asarray(box2)
#     # box2[0] = 1 - box2[0]
#     # # print(box2)
        
#     u_min1, v_min1, u_max1, v_max1 = box_to_corners(box1)
#     u_min2, v_min2, u_max2, v_max2 = box_to_corners(box2)
#     # print(f"{u_min1} <-> {u_min2}")
#     # print(f"{u_max1} <-> {u_max2}")

#     combined_u_min = min(u_min1, u_min2)
#     combined_v_min = min(v_min1, v_min2)
#     combined_u_max = max(u_max1, u_max2)
#     combined_v_max = max(v_max1, v_max2)

#     combined_u_center = (combined_u_min + combined_u_max) / 2
#     combined_v_center = (combined_v_min + combined_v_max) / 2
#     combined_u_size = combined_u_max - combined_u_min
#     combined_v_size = combined_v_max - combined_v_min

#     return np.array([[combined_u_center, combined_v_center, combined_u_size, combined_v_size]])

def line_with_width_coordinates(start, end, percentage, width, reference_level):
    # Calculate the coordinates of the point on the line segment
    point_on_line = start + (end - start) * percentage
    
    # Calculate the direction vector of the line segment
    direction = (end - start) / np.linalg.norm(end - start)
    
    # Calculate the offset for the new line
    offset = direction * (width / 2)
    
    # Calculate the start and end points of the new line
    new_start = point_on_line - offset
    new_end = point_on_line + offset
    
    new_start[2] = new_start[2] + reference_level
    new_end[2] = new_end[2] + reference_level
    
    return [new_start, new_end]

# def convert_to_center_size(box):
#     x1, y1, x2, y2 = box
#     u_center = (x1 + x2) / 2
#     v_center = (y1 + y2) / 2
#     u_size = x2 - x1
#     v_size = y2 - y1
#     return (u_center, v_center, u_size, v_size)

# def compute_area(box):
#     """Compute the area of a box."""
#     box = box_to_corners(box)
#     # Assuming box is in the format [x1, y1, x2, y2]
#     width = abs(box[2] - box[0])
#     height = abs(box[3] - box[1])
#     return width * height

# def find_and_merge_high_iou_boxes(boxes,info, threshold=0.6, size_diff_threshold=0.2):
#     """
#     Find and merge boxes with high IoU, taking size difference into account.
    
#     Args:
#         boxes (list): List of boxes in the format [x1, y1, x2, y2].
#         threshold (float): IoU threshold for merging.
#         size_diff_threshold (float): Maximum allowable ratio of box sizes for merging.

#     Returns:
#         list: List of merged boxes.
#     """
        
#     merged = True
#     while merged:
#         merged = False
#         num_boxes = len(boxes)
#         if num_boxes > 1:
#             for i in range(num_boxes):
#                 if merged:
#                     break
#                 for j in range(i + 1, num_boxes):
#                     iou = compute_iou(boxes[i], boxes[j])
#                     if iou > threshold:
#                         score_i = info[i][-1]
#                         score_j = info[j][-1]
#                         if score_j > score_i:
#                             del boxes[i]
#                             del info[i]
#                             num_boxes = len(boxes)
#                             break
#                         elif score_i > score_j:
#                             del boxes[j]
#                             del info[j]
#                             num_boxes = len(boxes)
#                             break
#     return boxes, info

# def calculate_percentage_black_pixels(image):
#     # Check if the image is a color image (3 channels) or a grayscale image (1 channel)
#     if len(image.shape) == 3 and image.shape[2] == 3:
#         # Convert the color image to grayscale
#         grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     elif len(image.shape) == 2:
#         # Image is already grayscale
#         grayscale_image = image
#     else:
#         # Handle other cases or raise an error
#         raise ValueError("Unsupported image format. Expected color (3 channels) or grayscale (1 channel).")

#     # # Convert the image to grayscale
#     # grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#     # Count black pixels
#     black_pixel_count = cv2.countNonZero(grayscale_image)

#     # Calculate total number of pixels
#     total_pixels = grayscale_image.shape[0] * grayscale_image.shape[1]

#     # Calculate percentage of black pixels
#     percentage_black_pixels = (black_pixel_count / total_pixels)

#     return float(1-percentage_black_pixels)

# # Function to calculate the distance between two points
# def calculate_distance(point1, point2):
#     return np.linalg.norm(np.array(point1) - np.array(point2))

def write_obj_with_submeshes(filename, meshes, mesh_names):
    """
    Write multiple Open3D TriangleMesh objects to a single OBJ file with submeshes.

    Parameters:
    - filename: str, the name of the output OBJ file.
    - meshes: list of open3d.geometry.TriangleMesh, the meshes to write.
    - mesh_names: list of str, the names of the submeshes.
    """
    if len(meshes) != len(mesh_names):
        raise ValueError("meshes and mesh_names must have the same length")

    vertex_offset = 1  # OBJ files are 1-indexed
    with open(filename, 'w') as file:
        for mesh, name in zip(meshes, mesh_names):
            file.write(f"g {name}\n")  # Start a new group for the submesh

            # Write vertices
            for vertex in mesh.vertices:
                file.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")

            # Write faces, adjusting indices based on the current offset
            for triangle in mesh.triangles:
                adjusted_triangle = triangle + vertex_offset
                file.write(f"f {adjusted_triangle[0]} {adjusted_triangle[1]} {adjusted_triangle[2]}\n")

            # Update the vertex offset for the next mesh
            vertex_offset += len(mesh.vertices)
            
def doors_to_json(doorNodes: list[MeshNode]) -> Dict:
    """
    Converts wall nodes data to a Dictionary ready for saving or further processing.

    Parameters:
        wallNodes (list): A list of nodes, each representing a wall with attributes like box, name, and height.
        file_name (str): The file name from which the walls are derived.

    Returns:
        str: A Dictionary representing the wall data.
    
    Raises:
        ValueError: If the input data is not in the expected format or missing required data.
    """
    objects=[]
    # Fill JSON with node data
    for n in doorNodes:
        objects.append({
            # "name": n.name,
            "id":int(n.object_id),
            "width": float(n.doorWidth),
            "height": float(n.height),
            "depth": float(n.depth),
            "loc": list(map(float, np.asarray(n.center))),                           
            "rotation": float(n.rotation),
            "host_id": int(n.host.derivedFrom.object_id),
            })
      
    # Convert the Python dictionary to a JSON string
    return objects

def max_2d_distance(points):
    def distance_2d(p1, p2):
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
    
    max_dist = 0
    num_points = len(points)
    
    for i in range(num_points):
        for j in range(i + 1, num_points):
            dist = distance_2d(points[i], points[j])
            if dist > max_dist:
                max_dist = dist
    
    return max_dist

def match_graph_with_las(file_path,class_dict, nodes, getResources=True,getNormals=False)->List[PointCloudNode]:
    # get the point cloud data
    if getResources:
        # print(f'processing {ut.get_filename(file_path)}...')      
        las = laspy.read(file_path) 
        pcd=gmu.las_to_pcd(las)
        pcd.estimate_normals() if getNormals else None
        
        #match pcd to nodes
        for c in class_dict['classes']:
            idx=np.where((las['classes']==c['id']))[0]
            class_pcd=pcd.select_by_index(idx)
            object_labels=las['objects'][idx]
            
            for j in np.unique(object_labels):
                indices=np.where(object_labels==j)[0]
                object_pcd=class_pcd.select_by_index(indices)
                node=next((x for x in nodes if x.object_id == j), None)
                if not node == None:
                    node.pcd=object_pcd if node is not None else None
            
    return nodes

def get_angle_with_x_axis(start_point, end_point):
    # Calculate the direction vector of the line
    direction_vector = np.array(end_point) - np.array(start_point)
    
    # Calculate the angle with the x-axis
    # x-axis direction vector
    x_axis_vector = np.array([1, 0, 0])
    
    # Dot product of direction vector and x-axis vector
    dot_product = np.dot(direction_vector, x_axis_vector)
    
    # Magnitude (length) of the direction vector
    magnitude_direction_vector = np.linalg.norm(direction_vector)
    
    # Magnitude of the x-axis vector (which is 1)
    magnitude_x_axis_vector = np.linalg.norm(x_axis_vector)
    
    # Calculate the cosine of the angle
    cos_theta = dot_product / (magnitude_direction_vector * magnitude_x_axis_vector)
    
    # Calculate the angle in radians
    theta_radians = np.arccos(cos_theta)
    
    # Convert the angle to degrees
    theta_degrees = np.degrees(theta_radians)
    
    return theta_degrees

def compute_door_parameters(detectionbox, ortho, image_resolution = 0.01):
    opening_width = round(int(np.asarray(detectionbox)[0][2]*ortho.shape[1])* image_resolution, 2)
    # print("Opening Width:", opening_width)
    
    opening_height = round(int(np.asarray(detectionbox)[0][3]*ortho.shape[0]) * image_resolution, 2)
    
    detection_center_u = int(np.asarray(detectionbox)[0][0]*ortho.shape[1]) * image_resolution
    detection_center_v = int(np.asarray(detectionbox)[0][1]*ortho.shape[0]) * image_resolution
    reference_level = round((ortho.shape[0]*image_resolution) - (detection_center_v + opening_height/2), 2)
    
    image_resource = extract_box_with_margin(ortho, detectionbox[0])
    image_resource = image_resource[...,::-1] # BGR to RGB
    bl_px = calculate_percentage_black_pixels(image_resource)
    
    return opening_width, opening_height, detection_center_u, detection_center_v, reference_level, bl_px

def create_orthos(wallNodes, image_resolution = 0.01, octree_size = 10, fill_region = 12):
    #Create a messh from this point cloud 
    octree=pt.pcd_to_octree(gmu.join_geometries([n.pcd for n in wallNodes]),octree_size) #if octree is None else octree

    # Calculate the size of each octree node based on octree depth and overall size
    def calculate_node_size(octree_depth, octree_size):
        num_voxels_per_dim = 2 ** octree_depth
        voxel_size = octree_size / num_voxels_per_dim
        return voxel_size

    voxel_size = calculate_node_size(octree.max_depth, octree.size)
    # print("Voxelsize = ", voxel_size)
    
    voxelmesh=gmu.octree_to_voxelmesh(octree) #if mesh is None else mesh
    # o3d.io.write_triangle_mesh(os.path.join(output_folder, "mesh.obj"), voxelmesh)
    # path = os.path.join(output_folder, "mesh.obj")
    # print(f"Saved Mesh to {path}")
    
    #Create a identity array containing the color so this can be retrieved afterwards
    original_colors=np.asarray(voxelmesh.vertex_colors)

    indices=np.asarray(voxelmesh.triangles)[:,0]
    triangle_colors=original_colors[indices]

    #append black color at the end of the array for the invalid hits
    triangle_colors=np.vstack((triangle_colors,np.array([0,0,0])))
    
    cpu_mesh=o3d.t.geometry.TriangleMesh.from_legacy(voxelmesh)
    # o3d.io.write_triangle_mesh(os.path.join(output_folder, "mesh.obj"), cpu_mesh)
    # path = os.path.join(output_folder, "mesh.obj")
    # print(f"Saved Mesh to {path}")
    
    # Create raycasting scene
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(cpu_mesh) 
    
    wall = 0
    for n in wallNodes:
        length = np.sqrt(np.sum((n.endpoint - n.startpoint)**2))
        surface = length * n.height
        n.orthos = []
        
        if not surface < 3 and n.height > 1.5 and length > 0.8:  
            #Create an ortho of the dominant side of the wall      
            ortho = create_wall_ortho(startpoint = n.startpoint, endpoint= n.endpoint, height= n.height, resolution = image_resolution, direction = n.normal, scene=scene, triangle_colors = triangle_colors, show = False)
            ortho = fill_black_pixels(ortho, region = fill_region)
            n.orthos.append(ortho)
            # if np.max(ortho) <= 1.0:
            #         ortho = ortho * 255.0
            # ortho_uint8 = ortho.astype(np.uint8)
            # Image.fromarray(ortho_uint8).save(os.path.join(output_folder,("Wall_" + str(n.object_id) + "-Face_" + str(1) +'.png')))
            
            #Also create an ortho of the other side of the wall
            if not n.singleFaced: #Single faced wall only needs one side
                ortho = create_wall_ortho(startpoint = n.startpoint, endpoint= n.endpoint, height= n.height, resolution = image_resolution, direction = -n.normal, scene=scene, triangle_colors = triangle_colors, dominant = False)
                ortho = fill_black_pixels(ortho, region = fill_region)
                n.orthos.append(ortho)
                # if np.max(ortho) <= 1.0:
                #     ortho = ortho * 255.0
                # ortho_uint8 = ortho.astype(np.uint8)
                # Image.fromarray(ortho_uint8).save(os.path.join(output_folder,("Wall_" + str(n.object_id) + "-Face_" + str(1) +'.png')))

        wall +=1

def merge_door_nodes(nodes: List[PointCloudNode],threshold_clustering_distance: float = 0.5,threshold_clustering_normals: float = 0.9) -> List[PointCloudNode]:
    """
    Merges wall nodes based on spatial and normal vector proximity.
 
    Parameters:
        nodes (List[Dict]): A list of wall node dictionaries. Each dictionary should have keys like 'resource' (o3d.geometry.PointCloud) and 'plane' (numpy array of plane coefficients).
        threshold_clustering_distance (float): Distance threshold for considering two nodes as neighbors.
        threshold_clustering_normals (float): Cosine of the angle threshold for considering normals as similar.
 
    Returns:
        List[Dict]: A list of merged wall nodes.
    """
    #next, we will cluster the planes based on distance and normal similarity (starting from the largest plane)
   
    clustered_nodes=[]
   
    # #sort the nodes from largest to smallest                                                    
    sorted_list = sorted(nodes, key=lambda x: np.asarray(x.resource.get_volume()),reverse=True)
   
    while len(sorted_list) not in [0,1]:
        n=sorted_list[0]
        sorted_list.pop(0)
       
        # retrieve neirest neighbors of same class_id
        inliers=np.asarray(n.resource.vertices)
        joined_pcd,identityArray=gmu.create_identity_point_cloud([p.resource for p in sorted_list])       #this is a little bit silly
        indices,distances=gmu.compute_nearest_neighbors(inliers,np.asarray(joined_pcd.points),n=1)
        indices=indices[:,0]
        distances=distances[:,0]
 
       
        #filter on normal similarity        
        candidate_normals = np.asarray([p.normal for p in sorted_list])

        normals=np.array([p for p in candidate_normals[identityArray[indices].astype(int)]]) 
        dotproducts = np.abs(np.einsum('ij,...j->...i', np.array([n.normal]), normals))[:,0]    
        indices=indices[(dotproducts>threshold_clustering_normals)]
        inliers=inliers[(dotproducts>threshold_clustering_normals)]
        distances=distances[(dotproducts>threshold_clustering_normals)]
        #and check again
        if len(indices)==0:
            clustered_nodes.append(n)
            continue
     
        #filter on distance  
        # orthogonal_distance = np.abs(np.einsum('ij,...j->...i', np.array([n.plane[:3]]), np.asarray(inliers)-np.asarray(joined_pcd.points)[indices] ))[:,0]    
        # indices=indices[((orthogonal_distance<threshold_clustering_orthogonal_distance) & (distances<threshold_clustering_coplanar_distance)) |
        #                 (distances<threshold_clustering_distance)]      
        indices=indices[(distances<threshold_clustering_distance)]  
        #check if there are any indices left
        if len(indices)==0:
            clustered_nodes.append(n)
            continue
       
   
        #merge the point clouds
        unique=np.unique(identityArray[indices])            
        pcd=gmu.join_geometries([n.resource]+[p.resource for i,p in enumerate(nodes) if i in unique])    
        # n.resource=pcd
        # n.height = np.max([n.height]+[p.height for i,p in enumerate(nodes) if i in unique])
        # n.doorWidth = np.average([n.doorWidth]+[p.doorWidth for i,p in enumerate(nodes) if i in unique])
        # n.startpoint = np.mean([n.startpoint] + [nodes[i].startpoint for i in unique], axis=0)
        # n.endpoint = np.mean([n.endpoint] + [nodes[i].endpoint for i in unique], axis=0)
        # n.axis=o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(np.vstack([n.startpoint, n.endpoint])),lines=o3d.utility.Vector2iVector([[0,1]])).paint_uniform_color([0,0,1])
        #delete elements from sorted_objectPcdNodes if they are in unique
        indices_to_remove = sorted(set(unique), reverse=True)
        [sorted_list.pop(idx) for idx in indices_to_remove]
       
        #reinsert the current cluster
        sorted_list.insert(0,n)
   
    #add remaining elements
    if len(sorted_list)!=0:
        for n in sorted_list:
            clustered_nodes.append(n)
   
    return clustered_nodes

def graph_to_wallnodes(f_pcd, f_rdf):
    wallNodes=tl.graph_path_to_nodes(f_rdf)
    for n in wallNodes:
        n.resource=o3d.geometry.TriangleMesh.create_from_oriented_bounding_box(o3d.geometry.OrientedBoundingBox.create_from_points(o3d.utility.Vector3dVector(n.orientedBounds)))
        n.pcd = None
    print(f'{len(wallNodes)} wallNodes detected!')
    
    # wallNodes = t8.match_graph_with_las(f_pcd,class_dict, nodes = wallNodes, getResources=True, getNormals=False)
 
    las = laspy.read(f_pcd) #if 'las' not in globals() else las
    pcd=gmu.las_to_pcd(las) #if 'pcd' not in globals() else pcd # this is the slowest step

    #create an identity point cloud of all the objectNodes
    identityPcd,identityArray=gmu.create_identity_point_cloud([n.resource for n in wallNodes],resolution=0.05)

    #remove all far away points
    dists = pcd.compute_point_cloud_distance(identityPcd)
    dists = np.asarray(dists)
    ind = np.where(dists <0.8)[0]
    inlier_pcd = pcd.select_by_index(ind)    

    #compute nearest neighbors
    indices,distances=gmu.compute_nearest_neighbors(np.asarray(inlier_pcd.points),np.asarray(identityPcd.points))    
    indices=indices.flatten()

    #compute the object and class scalars based on threshold distance
    labels = identityArray[indices[np.where(distances <= 0.8)[0]]]
    for u in np.unique( labels):
        object_pcd=inlier_pcd.select_by_index(np.where(labels==u)[0])
        wallNodes[u].pcd=object_pcd
        
    pcd_count = 0
    for n in wallNodes:
        if not n.pcd == None:
            pcd_count += 1
    print(f'Matched {pcd_count} pointclouds to nodes')

    for n in wallNodes: 
        n.derivedFrom = next((p for p in wallNodes if n.object_id == p.object_id), None)
        n.startpoint = ut.literal_to_array(n.start_pt) #np.asarray(n.start_pt[1:-1].split(), dtype=float)
        n.startpoint[2] = 0
        n.endpoint = ut.literal_to_array(n.end_pt) #np.asarray(n.end_pt[1:-1].split(), dtype=float)
        n.endpoint[2] = 0
        n.normal = ut.literal_to_array(n.normal) #np.asarray(n.normal[1:-1].split(), dtype=float)
        n.height = float(n.height)
        n.name = n.subject.split('///')[-1]
        if n.width == 0.127 or n.width == 0.2:
            n.singleFaced = True
        else:
            n.singleFaced = False
    
    return wallNodes

def object_detection(wallNodes, model, TEXT_PROMPT, BOX_TRESHOLD = 0.2, TEXT_TRESHOLD = 0.2):
    
    for n in wallNodes:
        n.boxes = []
        n.logits = []
        n.phrases = []
            
        if len(n.orthos) > 0:
            for ortho in n.orthos:
                
                boxes = None
                image = load_image(Image.fromarray((ortho * 255).astype(np.uint8)))

                boxes, logits, phrases = predict(
                    model=model, 
                    image=image, 
                    caption=TEXT_PROMPT, 
                    box_threshold=BOX_TRESHOLD, 
                    text_threshold=TEXT_TRESHOLD
                )
                n.boxes.append(boxes)
                n.logits.append(logits)
                n.phrases.append(phrases)
                
    return wallNodes

def process_detections(wallNodes, t_score = 0.5, image_resolution = 0.01):
    door_count = 0
    doorNodes = []
    
    for n in wallNodes:
        if (len(n.orthos) == 1 and len(n.boxes[0]) > 0) or len(n.orthos) == 2 and (len(n.boxes[0]) > 0 or len(n.boxes[1]) > 0) :
            for j, boxes in enumerate(n.boxes):
                if len(boxes) > 0: 
                    for i, box in enumerate(boxes):
                        probability = float(n.logits[j][i])
                        opening_width = round(int(np.asarray(box)[2]*n.orthos[j].shape[1])* image_resolution, 2)    
                        opening_height = round(int(np.asarray(box)[3]*n.orthos[j].shape[0]) * image_resolution, 2)
                        
                        detection_center_u = int(np.asarray(box)[0]*n.orthos[j].shape[1]) * image_resolution
                        detection_center_v = int(np.asarray(box)[1]*n.orthos[j].shape[0]) * image_resolution
                        
                        reference_level = round((n.orthos[j].shape[0]*image_resolution) - (detection_center_v + opening_height/2), 2)

                        score = is_door(probability, opening_width, opening_height, reference_level)
                        
                        if score >= t_score:
                            # annotated_frame = annotate(image_source= n.orthos[j], boxes=box.unsqueeze(0), logits=torch.from_numpy(np.array([score])), phrases=[n.phrases[j][i]])
                            # annotated_frame = annotated_frame[...,::-1] # BGR to RGB
                            # Image.fromarray(annotated_frame).save(os.path.join(output_folder,("Wall_" + str(n.object_id) + "-Face_" + str(j) +"-"+ str(score) +'-DOOR.png')))
                            
                            box1 = copy.deepcopy(box)
                            box = box.unsqueeze(0)

                            if j == 1:
                                box1[0] = 1-box1[0]
                            
                            boundaryPoints = line_with_width_coordinates(n.startpoint, n.endpoint, np.asarray(box1)[0], opening_width, 0)
                                
                                
                            doornode = BIMNode(
                                
                                name= "Doors_" + str(door_count),
                                axis=o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(boundaryPoints),lines=o3d.utility.Vector2iVector([[0,1]])).paint_uniform_color([0,0,1]),
                                startpoint= boundaryPoints[0],
                                endpoint= boundaryPoints[1],
                                doorWidth = np.round(np.linalg.norm(boundaryPoints[0] - boundaryPoints[1]),2),
                                height = opening_height,
                                doornessScore = score,
                                singleFaced = True,
                                depth = n.width,
                                object_id = 4000 + door_count,
                                reference_level = reference_level,
                                host = n,
                                probability = probability)
                            doorNodes.append(doornode)
                            door_count += 1
    return doorNodes


def create_geometry(doorNodes):
    if len(doorNodes) >= 1:
        for n in doorNodes:
            pointList=[]
            points=np.asarray(n.axis.points)
            pointList.extend(points+n.host.normal*n.host.width/2)
            pointList.extend(points-n.host.normal*n.host.width/2)
            pointList.extend(np.array(pointList)+np.array([0,0,n.height]))
            
            pcd=o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(pointList))
            box=pcd.get_oriented_bounding_box()
            
            n.resource = o3d.geometry.TriangleMesh.create_from_oriented_bounding_box(box)
            n.resource.paint_uniform_color(ut.literal_to_array(n.host.color))
            n.doorBox=o3d.geometry.LineSet.create_from_oriented_bounding_box(box)
            n.doorBox.paint_uniform_color([0,0,1])
            n.center = compute_center(n.startpoint, n.endpoint)
            n.cartesianTransform = copy.deepcopy(n.host.cartesianTransform)
            n.cartesianTransform[:3,3] = n.center
            n.normal = n.host.normal
            n.rotation = get_angle_with_x_axis(n.startpoint, n.endpoint)

def filter_door_nodes(doorNodes, min_width = 0.5, max_width = 1.2, min_height = 1.5, max_height = 2.5, max_distance_to_ground = 0.5):
    new_nodes = []
    for n in doorNodes:
        if  min_width > n.doorWidth or n.doorWidth > max_width:
            continue
        if min_height > n.height or n.height > max_height:
            continue
        if n.reference_level > max_distance_to_ground:
            continue
        new_nodes.append(n)
    return new_nodes

def export_door_nodes(doorNodes, output_folder, filename):
    write_obj_with_submeshes(os.path.join(output_folder,f'{filename}_doors.obj') , [n.resource for n in doorNodes], [n.name for n in doorNodes])

    reform_name='_'.join(filename.split('_')[:4])+'_doors'

    json_data=doors_to_json(doorNodes)
    with open(os.path.join(output_folder,reform_name+'.json'), 'w') as file:
        json.dump(json_data, file, indent=4)
    print("JSON data written to file:", os.path.join(output_folder,reform_name+'.json'))
