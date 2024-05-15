

import time
import os
import laspy
import geomapi
import geomapi.utils.geometryutils as gmu
import geomapi.utils as ut
import numpy as np
import torch
import cv2
from utils import timer
import open3d as o3d
import copy
from typing import Tuple

@timer
def create_top_view_point_cloud_image(las: laspy.LasData, pixel_resolution: float = 0.02) -> Tuple[np.ndarray, float, float, float]:
    """
    Create a top view image of a LiDAR point cloud from a LAS file.
    
    Args:
        las (laspy.LasData): A LasData object containing the point cloud data.
        pixel_resolution (float): The pixel resolution in meters.

    Returns:
        Tuple[np.ndarray, float, float, float]: A tuple containing the top view image of the point cloud,
                                               and the minimum x, y coordinates and pixel resolution.
    """
    # Get the coordinates and RGB data
    points = np.vstack((las.x, las.y)).transpose()
    colors = np.vstack((las.red, las.green, las.blue)).transpose()

    # Normalize the coordinates to fit in an image
    min_x, min_y = np.min(points, axis=0)
    max_x, max_y = np.max(points, axis=0)

    # Calculate the required image size based on the extent and pixel resolution
    image_width = int((max_x - min_x) / pixel_resolution) + 1
    image_height = int((max_y - min_y) / pixel_resolution) + 1

    # Create an image with a white background
    image = np.full((image_height, image_width, 3), 255, dtype=np.uint8)

    # Calculate the offsets for mapping points to pixels
    offset_x = min_x
    offset_y = min_y  

    # Map the points to pixel coordinates
    scaled_points = ((points - [offset_x, offset_y]) / pixel_resolution).astype(int)

    # Normalize RGB values (assuming 16-bit per channel, common in LAS files)
    colors = (colors / 65535 * 255).astype(np.uint8)

    # Draw each point on the image using its RGB color
    # for (x, y), (r, g, b) in zip(scaled_points, colors):
    #     if 0 <= x < image_width and 0 <= y < image_height:
    #         cv2.circle(image, (x, y), 1, (int(b), int(g), int(r)), -1)  # OpenCV uses BGR format
            
    # Check and adjust the points within the image bounds
    valid_indices = (0 <= scaled_points[:, 0]) & (scaled_points[:, 0] < image_width) & \
                    (0 <= scaled_points[:, 1]) & (scaled_points[:, 1] < image_height)
    scaled_points = scaled_points[valid_indices]
    colors = colors[valid_indices]

    # Adjust y-coordinates for image's coordinate system (flipping y-axis)
    scaled_points[:, 1] = image_height - scaled_points[:, 1] - 1

    # Draw each point on the image using its RGB color
    image[scaled_points[:, 1], scaled_points[:, 0]] = colors[:, [2, 1, 0]]  # Convert RGB to BGR for OpenCV
          
    return image, offset_x, offset_y, pixel_resolution

def slice_point_cloud(pcd: o3d.geometry.PointCloud, z_min: float, z_max: float) -> o3d.geometry.PointCloud:
    """
    Slice a point cloud based on the z-axis range.

    Args:
        pcd (o3d.geometry.PointCloud): An Open3D PointCloud object.
        z_min (float): The minimum z-coordinate for slicing.
        z_max (float): The maximum z-coordinate for slicing.

    Returns:
        o3d.geometry.PointCloud: A sliced Open3D PointCloud object within the specified z-axis range.
    """
    # Convert Open3D PointCloud to numpy array for slicing
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors) if pcd.colors else None

    # Create a mask for points within the specified z-range
    mask = (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
    sliced_points = points[mask]

    # Create a new PointCloud with sliced points and corresponding colors
    sliced_pcd = o3d.geometry.PointCloud()
    sliced_pcd.points = o3d.utility.Vector3dVector(sliced_points)
    if colors is not None:
        sliced_colors = colors[mask]
        sliced_pcd.colors = o3d.utility.Vector3dVector(sliced_colors)

    return sliced_pcd
