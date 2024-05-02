

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

@timer
def create_top_view_point_cloud_image(las, pixel_resolution=0.02)->(np.ndarray, float, float, float ):
    """
    Create a top view image of a LiDAR point cloud from a LAS file.
    Args:
        las: A laspy.LasData object containing the point cloud data.
        pixel_resolution: The pixel resolution in meters.
    Returns:
        A top view image of the point cloud.
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
    # offset_y = max_y
    

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

def slice_point_cloud(pcd, z_min, z_max)->np.ndarray:
    """
    Slice a point cloud based on the z-axis.
    Args:
        pcd: A numpy array containing the point cloud data.
        z_min: The minimum z-coordinate.
        z_max: The maximum z-coordinate.
    Returns:
        A numpy array containing the sliced point cloud data.
    """
    points=np.asarray(pcd.points)
    
    #retain only the points that have z value between min_z and max_z
    mask=(points[:,2]>z_min) & (points[:,2]<z_max)
    points=points[mask]
    colors=np.asarray(pcd.colors)[mask]
    pcd_slice=o3d.geometry.PointCloud()
    pcd_slice.points=o3d.utility.Vector3dVector(points)
    pcd_slice.colors=o3d.utility.Vector3dVector(colors)
    
    return pcd_slice
