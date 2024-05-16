
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
from  geomapi.nodes import *
import geomapi.tools as tl
import geomapi.tools.progresstools as pt
from typing import Dict, Any, Tuple,List


def compute_base_constraint(wallNodes:List[MeshNode],levelNodes:List[MeshNode],threshold_level_height:float)->None:
    for n in wallNodes:
        #compute minheight of the resource at 0.1% of the height (absolute minimum might be wrong)
        z_values = np.sort(np.asarray(n.resource.points)[:,2])
        minheight = np.percentile(z_values, 0.1)

        #compute base constraint. select the intersecting level that is closest to the bottom of the wall. Else, just take first levelNode.
        nearby_ref_levels= tl.select_nodes_with_intersecting_bounding_box(n,levelNodes)
        n.base_constraint= next((n for n in nearby_ref_levels if np.absolute(n.height-minheight)<threshold_level_height),levelNodes[0])  if nearby_ref_levels else levelNodes[0] #this is a link!
        
        #compute base offset
        n.base_offset=minheight-n.base_constraint.height
        print(f'name: {n.name}, base_constraint: {n.base_constraint.name}, base_offset: {n.base_offset}')
        
def compute_top_constraint(wallNodes:List[MeshNode],levelNodes:List[MeshNode],threshold_level_height:float)->None:        
    for n in wallNodes:
        #compute maxheight of the resource at 0.1% of the height (absolute minimum might be wrong)
        z_values = np.sort(np.asarray(n.resource.points)[:,2])
        minheight = np.percentile(z_values, 0.1)
        maxheight = np.percentile(z_values, 99.9)

        #compute base constraint. select the intersecting level that is closest to the top of the wall. Else, just take last levelNode.
        nearby_ref_levels= tl.select_nodes_with_intersecting_bounding_box(n,levelNodes)
        n.top_constraint= next((n for n in nearby_ref_levels if np.absolute(n.height-maxheight)<threshold_level_height),levelNodes[-1]) if nearby_ref_levels else levelNodes[-1] #this is a link!
        
        #compute base offset
        n.top_offset=maxheight-n.top_constraint.height

        #compute wall height
        n.height=maxheight-minheight
        print(f'name: {n.name}, top_constraint: {n.top_constraint.name}, top_offset: {n.top_offset}')

def compute_wall_orientation(wallNodes:List[MeshNode],referenceNodes:List[MeshNode],t_thickness:float=0.12,t_distance:float=0.7,t_inliers:float=0.5)->None:
    for n in wallNodes:    
        #Compute the dominant plane on the point cloud
        n.plane_model, inliers = n.resource.segment_plane(distance_threshold=0.03,
                                                ransac_n=3,
                                                num_iterations=1000)
        
        #get center of the face and postion it on the correct height (base constraint + base offset)   
        n.faceCenter=n.resource.select_by_index(inliers).get_center()  
        n.faceCenter[2]=n.base_constraint.height + n.base_offset

        #compute the normal of the plane in 2D (third component should be zero, normal should point outwards of the wall)
        n.normal=n.plane_model[:3]
        n.normal[2]=0
        n.normal/=np.linalg.norm(n.normal)

        #compute the sign by evaluating the dot product between the normal and the vector between the center of the box and the center of the face
        boxCenter=n.orientedBoundingBox.get_center()
        boxCenter[2]=n.base_constraint.height + n.base_offset
        n.sign=np.sign(np.dot(n.normal,n.faceCenter-boxCenter)) # this should be negative!
        n.flipped=False
        
        #check if there is an opposing plane as well, with sufficient inliers
        #if not, take a look at the ceiling and floor nodes to see on which side they are, and use them to spawn the wall away from these nodes
        outlier_pcd=n.resource.select_by_index(inliers,invert=True)
        
        if np.asarray(outlier_pcd.points).shape[0]>t_inliers*len(inliers):
            #compute second dominant plane on the point cloud
            _, second_inliers = outlier_pcd.segment_plane(distance_threshold=0.03,
                                                    ransac_n=3,
                                                    num_iterations=1000)

            if (len(second_inliers)<t_inliers*len(inliers)):# or (n.orientedBoundingBox.extent[2]<=0.10):
                
                #create reference pcd from ceiling and floors
                referencePcd,_=gmu.create_identity_point_cloud([n.resource for n in referenceNodes if n.resource is not None])
                #find nearest point near the top and the bottom 
                topPoint=copy.deepcopy(n.faceCenter) 
                topPoint[2]=n.base_constraint.height + n.base_offset+n.height
                bottomPoint=n.faceCenter
                #compute distance to the ceiling and floor points
                idx,_=gmu.compute_nearest_neighbors(np.asarray([topPoint,bottomPoint]),np.asarray(referencePcd.points)) 
                points=np.asarray(referencePcd.points)[idx[:,0]]
                #compute orthogonal distance to the plane and select node with lowest distance
                idx=idx[np.argmin(np.absolute(np.einsum('i,ji->j',n.normal,points-n.faceCenter))) ][0] 
                point=np.asarray(referencePcd.points)[idx]
                point[2]=n.base_constraint.height + n.base_offset
                n.sign=np.sign(np.dot(n.normal,point-n.faceCenter))
                n.flipped=True
        else: # this is in case 1 exact plane is found with no outliers
            topPoint=copy.deepcopy(n.faceCenter) 
            topPoint[2]=n.base_constraint.height + n.base_offset+n.height
            bottomPoint=n.faceCenter
            #compute distance to the ceiling and floor points
            referencePcd,_=gmu.create_identity_point_cloud([n.resource for n in referenceNodes if n.resource is not None])
            idx,_=gmu.compute_nearest_neighbors(np.asarray([topPoint,bottomPoint]),np.asarray(referencePcd.points)) 
            points=np.asarray(referencePcd.points)[idx[:,0]]
            #compute orthogonal distance to the plane and select node with lowest distance
            idx=idx[np.argmin(np.absolute(np.einsum('i,ji->j',n.normal,points-n.faceCenter))) ][0] 
            point=np.asarray(referencePcd.points)[idx]
            point[2]=n.base_constraint.height + n.base_offset
            n.sign=np.sign(np.dot(n.normal,point-n.faceCenter))
            n.flipped=True
        
        #flip the normal if it points inwards
        n.normal*=-1 if n.sign==-1 else 1

        print(f'name: {n.name}, plane: {n.plane_model}, inliers: {len(inliers)}/{len(np.asarray(n.resource.points))}')      

def compute_wall_thickness(wallNodes:List[MeshNode],t_thickness:float=0.12,t_distance:float=0.7)->None:
    for n in wallNodes:
        #compute the normals of the wall
        pcd_tree = o3d.geometry.KDTreeFlann(n.resource)
        n.resource.estimate_normals() if not n.resource.has_normals() else None

        #for every 100th point in P, that has the same normal as the dominant plane, select nearest points in P that meet a distance threshold    
        points=np.asarray(n.resource.points)[::100]
        normals=np.asarray(n.resource.normals)[::100]
        idx=np.where(np.absolute(np.einsum('i,ji->j',n.normal,normals))>0.9)
        points=points[idx]
        normals=normals[idx]
        distances=[]

        for p,q in zip(points,normals):
            #compute distances
            [k, idx, _] = pcd_tree.search_radius_vector_3d(p, t_distance)        
            #retain only the distances for which the normal is within 0.7 radians of the normal of the point
            kNormals=np.asarray(n.resource.normals)[idx]
            indices=np.asarray(idx)[np.where(np.absolute(np.einsum('i,ji->j',q,kNormals))>0.9)]
            #compute the dotproduct between vectors (p-q) and the normals of the q in the radius
            vectors=p-np.asarray(n.resource.select_by_index(indices).points)            
            #extend distances with all distances larger than t_thickness
            distances.extend([d for d in np.absolute(np.einsum('i,ji->j', q, vectors)) if d > 0.9*t_thickness])

        #keep most frequent distance with bins of 1cm
        if len(distances)>0:
            d=np.array(distances)
            bin_width = 0.01
            bins = np.arange(0, np.max(d) + bin_width, bin_width)
            hist, bin_edges = np.histogram(d, bins=bins)
            max_bin_index = np.argmax(hist)
            distance = (bin_edges[max_bin_index] + bin_edges[max_bin_index + 1]) / 2
        else:
            distance = t_thickness

        #set distance to t_thickness if distance is smaller than t_thickness
        n.wallThickness=distance if distance >= t_thickness else t_thickness

        print(f'name: {n.name}, BB_extent: {n.orientedBoundingBox.extent}, wallThickness: {n.wallThickness}')


def compute_wall_axis(wallNodes:List[MeshNode])->None:
    for n in wallNodes:     
    
        #offset the center of the plane with half the wall thickness in the direction of the normal of the plane  
        wallCenter=n.faceCenter-n.normal*n.wallThickness/2 

        wallCenter[2]=n.base_constraint.height + n.base_offset

        #project axis aligned bounding points on the plane
        box=n.resource.get_axis_aligned_bounding_box()    
        points=np.asarray(box.get_box_points())
        points[:,2]=n.base_constraint.height + n.base_offset

        #translate the points to the plane
        vectors=points-wallCenter
        translation=np.einsum('ij,j->i',vectors,n.normal)
        points=points - translation[:, np.newaxis] * n.normal

        # Calculate the pairwise distances between all boundary points
        distances = np.linalg.norm(points[:, np.newaxis] - points, axis=2)

        # Get the indices of the two points with the maximum distance
        max_indices = np.unravel_index(np.argmax(distances), distances.shape)

        # Retain only the two points with the maximum distance
        n.boundaryPoints = points[max_indices,:]

        #create the axis
        n.axis=o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(n.boundaryPoints),lines=o3d.utility.Vector2iVector([[0,1]])).paint_uniform_color([0,0,1])
        n.startPoint=n.boundaryPoints[0]
        n.endPoint=n.boundaryPoints[1]
        # Calculate the length
        n.wallLength = np.linalg.norm(n.boundaryPoints[0] - n.boundaryPoints[1])

        print(f'name: {n.name}, wallLength: {n.wallLength}')

def compute_wall_geometry(wallNodes:List[MeshNode])->None:
    for n in wallNodes:
        pointList=[]
        points=np.asarray(n.axis.points)
        pointList.extend(points+n.normal*n.wallThickness/2)
        pointList.extend(points-n.normal*n.wallThickness/2)

        pointList.extend(np.array(pointList)+np.array([0,0,n.height]))
        pcd=o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(pointList))

        box=pcd.get_oriented_bounding_box()
        n.wall=o3d.geometry.TriangleMesh.create_from_oriented_bounding_box(box)
        n.wall.paint_uniform_color(ut.literal_to_array(n.color))
        n.wallBox=o3d.geometry.LineSet.create_from_oriented_bounding_box(box)
        n.wallBox.paint_uniform_color([0,0,1])

        print(f'name: {n.name}, wall: {n.wall}')


def walls_to_json(wallNodes: list[MeshNode], file_name: str) -> Dict:
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
    
    # Prepare JSON data structure
    json_data = {
        "filename": f"{ut.get_filename(file_name)}_walls.obj",
        "objects": []
    }
    
    # Fill JSON with node data
    for n in wallNodes:
        if not hasattr(n, 'base_constraint') or not hasattr(n, 'base_offset') or not hasattr(n, 'top_constraint') or not hasattr(n, 'top_offset') or not hasattr(n, 'height') or not hasattr(n, 'wallThickness') or not hasattr(n, 'wallLength') or not hasattr(n, 'normal') or not hasattr(n, 'boundaryPoints') :
            raise ValueError("Node is missing required attributes (base_constraint, base_offset, top_constraint,top_offset,height,wallThickness,wallLength,normal, boundaryPoints ).")
        
        try:
            #fill json
            obj = {
                    "name": n.name,
                    "id": n.object_id,
                    "base_constraint":n.base_constraint.name,
                    "base_offset":n.base_offset,
                    "top_constraint":n.top_constraint.name,
                    "top_offset":n.top_offset,
                    "height": n.height,
                    "width": n.wallThickness,
                    "wallLength": n.wallLength,
                    "normal": list(n.normal),
                    "start_pt": list(n.boundaryPoints[0]),
                    "end_pt": list(n.boundaryPoints[1]),
                    "neighbor_wall_ids_at_start": [],
                    "neighbor_wall_ids_at_end": []
                    }
            json_data["objects"].append(obj)
        except Exception as e:
            raise ValueError(f"Error processing node {n.name}: {str(e)}")
    
    # Convert the Python dictionary to a JSON string
    return json_data
