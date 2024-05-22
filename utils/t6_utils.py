
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
from sklearn.neighbors import NearestNeighbors # to compute nearest neighbors


def compute_base_constraint(wallNodes:List[MeshNode],levelNodes:List[MeshNode],threshold_level_height:float)->None:
    for n in wallNodes:
        #compute minheight of the resource at 0.1% of the height (absolute minimum might be wrong)
        z_values = np.sort(np.asarray(n.resource.points)[:,2])
        minheight = np.percentile(z_values, 0.1)

        #compute base constraint. select the intersecting level that is closest to the bottom of the wall. Else, take the closest level to the minheight
        nearby_ref_levels= tl.select_nodes_with_intersecting_bounding_box(n,levelNodes)
        if nearby_ref_levels:
            n.base_constraint=next((n for n in nearby_ref_levels if np.absolute(n.height-minheight)<threshold_level_height),nearby_ref_levels[np.argmin([np.abs(n.height-minheight) for n in nearby_ref_levels ])] ) 
        else:  
            n.base_constraint=next((n for n in levelNodes if np.absolute(n.height-minheight)<threshold_level_height),levelNodes[np.argmin([np.abs(n.height-minheight) for n in levelNodes ])] ) 
            
            
        #compute base offset
        n.base_offset=minheight-n.base_constraint.height
        print(f'name: {n.name}, base_constraint: {n.base_constraint.name}, base_offset: {n.base_offset}')
        
def compute_top_constraint(wallNodes:List[MeshNode],levelNodes:List[MeshNode],threshold_level_height:float)->None:        
    for n in wallNodes:
        #compute maxheight of the resource at 0.1% of the height (absolute minimum might be wrong)
        z_values = np.sort(np.asarray(n.resource.points)[:,2])
        minheight = np.percentile(z_values, 0.1)
        maxheight = np.percentile(z_values, 99.9)

        #compute top constraint. select the intersecting level that is closest to the top of the wall. Else, just take closest levelNode.
        nearby_ref_levels= tl.select_nodes_with_intersecting_bounding_box(n,levelNodes)
        if nearby_ref_levels:
            n.top_constraint=next((n for n in nearby_ref_levels if np.absolute(n.height-maxheight)<threshold_level_height),nearby_ref_levels[np.argmin([np.abs(n.height-maxheight) for n in nearby_ref_levels ])] ) 
        else:  
            n.top_constraint=next((n for n in levelNodes if np.absolute(n.height-maxheight)<threshold_level_height),levelNodes[np.argmin([np.abs(n.height-maxheight) for n in levelNodes ])] ) 
        
        #check if the top constraint is the same as the base constraint, if so, take the next level
        n.top_constraint=n.top_constraint if n.top_constraint!= n.base_constraint else next((e for e in levelNodes if e.height>n.top_constraint.height),levelNodes[-1])
            
        #compute top offset
        n.top_offset=maxheight-n.top_constraint.height

        #compute wall height
        # n.height=maxheight-minheight #CVPR RULES
        n.top=n.top_constraint.height + n.top_offset if np.abs(n.top_offset)>1 else n.top_constraint.height
        n.bottom=n.base_constraint.height + n.base_offset if np.abs(n.base_offset)>1 else n.base_constraint.height
        n.height=n.top-n.bottom
        
        #CVPR DOESNT DO OFFSETS
        n.height=n.top_constraint.height-n.base_constraint.height if n.top_constraint!=n.base_constraint else n.height
        
        print(f'name: {n.name}, top_constraint: {n.top_constraint.name}, top_offset: {n.top_offset}')

def compute_wall_orientation(wallNodes:List[MeshNode],referenceNodes:List[MeshNode],t_thickness:float=0.12,t_distance:float=0.7,t_inliers:float=0.5)->None:
    for n in wallNodes:    
        #Compute the dominant plane on the point cloud
        n.plane_model, n.inliers = n.resource.segment_plane(distance_threshold=0.03,
                                                ransac_n=3,
                                                num_iterations=1000)
        
        #get center of the face and postion it on the correct height (base constraint + base offset)   
        n.faceCenter=n.resource.select_by_index(n.inliers).get_center()  
        n.faceCenter[2]=n.base_constraint.height#CVPR RULES + n.base_offset

        #compute the normal of the plane in 2D (third component should be zero, normal should point outwards of the wall)
        n.normal=n.plane_model[:3]
        n.normal[2]=0
        n.normal/=np.linalg.norm(n.normal)

        #compute the sign by evaluating the dot product between the normal and the vector between the center of the box and the center of the face
        boxCenter=n.orientedBoundingBox.get_center()
        boxCenter[2]=n.base_constraint.height#CVPR RULES + n.base_offset
        n.sign=np.sign(np.dot(n.normal,n.faceCenter-boxCenter)) # this should be negative!
        n.flipped=False
        
        #check if there is an opposing plane as well, with sufficient inliers
        #if not, take a look at the ceiling and floor nodes to see on which side they are, and use them to spawn the wall away from these nodes
        outlier_pcd=n.resource.select_by_index(n.inliers,invert=True)
        
        
        
        if np.asarray(outlier_pcd.points).shape[0]>t_inliers*len(n.inliers):
            #compute second dominant plane on the point cloud
            _, second_inliers = outlier_pcd.segment_plane(distance_threshold=0.03,
                                                    ransac_n=3,
                                                    num_iterations=1000)
            
            if (len(second_inliers)<t_inliers*len(n.inliers)):# or (n.orientedBoundingBox.extent[2]<=0.10):
                
                #create reference pcd from ceiling and floors
                referencePcd,_=gmu.create_identity_point_cloud([n.resource for n in referenceNodes if n.resource is not None])
                #find nearest point near the top and the bottom 
                topPoint=copy.deepcopy(n.faceCenter) 
                topPoint[2]=n.base_constraint.height#CVPR RULES + n.base_offset+n.height
                bottomPoint=n.faceCenter
                #compute distance to the ceiling and floor points
                idx,_=gmu.compute_nearest_neighbors(np.asarray([topPoint,bottomPoint]),np.asarray(referencePcd.points)) 
                points=np.asarray(referencePcd.points)[idx[:,0]]
                #compute orthogonal distance to the plane and select node with lowest distance
                idx=idx[np.argmin(np.absolute(np.einsum('i,ji->j',n.normal,points-n.faceCenter))) ][0] 
                point=np.asarray(referencePcd.points)[idx]
                point[2]=n.base_constraint.height #CVPR RULES+ n.base_offset
                n.sign=np.sign(np.dot(n.normal,point-n.faceCenter))
                n.flipped=True
            
                
        else: # this is in case 1 exact plane is found with no outliers
            topPoint=copy.deepcopy(n.faceCenter) 
            topPoint[2]=n.base_constraint.height#CVPR RULES + n.base_offset+n.height
            bottomPoint=n.faceCenter
            #compute distance to the ceiling and floor points
            referencePcd,_=gmu.create_identity_point_cloud([n.resource for n in referenceNodes if n.resource is not None])
            idx,_=gmu.compute_nearest_neighbors(np.asarray([topPoint,bottomPoint]),np.asarray(referencePcd.points)) 
            points=np.asarray(referencePcd.points)[idx[:,0]]
            #compute orthogonal distance to the plane and select node with lowest distance
            idx=idx[np.argmin(np.absolute(np.einsum('i,ji->j',n.normal,points-n.faceCenter))) ][0] 
            point=np.asarray(referencePcd.points)[idx]
            point[2]=n.base_constraint.height #CVPR RULES+ n.base_offset
            n.sign=np.sign(np.dot(n.normal,point-n.faceCenter))
            n.flipped=True
            
            
        
        #flip the normal if it points inwards
        n.normal*=-1 if n.sign==-1 else 1

        print(f'name: {n.name}, plane: {n.plane_model}, inliers: {len(n.inliers)}/{len(np.asarray(n.resource.points))}')      

def compute_wall_thickness(wallNodes:List[MeshNode],t_thickness:float=0.12,t_distance:float=0.7)->None:
    for n in wallNodes:
        distance=0

        #filter outlier_pcd so that their normals are within 0.9 radians of the normal of the plane
        outlier_pcd=n.resource.select_by_index(n.inliers,invert=True)
        normals=np.asarray(outlier_pcd.normals)
        idx=np.where(np.absolute(np.einsum('i,ji->j',n.normal,normals))>0.9)[0]
        outlier_pcd=outlier_pcd.select_by_index(idx)
        #remove all the points closer than 0.08m
        distances=np.asarray(outlier_pcd.compute_point_cloud_distance(n.resource.select_by_index(n.inliers)))
        idx=np.where(distances>0.08)[0]
        outlier_pcd=outlier_pcd.select_by_index(idx)
        
        if np.asarray(outlier_pcd.points).shape[0]>0.1*len(n.inliers) or np.asarray(outlier_pcd.points).shape[0]>500:
            #compute second dominant plane on the point cloud
            _, second_inliers = outlier_pcd.segment_plane(distance_threshold=0.03,
                                                    ransac_n=3,
                                                    num_iterations=1000)
     
            #get average of the outliers
            second_plane_center=np.mean(np.asarray(outlier_pcd.select_by_index(second_inliers).points),axis=0)
            second_plane_center[2]=n.base_constraint.height#CVPR RULES + n.base_offset
            #get average of the inliers
            first_plane_center=np.mean(np.asarray(n.resource.select_by_index(n.inliers).points),axis=0)
            first_plane_center[2]=n.base_constraint.height #CVPR RULES+ n.base_offset              
            
            #compute the distance between the two planes along the normal of the first plane
            distance=np.absolute(np.dot(n.normal,second_plane_center-first_plane_center)) 
        
        #ALTERNATIVE METHOD
        # #compute the normals of the wall -> this method had 34/150 errors
        # pcd_tree = o3d.geometry.KDTreeFlann(n.resource)
        # n.resource.estimate_normals() if not n.resource.has_normals() else None

        # #for every 100th point in P, that has the same normal as the dominant plane, select nearest points in P that meet a distance threshold    
        # points=np.asarray(n.resource.points)[::100]
        # normals=np.asarray(n.resource.normals)[::100]
        # idx=np.where(np.absolute(np.einsum('i,ji->j',n.normal,normals))>0.9)
        # points=points[idx]
        # normals=normals[idx]
        # distances=[]

        # for p,q in zip(points,normals):
        #     #compute distances
        #     [k, idx, _] = pcd_tree.search_radius_vector_3d(p, t_distance)        
        #     #retain only the distances for which the normal is within 0.7 radians of the normal of the point
        #     kNormals=np.asarray(n.resource.normals)[idx]
        #     indices=np.asarray(idx)[np.where(np.absolute(np.einsum('i,ji->j',q,kNormals))>0.9)]
        #     #compute the dotproduct between vectors (p-q) and the normals of the q in the radius
        #     vectors=p-np.asarray(n.resource.select_by_index(indices).points)            
        #     #extend distances with all distances larger than t_thickness
        #     distances.extend([d for d in np.absolute(np.einsum('i,ji->j', q, vectors)) if d > 0.9*t_thickness])

        # #keep most frequent distance with bins of 1cm
        # if len(distances)>0:
        #     d=np.array(distances)
        #     bin_width = 0.01
        #     bins = np.arange(0, np.max(d) + bin_width, bin_width)
        #     hist, bin_edges = np.histogram(d, bins=bins)
        #     max_bin_index = np.argmax(hist)
        #     distance = (bin_edges[max_bin_index] + bin_edges[max_bin_index + 1]) / 2
        # else:
        #     distance = t_thickness

        #set distance to t_thickness if distance is smaller than t_thickness
        n.wallThickness=distance if distance >= t_thickness else t_thickness

        print(f'name: {n.name}, BB_extent: {n.orientedBoundingBox.extent}, wallThickness: {n.wallThickness}')

def compute_wall_axis(wallNodes:List[MeshNode])->None:
    for n in wallNodes:     
    
        #offset the center of the plane with half the wall thickness in the direction of the normal of the plane  
        wallCenter=n.faceCenter-n.normal*n.wallThickness/2 

        wallCenter[2]=n.base_constraint.height #CVPR n.bottom

        #project axis aligned bounding points on the plane
        box=n.resource.get_axis_aligned_bounding_box()    
        points=np.asarray(box.get_box_points())
        points[:,2]=n.base_constraint.height #CVPRn.bottom

        #translate the points to the plane
        vectors=points-wallCenter
        translation=np.einsum('ij,j->i',vectors,n.normal)
        points=points - translation[:, np.newaxis] * n.normal

        # Calculate the pairwise distances between all boundary points
        distances = np.linalg.norm(points[:, np.newaxis] - points, axis=2)

        # Get the indices of the two points with the maximum distance
        max_indices = np.unravel_index(np.argmax(distances), distances.shape)

        # Retain only the two points with the maximum distance
        boundaryPoints = points[max_indices,:]

        #create the axis
        n.axis=o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(boundaryPoints),lines=o3d.utility.Vector2iVector([[0,1]])).paint_uniform_color([0,0,1])
        # n.startPoint=n.boundaryPoints[0]
        # n.endPoint=n.boundaryPoints[1]
        # Calculate the length
        n.wallLength = np.linalg.norm(boundaryPoints[0] - boundaryPoints[1])

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
        if not hasattr(n, 'base_constraint') or not hasattr(n, 'base_offset') or not hasattr(n, 'top_constraint') or not hasattr(n, 'top_offset') or not hasattr(n, 'height') or not hasattr(n, 'wallThickness') or not hasattr(n, 'wallLength') or not hasattr(n, 'normal') or not hasattr(n, 'axis') :
            raise ValueError("Node is missing required attributes (base_constraint, base_offset, top_constraint,top_offset,height,wallThickness,wallLength,normal, boundaryPoints ).")
        
        try:
            #fill json
            obj = {
                    "name": n.name,
                    "id": n.object_id,
                    "class_name": n.class_name,
                    "class_id": n.class_id,
                    "base_constraint":n.base_constraint.name,
                    "base_offset":n.base_offset,
                    "top_constraint":n.top_constraint.name,
                    "top_offset":n.top_offset,
                    "height": n.height,
                    "width": n.wallThickness,
                    "wallLength": n.wallLength,
                    "normal": list(n.normal),
                    "start_pt": list(np.asarray(n.axis.points)[0]),
                    "end_pt": list(np.asarray(n.axis.points)[1]),
                    "neighbor_wall_ids_at_start": n.neighbor_wall_ids_at_start,
                    "neighbor_wall_ids_at_end": n.neighbor_wall_ids_at_end,
                    }
            json_data["objects"].append(obj)
        except Exception as e:
            raise ValueError(f"Error processing node {n.name}: {str(e)}")
    
    # Convert the Python dictionary to a JSON string
    return json_data


def compute_nearest_neighbors(query_points: np.ndarray, reference_points: np.ndarray, maxDist: float = 5, n: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """Compute nearest neighbors (indices and distances) from query to reference points.

    Args:
        query_points (np.ndarray): points to calculate the distance from.
        reference_points (np.ndarray): points used as a reference.
        n (int, optional): number of neighbors. Defaults to 1.
        maxDist (float, optional): max distance to search.

    Returns:
        Tuple[np.ndarray, np.ndarray]: indices, distances
    """
    nbrs = NearestNeighbors(n_neighbors=n, algorithm='kd_tree').fit(reference_points)
    distances, indices = nbrs.kneighbors(query_points)
    
    # Ensure arrays are at least 2D
    if indices.ndim == 1:
        indices = np.expand_dims(indices, axis=0)
    if distances.ndim == 1:
        distances = np.expand_dims(distances, axis=0)
        
    # Efficient filtering using NumPy
    mask = distances < maxDist
    filtered_indices = [idx[mask_row] for idx, mask_row in zip(indices, mask)]
    filtered_distances = [dist[mask_row] for dist, mask_row in zip(distances, mask)]
    
    # Convert lists back to arrays with dtype=object
    filtered_indices = np.array(filtered_indices, dtype=object)
    filtered_distances = np.array(filtered_distances, dtype=object)

    return filtered_indices, filtered_distances

def compute_potential_wall_connections(wallNodesBIM: List[Node], weight_intersection: float = 1, weight_orthogonal: float = 1, weight_direct: float = 1,
                                       t_intersection_extension: float = 0.25, t_ortho_extension: float = 0.25, t_direct_extension: float = 0.7) -> List:
    for node in wallNodesBIM:
        node.potential_connections = []
        counter = 0
        axis = node.axis
        start_pt = node.start_pt
        end_pt = node.end_pt
        normal = node.normal
        direction = end_pt - start_pt
        node.direction = direction / np.linalg.norm(direction)

        # Check and compute potential connections for the start point
        if 0 not in node.pts_used_in_intersections:
            # Create intersection line in the direction of the axis for the start point
            intersection_extension = 0.25 * node.wallLength if 0.25 * node.wallLength < t_intersection_extension else t_intersection_extension
            node.start_intersection_line = o3d.geometry.LineSet(
                o3d.utility.Vector3dVector([start_pt - node.direction * intersection_extension, start_pt]),
                o3d.utility.Vector2iVector([[0, 1]]))
            node.potential_connections.append(LinesetNode(name=f'intersection_{counter}',
                                                          derivedFrom=node.name,
                                                          derivedFromPt=0,
                                                          resource=node.start_intersection_line,
                                                          weight=weight_intersection))
            counter += 1

            # Create orthogonal lines for the start point
            node.start_ortho_line1 = o3d.geometry.LineSet(
                o3d.utility.Vector3dVector([start_pt, start_pt + normal * t_ortho_extension]),
                o3d.utility.Vector2iVector([[0, 1]]))
            node.potential_connections.append(LinesetNode(name=f'ortho_{counter}',
                                                          derivedFrom=node.name,
                                                          derivedFromPt=0,
                                                          resource=node.start_ortho_line1,
                                                          weight=weight_orthogonal))
            counter += 1
            node.start_ortho_line2 = o3d.geometry.LineSet(
                o3d.utility.Vector3dVector([start_pt, start_pt - normal * t_ortho_extension]),
                o3d.utility.Vector2iVector([[0, 1]]))
            node.potential_connections.append(LinesetNode(name=f'ortho_{counter}',
                                                          derivedFrom=node.name,
                                                          derivedFromPt=0,
                                                          resource=node.start_ortho_line2,
                                                          weight=weight_orthogonal))
            counter += 1

            # Create direct lines for the start point
            all_points = np.array([e.start_pt for e in wallNodesBIM if e != node] + [e.end_pt for e in wallNodesBIM if e != node])
            indices, distances = compute_nearest_neighbors(np.array([start_pt]), all_points, maxDist=t_direct_extension, n=5)
            start_pts_direct = [all_points[i] for i in indices[0]]
            node.start_direct_lines = [o3d.geometry.LineSet(
                o3d.utility.Vector3dVector([start_pt, start_pt_direct]),
                o3d.utility.Vector2iVector([[0, 1]])) for start_pt_direct in start_pts_direct]
            for i in range(len(node.start_direct_lines)):
                node.potential_connections.append(LinesetNode(name=f'direct_{counter}',
                                                              derivedFrom=node.name,
                                                              derivedFromPt=0,
                                                              resource=node.start_direct_lines[i],
                                                              weight=weight_direct))
                counter += 1

        # Check and compute potential connections for the end point
        if 1 not in node.pts_used_in_intersections:
            # Create intersection line in the node.direction of the axis for the end point
            intersection_extension = 0.25 * node.wallLength if 0.25 * node.wallLength < t_intersection_extension else t_intersection_extension
            node.end_intersection_line = o3d.geometry.LineSet(
                o3d.utility.Vector3dVector([end_pt, end_pt + node.direction * intersection_extension]),
                o3d.utility.Vector2iVector([[0, 1]]))
            node.potential_connections.append(LinesetNode(name=f'intersection_{counter}',
                                                          derivedFrom=node.name,
                                                          derivedFromPt=1,
                                                          resource=node.end_intersection_line,
                                                          weight=weight_intersection))
            counter += 1

            # Create orthogonal lines for the end point
            node.end_ortho_line1 = o3d.geometry.LineSet(
                o3d.utility.Vector3dVector([end_pt, end_pt + normal * t_ortho_extension]),
                o3d.utility.Vector2iVector([[0, 1]]))
            node.potential_connections.append(LinesetNode(name=f'ortho_{counter}',
                                                          derivedFrom=node.name,
                                                          derivedFromPt=1,
                                                          resource=node.end_ortho_line1,
                                                          weight=weight_orthogonal))
            counter += 1
            node.end_ortho_line2 = o3d.geometry.LineSet(
                o3d.utility.Vector3dVector([end_pt, end_pt - normal * t_ortho_extension]),
                o3d.utility.Vector2iVector([[0, 1]]))
            node.potential_connections.append(LinesetNode(name=f'ortho_{counter}',
                                                          derivedFrom=node.name,
                                                          derivedFromPt=1,
                                                          resource=node.end_ortho_line2,
                                                          weight=weight_orthogonal))
            counter += 1

            # Create direct lines for the end point
            all_points = np.array([e.start_pt for e in wallNodesBIM if e != node] + [e.end_pt for e in wallNodesBIM if e != node])
            indices, distances = compute_nearest_neighbors(np.array([end_pt]), all_points, maxDist=t_direct_extension, n=5)
            end_pts_direct = [all_points[i] for i in indices[0]]
            node.end_direct_lines = [o3d.geometry.LineSet(
                o3d.utility.Vector3dVector([end_pt, end_pt_direct]),
                o3d.utility.Vector2iVector([[0, 1]])) for end_pt_direct in end_pts_direct]
            for i in range(len(node.end_direct_lines)):
                node.potential_connections.append(LinesetNode(name=f'direct_{counter}',
                                                              derivedFrom=node.name,
                                                              derivedFromPt=1,
                                                              resource=node.end_direct_lines[i],
                                                              weight=weight_direct))
                counter += 1

    # Gather all potential connections
    all_potential_connections = []
    for node in wallNodesBIM:
        all_potential_connections.extend(node.potential_connections)
    return all_potential_connections

def trim_and_extend_wall_nodes(wallNodes: List[Node], t_trim: float = 0.2,t_extend:float=0.5) -> None:
    """
    Trim and extend the wall nodes by adjusting their axes to the intersection points.

    Parameters:
        wallNodes: List of Node objects, each representing a wall.
        t_trim: Threshold distance within which points are considered for trimming.
        t_extend: Threshold distance to which points are extended.
    """
    for node in wallNodes:
        # Initialize the list of points used in intersections
        node.pts_used_in_intersections = []
        # Create a temporary deep copy of the node's axis
        node.new_axis = copy.deepcopy(node.axis)
        
        # Find intersection points with other wall nodes
        ref_wall_nodes = [e for e in wallNodes if e != node]
        intersection_points = find_line_intersection_or_extension(node.axis, [e.axis for e in ref_wall_nodes],t_extend=t_extend)
        node.neighbor_wall_ids_at_start=[]
        node.neighbor_wall_ids_at_end=[]
        if len([l for l in intersection_points if l is not None]):
            
            #get neighbor wall ids
            neigboring_walls=[ref_wall_nodes[i].object_id for i,intersection_point in enumerate(intersection_points) if intersection_point is not None]
       
            #filter None values
            intersection_points=[l for l in intersection_points if l is not None]
            # node.intersection_points = intersection_points
            
            if len(intersection_points) == 1:
                # If there is only one intersection point, determine if it is closer to the start or end point
                distances = np.abs([
                    np.linalg.norm(np.asarray(node.new_axis.points)[0] - intersection_points[0]),
                    np.linalg.norm(np.asarray(node.new_axis.points)[1] - intersection_points[0])
                ])
                
                # Find the index of the closest point and check if it is within the threshold
                index = np.argmin(distances) if np.min(distances) < t_trim else None
                if index is not None:
                    # Replace the axis point at the index with the intersection point
                    axis_points = np.asarray(node.new_axis.points)
                    axis_points[index] = intersection_points[0]
                    node.new_axis.points = o3d.utility.Vector3dVector(axis_points)
                    node.pts_used_in_intersections.append(index)
                    
                    # Color the axis (optional)
                    node.new_axis.paint_uniform_color([1, 0, 0])
                    
                    #assign neighbor wall ids
                    if index==0:
                        node.neighbor_wall_ids_at_start.append(neigboring_walls[0])
                    else:
                        node.neighbor_wall_ids_at_end.append(neigboring_walls[0])
            
            else:
                # Compute distances from the start and end points to all intersection points
                distances_start = np.array([np.linalg.norm(node.axis.points[0] - l) for l in intersection_points])
                distances_end = np.array([np.linalg.norm(node.axis.points[1] - l) for l in intersection_points])
                
                # Find indices of the closest points within the threshold
                index_start = np.argmin(distances_start) if np.min(distances_start) < t_trim else None
                index_end = np.argmin(distances_end) if np.min(distances_end) < t_trim else None
                
                if index_start is not None and index_start != index_end:
                    # Replace the start point of the axis with the intersection point
                    axis_points = np.asarray(node.new_axis.points)
                    axis_points[0] = intersection_points[index_start]
                    node.new_axis.points = o3d.utility.Vector3dVector(axis_points)
                    node.pts_used_in_intersections.append(0)
                    
                    # Color the axis (optional)
                    node.new_axis.paint_uniform_color([0, 1, 0])
                    
                    #assign neighbor wall ids
                    node.neighbor_wall_ids_at_start.append(neigboring_walls[index_start])
                
                if index_end is not None and index_start != index_end:
                    # Replace the end point of the axis with the intersection point
                    axis_points = np.asarray(node.new_axis.points)
                    axis_points[1] = intersection_points[index_end]
                    node.new_axis.points = o3d.utility.Vector3dVector(axis_points)
                    node.pts_used_in_intersections.append(1)
                    # Color the axis (optional)
                    node.new_axis.paint_uniform_color([0, 1, 0])
                    
                    #assign neighbor wall ids
                    node.neighbor_wall_ids_at_end.append(neigboring_walls[index_end])
        
    # Update the node's axis and delete the temporary copy
    for n in wallNodes:
        n.axis = n.new_axis
        del n.new_axis

def find_line_intersection_or_extension(line, ref_lines,t_extend:float=0) -> np.ndarray:
    """
    Find the intersection points between a line and several reference lines defined by points (p1, p2).

    Parameters:
    line: an object with a 'points' attribute, where points are defined as a list or array of coordinates.
    ref_lines: a list of objects with 'points' attributes, each defining a line segment.

    Returns:
    A list of intersection points or None if no intersections are found.
    """    

    p1 = np.asarray(line.points)[0, :2].astype(np.float64)
    p2 = np.asarray(line.points)[1, :2].astype(np.float64)
    
    intersection_points = []
    tolerance = 1e-9
    
    for ref_line in ref_lines:
        #we can represent these lines as:
        #Line 1: p(t1)=p1+t1⋅(p2−p1)
        #Line 2: q(t2)=p3+t2⋅(p4−p3)
        #and solve this for parameters t1 and t2
        
        # Convert points to numpy arrays
        p3 = np.asarray(ref_line.points)[0, :2].astype(np.float64)
        p4 = np.asarray(ref_line.points)[1, :2].astype(np.float64)
        
        # Direction vectors of the lines
        d1 = p2 - p1
        d2 = p4 - p3
        
        #get lengths of the lines
        length1=np.abs(np.linalg.norm(d1))   
        length2=np.abs(np.linalg.norm(d2))   
        
        # Compute the determinant (cross product) of direction vectors
        determinant = np.cross(d1, d2)
        
        # If the determinant is close to zero, lines are parallel or collinear
        if np.isclose(determinant, 0, atol=tolerance):
            continue
        
        # Solve for t1 and t2 in the parametric equations
        A = np.array([d1, -d2]).T
        b = p3 - p1
        try:
            t = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            # If A is singular, skip this pair of lines
            continue
        t1, t2 = t
        # print('new line')
        # print(t1,t2)
        # print(np.abs(np.linalg.norm(t1%1*d1/np.linalg.norm(d1))))
        # print(np.abs(np.linalg.norm(t2%1*d2/np.linalg.norm(d2))) )       

        # condition_0=(0-t_extend/length1 <= t1 <= 1+t_extend/length1 and 0-t_extend/length2 <= t2 <= 1+t_extend/length2)
        # condition_1=(0 <= t1 <= 1 and 0 <= t2 <= 1) #intersection point lies within both segments
        # condition_2=(np.abs(np.linalg.norm(t1%1*d1/np.linalg.norm(d1)))<t_extend and 0 <= t2 <= 1) #extend the line if t1*d1<t_extension
        # condition_3=(0 <= t1 <= 1 and np.abs(np.linalg.norm(t2%1*d2/np.linalg.norm(d2)))<t_extend) #extend the line if t2*d2<t_extension
        # condition_4=(np.abs(np.linalg.norm(t1%1*d1/np.linalg.norm(d1)))<t_extend and np.abs(np.linalg.norm(t2%1*d2/np.linalg.norm(d1)))<t_extend) #extend the line if t1*d1<t_extension and t2*d2<t_extension
        # print(condition_1 , condition_2 , condition_3 , condition_4)

        if (0-t_extend/length1 <= t1 <= 1+t_extend/length1 and 0-t_extend/length2 <= t2 <= 1+t_extend/length2): #or condition_2 or condition_3 or condition_4:
            
            intersection_point = p1 + t1 * d1        
            # Add the z-coordinate 
            z_coord = np.asarray(line.points)[0, 2]
            intersection_point = np.array([intersection_point[0], intersection_point[1], z_coord])
            intersection_points.append(intersection_point)

        else:
            intersection_points.append(None)
    return intersection_points[0] if len(intersection_points)==1 else intersection_points
