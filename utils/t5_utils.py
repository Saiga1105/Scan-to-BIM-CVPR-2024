
import time
import os
import laspy
import geomapi.utils.geometryutils as gmu
import geomapi.utils as ut
import numpy as np
from utils import timer
import open3d as o3d
from  geomapi.nodes import *
from typing import Tuple, List, Optional,Dict,Any
import copy 
from scipy.spatial.transform import Rotation   
import json

def create_level_nodes(
    nodes: List[PointCloudNode],
    threshold_vertical_clustering: float = 1.5,
    threshold_horizontal_clustering: float = 5.0
) -> List[SessionNode]:
    """
    Cluster nodes into levels based on spatial proximity and create level session nodes.

    Parameters:
        nodes (List[PointCloudNode]): A list of point cloud nodes to cluster.
        threshold_vertical_clustering (float): Vertical distance threshold for clustering nodes.
        threshold_horizontal_clustering (float): Horizontal distance threshold for clustering nodes.

    Returns:
        List[SessionNode]: A list of session nodes, each representing a cluster of point cloud nodes as a level.
    """
   
    clustered_nodes=[]    
    #sort the nodes from largest to smallest                                                    
    sorted_list = sorted(nodes, key=lambda x: len(np.asarray(x.resource.points)),reverse=True)
    
    #initialize candidate lists
    for n in nodes:
        n.candidates=[]
    
    #iterate over the sorted list    
    while len(sorted_list) >1:
        #take first element
        n=sorted_list[0]
        sorted_list.pop(0) 
        
        # retrieve neirest neighbors of same class_id
        candidate_list=[p for p in sorted_list if np.abs(p.resource.get_center()[2]-n.resource.get_center()[2])<=threshold_vertical_clustering]
        # print(f'len candidate list:: {len(candidate_list)}')
        if len(candidate_list)==0:
            clustered_nodes.append([n]+n.candidates)
            continue
        joined_pcd,identityArray=gmu.create_identity_point_cloud([p.resource for p in candidate_list ])       #this is a little bit silly
        indices,distances=gmu.compute_nearest_neighbors(np.asarray(n.resource.points),np.asarray(joined_pcd.points),n=1) 
        indices=indices[:,0]
        distances=distances[:,0]
        
        reference_points=np.asarray(joined_pcd.points)[indices]
        #compute vertical ditance
        vertical_distances=np.abs(reference_points[:,2]-np.asarray(n.resource.points)[:,2])
        #compute horizontal distance
        horizontal_distances=np.abs(np.linalg.norm(reference_points[:,:2]-np.asarray(n.resource.points)[:,:2],axis=1))
                      
        #filter on distance
        indices=indices[(vertical_distances<threshold_vertical_clustering) & (horizontal_distances<threshold_horizontal_clustering)]              
         
        #check if there are any indices left
        if len(indices)==0:
            clustered_nodes.append([n]+n.candidates)
            continue
        
        #group the nodes 
        unique=np.unique(identityArray[indices])
        candidates=[p for i,p in enumerate(candidate_list) if i in unique] #sorted_list
        n.candidates.extend(candidates)

        #remove candidates from sorted_list that are also in candidates
        indices_to_remove = sorted(set([i for i,p in enumerate(sorted_list) if p in candidates]), reverse=True)
        [sorted_list.pop(idx) for idx in indices_to_remove]

        #if there were candidates, add n back to list so it can continue to look for candidates with its updates geometry
        n.resource=gmu.join_geometries([n.resource]+[p.resource for p in candidates])
        #add n at front of list again
        sorted_list.insert(0,n)
    
    #add remaining elements
    if len(sorted_list)!=0:
        for n in sorted_list:
            clustered_nodes.append([n]+n.candidates)
            
    levelNodes=[]   
    #define a sessionNode for each cluster
    for i,nodes in enumerate(clustered_nodes):
        #create sessionNode
        name='level_'+str(i)+'0'
        referenceNode=SessionNode(linkedNodes=nodes,
                                    name=name,
                                    color=ut.random_color())    
        
        #determine height -> note that this can be negative
        weights=[float(len(np.asarray(n.resource.points))) for n in nodes]
        heights= [float(n.cartesianTransform[2,3]) for n in nodes]
        weighted_height= np.average(heights, weights=weights)

        #compute plane from cornerpoints orientedbounding box
        vertices=np.array([np.hstack((referenceNode.orientedBounds[0][0:2],weighted_height)),
                        np.hstack((referenceNode.orientedBounds[1][0:2],weighted_height)),
                        np.hstack((referenceNode.orientedBounds[2][0:2],weighted_height)),
                        np.hstack((referenceNode.orientedBounds[4][0:2],weighted_height))])#,
        vertices=o3d.utility.Vector3dVector(vertices)
        triangles=o3d.utility.Vector3iVector(np.array([[0,1,2],[2,1,3]]))
        plane=o3d.geometry.TriangleMesh(vertices,triangles)
        
        plane2=copy.deepcopy(plane).translate(np.array([0,0,0.1]))
        new_vertices=o3d.utility.Vector3dVector(np.vstack([np.asarray(plane.vertices),np.asarray(plane2.vertices)]))
        referenceNode.box=o3d.geometry.OrientedBoundingBox.create_from_points(new_vertices)
        referenceNode.box.color=[1,0,0]
        
        #assign information to referenceNode
        referenceNode.plane=plane
        referenceNode.height=weighted_height
        
        levelNodes.append(referenceNode)
    
    #sort the levelNodes by height
    levelNodes=sorted(levelNodes,key=lambda x: x.height)
    for i,n in enumerate(levelNodes):
        n.name='level_'+str(i)+'0'
        n.subject=n.name
            
    return levelNodes

def levels_to_json(levelNodes: list[SessionNode], file_name: str) -> Dict:
    """
    Converts level nodes data to a Dictionary ready for saving or further processing.

    Parameters:
        levelNodes (list): A list of nodes, each representing a level with attributes like box, name, and height.
        file_name (str): The file name from which the levels are derived.

    Returns:
        str: A Dictionary representing the level data.
    
    Raises:
        ValueError: If the input data is not in the expected format or missing required data.
    """
    
    # Prepare JSON data structure
    json_data = {
        "filename": f"{ut.get_filename(file_name)}_levels.obj",
        "objects": []
    }
    
    # Fill JSON with node data
    for n in levelNodes:
        if not hasattr(n, 'box') or not hasattr(n, 'name') or not hasattr(n, 'height'):
            raise ValueError("Node is missing required attributes (box, name, height).")
        
        try:
            # Safe copy of rotation matrix to ensure no unintended mutations
            rotation_matrix = copy.deepcopy(n.box.R)
            # Create a rotation object from the matrix and convert to Euler angles
            r = Rotation.from_matrix(np.asarray(rotation_matrix))
            rotations = r.as_euler("zyx", degrees=True)
            
            # Construct object dictionary for each node
            obj = {
                "name": n.name,
                "centroid": {
                    "x": n.box.center[0],
                    "y": n.box.center[1],
                    "z": n.height
                },
                "dimensions": {
                    "length": n.box.extent[0],
                    "width": n.box.extent[1],
                    "height": n.height
                },
                "rotations": {
                    "x": 0,  # Assuming fixed rotations for x and y as per the initial code
                    "y": 0,
                    "z": rotations[0]  # Typically, z is the vertical axis in most coordinate systems
                }
            }
            json_data["objects"].append(obj)
        except Exception as e:
            raise ValueError(f"Error processing node {n.name}: {str(e)}")
    
    # Convert the Python dictionary to a JSON string
    return json_data
