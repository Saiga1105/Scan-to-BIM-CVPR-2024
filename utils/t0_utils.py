'''This module contains utility functions for the t0 pipeline. The functions are used to parse JSON files, create Open3D objects, and process point clouds.'''

import time
import json
import open3d as o3d
from scipy.spatial.transform import Rotation   
import numpy as np
import geomapi
from geomapi.nodes import *
from geomapi import utils as ut
from geomapi.utils import geometryutils as gmu
from typing import Dict, Any, Tuple,List


def parse_json(file_path: str, objects_dict: Dict) -> Dict:
    """
    Parses a JSON file to extract object properties and store in a dictionary.
    """
    try:
        with open(file_path, 'r') as f:
            json_data = f.read()
            print("Data read from file:", file_path.split('/')[-1].split('.')[0])  # Check the data from the file.
            data = json.loads(json_data)
    except data:
        print("File not found.")
    except json.JSONDecodeError as e:
        print("Failed to decode JSON:", e)
    except Exception as e:
        print("An error occurred:", e)
    
    for item in data:
        item_class = file_path.split('/')[-1].split('.')[0].split('_')[-1]    
        source='_'.join(file_path.split('/')[-1].split('.')[0].split('_')[0:-1])
        # Collect details based on type
        if item_class == 'columns':
            objects_dict[item['id']] = {
                'type': item_class,
                'width': item['width'],
                'depth': item['depth'],
                'height': item['height'],
                'loc': item['loc'],
                'rotation': item['rotation'],
                'source':source
            }
            column=create_column(objects_dict[item['id']])
            objects_dict[item['id']]['resource'] = column 

        elif item_class == 'doors':
            objects_dict[item['id']] = {
                'type': item_class,
                'width': item['width'],
                'depth': item['depth'],
                'height': item['height'],
                'loc': item['loc'],
                'rotation': item['rotation'],
                'host_id': item['host_id'],
                'source':source
            }
            door=create_door(objects_dict[item['id']])     
            objects_dict[item['id']]['resource'] = door     

        elif item_class == 'walls':
            objects_dict[item['id']] = {
                'type': item_class,
                'start_pt': item['start_pt'],
                'end_pt': item['end_pt'],
                'width': item['width'],
                'height': item['height'],
                'neighbor_wall_ids_at_start': item['neighbor_wall_ids_at_start'],
                'neighbor_wall_ids_at_end': item['neighbor_wall_ids_at_end'],
                'source':source
            }
            line,wall=create_wall( objects_dict[item['id']])
            objects_dict[item['id']]['line'] = line
            objects_dict[item['id']]['resource'] = wall   

    return objects_dict

def create_object_nodes(objects_dict: Dict, class_dict: Dict) -> List[MeshNode]:
    """
    Creates and returns a list of MeshNode objects from object definitions and class mappings.

    Parameters:
        objects_dict (Dict[str, Any]): Dictionary containing object specifications.
        class_dict (Dict[str, Any]): Dictionary containing class definitions and a default class ID.

    Returns:
        List[MeshNode]: List of MeshNode instances created based on the objects and class mappings.
    """        
    nodes=[]
    #match objects to classes
    id_dict = {item['name']: item['id'] for item in class_dict['classes']}
    computed_ids = {key: id_dict.get(value['type'], class_dict['default']) for key, value in objects_dict.items()}

    #create geomapi nodes for each object
    for key,id in zip(objects_dict.keys(),computed_ids.values()):
        if objects_dict[key]['type'] == 'columns':
            nodes.append(MeshNode(name='columns_'+str(key), 
                                resource= objects_dict[key]['resource'],
                                class_id=id,
                                object_id=key,
                                width=objects_dict[key]['width'],
                                depth=objects_dict[key]['depth'],
                                height=objects_dict[key]['height'],
                                loc=objects_dict[key]['loc'],
                                rotation=objects_dict[key]['rotation'],                                    
                                class_name=objects_dict[key]['type'],
                                derivedFrom=objects_dict[key]['source'],
                                color=ut.random_color()))
        elif objects_dict[key]['type'] == 'doors':
            nodes.append(MeshNode(name='doors_'+str(key), 
                                resource= objects_dict[key]['resource'],
                                class_id=id,
                                object_id=key,
                                width=objects_dict[key]['width'],
                                depth=objects_dict[key]['depth'],
                                height=objects_dict[key]['height'],
                                loc=objects_dict[key]['loc'],
                                rotation=objects_dict[key]['rotation'],
                                class_name=objects_dict[key]['type'],
                                derivedFrom=objects_dict[key]['source'],
                                color=ut.random_color()))
        elif objects_dict[key]['type'] == 'walls':
            nodes.append(MeshNode(name='walls_'+str(key), 
                                resource= objects_dict[key]['resource'],
                                line=objects_dict[key]['line'],
                                class_id=id,
                                object_id=key,
                                width=objects_dict[key]['width'],
                                height=objects_dict[key]['height'],
                                neighbor_wall_ids_at_start=objects_dict[key]['neighbor_wall_ids_at_start'],
                                neighbor_wall_ids_at_end=objects_dict[key]['neighbor_wall_ids_at_end'],
                                class_name=objects_dict[key]['type'],
                                derivedFrom=objects_dict[key]['source'],
                                color=ut.random_color()))
    return nodes


def create_column(object_data: Dict) -> o3d.geometry.TriangleMesh:
    """
    Creates a 3D column mesh based on given specifications.

    Parameters:
        object_data (Dict[str, Any]): Dictionary containing dimensions and location for the column.

    Returns:
        o3d.geometry.TriangleMesh: 3D mesh of the column.
    """    
    width=object_data['width']
    depth=object_data['depth']
    height=object_data['height']
    loc=object_data['loc']
    rotation=object_data['rotation']
    
    # Create a box with the given dimensions
    column = o3d.geometry.TriangleMesh.create_box(width, depth, height)
    center=column.get_center()

    # Rotate the box
    rotation_matrix = Rotation.from_euler('z', rotation, degrees=True).as_matrix()
    column.rotate(rotation_matrix)
    
    # Translate the box minus its current center (loc is the true center of the column)
    column.translate(loc-np.array([center[0],center[1],0]))
    
    return column

def create_door(object_data: Dict) -> o3d.geometry.TriangleMesh:
    """
    Creates a 3D door mesh based on given specifications.

    Parameters:
        object_data (Dict[str, Any]): Dictionary containing dimensions and location for the door.

    Returns:
        o3d.geometry.TriangleMesh: 3D mesh of the door.
    """    
    width=object_data['width']
    depth=object_data['depth']
    height=object_data['height']
    loc=object_data['loc']
    rotation=object_data['rotation']
    
    # Create a box with the given dimensions
    door = o3d.geometry.TriangleMesh.create_box(width, depth, height)
    center=door.get_center()

    # Rotate the box
    rotation_matrix = Rotation.from_euler('z', rotation, degrees=True).as_matrix()
    door.rotate(rotation_matrix)
    
    # Translate the box minus its current center (loc is the true center of the column)
    door.translate(loc-np.array([center[0],center[1],0]))

    return door

def create_wall(object_data: Dict) -> Tuple[o3d.geometry.LineSet, o3d.geometry.TriangleMesh]:
    """
    Creates a 3D wall represented by a line and a mesh from provided specifications.

    Parameters:
        object_data (Dict[str, Any]): Dictionary containing the start and end points, width, and height of the wall.

    Returns:
        Tuple[o3d.geometry.LineSet, o3d.geometry.TriangleMesh]: LineSet and mesh representing the wall.
    """   
    start_pt=object_data['start_pt']
    end_pt=object_data['end_pt']
    width=object_data['width']
    height=object_data['height']
    
    # Create points array
    points = np.array([start_pt, end_pt], dtype=np.float64)

    # Create lines array
    lines = np.array([[0, 1]])  # This indicates a single line from the first point to the second

    # Create a LineSet object and set its points and lines
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    
    #compute direction
    vector1 = points[1] - points[0]

    # Compute normal from cross product with z-axis
    normal = np.cross(vector1, np.array([0, 0, 1]))
    
    # Normalize the normal vector
    normal /= np.linalg.norm(normal)
    
    # Create a point list for the wall by translating the points by the normal vector and the width
    pointList=[]
    points=np.asarray(line_set.points)
    pointList.extend(points+normal*width/2)
    pointList.extend(points-normal*width/2)
    pointList.extend(np.array(pointList)+np.array([0,0,height]))
    pcd=o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(pointList))

    # Create a mesh from the point list
    box=pcd.get_oriented_bounding_box()
    mesh=o3d.geometry.TriangleMesh.create_from_oriented_bounding_box(box)

    return line_set, mesh

def write_obj_with_submeshes(filename: str, meshes: List[o3d.geometry.TriangleMesh], mesh_names: List[str]):
    """
    Writes multiple Open3D TriangleMesh objects to a single OBJ file with named submeshes.

    Parameters:
        filename (str): Path to the output file.
        meshes (List[o3d.geometry.TriangleMesh]): List of meshes to write.
        mesh_names (List[str]): List of names for each submesh.
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
            
def process_point_cloud(pcdNode: PointCloudNode, objectNodes: List[MeshNode], distance_threshold: float = 0.1, resolution: float = 0.03) -> Tuple[np.ndarray, np.ndarray]:
    """
    Process a point cloud to compute object and class scalars based on the proximity of the points to provided nodes.

    Parameters:
        pcdNode (MeshNode): The node containing the main point cloud.
        objectNodes (List[MeshNode]): List of nodes derived from the main point cloud.
        distance_threshold (float): Distance threshold to consider when assigning objects and class scalars.
        resolution (float): The resolution at which to sample the point cloud for identity.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple containing arrays of class and object scalars.
    """    
    #create scalars for the point cloud
    object_scalar = np.full(len(pcdNode.resource.points), 0, dtype=np.uint8)
    class_scalar = np.full(len(pcdNode.resource.points), 255, dtype=np.uint8)
    for i,n in objectNodes:
        n.object_id=i+1

    #create an identity point cloud of all the objectNodes
    identityPcd,objectArray=gmu.create_identity_point_cloud([n.resource for n in objectNodes if n.derivedFrom==pcdNode.name],resolution=resolution)
    classArray=np.array([int(n.class_id) for n in objectNodes if n.derivedFrom==pcdNode.name])[objectArray.astype(int)]

    #compute nearest neighbors
    indices,distances=gmu.compute_nearest_neighbors(np.asarray(pcdNode.resource.points),np.asarray(identityPcd.points))
    indices=indices.flatten()
    
    #compute the object and class scalars based on threshold distance
    threshold_indices = np.where(distances <= distance_threshold)[0]
    object_scalar[threshold_indices] = objectArray[indices[threshold_indices]].astype(int)
    class_scalar[threshold_indices] = classArray[indices[threshold_indices]]
        
    return class_scalar,object_scalar