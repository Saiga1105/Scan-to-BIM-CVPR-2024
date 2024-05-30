"""
Utilities 

"""
import numpy as np
import time
from pathlib import Path
import geomapi.utils as ut
import laspy
import geomapi.utils.geometryutils as gmu
from typing import Dict, Any, Tuple,List
from geomapi.nodes import PointCloudNode
import geomapi.tools as tl
import open3d as o3d

def timer(func):
    """
    Decorator that measures and prints the execution time of a function.
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()  # Capture the start time
        result = func(*args, **kwargs)  # Call the function with its arguments
        end_time = time.time()  # Capture the end time
        print(f"Function {func.__name__} took {end_time - start_time:.4f} seconds to execute.")
        return result  # Return the result of the function
    return wrapper

def get_list_of_files(folder: Path | str , ext: str = None) -> list:
    """
    Get a list of all filepaths in the folder and subfolders that match the given file extension.

    Args:
        folder: The path to the folder as a string or Path object
        ext: Optional. The file extension to filter by, e.g., ".txt". If None, all files are returned.

    Returns:
        A list of filepaths that match the given file extension.
    """
    folder = Path(folder)  # Ensure the folder is a Path object
    allFiles = []
    # Iterate over all the entries in the directory
    for entry in folder.iterdir():
        # Create full path
        fullPath = entry
        # If entry is a directory then get the list of files in this directory 
        if fullPath.is_dir():
            allFiles += get_list_of_files(fullPath, ext=ext)
        else:
            # Check if file matches the extension
            if ext is None or fullPath.suffix.lower() == ext.lower():
                allFiles.append(fullPath.as_posix())
    return allFiles



@timer
def match_graph_with_las(file_path,las,pcd,class_dict,getResources=True,getNormals=False)->List[PointCloudNode]:

    #import graph
    f_g=Path(file_path).with_suffix('.ttl')
    pcdNodes=tl.graph_path_to_nodes(graphPath=str(f_g))

    # get the point cloud data
    if getResources:
          
        pcd.estimate_normals() if getNormals or not pcd.has_normals()  else None
        
        #match pcd to nodes
        for c in class_dict['classes']:
            idx=np.where((las['classes']==c['id']))[0]
            class_pcd=pcd.select_by_index(idx)
            object_labels=las['objects'][idx]
            
            for j in np.unique(object_labels):
                indices=np.where(object_labels==j)[0]
                object_pcd=class_pcd.select_by_index(indices)
                pcdNode=next((x for x in pcdNodes if x.object_id == j), None)
                pcdNode.resource=object_pcd if pcdNode is not None else None
            
    return pcdNodes

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



def load_obj_and_create_meshes(file_path: str) -> Dict[str, o3d.geometry.TriangleMesh]:
    """
    Loads an OBJ file and creates TriangleMeshes for each object group.

    Args:
        file_path (str): Path to the OBJ file.

    Returns:
        Dict[str, o3d.geometry.TriangleMesh]: A dictionary mapping object group names to their corresponding TriangleMeshes.
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()

    vertices = []
    faces = {}
    current_object = None

    for line in lines:
        if line.startswith('v '):
            parts = line.strip().split()
            vertex = [float(parts[1]), float(parts[2]), float(parts[3])]
            vertices.append(vertex)
        elif line.startswith('f '):
            if current_object is not None:
                parts = line.strip().split()
                face = [int(parts[1].split('/')[0]) - 1, int(parts[2].split('/')[0]) - 1, int(parts[3].split('/')[0]) - 1]
                faces[current_object].append(face)
        elif line.startswith('g '):
            current_object = line.strip().split()[1]
            if current_object not in faces:
                faces[current_object] = []

    meshes = {}
    for object_name, object_faces in faces.items():
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(object_faces)
        mesh.compute_vertex_normals()
        meshes[object_name] = mesh
    
    return meshes