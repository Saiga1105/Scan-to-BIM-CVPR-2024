
import time
import json
import open3d as o3d
from scipy.spatial.transform import Rotation   
import numpy as np
import geomapi
from geomapi.nodes import *
from geomapi import utils as ut
from geomapi.utils import geometryutils as gmu


def t1_time_funtion(func, *args):
    """Measures how long the functions takes to run and returns the result 

    Args:
        func (function): The funtion to measure, write without ()
        *args (Tuple) : The arguments for the funtion, pass as a tuple with a * in front to pass the arguments seperatly

    Returns:
        object: The result of the function
    """

    start = time.time()
    result = func(*args)
    end = time.time()
    print("Completed function `" + func.__name__ + "()` in", np.round(end - start,3), "seconds")
    return result 

def parse_json(file,objects_dict):
    try:
        with open(file, 'r') as f:
            json_data = f.read()
            print("Data read from file:", file.split('/')[-1].split('.')[0])  # Check the data from the file.
            data = json.loads(json_data)
    except data:
        print("File not found.")
    except json.JSONDecodeError as e:
        print("Failed to decode JSON:", e)
    except Exception as e:
        print("An error occurred:", e)
    
    for item in data:
        item_class = file.split('/')[-1].split('.')[0].split('_')[-1]    
        source='_'.join(file.split('/')[-1].split('.')[0].split('_')[0:-1])
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

def create_object_nodes(objects_dict,class_dict):
        
        nodes=[]
        #match objects to classes
        id_dict = {item['name']: item['id'] for item in class_dict['classes']}
        computed_ids = {key: id_dict.get(value['type'], class_dict['default']) for key, value in objects_dict.items()}

        #create geomapi nodes for each object
        for key,id in zip(objects_dict.keys(),computed_ids.values()):
            if objects_dict[key]['type'] == 'columns':
                nodes.append(MeshNode(name=key, 
                                    resource= objects_dict[key]['resource'],
                                    class_id=id,
                                    width=objects_dict[key]['width'],
                                    depth=objects_dict[key]['depth'],
                                    height=objects_dict[key]['height'],
                                    loc=objects_dict[key]['loc'],
                                    rotation=objects_dict[key]['rotation'],                                    
                                    class_name=objects_dict[key]['type'],
                                    derivedFrom=objects_dict[key]['source'],
                                    color=ut.random_color()))
            elif objects_dict[key]['type'] == 'doors':
                nodes.append(MeshNode(name=key, 
                                    resource= objects_dict[key]['resource'],
                                    class_id=id,
                                    width=objects_dict[key]['width'],
                                    depth=objects_dict[key]['depth'],
                                    height=objects_dict[key]['height'],
                                    loc=objects_dict[key]['loc'],
                                    rotation=objects_dict[key]['rotation'],
                                    class_name=objects_dict[key]['type'],
                                    derivedFrom=objects_dict[key]['source'],
                                    color=ut.random_color()))
            elif objects_dict[key]['type'] == 'walls':
                nodes.append(MeshNode(name=key, 
                                    resource= objects_dict[key]['resource'],
                                    line=objects_dict[key]['line'],
                                    class_id=id,
                                    width=objects_dict[key]['width'],
                                    height=objects_dict[key]['height'],
                                    neighbor_wall_ids_at_start=objects_dict[key]['neighbor_wall_ids_at_start'],
                                    neighbor_wall_ids_at_end=objects_dict[key]['neighbor_wall_ids_at_end'],
                                    class_name=objects_dict[key]['type'],
                                    derivedFrom=objects_dict[key]['source'],
                                    color=ut.random_color()))
        return nodes

def create_column(object_data)->o3d.geometry.TriangleMesh:
    
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

def create_door(object_data)->o3d.geometry.TriangleMesh:
    
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

def create_wall(object_data):
    """
    Creates a LineSet object in Open3D from two 3D points.

    Parameters:
        object_data (dict): A dictionary containing 'start_pt',  'end_pt', both of which are
                            lists or tuples of x, y, z coordinates of the points. Also contains 'width' and 'height' of the wall.

    Returns:
        o3d.geometry.LineSet,o3d.geometry.TriangleMesh: LineSet object representing the line between start_pt and end_pt, and the oriented bounding box representing the wall.
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
    # wallBox=o3d.geometry.LineSet.create_from_oriented_bounding_box(box)


    return line_set, mesh

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
            
def process_point_cloud(pcdNode,objectNodes,distance_threshold:0.1,resolution:0.03):
    
    #create scalars for the point cloud
    object_scalar = np.full(len(pcdNode.resource.points), 0, dtype=np.uint8)
    class_scalar = np.full(len(pcdNode.resource.points), 255, dtype=np.uint8)
    for i,n in objectNodes:
        n.object_id=i+1

    #create an identity point cloud of all the objectNodes
    identityPcd,objectArray=gmu.create_identity_point_cloud([n.resource for n in objectNodes if n.derivedFrom==pcdNode.name],resolution=resolution)
    classArray=np.array([int(n.class_id) for n in objectNodes if n.derivedFrom==pcdNode.name])[objectArray.astype(int)]
    # print(len(classArray),len(objectArray))

    #compute nearest neighbors
    indices,distances=gmu.compute_nearest_neighbors(np.asarray(pcdNode.resource.points),np.asarray(identityPcd.points))
    indices=indices.flatten()
    
    #compute the object and class scalars based on threshold distance
    threshold_indices = np.where(distances <= distance_threshold)[0]
    object_scalar[threshold_indices] = objectArray[indices[threshold_indices]].astype(int)
    class_scalar[threshold_indices] = classArray[indices[threshold_indices]]
    
    #remap objectArray
    # names=np.array([int(n.name) for n in objectNodes])
    # object_scalar=names[object_scalar]
    
    return class_scalar,object_scalar