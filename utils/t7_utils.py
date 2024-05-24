import numpy as np
from scipy.spatial import ConvexHull
import numpy.linalg as LA
import matplotlib.pyplot as plt
import geomapi.tools as tl
from rdflib import Graph
import geomapi.utils as ut
from geomapi.nodes import PointCloudNode
import laspy
import os
from geomapi.utils import geometryutils as gmu
import json
import open3d as o3d
import context 
import utils as utl

def load_graph(laz, graph_path):

    # Parse the graph
    graph = Graph().parse(str(graph_path))

    nodes = tl.graph_to_nodes(graph)
    column_nodes = [n for n in nodes if 'columns' in n.subject.lower() and type(n)==PointCloudNode]

    # for n in column_nodes:
    #     idx = np.where(( laz ['classes']==n.class_id) & ( laz ['objects'] == n.object_id))
    #     pcd = o3d.geometry.PointCloud()
    #     pcd.points = o3d.utility.Vector3dVector(laz.xyz[idx])
    #     n.resource = pcd
    #     print(f'{len(column_nodes)} columnNodes detected!')

    return nodes, column_nodes
    
def load_levels(laz, graph_path):  

    # Parse the graph
    graph = Graph().parse(str(graph_path))
    nodes = tl.graph_to_nodes(graph)

    # Separate nodes by type
    ceilings_nodes = [n for n in nodes if 'ceilings' in n.subject.lower() and isinstance(n, PointCloudNode)]
    floors_nodes = [n for n in nodes if 'floors' in n.subject.lower() and isinstance(n, PointCloudNode)]
    # ceilings_nodes = [n for n in nodes if 'ceilings' in n.class_name.lower() and isinstance(n, PointCloudNode)]
    # floors_nodes = [n for n in nodes if 'floors' in n.class_name.lower() and isinstance(n, PointCloudNode)]
    # level_nodes = [n for n in nodes if 'level' in n.subject]
    
    floors_z = []
    print(len(floors_nodes), "floors detected")

    for n in floors_nodes: # Maybe if I have more floors?
        idx = np.where((laz['classes'] == n.class_id) & (laz['objects'] == n.object_id))
        z_values = laz.z[idx]
        avg_z =  np.mean(z_values)
        floors_z.append(avg_z)
    
    ceilings_z = []
    print(len(ceilings_nodes), "ceilings detected")

    for n in ceilings_nodes:
        idx = np.where((laz['classes'] == n.class_id) & (laz['objects'] == n.object_id))
        z_values = laz.z[idx]
        avg_z=  np.mean(z_values)
        ceilings_z.append(avg_z)
    
    floor_z = np.mean(floors_z)
    ceiling_z = np.mean (ceilings_z)
    print('floor_z_avg:', floor_z)
    print('ceiling_z_avg:', ceiling_z)

    return floors_z, ceilings_z, floor_z, ceiling_z
 
def load_point_cloud(file_name):

    laz  = laspy.read(file_name)
    pcd = gmu.las_to_pcd(laz)
    pcd.estimate_normals()
    pcd.orient_normals_to_align_with_direction()
    pcd_nodes = PointCloudNode(resource=pcd)
    normals=np.asarray(pcd.normals)
    
    return laz, pcd, pcd_nodes, normals

def compute_column_height(z_values):

    min_z = np.min(z_values)
    max_z = np.max(z_values)
    avg_z = np.mean(z_values)
    height = max_z-min_z

    return min_z, max_z, avg_z, height

def load_columns_data(laz, columns_nodes, avg_z, normals):

    columns_points = {}
    columns_points_2d = {}

    for node in columns_nodes:

        idx = np.where((laz['classes'] == node.class_id) & (laz['objects'] == node.object_id))
        
        if len(idx[0]) > 0:

            columns_points[node.object_id] = np.vstack((laz.x[idx], laz.y[idx], laz.z[idx], np.asarray(normals)[idx, 0], np.asarray(normals)[idx, 1], np.asarray(normals)[idx, 2])).transpose() 

            # Enable this to filter the points at min - max height
            z_values = columns_points[node.object_id][:, 2]
            min_z = np.min(z_values)
            max_z = np.max(z_values)
            
            idx = np.where((laz['classes'] == node.class_id) & (laz['objects'] == node.object_id) & (laz.z > min_z + 0.1) & (laz.z < max_z - 0.1))

            # Place points at avg_z
            columns_points_2d[node.object_id] = np.vstack((laz.x[idx], laz.y[idx], np.full_like(laz.z[idx], avg_z), np.asarray(normals)[idx, 0], np.asarray(normals)[idx, 1], np.asarray(normals)[idx, 2])).transpose() 
        
    return columns_points, columns_points_2d

def compute_middle_points(segments):
    
    print(segments)

    # Middle points of each segment
    middle_points_list = []
    for segment_name, segment in segments.items():
        x_middle = (segment[0][0] + segment[1][0])/2
        y_middle = (segment[0][1] + segment[1][1]) / 2
        middle_points_list.append((x_middle, y_middle))

    print("Middle points of each segment (in a list):")
    for i, middle_point in enumerate(middle_points_list):
        segment_name = list(segments.keys())[i]
        print(f"Segment name: {segment_name}, Point name: {middle_point}, Middle points: {middle_point}")

    print("Coordinates assigned to points A1, B1, C1, D1:")
    print(f"A1: {middle_points_list[0]}")
    print(f"B1: {middle_points_list[1]}")
    print(f"C1: {middle_points_list[2]}")
    print(f"D1: {middle_points_list[3]}")
    
    return middle_points_list[0], middle_points_list[1], middle_points_list[2], middle_points_list[3]

def compute_best_coverage_segments(rotated_points, histogram_step, coordinate, key, output_file):

    cases_h = np.arange(min([p[coordinate] for p in rotated_points]), max([p[coordinate] for p in rotated_points]), histogram_step)

    case_b = cases_h [: len(cases_h) // 3]
    hist_b, case_b = np.histogram([p[coordinate] for p in rotated_points], bins=case_b)

    if len(hist_b) == 0:
        print(f'Column: {key} - could not compute features')
        output_file.write(f'Column: {key} - could not compute features\n')
        return None, False
       
    case_d = cases_h [2*len(cases_h) // 3:]
    hist_d, case_d =  np.histogram([p[coordinate] for p in rotated_points], bins=case_d)

    if len(hist_d) == 0:
        print(f'Column: {key} - could not compute features')
        output_file.write(f'Column: {key} - could not compute features\n')
        return None, False

    max_index_case_b = np.argmax(hist_b)
    max_index_case_d = np.argmax(hist_d)

    print("case_b", case_b)
    print("case_d", case_d)
    coord_b = case_b[max_index_case_b] + histogram_step / 2
    coord_d = case_d[max_index_case_d] + histogram_step / 2

    print("Coordinate of minimum value in case_b along {coordinate} axis:", coord_b)
    print("Coordinate of maximum value in case_d along {coordinate} axis:", coord_d)

    # New bbox
    new_coordinate = (coord_b, coord_b, coord_d, coord_d)
    return new_coordinate, True

# Compute bounding box function
def compute_bounding_box(points):
    
    pi2 = np.pi/2.

    # get the convex hull for the points
    hull_points = points[ConvexHull(points).vertices]

    # calculate edge angles
    edges = np.zeros((len(hull_points)-1, 2))
    edges = hull_points[1:] - hull_points[:-1]

    angles = np.zeros((len(edges)))
    angles = np.arctan2(edges[:, 1], edges[:, 0])
    angles = np.abs(np.mod(angles, pi2))
    angles = np.unique(angles)

    # find rotation matrices
    # XXX both work
    rotations = np.vstack([np.cos(angles), np.cos(angles-pi2), np.cos(angles+pi2),np.cos(angles)]).T
    rotations = rotations.reshape((-1, 2, 2))

    # Covariance matrix
    cov = np.cov(points, rowvar=False)

    # Eigenvalues and eigenvectors of the covariance matrix
    # Eigenvectors {x1,y1, x2, y2} directions of bboxes, Eigenvalues {i1, i2} magnitude of eigenvectors
    eigenvalues, eigenvectors = LA.eig(cov)
    
    # apply rotations to the hull
    rot_points = np.dot(rotations, hull_points.T)

    # find the bounding points
    min_x = np.nanmin(rot_points[:, 0], axis=1)
    max_x = np.nanmax(rot_points[:, 0], axis=1)
    min_y = np.nanmin(rot_points[:, 1], axis=1)
    max_y = np.nanmax(rot_points[:, 1], axis=1)

    # find the box with the best area
    areas = (max_x - min_x) * (max_y - min_y)
    best_idx = np.argmin(areas)

    # return the best box
    x1 = max_x[best_idx]
    x2 = min_x[best_idx]
    y1 = max_y[best_idx]
    y2 = min_y[best_idx]
    r = rotations[best_idx]

    rval = np.zeros((4, 2))
    rval[0] = np.dot([x1, y2], r)
    rval[1] = np.dot([x2, y2], r)
    rval[2] = np.dot([x2, y1], r)
    rval[3] = np.dot([x1, y1], r)

    return rval, r


# def round_rotation (rotation, round):
#     round = [0, 90, 180, 360]
#     rotation_rounded = rotation % round
#     return rotation_rounded

# Compute column_featrues function
def column_features(column_points, bbox, rotation_matrix, floors_z, ceilings_z):
    center_r = np.mean(bbox, axis = 0)
    center = np.append(center_r, 0)
    print ('Center hey:', center)
    
    # Vertices
    A = bbox[0]
    B = bbox[1]
    C = bbox[2]
    D = bbox[3]

    # Dimensions
    horizontal_segment = (A[0]-B[0], A[1]-B[1])
    width = np.sqrt(horizontal_segment[0]**2 + horizontal_segment[1]**2)
    vertical_segment = (C[0]-B[0], C[1]-B[1])
    depth = np.sqrt(vertical_segment[0]**2 + vertical_segment[1]**2)
    # Rotation
    angle = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    rotation = np.degrees(angle)
    
    # round = [0, 90, 180, 360]
    # rotation_rounded = round_rotation (rotation, round)

    #  z_values = np.array(points)
    z_values = column_points[:, 2] #np.concatenate([value[:, 2] for value in column_points.values()])
    min_z = np.min(z_values)
    max_z = np.max(z_values)

    # find closest floor_z to min_z
    closest_floor_index = -1    
    floor_distance = 10000  

    for floor_index, z in enumerate(floors_z):
        distance = abs(min_z - z)
        if distance < floor_distance:
            closest_floor_index = floor_index
            floor_distance = distance
            # print ('Floor_index:', floor_index)
            # print ('distance:', distance)
    
    closest_ceiling_index = -1
    ceiling_distance = 10000

    for ceiling_index, z in enumerate(ceilings_z):
        distance = abs(max_z - z)
        if distance < ceiling_distance:
            closest_ceiling_index = ceiling_index
            ceiling_distance = distance

    min_z = floors_z[closest_floor_index]
    max_z = ceilings_z[closest_ceiling_index]
    
    height_column = max_z - min_z
    print(f'Floor index:" {floor_index} "Floor offset: :" {floor_distance}')
    print(f'Ceiling index: {ceiling_index} Ceiling offset : {ceiling_distance}')
    print (f'Column height: ', {height_column})

    return width, depth, height_column, center, rotation, min_z, max_z

# Compute rotate_points function
def rotate_points(points, center, rotation_matrix):

        rotated_points = []

        for point in points: 
            shifted = point - center   
            rotated = np.dot(shifted, rotation_matrix)
            rotated_points.append(rotated + center)
        rotated_points = np.array(rotated_points)

        return rotated_points

def json_export(output_folder, name, key, width, depth, height_column, center, rotation_rounded):

    #json_file_path = os.path.join(output_folder, f'{ut.get_filename(name)}_columns.json')
    json_file_path = os.path.join(output_folder, '_'.join(ut.get_filename(name).split('_')[:4]) + '_columns.json')

    # Check if the file exists and read its content
    if os.path.exists(json_file_path):
        with open(json_file_path, "r") as json_file:
            json_file_data = json.load(json_file)
    else:
        json_file_data = []

    # Construct the new object data
    obj = {
        "id": key,
        "width": width,
        "depth": depth,
        "height": height_column,
        "loc": [center[0], center[1], center[2]],
        "rotation": 0 # 0 for cvpr, rotation_rounded otherwise
    }

    # Append the new object to the list of objects
    json_file_data.append(obj)

    # Write the updated JSON data to file
    with open(json_file_path, "w") as json_file:
        json.dump(json_file_data, json_file, indent=4)

    print("JSON data written to file:", json_file_path)

def column_mesh(points, floor_z, ceiling_z, height, minimum_bounding_box, output_folder, name):  
    base_edges = []  
    top_edges = []
    vertical_edges = []
    faces = []

    print ("Vertex obj base", floor_z)
    print ("Vertex obj end", ceiling_z)

    base_vertices = [
                [minimum_bounding_box[0][0], minimum_bounding_box[0][1], floor_z], 
                [minimum_bounding_box[1][0], minimum_bounding_box[1][1], floor_z], 
                [minimum_bounding_box[2][0], minimum_bounding_box[2][1], floor_z],
                [minimum_bounding_box[3][0], minimum_bounding_box[3][1], floor_z]
]
    print ('Base vertices: ', base_vertices)

    end_vertices = [
                    [minimum_bounding_box[0][0], minimum_bounding_box[0][1], ceiling_z], 
                    [minimum_bounding_box[1][0], minimum_bounding_box[1][1], ceiling_z], 
                    [minimum_bounding_box[2][0], minimum_bounding_box[2][1], ceiling_z],
                    [minimum_bounding_box[3][0], minimum_bounding_box[3][1], ceiling_z]
]

    base_vertices = np.array(base_vertices)
    print(base_vertices.shape)
    end_vertices = np.array(end_vertices)
    vertices = np.vstack ((base_vertices, end_vertices))

    A0 = np.array([minimum_bounding_box[0][0], minimum_bounding_box[0][1], floor_z]) 
    B0 = np.array([minimum_bounding_box[1][0], minimum_bounding_box[1][1], floor_z])
    C0 = np.array([minimum_bounding_box[2][0], minimum_bounding_box[2][1], floor_z])
    D0 = np.array([minimum_bounding_box[3][0], minimum_bounding_box[3][1], floor_z])

    A1 = np.array([minimum_bounding_box[0][0], minimum_bounding_box[0][1], ceiling_z]) 
    B1 = np.array([minimum_bounding_box[1][0], minimum_bounding_box[1][1], ceiling_z])
    C1 = np.array([minimum_bounding_box[2][0], minimum_bounding_box[2][1], ceiling_z])
    D1 = np.array([minimum_bounding_box[3][0], minimum_bounding_box[3][1], ceiling_z])

    base_edges= np.array([[A0, B0], [B0, C0], [C0, D0], [D0, A0]])
    top_edges = np.array([[A1, B1], [B1, C1], [C1, D1], [D1, A1]])
    vertical_edges = np.array([[A0, A1], [B0, B1], [C0, C1], [D0, D1]])
    edges = np.vstack((base_edges, top_edges, vertical_edges))

    # Compute faces
    face_a = np.array([A0, B1, A1])
    face_b = np.array([A0, B0, B1])
    face_c = np.array([B0, B1, C0])
    face_d = np.array([C0, C1, B1])
    face_e = np.array([C0, C1, D0])
    face_f = np.array([C1, D1, D0])
    face_g = np.array([A0, D0, D1])
    face_h = np.array([A0, A1, D1])

    # Faces
    faces = np.array((face_a, face_b, face_c, face_d, face_e, face_f, face_g, face_h))
    print ('Faces:', faces)
    print ('Faces:', faces.shape)
    
    # Plotting
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Plot edges
    color_edges = 'red'
    lw_edges = 0.25
    markersize_vertex = 2
    color_points = 'red'
    markersize_points = 0.001
    points_column = 'blue'
    
    # Plot base vertices
    points = np.concatenate([points], axis=0)
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], marker = 'o', color = points_column, s = markersize_points, alpha = 0.90)

    # for vertices in base_vertices:
    x = vertices[:, 0]
    y = vertices[:, 1]
    z = vertices[:, 2]
    ax.plot(x, y, z, marker='o', color = color_points, markersize = markersize_vertex)
   
    # Flatten the edges array
    x_edges = edges[:, :, 0].flatten()
    y_edges = edges[:, :, 1].flatten()
    z_edges = edges[:, :, 2].flatten()

    # Plot edges as scatter
    ax.scatter(x_edges, y_edges, z_edges, color= color_edges, lw = lw_edges)

    for face in faces:
        # Close the loop by repeating the first vertex
        face = np.append(face, [face[0]], axis=0)
        # Plot the face
        ax.plot(face[:, 0], face[:, 1], face[:, 2])

    # Set labels
    ax.set_xlabel('x label')
    ax.set_ylabel('y label')
    ax.set_zlabel('z label')
    plt.gca().set_aspect('equal', adjustable='box')

    # Show plot
    #plt.show()

    # Prepare file for obj file
    A0 = np.array(base_vertices[0]) #1
    B0 = base_vertices[1] #2
    C0 = base_vertices[2] #3
    D0 = base_vertices[3] #4
    A1 = end_vertices[0] #5
    B1 = end_vertices[1] #6
    C1 = end_vertices[2] #7
    D1 = end_vertices[3] #8
    print ('A:', A0)
    print (f'A0, {A0[0]} {A0[1]} {A0[2]}')

    faces_obj = [
        [3, 7, 8],
        [3, 8, 4],
        [1, 5, 6],
        [1, 6, 2],
        [7, 3, 2],
        [7, 2, 6],
        [4, 8, 5],
        [4, 5, 1],
        [8, 7, 6],
        [8, 6, 5],
        [3, 4, 1],
        [3, 1, 2]
    ]

    file_name = output_folder / f"{name}.obj"

   # Create file
    with open(file_name, "w") as f:
        for v in vertices:
            f.write(f'v {v[0]:.3f} {v[1]} {v[2]}\n')
        # Write faces
        for face in faces_obj:
        # Convert face vertices to strings without brackets
            face_str = ' '.join([str(v) for v in face])
            f.write(f'f {face_str}\n')
    print ("Obj correctly generated!")




## ___________________________________________________________________________________________________________________
  # Function to read the graph
def load_TEST_graph(laz, graph_path):
    # Parse the graph
    graph = Graph().parse(str(graph_path))
    nodes = tl.graph_to_nodes(graph)  
    
    # Filter nodes for column nodes
    column_nodes = [n for n in nodes if 'columns' in n.subject.lower() and isinstance(n, PointCloudNode)]
    
    if column_nodes:
        for n in column_nodes:
                idx = np.where(( laz ['classes']==n.class_id) & ( laz ['objects'] == n.object_id))
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(laz.xyz[idx])
                n.resource = pcd
                print(f'{len(column_nodes)} columnNodes detected!')
                return graph, nodes, column_nodes
        
    else:
        print('No columns detected')
        #return graph, nodes, []
       
        return graph, nodes, column_nodes
    


