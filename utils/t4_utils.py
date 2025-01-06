
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
from typing import Tuple, List, Optional,Dict,Any
import copy

@timer
def fit_planes(point_cloud, distance_threshold=0.05, ransac_n=3, num_iterations=1000, min_inliers=1000,eps=0.5,min_cluster_points=200):
    """
    Segments a point cloud into planes using the RANSAC algorithm, continuing until no more planes with sufficient
    inliers can be found.

    Parameters:
        point_cloud (o3d.geometry.PointCloud): The point cloud from which to segment planes.
        distance_threshold (float): The maximum distance a point can be from the plane to be considered an inlier.
        ransac_n (int): The number of initial points to consider for a plane model.
        num_iterations (int): The number of iterations the RANSAC algorithm should run.
        min_inliers (int): The minimum number of inliers required for a plane to be considered valid.

    Returns:
        list(planes,pcds): A list of plane models where each plane model is defined by its coefficients.

    Raises:
        ValueError: If the input point cloud does not have the required methods.
    """
    #copy the point cloud
    pcd = point_cloud
    planes = []
    pcds=[]
    a=0
    while True:
        #fit plane
        plane_model, inliers = pcd.segment_plane(distance_threshold=distance_threshold, ransac_n=ransac_n, num_iterations=num_iterations)
        new_pcd = pcd.select_by_index(inliers) 
        #cluster inliers
        labels = np.array(new_pcd.cluster_dbscan(eps=eps, min_points=min_cluster_points))
        #filter out planes with less than min_inliers
        if len(np.array(inliers)[np.where(labels!=-1)[0]]) >= min_inliers:
            for u in np.unique(labels):
                if u != -1:                        
                    #select by index
                    inlier_pcd= pcd.select_by_index(np.array(inliers)[np.where(labels==u)[0]])
                    #add plane and point cloud
                    planes.append(plane_model)
                    pcds.append(inlier_pcd)
                
            #remove inliers that are already clustered
            inliers = np.delete(inliers, np.where(labels == -1)[0])
            pcd = pcd.select_by_index(inliers, invert=True)
                
        else:
            a+=1
            if a>10: # stop after 10 iterations of not finding planes
                break
            
    return planes,pcds

def split_point_cloud_in_planar_clusters(point_cloud, 
                                    distance_threshold=0.05, 
                                    sample_resolution=None, 
                                    ransac_n=3, 
                                    num_iterations=1000, 
                                    min_inliers=1000, 
                                    eps=0.5, 
                                    min_cluster_points=200,
                                    threshold_clustering_distance=0.7,
                                    threshold_clustering_normals=0.9):

    clustered_pcds=[]
    clustered_planes=[]
    
    #sample the point cloud (optional)
    point_cloud=gmu.sample_geometry(point_cloud, resolution=sample_resolution)[0] if sample_resolution is not None else point_cloud

    #fit planes
    planes,pcds=fit_planes(point_cloud,  
                            distance_threshold=distance_threshold,                                   
                                min_inliers=min_inliers,
                                min_cluster_points=min_cluster_points,
                                num_iterations=num_iterations,
                                ransac_n=ransac_n,
                                eps=eps)
    # print(f"Found {len(planes)} planes in {len(pcds)} point clouds")
    
    #also sort the planes                                                    
    sorted_list = sorted(zip(pcds,planes), key=lambda x: len(np.asarray(x[0].points)),reverse=True)
    
    counter=0        
    while len(sorted_list) not in [0,1]:
        n=sorted_list[0]
        sorted_list.pop(0) 
        
        # retrieve neirest neighbors of same class_id
        joined_pcd,identityArray=gmu.create_identity_point_cloud([pcd for pcd, _ in sorted_list])       #this is a little bit silly
        indices,distances=gmu.compute_nearest_neighbors(np.asarray(n[0].points),np.asarray(joined_pcd.points),n=1) 
        indices=indices[:,0]
        distances=distances[:,0]

                
        #filter on distance
        indices=indices[(distances<threshold_clustering_distance)]        
        #check if there are any indices left
        if len(indices)==0:
            clustered_pcds.append(n[0])
            clustered_planes.append(n[1])
            continue
        
        #filter on normal similarity        
        planes=np.array([plane[:3] for _, plane in sorted_list])        
        normals=np.array([p for p in planes[identityArray[indices].astype(int)]])        
        dotproducts = np.abs(np.einsum('ij,...j->...i', np.array([n[1][:3]]), normals))[:,0]    
        indices=indices[(dotproducts>threshold_clustering_normals)]
        #and check again
        if len(indices)==0:
            clustered_pcds.append(n[0])
            clustered_planes.append(n[1])
            continue
        
        #merge the point clouds
        unique=np.unique(identityArray[indices])
        if len(unique)>0:                
            pcd=gmu.join_geometries([n[0]]+[p[0] for i,p in enumerate(sorted_list) if i in unique])
            clustered_pcds.append(pcd)
            clustered_planes.append(n[1])
        else:
            clustered_pcds.append(n[0])
            clustered_planes.append(n[1])

        #increase counter
        counter+=1
        
        #delete elements from sorted_objectPcdNodes if they are in unique
        indices_to_remove = sorted(set(unique), reverse=True)
        [sorted_list.pop(idx) for idx in indices_to_remove]
    
    #add remaining elements
    if len(sorted_list)!=0:
        for n in sorted_list:
            clustered_pcds.append(n[0])
            clustered_planes.append(n[1])

    return clustered_pcds,clustered_planes

@timer
def split_point_cloud_in_planar_clusters2(
    point_cloud: o3d.geometry.PointCloud, 
    distance_threshold: float = 0.05, 
    sample_resolution: Optional[float] = None, 
    ransac_n: int = 3, 
    num_iterations: int = 1000, 
    min_inliers: int = 1000, 
    eps: float = 0.5,
    scale: float = 0.9,
    min_cluster_points: int = 200,
    threshold_clustering_distance: float = 0.7,
    threshold_clustering_normals: float = 0.9
) -> Tuple[List[o3d.geometry.PointCloud], List[np.ndarray]]:
    """
    Segments a point cloud into clusters based on planar segmentation and merges clusters based on distance and normal similarity.
    
    Parameters:
        point_cloud (o3d.geometry.PointCloud): Input point cloud.
        distance_threshold (float): Threshold distance for considering inliers in RANSAC.
        sample_resolution (Optional[float]): Resolution for downsampling the point cloud.
        ransac_n (int): Number of random points to estimate the plane model.
        num_iterations (int): Number of iterations for RANSAC.
        min_inliers (int): Minimum inliers for a cluster to be valid.
        eps (float): Epsilon value for DBSCAN clustering.
        min_cluster_points (int): Minimum points to form a cluster in DBSCAN.
        threshold_clustering_distance (float): Distance threshold for merging clusters.
        threshold_clustering_normals (float): Normal similarity threshold for merging clusters.

    Returns:
        Tuple[List[o3d.geometry.PointCloud], List[np.ndarray]]: A tuple containing clusters of point clouds and their respective plane models.
    """
    clustered_pcds=[]
    clustered_planes=[]
    clustered_plane_meshes=[]
    
    
    #sample the point cloud (optional)
    point_cloud=gmu.sample_geometry(point_cloud, resolution=sample_resolution)[0] if sample_resolution is not None else point_cloud

    #1.fit planes to create a set of potential clusters
    planes,pcds=fit_planes(point_cloud,  
                            distance_threshold=distance_threshold,                                   
                                min_inliers=min_inliers,
                                min_cluster_points=min_cluster_points,
                                num_iterations=num_iterations,
                                ransac_n=ransac_n,
                                eps=eps)
    
    #2.next, we will cluster the planes based on distance and normal similarity (starting from the largest plane)
    planar_meshes=[create_plane_mesh(pl[:3],np.asarray(pcd.points)) for pl,pcd in zip(planes,pcds)]
    
    #sort the planes starting from the largest one                                                   
    sorted_list = sorted(zip(pcds,planes,planar_meshes), key=lambda x: len(np.asarray(x[0].points)),reverse=True)
    # print([plane for _, plane,plane_mesh in sorted_list if np.abs(np.dot(sorted_list[0][1][:3],plane[:3]))<0.3])
    
    # counter=0        
    while len(sorted_list) not in [0,1]:
        n=sorted_list[0]
        sorted_list.pop(0)         
        
        #schrink planar mesh by 10% and only use points that are in the mesh
        local_mesh=scale_plane_mesh(create_plane_mesh(n[1][:3],np.asarray(n[0].points)),scale,scale)
        inliers, _=split_plane_mesh_points_inliers_outliers(local_mesh, np.asarray(n[0].points))        
        
        # retrieve neirest neighbors of same class_id
        joined_pcd,identityArray=gmu.create_identity_point_cloud([pcd for pcd, _,_ in sorted_list])       #this is a little bit silly
        indices,distances=gmu.compute_nearest_neighbors(inliers,np.asarray(joined_pcd.points),n=1) 
        indices=indices[:,0]
        distances=distances[:,0]
        
        #filter on distance
        inliers=inliers[(distances<threshold_clustering_distance)]
        indices=indices[(distances<threshold_clustering_distance)]        
        #check if there are any indices left
        if len(indices)==0:
            clustered_pcds.append(n[0])
            clustered_planes.append(n[1])
            clustered_plane_meshes.append(scale_plane_mesh(create_plane_mesh(n[1][:3],np.asarray(n[0].points)),scale,scale))
            continue
        
        #filter on normal similarity        
        planes=np.array([plane[:3] for _, plane,_ in sorted_list])        
        normals=np.array([p for p in planes[identityArray[indices].astype(int)]])        
        dotproducts = np.abs(np.einsum('ij,...j->...i', np.array([n[1][:3]]), normals))[:,0]    
        inliers=inliers[(dotproducts>threshold_clustering_normals)]
        indices=indices[(dotproducts>threshold_clustering_normals)]
        #and check again
        if len(indices)==0:
            clustered_pcds.append(n[0])
            clustered_planes.append(n[1])
            clustered_plane_meshes.append(scale_plane_mesh(create_plane_mesh(n[1][:3],np.asarray(n[0].points)),scale,scale))
            continue
        
        #filter based on collisions with other planes
        
        #CENTER TO CENTER APPROACH
        # target_points=np.array([sorted_list[i][0].get_center() for i in identityArray[indices]])
        # target_points=np.array([p[0].get_center() for i,p in enumerate(sorted_list) if i in list(np.unique(identityArray[indices]))])        
        # center = n[0].get_center().reshape(1, 3)
        # centers = np.repeat(center, target_points.shape[0], axis=0)
        # directions=target_points-n[0].get_center()
        # directions=directions/np.linalg.norm(directions,axis=1)[:,None]
        
        #INDIVIDUAL POINTS APPROACH
        #create rays between the inliers and their target points
        target_points=np.asarray(joined_pcd.points)[indices]
        directions=target_points-inliers
        # directions=directions/np.linalg.norm(directions,axis=1)[:,None]
        rays=np.hstack((inliers,directions))
        rays=o3d.core.Tensor(rays,dtype=o3d.core.Dtype.Float32)
        
        #create raycasting scene  
        #join all planar meshes that are near orthogonal to the plane        
        joined_plane_meshes=gmu.join_geometries([plane_mesh for _, plane,plane_mesh in sorted_list if np.abs(np.dot(n[1][:3],plane[:3]))<0.3]+ #old meshes
                                                [plane_mesh for plane, plane_mesh in zip(clustered_planes,clustered_plane_meshes) if np.abs(np.dot(n[1][:3],plane[:3]))<0.3])  # newly clustered meshes
        if joined_plane_meshes:    
            scene = o3d.t.geometry.RaycastingScene()    
            reference=o3d.t.geometry.TriangleMesh.from_legacy(joined_plane_meshes)
            scene.add_triangles(reference)
            #compute raycasting
            ans = scene.cast_rays(rays)  #collisions don't consider backside of planes!
            # check if vector does not collide with other planes            
            inliers=inliers[(ans['geometry_ids'].numpy()!=0)] #| (ans['t_hit'].numpy()*1/scale>threshold_clustering_distance)]
            indices=indices[(ans['geometry_ids'].numpy()!=0)]#| (ans['t_hit'].numpy()*1/scale>threshold_clustering_distance)]            
            directions=directions[(ans['geometry_ids'].numpy()!=0)]#| (ans['t_hit'].numpy()*1/scale>threshold_clustering_distance)]
            
            # # Create a boolean mask based on identityArray and ans['geometry_ids']
            # valid_ids = np.where(ans['geometry_ids'].numpy() == 0)[0]
            # valid_ids_mask = ~np.isin(identityArray[indices], valid_ids)
            # # Filter inliers, indices, and directions based on the mask
            # inliers = inliers[valid_ids_mask]
            # indices = indices[valid_ids_mask]
            
            # directions = directions[ans['geometry_ids'].numpy()!=0]
            # centers = centers[ans['geometry_ids'].numpy()!=0]
    
            #and check inliers again
            if len(indices)==0:
                clustered_pcds.append(n[0])
                clustered_planes.append(n[1])
                clustered_plane_meshes.append(scale_plane_mesh(create_plane_mesh(n[1][:3],np.asarray(n[0].points)),scale,scale))
                continue
            # if directions.shape[0]>1 and centers.shape[0]>1:
            #     rays=np.hstack((centers,directions))
            #     rays=o3d.core.Tensor(rays,dtype=o3d.core.Dtype.Float32)
            #     lines=gmu.rays_to_lineset(rays).paint_uniform_color([1,0,0])
            #     o3d.visualization.draw_geometries([lines,n[0].paint_uniform_color([0,0,1]),joined_plane_meshes.paint_uniform_color([0,1,0]),point_cloud.paint_uniform_color([0,0,0])])
         
        #merge the point clouds
        unique=np.unique(identityArray[indices])            
        pcd=gmu.join_geometries([n[0]]+[p[0] for i,p in enumerate(sorted_list) if i in unique])     
               
        # o3d.visualization.draw_geometries([n[0].paint_uniform_color([0,0,1]),pcd.paint_uniform_color([0,1,0]),point_cloud.paint_uniform_color([0,0,0])])
           
        #delete elements from sorted_objectPcdNodes if they are in unique
        indices_to_remove = sorted(set(unique), reverse=True)
        [sorted_list.pop(idx) for idx in indices_to_remove]
        
        #reinsert the current cluster
        sorted_list.insert(0,(pcd,n[1],scale_plane_mesh(create_plane_mesh(n[1][:3],np.asarray(pcd.points)),scale,scale)))
    
    #add remaining elements
    if len(sorted_list)!=0:
        for n in sorted_list:
            clustered_pcds.append(n[0])
            clustered_planes.append(n[1])
            clustered_plane_meshes.append(scale_plane_mesh(create_plane_mesh(n[1][:3],np.asarray(n[0].points)),scale,scale))


    return clustered_pcds,clustered_planes,clustered_plane_meshes


def split_plane_mesh_points_inliers_outliers(plane_mesh, points):
    
    vertices = np.asarray(plane_mesh.vertices)
    centroid = np.mean(vertices, axis=0)
    u = vertices[1] - vertices[0]
    v = vertices[2] - vertices[0]
    
    u /= np.linalg.norm(u)
    v /= np.linalg.norm(v)
    
    relative_positions = points - centroid
    proj_u = np.dot(relative_positions, u)
    proj_v = np.dot(relative_positions, v)
    
    min_u = np.min(np.dot(vertices - centroid, u))
    max_u = np.max(np.dot(vertices - centroid, u))
    min_v = np.min(np.dot(vertices - centroid, v))
    max_v = np.max(np.dot(vertices - centroid, v))
    
    inlier_mask = (min_u <= proj_u) & (proj_u <= max_u) & (min_v <= proj_v) & (proj_v <= max_v)
    inliers = points[inlier_mask]
    outliers = points[~inlier_mask]
    
    return inliers, outliers

def scale_plane_mesh(plane_mesh, scale_u, scale_v):
    vertices = np.asarray(plane_mesh.vertices)
    centroid = np.mean(vertices, axis=0)

    if len(vertices) < 4:
        raise ValueError("The plane mesh must have at least 4 vertices.")

    u = vertices[1] - vertices[0]
    v = vertices[2] - vertices[0]

    u /= np.linalg.norm(u)
    v /= np.linalg.norm(v)

    for i in range(len(vertices)):
        relative_pos = vertices[i] - centroid
        proj_u = np.dot(relative_pos, u)
        proj_v = np.dot(relative_pos, v)

        vertices[i] = centroid + scale_u * proj_u * u + scale_v * proj_v * v

    plane_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    plane_mesh.compute_vertex_normals()
    return plane_mesh

def create_plane_mesh(normal, points):
    # Normalize the normal vector
    normal = normal / np.linalg.norm(normal)

    # Compute the centroid of the points
    centroid = np.mean(points, axis=0)

    # Find two orthogonal vectors on the plane
    if np.allclose(normal, [0, 0, 1]) or np.allclose(normal, [0, 0, -1]):
        u = np.array([1, 0, 0])
    else:
        u = np.cross(normal, [0, 0, 1])
        u /= np.linalg.norm(u)
    v = np.cross(normal, u)

    # Project points onto the plane to get the bounding rectangle
    proj_points = points - centroid
    proj_u = np.dot(proj_points, u)
    proj_v = np.dot(proj_points, v)

    min_u, max_u = np.min(proj_u), np.max(proj_u)
    min_v, max_v = np.min(proj_v), np.max(proj_v)

    # Define the corners of the rectangle
    corners = [
        centroid + min_u * u + min_v * v,
        centroid + min_u * u + max_v * v,
        centroid + max_u * u + min_v * v,
        centroid + max_u * u + max_v * v
    ]

    # Create the mesh from the corners
    vertices = o3d.utility.Vector3dVector(corners)
    triangles = o3d.utility.Vector3iVector([
        [0, 1, 2],
        [2, 1, 3]
    ])

    plane_mesh = o3d.geometry.TriangleMesh(vertices, triangles)
    plane_mesh.compute_vertex_normals()
    return plane_mesh

def split_point_cloud_by_dbscan(class_pcd,eps=0.2,min_cluster_points=100):
    pcds=[]
    labels = np.array(class_pcd.cluster_dbscan(eps=eps, min_points=min_cluster_points))
    for u in np.unique(labels):
        if u != -1:      
            #get indices 
            indices=np.where(labels==u)[0]                 
            if indices.shape[0]>min_cluster_points:
                #select by index
                inlier_pcd= class_pcd.select_by_index(indices)
                pcds.append(inlier_pcd)
    return pcds if len(pcds)>0 else [pcds]
  
  
def create_floor_and_ceiling_nodes(
    las,
    pcd: o3d.geometry.PointCloud,
    f: str,
    c: dict,
    sample_resolution: float = 0.05,
    distance_threshold: float = 0.05,
    min_inliers: int = 1000,
    eps: float = 0.5,
    min_cluster_points: int = 200
) -> List[PointCloudNode]:
    """
    Creates nodes for floors and ceilings from point cloud data based on the provided class info.
    
    Parameters:
        las: LAS file data containing point classes.
        pcd (o3d.geometry.PointCloud): The point cloud to process.
        f (str): The name of the file being processed.
        c (dict): Information about the point class to process.
        sample_resolution (float): Resolution for downsampling.
        distance_threshold (float): Distance threshold for plane segmentation.
        min_inliers (int): Minimum number of inliers for a plane.
        eps (float): Epsilon value for DBSCAN.
        min_cluster_points (int): Minimum points in a cluster.

    Returns:
        List: A list of nodes representing floors and ceilings.
    """
    #select points of class    
    idx=np.where((las['classes']==c['id']))[0]    
    class_pcd=pcd.select_by_index(idx)
    
    #retrieve objects from planar clusters
    clustered_pcds,clustered_planes=split_point_cloud_in_planar_clusters(class_pcd,
                                                                            sample_resolution=sample_resolution,
                                                                            distance_threshold=distance_threshold, 
                                                                            min_inliers=min_inliers,
                                                                            eps=eps,
                                                                            min_cluster_points=min_cluster_points)
    #create nodes
    nodes=[]
    for i,n in enumerate(clustered_pcds):
        try: # planar point clouds generate an oriented bounding box error
            dim=n.get_oriented_bounding_box() 
            nodes.append(PointCloudNode(resource=n,
                                    class_id=c['id'],
                                    class_name=c['name'],
                                    object_id=i+1+c['id']*1000, #+1 because 0 is clutter
                                    plane=clustered_planes[i],
                                    color=ut.random_color(),
                                    name=ut.get_filename(f)+'_'+c['name']+'_'+str(i+1+c['id']*1000)))     
        except:
            dim=n.get_axis_aligned_bounding_box()
            nodes.append(PointCloudNode(resource=n,
                                    class_id=c['id'],
                                    class_name=c['name'],
                                    object_id=i+1+c['id']*1000, #+1 because 0 is clutter
                                    orientedBoundingBox=dim,
                                    plane=clustered_planes[i],
                                    color=ut.random_color(),
                                    name=ut.get_filename(f)+'_'+c['name']+'_'+str(i+1+c['id']*1000)))     
       

    return nodes

def create_wall_nodes(
    las: Any, 
    pcd: o3d.geometry.PointCloud, 
    f: str, 
    c: Dict, 
    sample_resolution: float = 0.03,
    distance_threshold: float = 0.03, 
    min_inliers: int = 200,
    eps: float = 0.5,
    threshold_min_cluster_points: int = 200,
    size: List[int] = [12, 12, 100],
    threshold_wall_verticality: float = 0.1,
    threshold_wall_dim: float = 0.5,
    threshold_clustering_distance: float = 0.6
) -> Tuple[List[PointCloudNode], o3d.geometry.PointCloud]:
    """
    Creates wall nodes from point cloud data by classifying and clustering wall segments.

    Parameters:
        las (Any): LAS file data containing point classes.
        pcd (o3d.geometry.PointCloud): The point cloud to process.
        f (str): Name of the file.
        c (Dict): Information about the point class to process (ID and name).
        sample_resolution (float): Resolution for downsampling.
        distance_threshold (float): Distance threshold for plane segmentation.
        min_inliers (int): Minimum number of inliers for a plane.
        eps (float): Epsilon value for DBSCAN.
        min_cluster_points (int): Minimum points in a cluster.
        size (List[int]): Dimensions to split the point cloud into smaller boxes.
        threshold_wall_verticality (float): Verticality threshold to determine if a plane can be considered a wall.
        threshold_wall_dim (float): Dimension threshold for size of the wall.
        threshold_clustering_distance (float): Distance threshold for merging clusters.

    Returns:
        Tuple[List[Dict], o3d.geometry.PointCloud]: A tuple containing a list of wall nodes and the remaining point cloud.
    """
    #initiliaze counter and rest point cloud
    counter=0
    rest_pcd=o3d.geometry.PointCloud()
    
    #select points of class  
    idx=np.where((las['classes']==c['id']))[0]
    class_pcd=pcd.select_by_index(idx)
    
    #split into boxes (otherwise you will get horizontal planes mostly)
    min_bound=pcd.get_min_bound()
    max_bound=pcd.get_max_bound()            
    box=o3d.geometry.AxisAlignedBoundingBox(min_bound=np.array([min_bound[0]-5,min_bound[1]-5,min_bound[2]-5]),
                                max_bound=np.array([max_bound[0]+5,max_bound[1]+5,max_bound[2]+5]))
    boxes,_=gmu.divide_box_in_boxes(box,size=size)
    
    # select indices per boxes
    sub_pcds=[]
    nodes=[]
    for box in boxes:
        idxLists=box.get_point_indices_within_bounding_box(class_pcd.points)
        sub_pcd=class_pcd.select_by_index(idxLists)
    
        #retrieve objects from planar clusters
        if len(np.asarray(sub_pcd.points))<threshold_min_cluster_points:
            continue
        clustered_pcds,clustered_planes,clustered_plane_meshes=split_point_cloud_in_planar_clusters2(sub_pcd,
                                                                            sample_resolution=sample_resolution,
                                                                            distance_threshold=distance_threshold, 
                                                                            min_inliers=min_inliers,
                                                                            eps=eps,
                                                                            min_cluster_points=threshold_min_cluster_points,
                                                                            threshold_clustering_distance=threshold_clustering_distance)
        #create nodes
        if len(clustered_pcds)!=0:
            for p,pl in zip(clustered_pcds,clustered_planes):
                try:
                    dim=p.get_oriented_bounding_box().extent
                except:
                    dim=p.get_axis_aligned_bounding_box().get_extent()
                #check if the plane is vertical enough and large enough
                if (np.abs(pl[2])<threshold_wall_verticality) & (dim[0]>threshold_wall_dim) & (dim[1]>threshold_wall_dim):                
                    counter+=1
                    nodes.append(PointCloudNode(resource=p,
                                                class_id=c['id'],
                                                class_name=c['name'],
                                                object_id=c['id']*1000+counter, 
                                                plane=pl,
                                                color=ut.random_color(),
                                                name=ut.get_filename(f)+'_'+c['name']+'_'+str(c['id']*1000+counter)))       
                else: #add to rest
                    rest_pcd+=p
    
    
    # nodes=merge_wall_nodes(nodes,threshold_clustering_distance=0.3*threshold_clustering_distance)
        
    return nodes,rest_pcd



def merge_wall_nodes(
    nodes: List[PointCloudNode],
    threshold_clustering_distance: float = 0.7,
    threshold_clustering_normals: float = 0.9
) -> List[PointCloudNode]:
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
    
    #sort the nodes from largest to smallest                                                    
    sorted_list = sorted(nodes, key=lambda x: len(np.asarray(x.resource.points)),reverse=True)
    
    counter=0        
    while len(sorted_list) not in [0,1]:
        n=sorted_list[0]
        sorted_list.pop(0) 
        
        # retrieve neirest neighbors of same class_id
        joined_pcd,identityArray=gmu.create_identity_point_cloud([p.resource for p in sorted_list])       #this is a little bit silly
        indices,distances=gmu.compute_nearest_neighbors(np.asarray(n.resource.points),np.asarray(joined_pcd.points),n=1) 
        indices=indices[:,0]
        distances=distances[:,0]

                
        #filter on distance
        indices=indices[(distances<threshold_clustering_distance)]        
        #check if there are any indices left
        if len(indices)==0:
            clustered_nodes.append(n)
            continue
        
        #filter on normal similarity        
        planes=np.array([p.plane[:3] for p in sorted_list])        
        normals=np.array([p for p in planes[identityArray[indices].astype(int)]])        
        dotproducts = np.abs(np.einsum('ij,...j->...i', np.array([n.plane[:3]]), normals))[:,0]    
        indices=indices[(dotproducts>threshold_clustering_normals)]
        #and check again
        if len(indices)==0:
            clustered_nodes.append(n)
            continue
        
        #merge the point clouds
        unique=np.unique(identityArray[indices])
        if len(unique)>0:                
            pcd=gmu.join_geometries([n.resource]+[p.resource for i,p in enumerate(sorted_list) if i in unique])
            n.resource=pcd
        clustered_nodes.append(n)

        #increase counter
        counter+=1
        
        #delete elements from sorted_objectPcdNodes if they are in unique
        indices_to_remove = sorted(set(unique), reverse=True)
        [sorted_list.pop(idx) for idx in indices_to_remove]
    
    #add remaining elements
    if len(sorted_list)!=0:
        for n in sorted_list:
            clustered_nodes.append(n)
    
    return clustered_nodes

def merge_wall_nodes2(
    nodes: List[PointCloudNode],
    threshold_clustering_distance: float = 0.7,
    threshold_clustering_normals: float = 0.9,
    threshold_clustering_orthogonal_distance: float = 0.1,
    threshold_clustering_coplanar_distance: float = 5
) -> List[PointCloudNode]:
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
    
    #sort the nodes from largest to smallest                                                    
    sorted_list = sorted(nodes, key=lambda x: len(np.asarray(x.resource.points)),reverse=True)
    
    while len(sorted_list) not in [0,1]:
        n=sorted_list[0]
        sorted_list.pop(0) 
        
        # retrieve neirest neighbors of same class_id
        inliers=np.asarray(n.resource.points)
        joined_pcd,identityArray=gmu.create_identity_point_cloud([p.resource for p in sorted_list])       #this is a little bit silly
        indices,distances=gmu.compute_nearest_neighbors(inliers,np.asarray(joined_pcd.points),n=1) 
        indices=indices[:,0]
        distances=distances[:,0]

        
        #filter on normal similarity        
        planes=np.array([p.plane[:3] for p in sorted_list])        
        normals=np.array([p for p in planes[identityArray[indices].astype(int)]])        
        dotproducts = np.abs(np.einsum('ij,...j->...i', np.array([n.plane[:3]]), normals))[:,0]    
        indices=indices[(dotproducts>threshold_clustering_normals)]
        inliers=inliers[(dotproducts>threshold_clustering_normals)]
        distances=distances[(dotproducts>threshold_clustering_normals)]
        #and check again
        if len(indices)==0:
            clustered_nodes.append(n)
            continue
     
        #filter on distance    or  orthogonal distance
        orthogonal_distance = np.abs(np.einsum('ij,...j->...i', np.array([n.plane[:3]]), np.asarray(inliers)-np.asarray(joined_pcd.points)[indices] ))[:,0]    
        indices=indices[((orthogonal_distance<threshold_clustering_orthogonal_distance) & (distances<threshold_clustering_coplanar_distance)) |
                        (distances<threshold_clustering_distance)]        
        #check if there are any indices left
        if len(indices)==0:
            clustered_nodes.append(n)
            continue
        
    
        #merge the point clouds
        unique=np.unique(identityArray[indices])            
        pcd=gmu.join_geometries([n.resource]+[p.resource for i,p in enumerate(sorted_list) if i in unique])     
        n.resource=pcd
               
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




def create_thrash_node(
    las: Any, 
    pcd: o3d.geometry.PointCloud, 
    f: str, 
    c: Dict
) -> PointCloudNode:
    """
    Creates a node for trash class points in a point cloud, which typically include points not belonging to any specific class.

    Parameters:
        las (Any): LAS file data containing point classes and other metadata.
        pcd (o3d.geometry.PointCloud): The point cloud from which to select trash points.
        f (str): Name of the LAS file, used for naming the node.
        c (Dict): Dictionary containing information about the class, particularly the class ID and name.

    Returns:
        Dict: A dictionary representing the trash node, including the point cloud resource, class details, and node name.
    """
    #select points of class    
    idx=np.where((las['classes']==c['id']))[0]
    class_pcd=pcd.select_by_index(idx)
    thrashNode=PointCloudNode(resource=class_pcd,
                                    class_id=c['id'],
                                    class_name=c['name'],
                                    object_id=0,
                                    color=ut.random_color(),
                                    name=ut.get_filename(f)+'_'+c['name']+'_0')
    return thrashNode

def create_column_nodes(
    las: Any, 
    pcd: o3d.geometry.PointCloud, 
    f: str, 
    c: Dict,
    eps: float = 1.0,
    min_cluster_points: int = 200,
    threshold_column_verticality: float = 0.1,
    threshold_column_height: float = 1.5,
    threshold_column_points: int = 1000
) -> Tuple[List[PointCloudNode], o3d.geometry.PointCloud]:
    """
    Creates column nodes from a point cloud based on specified class information and geometric thresholds.

    Parameters:
        las (Any): LAS file data containing point classes and other metadata.
        pcd (o3d.geometry.PointCloud): The point cloud from which to select column points.
        f (str): Name of the LAS file, used for node naming.
        c (Dict): Dictionary containing information about the class, particularly the class ID and name.
        eps (float): DBSCAN parameter for the maximum distance between two samples for them to be considered as in the same neighborhood.
        min_cluster_points (int): Minimum number of points in a cluster for DBSCAN.
        threshold_column_verticality (float): Verticality threshold for determining if a cluster can be considered a column.
        threshold_column_height (float): Minimum height of a cluster to be considered a column.
        threshold_column_points (int): Minimum number of points a cluster must have to be considered a column.

    Returns:
        Tuple[List[Dict], o3d.geometry.PointCloud]: A tuple containing a list of column nodes and the remaining point cloud.
    """
    nodes=[]
    rest_pcd=o3d.geometry.PointCloud()

    #select points of class    
    idx = np.where((las['classes'] == c['id']))[0]
    class_pcd=pcd.select_by_index(idx)
    object_labels=las['objects'][idx]
    
    #split clusters by distance
    object_pcds=[]
    for i in np.unique(object_labels): 
        if i==0:
            potential_object_pcds=split_point_cloud_by_dbscan(class_pcd.select_by_index(np.where(object_labels==i)[0]),eps=eps,min_cluster_points=min_cluster_points)
            #retain only clusters with sufficient height
            object_pcds.extend([p for p in potential_object_pcds])
        else:           
            object_pcds.append(class_pcd.select_by_index(np.where(object_labels==i)[0]))
    
    for i,p in enumerate(object_pcds):    
        nodes.append(PointCloudNode(resource=p,
                                    class_id=c['id'],
                                    class_name=c['name'],
                                    object_id=c['id']*1000+i,
                                    color=ut.random_color(),
                                    name=ut.get_filename(f)+'_'+c['name']+'_'+str(c['id']*1000+i)))
    
    #filter columns that are too small, too low or too close to each other
    newNodes=[]
    while len(nodes)>0:         
        n=nodes.pop(0)
        
        #scrap horizontal columns
        n.resource=n.resource.select_by_index(np.where(np.asarray(n.resource.normals)[:,2]<threshold_column_verticality)[0])
        box=n.resource.get_axis_aligned_bounding_box()
        
        if len(np.asarray(n.resource.points))>threshold_column_points and (box.max_bound[2]-box.min_bound[2])>threshold_column_height:
            merged=False
            for m in nodes:
                if n!=m and len(np.asarray(n.resource.points))<len(np.asarray(m.resource.points)) and np.linalg.norm(n.cartesianTransform[0:3,3] - m.cartesianTransform[0:3,3]) < eps:
                    m.resource+=n.resource
                    merged=True
                    break    
            if not merged:
                newNodes.append(n) 
                        
        else:
            rest_pcd+=n.resource
                    
    return newNodes,rest_pcd

def create_door_nodes(
    las: Any, 
    pcd: o3d.geometry.PointCloud, 
    f: str, 
    c: Dict,
    eps: float = 0.5,
    min_cluster_points: int = 200,
    threshold_door_dim: float = 0.5
) -> Tuple[List[PointCloudNode], o3d.geometry.PointCloud]:
    """
    Creates door nodes from a point cloud based on specified class information and geometric thresholds.

    Parameters:
        las (Any): LAS file data containing point classes and other metadata.
        pcd (o3d.geometry.PointCloud): The point cloud from which to select door points.
        f (str): Name of the LAS file, used for node naming.
        c (Dict): Dictionary containing information about the class, particularly the class ID and name.
        eps (float): DBSCAN parameter for the maximum distance between two samples for them to be considered as in the same neighborhood.
        min_cluster_points (int): Minimum number of points in a cluster for DBSCAN.
        threshold_door_dim (float): Minimum dimension threshold for a cluster to be considered a door.

    Returns:
        Tuple[List[Dict], o3d.geometry.PointCloud]: A tuple containing a list of door nodes and the remaining point cloud.
    """
    nodes=[]
    rest_pcd=o3d.geometry.PointCloud()
    
    #select points of class
    idx = np.where((las['classes'] == c['id']))[0]
    class_pcd=pcd.select_by_index(idx)
    object_labels=las['objects'][idx]
    
    #split clusters by distance
    object_pcds=[]
    for i in np.unique(object_labels): 
        if i==0:
            object_pcds.extend(split_point_cloud_by_dbscan(class_pcd.select_by_index(np.where(object_labels==i)[0]),eps=eps,min_cluster_points=min_cluster_points))
        else:           
            object_pcds.append(class_pcd.select_by_index(np.where(object_labels==i)[0]))
    #create nodes
    for i,p in enumerate(object_pcds):    
        #filter doors that are too small
        try:
            dim=p.get_oriented_bounding_box().extent
        except:
            dim=p.get_axis_aligned_bounding_box().get_extent()
        if (dim[0]>threshold_door_dim) & (dim[1]>threshold_door_dim):
            nodes.append(PointCloudNode(resource=p,
                                        class_id=c['id'],
                                        class_name=c['name'],
                                        object_id=c['id']*1000+i,
                                        color=ut.random_color(),
                                        name=ut.get_filename(f)+'_'+c['name']+'_'+str(c['id']*1000+i)))
        else:
            rest_pcd+=p
    
    return nodes,rest_pcd  
