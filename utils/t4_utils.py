
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
    return pcds if len(pcds)>0 else [class_pcd]
  