'''The provided code processes point cloud data and saves it in a format that can be used by PyTorch dataloader in PTV3 network. The code reads LAS/LAZ files, converts them to point cloud data, and remaps the classes to 0,1,2,3,4,5 based on the values in the configuration file. It then divides the point cloud into chunks based on the specified size or parts and saves the processed data as .pth files. The `preprocess_point_clouds_to_pth` function takes the file path, output folder, configuration file, size, and parts as input parameters and performs the preprocessing steps. The `handle_process` function processes the point cloud data and saves it in .pth format. The `revert_class_mapping` function reverts the class mapping from indices to labels based on the configuration file. The code also includes a timer decorator to measure the execution time of the functions.'''

import time
import os
import laspy
import geomapi
import geomapi.utils.geometryutils as gmu
import geomapi.utils as ut
import numpy as np
import torch
from typing import Any, Dict, List, Optional

from utils import timer
import open3d as o3d
from deprecated import deprecated
 
@timer
def preprocess_point_clouds_to_pth(file_path: str, output_folder: str, cfg: Dict, size: Optional[List[int]] = None, parts: Optional[List[int]] = None) -> None:
    """
    Preprocesses point cloud data for use in a PyTorch DataLoader.

    Args:
        file_path (str): File path of the point cloud data.
        output_folder (str): Directory to save the preprocessed files.
        cfg (Dict[str, Any]): Configuration for processing.
        size (Optional[List[int]]): Dimensions for chunking the point cloud.
        parts (Optional[List[int]]): How to divide the point cloud into parts.
    """
    #get file_name
    file_name=ut.get_filename(f)    
    print(f'processing {file_name} ...')

    # Read LAS/LAZ
    las = laspy.read(f)
    #las to pcd
    pcd = gmu.las_to_pcd(las)
    #estimate normals if no normals present
    pcd.estimate_normals() if not pcd.has_normals() else None
    
    # Remap las['classes'] to 0,1,2,3,4,5 based on values in cfg['data']['labels']
    if getattr(las,'classes',None) is not None:
        class_mapping = {label: i for i, label in enumerate(cfg['data']['labels'])}
        remapped_classes = np.array([class_mapping[label] for label in las['classes']])
                
    #divide the point cloud into chunks per part [7,7,1] or size e.g. [10m,10m,100m]
    if size or parts:
        box=o3d.geometry.AxisAlignedBoundingBox(min_bound=np.array([las.x.min(),las.y.min(),las.z.min()]),
                                        max_bound=np.array([las.x.max(),las.y.max(),las.z.max()]))
        #create open3d axis aligned bounding box of points
        boxes,names=gmu.divide_box_in_boxes(box,size=size) if size is not None else gmu.divide_box_in_boxes(box,parts=parts)
        # select indices per boxes
        pathLists=[]
        idxLists=[]
        for box,name in zip(boxes,names):
            pathLists.append(f'{name[0]}_{name[1]}_{name[2]}')
            idxLists.append(box.get_point_indices_within_bounding_box(pcd.points))
        #save the chunks
        for i,name in zip(idxLists,pathLists):
            #select points
            sub_pcd=pcd.select_by_index(i)
            #create chunk_dict
            chunk_name=f"{file_name}_{name}"
            if getattr(las,'classes',None) is not None:
                sub_labels=remapped_classes[i]
                chunk_dict = dict(coord=np.asarray(sub_pcd.points), color=np.asarray(sub_pcd.colors), normal=np.asarray(sub_pcd.normals), scene_id=chunk_name, semantic_gt=sub_labels)
            else:
                chunk_dict = dict(coord=np.asarray(sub_pcd.points), color=np.asarray(sub_pcd.colors), normal=np.asarray(sub_pcd.normals), scene_id=chunk_name)
      
            #save chunk_dict
            torch.save(chunk_dict, os.path.join(output_folder, f"{chunk_name}.pth")) 
            print(f'saved {chunk_name}')
    else:
        #save the whole point cloud
        chunk_name=f"{file_name}"
        if getattr(las,'classes',None) is not None:
            chunk_dict = dict(coord=np.asarray(pcd.points), color=np.asarray(pcd.colors), normal=np.asarray(pcd.normals), scene_id=chunk_name, semantic_gt=sub_labels)
        else:
            chunk_dict = dict(coord=np.asarray(pcd.points), color=np.asarray(pcd.colors), normal=np.asarray(pcd.normals), scene_id=chunk_name)
        #create chunk_dict
        torch.save(chunk_dict, os.path.join(output_folder, f"{chunk_name}.pth"))
        print(f'saved {file_name}')
        

def revert_class_mapping(cfg: Dict, indices: List[int]) -> List[str]:
    """
    Reverts class indices back to their original string labels based on a configuration mapping.

    Args:
        cfg (Dict[str, Any]): Configuration dictionary containing class labels.
        indices (List[int]): List of indices to revert back to labels.

    Returns:
        List[str]: List of class labels corresponding to the input indices.
    """
    # Create the original mapping from labels to indices
    class_mapping = {label: i for i, label in enumerate(cfg['data']['labels'])}
    
    # Invert the mapping to go from indices to labels
    inverted_class_mapping = {v: k for k, v in class_mapping.items()}
    
    # Use the inverted mapping to convert indices back to labels
    remapped_classes = [inverted_class_mapping[i] for i in indices]
    
    return remapped_classes