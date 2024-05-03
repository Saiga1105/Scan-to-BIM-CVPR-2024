
import time
import os
import laspy
import geomapi
import geomapi.utils.geometryutils as gmu
import geomapi.utils as ut
import numpy as np
import torch

from utils import timer
import open3d as o3d

@timer
def handle_process(file_name, output_folder,cfg,batch_size=20000000) -> None:
    print(f'processing {file_name} ...')

    coords = []
    scene_id = os.path.basename(file_name)
    name, _ = os.path.splitext(scene_id)    

    # Read LAS/LAZ
    las = laspy.read(file_name)
    
    #conver to pcd if no normals are present
    pcd = gmu.las_to_pcd(las)
    pcd.estimate_normals()
    pcd.orient_normals_to_align_with_direction()
    
    coords = np.stack([las.x, las.y, las.z], axis=1)
    colors = np.stack([las.red, las.green, las.blue], axis=1).astype(np.uint8)
    normals = np.asarray(pcd.normals)
    # verticality = np.nan_to_num(las.verticality)
    # max = np.max(verticality)
    # verticality = verticality / (max / 2.) - 1.
    
    # Remap las['classes'] to 0,1,2,3,4,5 based on values in cfg['data']['labels']
    class_mapping = {label: i for i, label in enumerate(cfg['data']['labels'])}
    remapped_classes = np.array([class_mapping[label] for label in las['classes']])
    
    #per 10 million points, save the data
    num_points = coords.shape[0]
    if num_points >= batch_size:
        num_chunks = num_points // batch_size
        for i in range(num_chunks):
            start_idx = i * batch_size
            end_idx = (i + 1) * batch_size
            chunk_coords = coords[start_idx:end_idx]
            chunk_colors = colors[start_idx:end_idx]
            chunk_normals = normals[start_idx:end_idx]
            chunk_semantic_gt = remapped_classes[start_idx:end_idx]
            chunk_save_dict = dict(coord=chunk_coords, color=chunk_colors, normal=chunk_normals, scene_id=scene_id, semantic_gt=chunk_semantic_gt)
            torch.save(chunk_save_dict, os.path.join(output_folder, f"{name}_{i}.pth"))
        remaining_coords = coords[num_chunks * batch_size:]
        remaining_colors = colors[num_chunks * batch_size:]
        remaining_normals = normals[num_chunks * batch_size:]
        remaining_semantic_gt = remapped_classes[num_chunks * batch_size:]
        remaining_save_dict = dict(coord=remaining_coords, color=remaining_colors, normal=remaining_normals, scene_id=scene_id, semantic_gt=remaining_semantic_gt)
        torch.save(remaining_save_dict, os.path.join(output_folder, f"{name}_{num_chunks}.pth"))
    else:
        save_dict = dict(coord=coords, color=colors, normal=normals, scene_id=scene_id, semantic_gt=remapped_classes)
        torch.save(save_dict, os.path.join(output_folder, f"{name}.pth"))    
    
    # save_dict = dict(coord=coords, color=colors, normal=normals, scene_id=scene_id, semantic_gt=las['classes'].astype(int))    

    #torch.save(save_dict, os.path.join(output_folder, f"{name}.pth"))
    
@timer
def preprocess_point_clouds_to_pth(f:str, output_folder:str,cfg,size=None,parts=None):
    """Preprocess point clouds to pth files used by PyTorch dataloader in PTV3 network.

    Args:
        f (str): file path to the point cloud
        output_folder (str): output directory
        cfg (custom): training/testing config file
        size (list, optional): size of parts to partition the point cloudsin X,Y,Z. Defaults to [20,20,100].
        parts (list, optional): number of parts to partition the point cloud in X,Y,Z. Defaults to [7,7,1].
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
        
def revert_class_mapping(cfg, indices):
    # Create the original mapping from labels to indices
    class_mapping = {label: i for i, label in enumerate(cfg['data']['labels'])}
    
    # Invert the mapping to go from indices to labels
    inverted_class_mapping = {v: k for k, v in class_mapping.items()}
    
    # Use the inverted mapping to convert indices back to labels
    remapped_classes = [inverted_class_mapping[i] for i in indices]
    
    return remapped_classes