
import time
import os
import laspy
import geomapi
import geomapi.utils.geometryutils as gmu
import geomapi.utils as ut
import numpy as np
import torch

from utils import timer


@timer
def handle_process(file_name, output_folder,cfg) -> None:
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
    save_dict = dict(coord=coords, color=colors, normal=normals, scene_id=scene_id, semantic_gt=remapped_classes)
    
    
    # save_dict = dict(coord=coords, color=colors, normal=normals, scene_id=scene_id, semantic_gt=las['classes'].astype(int))    

    torch.save(save_dict, os.path.join(output_folder, f"{name}.pth"))
    

