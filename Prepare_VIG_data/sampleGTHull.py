# Author: Wentao Yuan (wyuan1@cs.cmu.edu) 05/31/2018
# Modified by Rui Zeng 07/12/2020

import Imath
import OpenEXR
import argparse
import array
import numpy as np
import os
import  open3d as o3d
# from open3d.io import *
import matplotlib.pyplot as plt
import sys
import pdb

import copy

import torch

from torch_geometric.nn import fps

def read_exr(exr_path, height, width):
    file = OpenEXR.InputFile(exr_path)
    depth_arr = array.array('f', file.channel('R', Imath.PixelType(Imath.PixelType.FLOAT)))
    depth = np.array(depth_arr).reshape((height, width))
    depth[depth < 0] = 0
    depth[np.isinf(depth)] = 0
    return depth


def depth2pcd(depth, intrinsics, pose):
    inv_K = np.linalg.inv(intrinsics)
    inv_K[2, 2] = -1
    depth = np.flipud(depth)
    y, x = np.where(depth > 0)
    # image coordinates -> camera coordinates
    points = np.dot(inv_K, np.stack([x, y, np.ones_like(x)] * depth[y, x], 0))
    # camera coordinates -> world coordinates
    points = np.dot(pose, np.concatenate([points, np.ones((1, points.shape[1]))], 0)).T[:, :3]
    return points


if __name__ == '__main__':

    # print("Input nr views:")
    # nr_views= int(input())


    # angle = 360/nr_views

    # print("Input scale model:")
    # scale_model= float(input())

    nr_points_sample = 16384
    
    
    


    
    
    # path_meshes = os.path.join("/mnt/ssd1/Alex_data/Cuda_Data/Shapenet_with_image_transf/")
    
    folder_dataset = "3d_Objects_full"
    folder_dataset = "3d_Shape_small"
   
    root_folder = "/mnt/ssd1/Alex_data/Test_PCNBV_again_2/"

    

    

    path_pcd_gt = os.path.join(root_folder,"0_gt_pcd_scaled",folder_dataset)

    path_scale = os.path.join(root_folder,"0_gt_meshes_translated",folder_dataset)

    
    datatype_list = ["test"]
    
    print("Select sph y/n")
    otherview_sel = input()

    for data_type in datatype_list:
       

        if otherview_sel == "y":
            path_meshes = os.path.join(root_folder,"7_NBV_hull_sph",folder_dataset,data_type)
            output_dir = os.path.join(root_folder,"9_gt_pcd_hull_sph",folder_dataset, data_type)
        else:
            path_meshes = os.path.join(root_folder,"7_NBV_hull_selected",folder_dataset,data_type)
            output_dir = os.path.join(root_folder,"9_gt_pcd_hull",folder_dataset, data_type)


        ShapeNetv1_dir = os.path.join(path_meshes)

        

        

        

        # class_list = os.listdir(path_meshes)

        category_list = os.listdir(os.path.join(ShapeNetv1_dir))

        for category_id in category_list:

            print(category_id)
            model_list = os.listdir(os.path.join(ShapeNetv1_dir,category_id))

            for model_id in model_list:
                
                

                path_pcd_gt_model = os.path.join(path_pcd_gt, data_type, category_id, model_id,"gt.pcd")
                path_mesh_gt = os.path.join(path_meshes, category_id, model_id,"model_hull.obj")

                path_scale_model = os.path.join(path_scale, data_type, category_id, model_id,"scale.npy")

                scale_transform = np.load(path_scale_model)

                pcd_gt = o3d.io.read_point_cloud(path_pcd_gt_model)
                pcd_gt.paint_uniform_color([0, 1, 0])

                mesh_gt = o3d.io.read_triangle_mesh(path_mesh_gt)

                pcd_mesh_gt = mesh_gt.sample_points_uniformly(number_of_points=int(np.floor(nr_points_sample*scale_transform[0]*scale_transform[1]*scale_transform[2])))
        

                pcd_dir = os.path.join(output_dir, category_id, model_id)
                if not os.path.exists(pcd_dir):
                    os.makedirs(pcd_dir)
                o3d.io.write_point_cloud(os.path.join(pcd_dir,"gt_hull.pcd"), pcd_mesh_gt)
                # o3d.visualization.draw_geometries([pcd_mesh_gt,mesh_gt])

                
              
               



    print("Done")
                   
                
                
