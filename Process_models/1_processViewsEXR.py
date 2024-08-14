# Author: Wentao Yuan (wyuan1@cs.cmu.edu) 05/31/2018
# Modified by Rui Zeng 07/12/2020

import Imath
import OpenEXR
import argparse
import array
import numpy as np
import os
from open3d import *
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

    print("Input nr views:")
    nr_views= int(input())


    angle = 360/nr_views

    # print("Input scale model:")
    # scale_model= float(input())

    nr_points_fps=1024
    
    
    datatype_list = ["valid",]


    
    
    # path_meshes = os.path.join("/mnt/ssd1/Alex_data/Cuda_Data/Shapenet_with_image_transf/")
    
    folder_dataset = "3d_Shape_small"
    folder_dataset = "3d_Objects_full"
   
    root_folder = "/mnt/ssd1/Alex_data/Test_PCNBV_again_2/"

    path_meshes = os.path.join(root_folder,"0_gt_meshes",folder_dataset)

    
    ShapeNetv1_dir = os.path.join(path_meshes)
    
    model_dir_intrinsics = os.path.join(root_folder,'1_Export_blender_views_transformed_big',folder_dataset)

    for data_type in datatype_list:
        #model_dir_intrinsics = "/mnt/ssd1/Alex_data/Cuda_Data/Blender_views/New_data_6b/Blensor_Views_New_New/correct_deg_ISR_rand_rot_rotmin_0_rotmax_0_6_views_distmin_1_distmax_1"


        model_dir = os.path.join(model_dir_intrinsics, data_type)

        output_dir = os.path.join(root_folder,"1_views_blender_pcd_translated_big",folder_dataset, data_type)

        output_dir_error_sampling =  os.path.join(root_folder,"error_sampling", data_type)

        intrinsics = np.loadtxt(os.path.join(model_dir_intrinsics, 'intrinsics.txt'))
        width = int(intrinsics[0, 2] * 2)
        height = int(intrinsics[1, 2] * 2)

        class_list = os.listdir(model_dir)

        category_list = os.listdir(os.path.join(ShapeNetv1_dir, data_type))

        for category_id in category_list:

            print(category_id)
            model_list = os.listdir(os.path.join(ShapeNetv1_dir, data_type,category_id))

            for model_id in model_list:
                
                pcd_dir = os.path.join(output_dir, category_id, model_id)
                
               
                os.makedirs(pcd_dir, exist_ok=True)
                for i in range(nr_views):
                    exr_path = os.path.join(model_dir, 'exr',category_id, model_id, '%d.exr' % i)
                    pose_path = os.path.join(model_dir, 'pose',category_id, model_id, '%d.txt' % i)   

                    depth = read_exr(exr_path, height, width)

                    pose = np.loadtxt(pose_path)
                    points = depth2pcd(depth, intrinsics, pose)
          
                    if (points.shape[0] == 0):
                        points = np.array([(1.0,1.0,1.0)])
                    pcd = open3d.geometry.PointCloud()
                    #pcd.points = open3d.utility.Vector3dVector(np.asarray(points.cpu()))
                    pcd.points = open3d.utility.Vector3dVector(points)
                    pcd.paint_uniform_color([0, 0, 1])

                    points = np.array(pcd.points)

                    open3d.io.write_point_cloud(os.path.join(pcd_dir, '%d.pcd' % i), pcd,write_ascii=True)



    print("Done")
                   
                
                
