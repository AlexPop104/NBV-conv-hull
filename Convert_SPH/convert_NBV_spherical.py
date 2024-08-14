import numpy as np
import open3d as o3d
import os
import copy
import tqdm
import random

import Spherical_utils as sph_utils

from datetime import datetime

import time

def sampling_sphere(nlon,nlat,radius):
    """ Function for reconstructing a point cloud from its projection onto the unit sphere. """

    
    lon = np.linspace(0, nlon * np.pi * 2 / (nlon - 1), num=nlon, endpoint=False)
    lat = np.linspace(nlat // 2 * np.pi / -(nlat - 1), np.pi / 2, num=nlat, endpoint=True)
    lon = np.broadcast_to(lon.reshape((1, nlon)), (nlat, nlon))
    lat = np.broadcast_to(lat.reshape((nlat, 1)), (nlat, nlon))


   

    z = np.sin(lat)*radius 
    t = np.cos(lat)*radius
    x = t * np.cos(lon)
    y = t * np.sin(lon)

    pc = np.zeros((nlat,nlon,3))
    
    pc[:, :, 0] = x
    pc[:, :, 1] = y
    pc[:, :, 2] = -z  # must have the minus
    pc = pc.reshape((-1, 3))
    #pc = pc[flag, :]  # only the flagged polar angles must be used in the point cloud reconstruction
    #pc += origin

    return pc


if __name__ == '__main__':

    folder_dataset = "3d_Shape_small"
    #folder_dataset = "3d_Objects"

    l_max = 3

    root_folder = "/mnt/ssd1/Alex_data/Test_PCNBV_again_2/"

    selection_dataset = 1

    path_views_gt = os.path.join(root_folder,"5_NBV_vig_translated",folder_dataset)

    path_gt_pcds= os.path.join(root_folder,"0_gt_pcd_scaled",folder_dataset)

    export_folder = os.path.join(root_folder,"6_NBV_spherical_vig",str(l_max),folder_dataset)
    
    epsilon = 0.00707

    nr_points = 16384

    nr_max_iterations = 10

    
    

    
    epsilon = 0.00707

    nr_points = 16384

    nr_max_iterations = 3


    path_viewspace = "/home/alex/Alex_documents/Alex_work_new2/Alex_work/GeoA3/Extra_code_Neurocomputing/Cod_PC_ALEX/New_version_NBVnet/viewspace.txt"
    viewspace = np.loadtxt(path_viewspace)

    #datatype_list = ["valid","train"]

    datatype_list = ["train"]

    np.random.seed(0)

    for data_type in datatype_list:

        nr_cov_array = np.zeros(nr_max_iterations)
        avg_cov_array = np.zeros(nr_max_iterations)

        path_v_gt = os.path.join(path_views_gt,data_type)
        
        model_dir = os.path.join(path_gt_pcds,data_type)

        category_list = os.listdir(model_dir)

        for category in tqdm.tqdm(category_list):

            model_list = os.listdir(os.path.join(model_dir,category))

            for model in tqdm.tqdm(model_list):

                pcd_mesh_orig = o3d.io.read_point_cloud(os.path.join(model_dir,category,model,"gt.pcd"))

                pcd_mesh_orig.paint_uniform_color([1, 1, 0])

                R_before = pcd_mesh_orig.get_rotation_matrix_from_xyz((np.pi/2,0,0))

                view_select_list = os.listdir(os.path.join(path_v_gt,category,model))
                    

                for view_initial in view_select_list:
                #for view_initial in range(3):

                    file_list = os.listdir(os.path.join(path_v_gt,category,model,view_initial))

                    for file in file_list:

                        if(file.endswith(".pcd")):
                            
                            pcd_view = o3d.io.read_point_cloud(os.path.join(path_v_gt,category,model,view_initial,file))

                            pcd_view.paint_uniform_color([1, 0, 0])

                            # o3d.visualization.draw_geometries([pcd_mesh_orig,pcd_view])

                            points_view = np.asarray(pcd_view.points)

                            smooth_pc , _ , lat_lon = sph_utils.reconstruct_pcd_from_points(points=points_view,nr_points_fps=1024,lmax=l_max,sigma=25,choice=0,choice_filtering=0,choice_fps=0)

                            if(smooth_pc.shape[0]>1):

                                smooth_pcd = o3d.geometry.PointCloud()
                                smooth_pcd.points = o3d.utility.Vector3dVector(smooth_pc)
                                smooth_pcd.paint_uniform_color([0, 1, 0])

                                export_folder_model = os.path.join(export_folder,data_type,category,model,view_initial)


                                file_export = file.replace(".pcd","")
                                file_export = file_export + "_spherical.pcd"

                                if not os.path.exists(export_folder_model):
                                    os.makedirs(export_folder_model)

                                o3d.io.write_point_cloud(os.path.join(export_folder_model,file_export),smooth_pcd)
                            else:
                                print("Error in reconstruction")
                                print(os.path.join(path_v_gt,category,model,view_initial,file))
                               





                    
                    
                           
                            
                            

                    

                

        