import sys
# sys.path.append('/home/user/Code/PoinTr/')

sys.path.append('/home/alex/Alex/Work/Incercare_PointTr/PoinTr/')

import open3d as o3d

import mathutils
import copy


from tools import test_net
from utils import parser, dist_utils, misc
from utils.logger import *
from utils.config import *
import time
import os
import torch
import torch.nn as nn
import json
from tools import builder
import cv2
import numpy as np

class PC_Predictor():
    def __init__(self, model_path, config_path):
        self.config = cfg_from_yaml_file(config_path)
        self.base_model = builder.model_builder(self.config['model'])
        builder.load_model(self.base_model, model_path, logger = None)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.base_model.to(self.device)
        
        self.base_model.eval()  # set model to eval mode
        self.target = './vis'
        self.useful_cate = [
            "02691156", #plane
            "04379243", #table
            "03790512", #motorbike
            "03948459", #pistol
            "03642806", #laptop
            "03467517", #guitar
            "03261776", #earphone
            "03001627", #chair
            "02958343", #car
            "04090263", #rifle
            "03759954", # microphone
            ]
        
    '''
    def normalize(self, pc):
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc / m
        
        return pc, centroid, m
    '''
    
    def normalize(self, pc):
        centroid = torch.mean(pc, dim=1)
        pc = pc - centroid
        m = torch.max(torch.sqrt(torch.sum(pc**2, dim=2)))
        pc = pc / m
        return pc, centroid, m
    
    def denormalize(self, pc, centroid, m):
        return (pc * m) + centroid
    
    def predict(self, point_cloud):
        ## If not tensor, convert to tensor
        if isinstance(point_cloud, np.ndarray):
            points = torch.FloatTensor(point_cloud)

        else:
            points = point_cloud
        
        ## If not 3-dim, adda dim (batch size)
        if len(points.shape) == 2:
            points = torch.unsqueeze(points, 0)
        
        ## Send to GPU
        points = points.to(self.device)
        
        ## Nromalization
        points, centroid, max_pt = self.normalize(points)
        
        ## Downsampling
        if points.shape[1] > 2048:
            points_down = misc.fps(points, 2048)
        else:
            points_down = points
        
        ## Run model
        ret = self.base_model(points_down)
        coarse_points = ret[0]
        dense_points = ret[-1]
        
        ## Denormalize
        points = self.denormalize(dense_points, centroid, max_pt)
        
        ## Remove batch-dim and convert to numpy
        points_np = points.squeeze().detach().cpu().numpy()
        
        return points_np
    
    def select_view(self,pointr_model,pcd,viewspace,epsilon):
        point_cloud = np.asarray(pcd.points)
        point_cloud = torch.tensor(point_cloud).float()
        point_cloud = point_cloud.unsqueeze(0)
        
        new_points,centroid, m = pointr_model.normalize(point_cloud)
        predicted_points = pointr_model.predict(new_points)
        predicted_points_denormalized = pointr_model.denormalize(torch.tensor(predicted_points).unsqueeze(0), centroid, m).squeeze().detach().cpu().numpy()

        pcd_predicted = o3d.geometry.PointCloud()
        pcd_predicted.points = o3d.utility.Vector3dVector(predicted_points_denormalized)
        pcd_predicted.paint_uniform_color([1.0, 0.0, 0.0])

        # mesh_mv = o3d.geometry.TriangleMesh.create_coordinate_frame()
        # mesh_mv = mesh_mv.scale(0.1, center=(0, 0, 0))
        # mesh_mv = mesh_mv.translate((viewspace[view,0], viewspace[view,1], viewspace[view,2]), relative=False)

        distances = np.asarray(pcd_predicted.compute_point_cloud_distance(pcd))
        diff_ids = np.where(distances >= epsilon)[0]

        nr_points_list_view = []

        for iter_view in range(len(viewspace)):
            mesh_mv = o3d.geometry.TriangleMesh.create_coordinate_frame()
            mesh_mv = mesh_mv.scale(0.1, center=(0, 0, 0))
            

            cam_pose = mathutils.Vector((viewspace[iter_view,0], viewspace[iter_view,1], viewspace[iter_view,2]))
            center_pose = mathutils.Vector((0,0,0))
            direct = center_pose - cam_pose
            rot_quat = direct.to_track_quat('-Z', 'Y')
            rot_euler = rot_quat.to_euler()

            R_new = rot_quat.to_matrix()

            mesh_mv.rotate(R_new, center=(0, 0, 0))

            #mesh_mv = mesh_mv.translate((viewspace[iter_view,0], viewspace[iter_view,1], viewspace[iter_view,2]), relative=False)



            [x,y,z] = cam_pose
            [rx,ry,rz] = rot_euler


            pcd_temp = copy.deepcopy(pcd_predicted)

            

            

            
            # R = o3d.geometry.get_rotation_matrix_from_axis_angle([np.radians(-rx), np.radians(-ry), np.radians(-rz)])
            pcd_temp = pcd_temp.rotate(np.transpose(R_new), center=(0,0,0))
            pcd_temp = pcd_temp.translate((-x,-y,-z))

            # o3d.visualization.draw_geometries([mesh_mv,pcd_temp])

            #o3d.visualization.draw_geometries([mesh_mv,mesh_gt,pcd_temp])

            points_predicted = np.asarray(pcd_predicted.points)

            diameter = np.linalg.norm(np.asarray(pcd_temp.get_max_bound()) - np.asarray(pcd_temp.get_min_bound()))

            camera = [0, 0, 0] # camera is assumed to be at 0,0,0, because we moved the point cloud instead
            radius = diameter * 100
            _, hid_ids = pcd_temp.hidden_point_removal(camera, radius)

            
            visible_predicted_points = list(set(diff_ids).intersection(set(hid_ids)))

            eliminated_predicted_points = list(set(diff_ids).difference(set(visible_predicted_points)))
            

            # pcd_chosen = pcd_predicted.select_by_index(visible_predicted_points)
            # pcd_eliminated = pcd_predicted.select_by_index(eliminated_predicted_points)

            # pcd_chosen = pcd_chosen.rotate(np.transpose(R_new), center=(0,0,0))
            # pcd_chosen = pcd_chosen.translate((-x,-y,-z))

            # pcd_eliminated = pcd_eliminated.rotate(np.transpose(R_new), center=(0,0,0))
            # pcd_reliminated = pcd_eliminated.translate((-x,-y,-z))

            # pcd_chosen.paint_uniform_color([0.0, 0, 1.0])
            # pcd_eliminated.paint_uniform_color([1.0, 0, 0.0])
            # o3d.visualization.draw_geometries([pcd_chosen, pcd_eliminated,mesh_mv])
            
            
            
            nr_points_list_view.append(len(visible_predicted_points))

        # print("Number of points in each view: ", nr_points_list_view)

        target_values = np.array(nr_points_list_view)

        return target_values


if __name__ == "__main__":
    # pointr_model = PC_Predictor(model_path='/home/dhami/Code/PoinTr/pretrained/90_0.25.pth')
    # point_cloud = np.load('/home/dhami/Code/PoinTr/data/ShapeNet55-34/shapenet_pc/02828884-3d2ee152db78b312e5a8eba5f6050bab.npy')

    path_viewspace = "/home/alex/Alex/Work/Incercare_PointTr/Pred-NBV/new_simulation/viewspace.txt"
    viewspace = np.loadtxt(path_viewspace)

    epsilon = 0.00707

    # path_mesh_model = "/home/alex/Alex/Work/3d_models_normalized_2/valid/LM5/LM5/model.obj"
    # mesh_gt = o3d.io.read_triangle_mesh(path_mesh_model)

    # R = mesh_gt.get_rotation_matrix_from_xyz((np.pi/2, 0, 0))
    # mesh_gt.rotate(R, center=(0, 0, 0))


    modelPath = '/home/alex/Alex/Work/Incercare_PointTr/Pred-NBV/models/PoinTr_C_final.pth'
    pathConfig = '/home/alex/Alex/Work/Incercare_PointTr/PoinTr/cfgs/PCN_models/PoinTr.yaml'
    pathConfig = "/home/alex/Alex/Work/Incercare_PointTr/PoinTr/cfgs/ShapeNet55_models/PoinTr.yaml"

    pointr_model = PC_Predictor(model_path=modelPath, config_path=pathConfig)


    folder_mesh = "/home/alex/Alex/Work/3d_models_normalized_2/valid/"

    folder_pcd = "/home/alex/Alex/Work/Test_all_again_2/1_views_blender_pcd/3d_Objects_full/valid/"

    category_list = os.listdir(folder_mesh)

    for category in category_list[3:5]:

        model_list = os.listdir(os.path.join(folder_pcd, category))

        for model in model_list:

            path_mesh_model = os.path.join(folder_mesh, category, model, "model.obj")
            mesh_gt = o3d.io.read_triangle_mesh(path_mesh_model)

            R = mesh_gt.get_rotation_matrix_from_xyz((np.pi/2, 0, 0))
            mesh_gt.rotate(R, center=(0, 0, 0))

            for view in range(len(viewspace)):

                path_pcd = os.path.join(folder_pcd, category, model, str(view)+".pcd")

                if os.path.exists(path_pcd):
                    pcd = o3d.io.read_point_cloud(path_pcd)
                    pcd.paint_uniform_color([0.0, 1.0, 0.0])

                    predicted_list = pointr_model.select_view(pointr_model=pointr_model,pcd=pcd,viewspace=viewspace,epsilon=epsilon)
                    print(predicted_list)
    # point_cloud = np.load('/home/dhami/Code/PoinTr/data/ShapeNet55-34/shapenet_pc/02828884-3d2ee152db78b312e5a8eba5f6050bab.npy')
    
   