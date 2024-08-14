import numpy as np
import os

import open3d as o3d

import copy




folder_dataset = "3d_Shape_small"
folder_dataset = "3d_Objects_full"

path_root = "/mnt/ssd1/Alex_data/Test_PCNBV_again_2/"

root_meshes = os.path.join(path_root,"0_gt_meshes",folder_dataset)

root_transforms = os.path.join(path_root,"0_gt_meshes_translated_3",folder_dataset)



path_export = os.path.join(path_root,'0_gt_pcd_scaled_big',folder_dataset) 

nr_points_sample = 16384

dataset_type = "valid"

category_list = os.listdir(os.path.join(root_meshes,dataset_type))

nr_broken = np.zeros(len(category_list))

for sel_category in range(len(category_list)):

    category = category_list[sel_category]

    model_list = os.listdir(os.path.join(root_meshes, dataset_type,category))

    for model in model_list:

        print(model)

        mesh_path = os.path.join(root_meshes,dataset_type,category,model,"model.obj")

        path_scale = os.path.join(root_transforms,dataset_type,category,model,"scale.npy")

        scale_transform = np.load(path_scale)

        

        

        mesh_gt = o3d.io.read_triangle_mesh(mesh_path)

        vertices = np.asarray(mesh_gt.vertices)

        vertices = vertices * scale_transform
        
        mesh_gt.vertices = o3d.utility.Vector3dVector(vertices)

        R_before = mesh_gt.get_rotation_matrix_from_xyz((np.pi/2,0,0))

        mesh_gt = mesh_gt.rotate(R_before,center = (0,0,0))

        


        #pcd_gt = mesh_gt.sample_points_uniformly(number_of_points=int(np.floor(nr_points_sample*scale_transform[0]*scale_transform[1]*scale_transform[2])))
        
        pcd_gt = mesh_gt.sample_points_uniformly(number_of_points=nr_points_sample)
        
        pcd_gt.paint_uniform_color([0, 1, 0])

        

        # o3d.visualization.draw_geometries([mesh_gt,pcd_gt],zoom=1, front=[0,-1,-1], lookat=[0,0,0], up=[0,-1,0])

        pcd_gt.paint_uniform_color([0, 1, 0])

        pcd_gt_new = copy.deepcopy(pcd_gt)

        pcd_gt_new = pcd_gt_new.farthest_point_down_sample(1024)

        #o3d.visualization.draw_geometries([pcd_gt])

        

        pcd_views = o3d.geometry.PointCloud()

       

        hull, _ = pcd_gt_new.compute_convex_hull()
                                

        volume=0
        
        if (hull.is_watertight()):

            path_export_model = os.path.join(path_export,dataset_type,category,model)

            if not os.path.exists(path_export_model):
                os.makedirs(path_export_model)

            path_export_pcd = os.path.join(path_export_model,"gt.pcd")
            path_export_volume = os.path.join(path_export_model,"volume.npy")

            
            volume=hull.get_volume()

            np.save(path_export_volume,volume)

            o3d.io.write_point_cloud(path_export_pcd, pcd_gt, write_ascii=True)

            print("Saved: ", os.path.join(path_export_model,"gt.pcd"))

        else:
            nr_broken[sel_category] += 1
            print(mesh_path)
            print("Not watertight")

print(nr_broken)
        

        

            

            

                

            
