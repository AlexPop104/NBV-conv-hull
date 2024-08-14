import numpy as np
import os

import open3d as o3d

import copy


def scalePoints(points, scale_x, scale_y, scale_z):
    points[:,0] = points[:,0] * scale_x
    points[:,1] = points[:,1] * scale_y
    points[:,2] = points[:,2] * scale_z

    return points

def translatePoints(points, translate_x, translate_y, translate_z):
    points[:,0] = points[:,0] + translate_x
    points[:,1] = points[:,1] + translate_y
    points[:,2] = points[:,2] + translate_z

    return points

folder_dataset = "3d_Shape_small"
folder_dataset = "3d_Objects_full"

path_root = "/mnt/ssd1/Alex_data/Test_PCNBV_again_2/"

root_meshes = os.path.join(path_root,"0_gt_meshes",folder_dataset)

path_export = os.path.join(path_root,'0_gt_meshes_translated_3',folder_dataset) 

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

        mesh_gt = o3d.io.read_triangle_mesh(mesh_path)

        mesh_gt_copy = copy.deepcopy(mesh_gt)

        max_val = 20
        min_val = 10

        translate_x = min_val+ (max_val-min_val) * np.random.rand()
        translate_y = min_val+ (max_val-min_val) * np.random.rand()
        translate_z = min_val+ (max_val-min_val) * np.random.rand()

        scale_x = min_val+ (max_val-min_val) * np.random.rand()
        scale_y = min_val+ (max_val-min_val) * np.random.rand()
        scale_z = min_val+ (max_val-min_val) * np.random.rand()

        # mesh_gt_vertex = np.asarray(mesh_gt.vertices)

        # mesh_gt_vertex = scalePoints(mesh_gt_vertex, scale_x, scale_y, scale_z)
        #mesh_gt_vertex = translatePoints(mesh_gt_vertex, translate_x, translate_y, translate_z)

        #mesh_gt.vertices = o3d.utility.Vector3dVector(mesh_gt_vertex)

        #o3d.visualization.draw_geometries([mesh_gt, mesh_gt_copy])

        scale_export = np.asarray([scale_x, scale_y, scale_z])
        translate_export = np.asarray([translate_x, translate_y, translate_z])

        path_export_model = os.path.join(path_export,dataset_type,category,model)

        if not os.path.exists(path_export_model):
            os.makedirs(path_export_model)

        #o3d.io.write_triangle_mesh(os.path.join(path_export_model,"model.obj"),mesh_gt)
        np.save(os.path.join(path_export_model,"scale.npy"),scale_export)
        np.save(os.path.join(path_export_model,"translate.npy"),translate_export)

                                

        

        

            

            

                

            
