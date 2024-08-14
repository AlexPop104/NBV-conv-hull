import numpy as np
import open3d as o3d
import os

import torch

import copy

import random
import tqdm

import time


from pc_nbv import AutoEncoder, Encoder, Decoder

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('conv') != -1:
        m.weight.data.normal_(0.0, 0.1)
        m.bias.data.fill_(0)

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

    

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Assume that we are on a CUDA machine, then this should print a CUDA device:
    print(device)

    path_viewspace = "/home/alex/Alex_documents/Alex_work_new2/Alex_work/GeoA3/Extra_code_Neurocomputing/Cod_PC_ALEX/New_version_NBVnet/viewspace.txt"
    viewspace = np.loadtxt(path_viewspace)

    nr_views = viewspace.shape[0]

    
    path_model = "/home/alex/Alex_documents/PC-NBV_pytorch/log/best.pth"

    path_model = "/mnt/ssd1/Alex_data/PC-NBV/Trained_models/PCNBV_remake_33_09_07_24_09:25:54/model9.pt"

    

    #path_model = "/mnt/ssd1/Alex_data/PC-NBV/Trained_models/PCNBV_remake_33_07_07_24_17:36:02/model63.pt"

    num_classes = 33 
    nr_points= 512

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Training on {device}")

    net = AutoEncoder(views=num_classes)
    net.load_state_dict(torch.load(path_model))
    net = net.to(device)

    net.eval()

    folder_dataset = "3d_Shape_small"
    #folder_dataset = "3d_Objects_full"


    root_folder = "/mnt/ssd1/Alex_data/Test_PCNBV_again_2/"

    selection_dataset = 1

    if selection_dataset == 0:
        root_views = os.path.join(root_folder,"3_blensor_hull_views",folder_dataset)
        path_root_meshes = os.path.join(root_folder,"2_hull_meshes_blensor",folder_dataset)

        #root_views_gt = os.path.join(root_folder,"1_views_blensor_pcd",folder_dataset)

        root_views_gt = os.path.join(root_folder,"1_views_blender_pcd",folder_dataset)

    else:
        root_views = os.path.join(root_folder,"3_blender_hull_views",folder_dataset)
        path_root_meshes = os.path.join(root_folder,"2_hull_meshes_blender",folder_dataset)

        root_views_gt = os.path.join(root_folder,"1_views_blender_pcd",folder_dataset)

    path_gt_pcds= os.path.join(root_folder,"0_gt_pcd",folder_dataset)
    
    epsilon = 0.00707

    nr_points = 16384

    nr_max_iterations = 10
    

    
    epsilon = 0.00707

    nr_points = 16384

    nr_points_net = 1024

    nr_max_iterations = 10


   

    path_viewspace = "/home/alex/Alex_documents/Alex_work_new2/Alex_work/GeoA3/Extra_code_Neurocomputing/Cod_PC_ALEX/New_version_NBVnet/viewspace.txt"
    viewspace = np.loadtxt(path_viewspace)


    np.random.seed(0)
    

    datatype_list = ["test"]

    for data_type in datatype_list:

        nr_cov_array = np.zeros(nr_max_iterations)
        avg_cov_array = np.zeros(nr_max_iterations)

        root_pcd_gt = os.path.join(path_gt_pcds,data_type)

        path_views_gt = os.path.join(root_views_gt,data_type)

        model_dir = os.path.join(path_root_meshes,data_type)

        category_list = os.listdir(model_dir)

        for category in tqdm.tqdm(category_list):

            model_list = os.listdir(os.path.join(model_dir,category))

            for model in tqdm.tqdm(model_list):

                for index_transf in range(1):

                    #path_model = os.path.join(model_dir,category,model,"rot_"+str(index_transf),"model_hull.obj")
                    # path_views = os.path.join(root_views,data_type,category,model,"rot_"+str(index_transf))
                    path_model_hull = os.path.join(model_dir,category,model,"model_hull.obj")
                    path_views = os.path.join(root_views,data_type,category,model,)

                    mesh_hull = o3d.io.read_triangle_mesh(path_model_hull)
                    pcd_gt_hull = mesh_hull.sample_points_uniformly(number_of_points=nr_points)
                    pcd_gt_hull.paint_uniform_color([0, 1, 0])


                    pcd_mesh_orig = o3d.io.read_point_cloud(os.path.join(root_pcd_gt,category,model,"gt.pcd"))

                    

                    # path_mesh_orig = os.path.join(root_mesh_gt,category,model,"model.obj")
                    # mesh_orig_gt = o3d.io.read_triangle_mesh(path_mesh_orig)
                    # pcd_mesh_orig = mesh_orig_gt.sample_points_uniformly(number_of_points=16384)

                    pcd_mesh_orig.paint_uniform_color([1, 1, 0])

                    R_before = pcd_gt_hull.get_rotation_matrix_from_xyz((np.pi/2,0,0))
                    
                    pcd_gt_hull = pcd_gt_hull.rotate(R_before,center = (0,0,0))

                    pcd_mesh_orig_copy = copy.deepcopy(pcd_mesh_orig)

                    pcd_mesh_orig_copy = pcd_mesh_orig_copy.rotate(R_before,center = (0,0,0))

                    
                    views_select_list = [1,3,7,10,15,24,27,30]

                    
                    for view_initial in views_select_list:
                    #for view_initial in range(3):

                        
                        total_time_net = 0
                        total_time_sampling = 0

                        pcd_all_views_selected = o3d.geometry.PointCloud()

                        #path_view_initial = os.path.join(path_views_gt,category,model,"rot_"+str(index_transf),str(view_initial)+".npy")
                        path_view_initial = os.path.join(path_views_gt,category,model,str(view_initial)+".pcd")

                        pcd_view = o3d.io.read_point_cloud(path_view_initial)

                        points_view_initial = np.asarray(pcd_view.points)

                       
                        pcd_view.paint_uniform_color([1, 0, 0])

                        pcd_all_views_selected += pcd_view

                        # o3d.visualization.draw_geometries([pcd_mesh_orig,pcd_all_views_selected])
                        #o3d.visualization.draw_geometries([pcd_all_views_selected])

                
                        distances_gt = np.asarray(pcd_mesh_orig.compute_point_cloud_distance(pcd_all_views_selected))

                        indices_selected_gt = np.where(distances_gt < epsilon
                                                    )[0]

                        coverage_view = float(indices_selected_gt.shape[0])/nr_points
                        

                        avg_cov_array[0] += coverage_view
                        nr_cov_array[0] += 1

                        viewstate = np.zeros((1,num_classes))
                        viewstate[0,view_initial] = 1

                        viewstate = torch.tensor(viewstate).float().to(device)


                        pcd_net = copy.deepcopy(pcd_all_views_selected)

                        points_pcd_net = np.asarray(pcd_net.points)

                        if points_pcd_net.shape[0] > nr_points:
                                start_time = time.time()
                                pcd_net = pcd_net.farthest_point_down_sample(nr_points_net)
                                total_time_sampling += time.time()-start_time

                        points_pcd_net = np.asarray(pcd_net.points)

                        points_pcd_net = torch.tensor(points_pcd_net).float().to(device)

                        points_pcd_net = points_pcd_net.unsqueeze(0)

                        points_pcd_net = points_pcd_net.permute(0,2,1)

                        
    
                        for view_iter in range(nr_max_iterations-1):
                            start_time = time.time()
                            features , outputs = net(points_pcd_net,viewstate)
                            total_time_net += time.time()-start_time

                            outputs = outputs - torch.mul(outputs,viewstate)

                            view_select = torch.max(outputs.data, 1)[1]

                            path_view = os.path.join(path_views_gt,category,model,str(view_select.item())+".pcd")

                            pcd_view = o3d.io.read_point_cloud(path_view)

                            points_view = np.asarray(pcd_view.points)

                           
                            pcd_view.paint_uniform_color([1, 0, 0])

                            pcd_all_views_selected += pcd_view

                            distances_gt = np.asarray(pcd_mesh_orig.compute_point_cloud_distance(pcd_all_views_selected))


                            #o3d.visualization.draw_geometries([pcd_mesh_orig,pcd_all_views_selected])

                            indices_selected_gt = np.where(distances_gt < epsilon
                                                        )[0]

                            coverage_view = float(indices_selected_gt.shape[0])/nr_points

                           

                            avg_cov_array[view_iter+1] += coverage_view
                            nr_cov_array[view_iter+1] += 1

                            if view_iter <= nr_max_iterations-2:
                                viewstate[0,view_select] = 1
                                pcd_net = copy.deepcopy(pcd_all_views_selected)
                                points_pcd_net = np.asarray(pcd_net.points)

                                
                                if points_pcd_net.shape[0] > nr_points:
                                        start_time = time.time()
                                        pcd_net = pcd_net.farthest_point_down_sample(nr_points_net)
                                        total_time_sampling += time.time()-start_time

                                # print("Iteration: "+str(iteration))
                                # print("Time for sampling: ",time.time()-start_time)

                                

                                points_pcd_net = np.asarray(pcd_net.points)

                                points_pcd_net = torch.tensor(points_pcd_net).float().to(device)

                                points_pcd_net = points_pcd_net.unsqueeze(0)

                                points_pcd_net = points_pcd_net.permute(0,2,1)



                            # if(view_iter==8):
                            #     print("ceva")
                        print(total_time_net)
                        print(total_time_sampling)
        print(avg_cov_array/nr_cov_array)
                            
            

      
                        
                    

        