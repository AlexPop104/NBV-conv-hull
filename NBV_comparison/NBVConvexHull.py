import numpy as np
import open3d as o3d
import os
import copy
import tqdm

import time

from datetime import datetime



if __name__ == '__main__':

    
    #folder_dataset = "3d_Shape_small"
    folder_dataset = "3d_Objects_full"


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


   

    path_viewspace = "/home/alex/Alex_documents/Alex_work_new2/Alex_work/GeoA3/Extra_code_Neurocomputing/Cod_PC_ALEX/New_version_NBVnet/viewspace.txt"
    viewspace = np.loadtxt(path_viewspace)

    datatype_list = ["valid"]

    np.random.seed(0)

    for data_type in datatype_list:

        nr_cov_array = np.zeros(nr_max_iterations)
        avg_cov_array = np.zeros(nr_max_iterations)

        root_pcd_gt = os.path.join(path_gt_pcds,data_type)

        path_views_gt = os.path.join(root_views_gt,data_type)

        model_dir = os.path.join(path_root_meshes,data_type)

        category_list = os.listdir(model_dir)

        category_coverage = np.zeros((len(category_list),nr_max_iterations))
        nr_models_categ = np.zeros((len(category_list),nr_max_iterations))

        indices_models = range(len(category_list))
        

        dict_models = {category_list[i]: indices_models[i] for i in range(len(indices_models))}

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

                    mesh_hull = mesh_hull.rotate(R_before,center = (0,0,0))

                    pcd_mesh_orig_copy = copy.deepcopy(pcd_mesh_orig)

                    pcd_mesh_orig_copy = pcd_mesh_orig_copy.rotate(R_before,center = (0,0,0))


                    #R_something = pcd_gt.get_rotation_matrix_from_xyz((np.pi,0,0))
                    #pcd_mesh_orig = pcd_mesh_orig.rotate(R_before,center = (0,0,0))

                    #o3d.visualization.draw_geometries([pcd_gt_hull,pcd_mesh_orig_copy])
                    

                    views = os.listdir(path_views)

                    nr_views = viewspace.shape[0]

                    pcd_all_views = o3d.geometry.PointCloud()

                    index_list = []

                    list_nr_elements = []

                    indices = range(nr_points)

                    for iter_view in range(nr_views):

                        path = os.path.join(path_views,str(iter_view)+".npy")

                        if os.path.exists(path):

                            points_view = np.load(path)

                            pcd_selected_view = o3d.geometry.PointCloud()
                            pcd_selected_view.points = o3d.utility.Vector3dVector(points_view)
                            
                            #points_view = np.asarray(pcd_selected_view.points)

                            if points_view.shape[0] != 0:
                                

                                pcd_view_hull = o3d.geometry.PointCloud()
                                pcd_view_hull.points = o3d.utility.Vector3dVector(points_view)
                                pcd_view_hull.paint_uniform_color([1, 0, 0])

                                axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=viewspace[iter_view,:])

                                o3d.visualization.draw_geometries([mesh_hull,axis])

                                o3d.visualization.draw_geometries([mesh_hull,pcd_gt_hull,axis])

                                o3d.visualization.draw_geometries([mesh_hull,pcd_gt_hull,pcd_view_hull,axis])

                                distances = np.asarray(pcd_gt_hull.compute_point_cloud_distance(pcd_view_hull))

                                indices_selected = np.where(distances < epsilon
                                                            )[0]
                                
                                indices_unselected = np.where(distances >= epsilon)[0]

                                list_nr_elements.append(indices_selected.shape[0])

                                pcd_selected_view = pcd_gt_hull.select_by_index(indices_selected)
                                pcd_selected_view.paint_uniform_color([1, 0, 1])

                                pcd_unselected_view = pcd_gt_hull.select_by_index(indices_unselected)
                                pcd_unselected_view.paint_uniform_color([0, 1, 0])

                                o3d.visualization.draw_geometries([mesh_hull,pcd_selected_view,axis])
                                # o3d.visualization.draw_geometries([mesh_hull,pcd_unselected_view,axis])


                                index_list.append(indices_selected)

                                #o3d.visualization.draw_geometries([pcd_gt_hull,pcd_view_hull,axis,pcd_mesh_orig_copy])

                                pcd_all_views += pcd_view_hull

                            else:
                                    list_nr_elements.append(0)
                                    index_list.append([])


                    o3d.visualization.draw_geometries([pcd_gt_hull])

                    


                    #initial_view = np.random.randint(viewspace.shape[0])

                    
                    views_select_list = [1,3,7,10,15,24,27,30]

                    for initial_view in views_select_list:
                    #for initial_view in range(3):

                        list_nr_elements_copy = copy.deepcopy(list_nr_elements)
                        index_list_copy = copy.deepcopy(index_list)

                    #initial_view = 1



                        selected_views = []

                        iteration_nr = 0

                        nr_elements_check = 0

                        start_time = time.time()


                        selected_views.append(initial_view)
                        current_list_elements = copy.deepcopy(index_list_copy[initial_view])

                        for i in range(len(index_list)):
                                index_list_copy[i] = set(index_list_copy[i]).difference(current_list_elements)
                                list_nr_elements_copy[i] = len(index_list_copy[i])

                        #print(str(iteration_nr)+"Initial Iteration Done")
                        #print(str(len(current_list_elements))+" elements selected")

                        nr_elements_check += len(current_list_elements)



                    

                        while(np.max(list_nr_elements_copy)>0) and iteration_nr < nr_max_iterations:
                            iteration_nr += 1

                            selected_views.append(np.argmax(list_nr_elements_copy))  

                            current_list_elements = copy.deepcopy(index_list_copy[np.argmax(list_nr_elements_copy)])

                            for i in range(len(index_list_copy)):
                                index_list_copy[i] = set(index_list_copy[i]).difference(current_list_elements)
                                list_nr_elements_copy[i] = len(index_list_copy[i])

                            #print(str(iteration_nr)+" Iteration Done")
                            #print(str(len(current_list_elements))+" elements selected")

                            nr_elements_check += len(current_list_elements)


                        #print("Time for SCO: "+str(time.time()-start_time))
                        #print("Total elements selected hull mesh: "+str(nr_elements_check))

                        pcd_all_views_selected_v2 = o3d.geometry.PointCloud()

                        # if len(selected_views) < nr_max_iterations:
                        #      print("Hopa")
                        indices_coverage_list = []

                        points_pcd_gt = np.asarray(pcd_mesh_orig.points)

                        

                        check_coverage_pcd = o3d.geometry.PointCloud()
                        check_hull_pcd = o3d.geometry.PointCloud()

                        axis_check = o3d.geometry.TriangleMesh()

                        hull_check, _ = pcd_gt_hull.compute_convex_hull()

                        hull_ls_check = o3d.geometry.LineSet.create_from_triangle_mesh(hull_check)

                        for view_iter in range(nr_max_iterations):

                            if (view_iter>=len(selected_views)):
                                view_select = np.random.randint(viewspace.shape[0])
                            else:
                                view_select = selected_views[view_iter]

                            #path_view = os.path.join(path_views_gt,category,model,"rot_"+str(index_transf),str(view_select)+".npy")

                            path_view = os.path.join(path_views_gt,category,model,str(view_select)+".pcd")

                            pcd_view = o3d.io.read_point_cloud(path_view)

                            
                            path_view_hull = path = os.path.join(path_views,str(view_select)+".npy")

                            points_view_hull = np.load(path_view_hull)

                            pcd_view_hull = o3d.geometry.PointCloud()
                            pcd_view_hull.points = o3d.utility.Vector3dVector(points_view_hull)

                            # pcd_view_hull.paint_uniform_color([1, 0, 1])

                            # color_view = [random.random(), random.random(), random.random()]

                            color_view = [0,0,1]


                            pcd_view_hull.paint_uniform_color([color_view[0], color_view[1], color_view[2]])

                            check_hull_pcd += pcd_view_hull

                            

                            #pcd_view = pcd_view.farthest_point_down_sample(2000)

                            points_view = np.asarray(pcd_view.points)

                            
                            pcd_view.paint_uniform_color([1, 0, 0])


                            pcd_all_views_selected_v2 += pcd_view




                         
                            
                            distances_gt_v2 = np.asarray(pcd_mesh_orig.compute_point_cloud_distance(pcd_all_views_selected_v2))
                            indices_selected_gt_v2 = np.where(distances_gt_v2 < epsilon
                                                        )[0]
                            
                            indices_coverage_list.append(indices_selected_gt_v2)
                            indices_coverage_final = list(set().union(*indices_coverage_list))

                            points_all_views_covered = points_pcd_gt[indices_coverage_final,:]

                            check_coverage_pcd.points = o3d.utility.Vector3dVector(points_all_views_covered)
                            check_coverage_pcd.paint_uniform_color([0,0,1])

                            check_coverage_pcd = check_coverage_pcd.rotate(R_before,center = (0,0,0))

                            axis_check += o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.12, origin=viewspace[view_select,:])


                            

                            #o3d.visualization.draw_geometries([hull_ls_check,pcd_mesh_orig_copy,check_coverage_pcd,axis_check],zoom=0.8,front=[-4,1,-5],lookat=[0.,0,0],up=[0.5, -0.5,1])

                           
                            #o3d.visualization.draw_geometries([hull_ls_check,check_coverage_pcd,axis_check],zoom=0.8,front=[5,0,0],lookat=[0.,0,0],up=[0, -1, 0])
                           

                           

                            coverage_view_2 = float(len(indices_coverage_final))/nr_points

                            #print(coverage_view_2)

                           
                            #print(coverage_view_2)

                            # print(coverage_view)
                            #o3d.visualization.draw_geometries([pcd_mesh_orig,pcd_all_views_selected_v2])

                            avg_cov_array[view_iter] += coverage_view_2
                            nr_cov_array[view_iter] += 1

                            

                            category_coverage[dict_models[category],view_iter] += coverage_view_2

                            # print(coverage_view_2)
                            nr_models_categ[dict_models[category],view_iter] += 1

                            
            #print(category_coverage[dict_models[category]]/nr_models_categ[dict_models[category]])

        print(avg_cov_array/nr_cov_array)

        # ceva = category_coverage/nr_models_categ

        # print(category_coverage/nr_models_categ)
                            
                            

                    

                

        