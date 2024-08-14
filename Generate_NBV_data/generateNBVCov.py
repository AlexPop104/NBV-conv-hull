import os
import numpy as np
import open3d as o3d
import random
import time
import pdb

import copy

import tqdm



if __name__ == '__main__':

    # os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    
    
    view_num = 33
    nr_views_choose= 2
    nr_steps_ahead=9

    nr_points_fps = 1024

    epsilon = 0.00737

    # root_folder = "/mnt/ssd1/Alex_data/Cuda_Data/Shapenet_with_image_transf"
    #folder_name = "3d_Objects_again"

    root_folder = "/mnt/ssd1/Alex_data/Test_PCNBV_again_2"
    folder_name = "3d_Shape_small"



    # path
    data_type = 'train'

    
    ShapeNetv1_dir = os.path.join(root_folder,'0_gt_pcd_scaled', folder_name)
    pc_dir = os.path.join(root_folder,'1_views_blender_pcd_translated', folder_name)

    scale_dir = os.path.join(root_folder,'0_gt_meshes_translated', folder_name)
    scale_dir_2 = os.path.join(root_folder,'0_gt_meshes_translated_2', folder_name)


    save_dir = os.path.join(root_folder,'5_NBV_coverage_translated', folder_name)
    class_dir = os.path.join(ShapeNetv1_dir,data_type)

    class_list = os.listdir(class_dir)

    f=open('generate_nbv.log', 'w+')

    for class_id in tqdm.tqdm(class_list):

        model_list = os.listdir(os.path.join(ShapeNetv1_dir, data_type, class_id))

        for model in tqdm.tqdm(model_list):

            path_gt_points = os.path.join(ShapeNetv1_dir, data_type, class_id, model, 'gt.pcd')
            save_model_path = os.path.join(save_dir,  data_type, class_id, model)
          
            gt_pcd = o3d.io.read_point_cloud(path_gt_points)
            gt_pcd.paint_uniform_color([0, 0, 1])

            
            scale_model_1 = np.load(os.path.join(scale_dir, data_type, class_id, model, 'scale.npy'))
            scale_model_2 = np.load(os.path.join(scale_dir_2, data_type, class_id, model, 'scale.npy'))

            gt_points = np.asarray(gt_pcd.points)

            


            # gt_points = gt_points * scale_model_1


           
            part_points_list = []

            view_selection_list = []
            
            for i in range(view_num):
                pcd_path = os.path.join(pc_dir,data_type,class_id, model, str(i) + ".pcd")
                if os.path.exists(pcd_path):
                    cur_pc = o3d.io.read_point_cloud(pcd_path)
                    cur_pc.paint_uniform_color([1,0,0])

                    R = cur_pc.get_rotation_matrix_from_xyz((-np.pi/2,0,0))
                    cur_pc = cur_pc.rotate(R, center=(0, 0, 0))

                    cur_points = np.array(cur_pc.points)
                    cur_points = cur_points * 1/ scale_model_1
                    cur_points = cur_points * scale_model_2
                    
                    cur_pc.points = o3d.utility.Vector3dVector(cur_points)

                    R = cur_pc.get_rotation_matrix_from_xyz((np.pi/2,0,0))
                    cur_pc = cur_pc.rotate(R, center=(0, 0, 0))

                    cur_points = np.asarray(cur_pc.points)  

                    # o3d.visualization.draw_geometries([gt_pcd,cur_pc])

                    view_selection_list.append(i)
                else:
                    cur_points = np.zeros((1,3))

                part_points_list.append(cur_points)

            # reconstruct from different views 1 times
            selected_init_view = []

            
            random.shuffle(view_selection_list)
            view_selection_list = view_selection_list[0:nr_views_choose]

            start = time.time() 

            for ex_index in view_selection_list: 
            #for ex_index in range(16):  

                

                cur_ex_dir = os.path.join(save_dir,data_type,class_id, model, str(ex_index))
                if not os.path.exists(cur_ex_dir):
                    os.makedirs(cur_ex_dir) 

                # init view state
                view_state = np.zeros(view_num) # 0 unselected, 1 selected, 2 cur
                # init start view
                # while (True):
                #     cur_view = random.randint(0, view_num - 1)
                #     #cur_view = random.randint(0, nr_views_choose-1)
                #     if not cur_view in selected_init_view:
                #         selected_init_view.append(cur_view)
                #         break   
                cur_view = ex_index

                view_state[cur_view] = 1
                #view_state[0]=1

                acc_pc_points = part_points_list[cur_view]  

                acc_pcd = o3d.geometry.PointCloud()
                acc_pcd.points = o3d.utility.Vector3dVector(acc_pc_points)

                dist2_new = np.asarray(gt_pcd.compute_point_cloud_distance(acc_pcd))
                indices_selected_gt_v2 = np.where(dist2_new < epsilon)[0]

                cur_cov = float(indices_selected_gt_v2.shape[0])/gt_points.shape[0]

                

                # max scan 10 times
                for scan_index in range(nr_steps_ahead):   

                    if cur_cov < 0.9999: 


                        target_value = np.zeros((view_num, 1)) # surface coverage, register coverage, moving cost for each view         

                        max_view_index = 0
                        max_view_cov = 0
                        max_new_pc = np.zeros((1,3))

                        batch_acc_pcd = o3d.geometry.PointCloud()
                        batch_acc_pcd.points = o3d.utility.Vector3dVector(acc_pc_points)
                        batch_acc_pcd.paint_uniform_color([1, 0, 0])

                        # print(batch_acc_pcd)

                        # o3d.visualization.draw_geometries([batch_acc_pcd, gt_pcd])

                        # # # accumulate points coverage
                        # batch_acc = acc_pc_points[np.newaxis, :, :]
                        # batch_gt = gt_points[np.newaxis, :, :]
                    
                        # evaluate all the views
                        for i in range(view_num):   

                          
                            batch_part_cur = part_points_list[i]

                            batch_part_cur_pcd = o3d.geometry.PointCloud()
                            batch_part_cur_pcd.points = o3d.utility.Vector3dVector(batch_part_cur)

                            dist1_new =  np.asarray(batch_part_cur_pcd.compute_point_cloud_distance(batch_acc_pcd))
                            dis_flag_new = dist1_new < epsilon 

                            # # new pc
                            # dist1_new = sess.run(dist1, feed_dict={part_tensor:batch_part_cur, gt_tensor:batch_acc})
                            # dis_flag_new = dist1_new < 0.00005  

                            

                            pc_register = batch_part_cur[dis_flag_new]
                            pc_new = batch_part_cur[~dis_flag_new]

                            pcd_register = o3d.geometry.PointCloud()
                            pcd_register.points = o3d.utility.Vector3dVector(pc_new)
                            pcd_register.paint_uniform_color([random.random(), random.random(), random.random()])


                            batch_new_pcd =  pcd_register+batch_acc_pcd

                            

                            batch_new = pc_new[np.newaxis, :, :]    

                            # test new coverage
                            if batch_new.shape[1] != 0:

                                dist2_new =  np.asarray(gt_pcd.compute_point_cloud_distance(batch_new_pcd))
                                indices_selected_gt_v2 = np.where(dist2_new < epsilon)[0]

                                

                                cover_new = float(indices_selected_gt_v2.shape[0])/gt_points.shape[0]

                                target_value[i, 0] = np.abs(cover_new - cur_cov)

                                #target_value[i, 0] = cover_new

                                

                            else:
                                cover_new = 0  
                                target_value[i, 0] = 0
                                #target_value[i, 0] = cur_cov

                            

                            if ( target_value[i, 0] > max_view_cov ):
                                max_view_index = i
                                max_view_cov = target_value[i, 0]
                                max_new_pc = pc_new 

                        if max_view_cov <= 0.000001:
                            # print('coverage not increase, break')
                            # f.write('coverage not increase, break' +'\n')
                            break

                        np.save(os.path.join(cur_ex_dir, str(scan_index) + "_viewstate.npy") ,view_state)

                        batch_acc_pcd_export = copy.deepcopy(batch_acc_pcd)

                        if acc_pc_points.shape[0] > nr_points_fps:
                            batch_acc_pcd_export = batch_acc_pcd_export.farthest_point_down_sample(nr_points_fps)

                        o3d.io.write_point_cloud(os.path.join(cur_ex_dir, str(scan_index) + "_acc_pc.pcd"), batch_acc_pcd_export, write_ascii=True)


                        #np.save(os.path.join(cur_ex_dir, str(scan_index) + "_acc_pc.npy"), acc_pc_points)

                        np.save(os.path.join(cur_ex_dir, str(scan_index) + "_target_value.npy"), target_value)  

                        # print("choose view:" + str(max_view_index) + " add coverage:" + str(max_view_cov)) 

                        # f.write("choose view:" + str(max_view_index) + " add coverage:" + str(max_view_cov) +'\n')  

                        

                        

                        cur_view = max_view_index

                        cur_cov = cur_cov + target_value[max_view_index, 0]
                        # cur_cov =  target_value[max_view_index, 0]

                        view_state[cur_view] = 1
                        acc_pc_points = np.append(acc_pc_points, max_new_pc, axis=0)   

                        

                        batch_acc_pcd.points = o3d.utility.Vector3dVector(acc_pc_points)

                        batch_acc_pcd.paint_uniform_color([1, 0, 0])

                        #o3d.visualization.draw_geometries([batch_acc_pcd, gt_pcd])

                        # print('scan %s done, time=%.4f sec' % (scan_index, time.time() - start))

                        # f.write('scan %s done, time=%.4f sec' % (scan_index, time.time() - start) +'\n')
            print('model %s done, time=%.4f sec' % (model, time.time() - start))

            


