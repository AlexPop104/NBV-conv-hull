import os
import numpy as np
import open3d as o3d
import random
import time
import pdb

import copy



if __name__ == '__main__':

    # os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    
    
    view_num = 33
    nr_views_choose= 4
    nr_steps_ahead=10

    nr_points_fps = 1024

    epsilon = 0.00737

    root_folder = "/mnt/ssd1/Alex_data/Test_PCNBV_again_2"
    folder_name = "3d_Shape_small"

    # path
    data_type = 'train'

    print("Start class")
    class_start = int(input())

    print("End class")
    class_end = int(input())

    
    ShapeNetv1_dir = os.path.join(root_folder,'0_gt_pcd_scaled', folder_name)
    pc_dir = os.path.join(root_folder,'1_views_blender_pcd_translated', folder_name)

    scale_dir = os.path.join(root_folder,'0_gt_meshes_translated', folder_name)
    scale_dir_2 = os.path.join(root_folder,'0_gt_meshes_translated_2', folder_name)


    save_dir = os.path.join(root_folder,'5_NBV_vig_translated', folder_name)
    class_dir = os.path.join(ShapeNetv1_dir,data_type)

    class_list = os.listdir(class_dir)

    f=open('generate_nbv.log', 'w+')

    for class_id in class_list[class_start:class_end]:

        model_list = os.listdir(os.path.join(ShapeNetv1_dir, data_type, class_id))

        for model in model_list:

            path_gt_points = os.path.join(ShapeNetv1_dir, data_type, class_id, model, 'gt.pcd')
            path_gt_volume = os.path.join(ShapeNetv1_dir, data_type, class_id, model, 'volume.npy')

            save_model_path = os.path.join(save_dir,  data_type, class_id, model)
          
            gt_pcd = o3d.io.read_point_cloud(path_gt_points)

            scale_model_1 = np.load(os.path.join(scale_dir, data_type, class_id, model, 'scale.npy'))
            scale_model_2 = np.load(os.path.join(scale_dir_2, data_type, class_id, model, 'scale.npy'))

            gt_volume = np.load(path_gt_volume)
            
            gt_points = np.array(gt_pcd.points)
           
            part_points_list = []

            view_selection_list = []
            
            for i in range(view_num):
                pcd_path = os.path.join(pc_dir,data_type,class_id, model, str(i) + ".pcd")
                if os.path.exists(pcd_path):
                    cur_pc = o3d.io.read_point_cloud(pcd_path)
                    R = cur_pc.get_rotation_matrix_from_xyz((-np.pi/2,0,0))
                    cur_pc = cur_pc.rotate(R, center=(0, 0, 0))

                    cur_points = np.array(cur_pc.points)
                    cur_points = cur_points * 1/ scale_model_1
                    cur_points = cur_points * scale_model_2
                    
                    cur_pc.points = o3d.utility.Vector3dVector(cur_points)

                    R = cur_pc.get_rotation_matrix_from_xyz((np.pi/2,0,0))
                    cur_pc = cur_pc.rotate(R, center=(0, 0, 0))

                    if cur_points.shape[0] > nr_points_fps:
                        cur_pc = cur_pc.farthest_point_down_sample(512)
                    

                    cur_points = np.asarray(cur_pc.points)  

                    view_selection_list.append(i)
                else:
                    cur_points = np.zeros((1,3))

                part_points_list.append(cur_points)

            # reconstruct from different views 1 times
            selected_init_view = []

            
            random.shuffle(view_selection_list)
            view_selection_list = view_selection_list[0:nr_views_choose]

            for ex_index in view_selection_list: 
            #for ex_index in range(16):  

                start = time.time() 

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
                acc_pcd.paint_uniform_color([1, 0, 0])

                try :
                    hull_part , _ = acc_pcd.compute_convex_hull()
                
                

                    if hull_part.is_watertight():

                        volume_part = hull_part.get_volume()

                        cur_cov = 1- np.abs(1-volume_part/gt_volume) 

                    else:
                        cur_cov = 0
                except:
                    cur_cov = 0

                # dist2_new = np.asarray(gt_pcd.compute_point_cloud_distance(acc_pcd))
                # indices_selected_gt_v2 = np.where(dist2_new < epsilon)[0]

                # cur_cov = float(indices_selected_gt_v2.shape[0])/gt_points.shape[0]

                

                # max scan 10 times
                for scan_index in range(nr_steps_ahead):   

                    if cur_cov < 0.999: 

                        print("coverage:" + str(cur_cov) + " in scan round " + str(scan_index)) 

                        f.write("coverage:" + str(cur_cov) + " in scan round " + str(scan_index) +'\n')

                        
                        # np.savetxt(os.path.join(cur_ex_dir, str(scan_index) + "_acc_pc.xyz"), acc_pc_points)    

                        target_value = np.zeros((view_num, 1)) # surface coverage, register coverage, moving cost for each view         

                        max_view_index = 0
                        max_view_cov = 0
                        max_new_pc = np.zeros((1,3))

                        batch_acc_pcd = o3d.geometry.PointCloud()
                        batch_acc_pcd.points = o3d.utility.Vector3dVector(acc_pc_points)
                        batch_acc_pcd.paint_uniform_color([1, 0, 0])

                        print(batch_acc_pcd)

                        
                        # evaluate all the views
                        for i in range(view_num):   


                            batch_part_cur = part_points_list[i]

                            batch_part_cur_pcd = o3d.geometry.PointCloud()
                            batch_part_cur_pcd.points = o3d.utility.Vector3dVector(batch_part_cur)

                            batch_new_pcd =  batch_part_cur_pcd+batch_acc_pcd

                            batch_new_pcd.paint_uniform_color([0, 1, 0])

                           

                            try :

                                hull_part , _ = batch_new_pcd.compute_convex_hull()

                                if hull_part.is_watertight():

                                    volume_part = hull_part.get_volume()

                                    cover_new = 1- np.abs(1-volume_part/gt_volume) 

                                  
                                else:
                                    cover_new = 0

                            except:
                                cover_new = 0


                            target_value[i, 0] = cover_new

                            if ( target_value[i, 0] > max_view_cov ):
                                if view_state[i] == 0:
                                    max_view_index = i
                                    max_view_cov = target_value[i, 0]
                                    max_new_pc = batch_part_cur 

                        if max_view_cov <= cur_cov:
                            print('coverage not increase, break')
                            f.write('coverage not increase, break' +'\n')
                            break

                        np.save(os.path.join(cur_ex_dir, str(scan_index) + "_viewstate.npy") ,view_state)

                        batch_acc_pcd_export = copy.deepcopy(batch_acc_pcd)

                        if acc_pc_points.shape[0] > nr_points_fps:
                            batch_acc_pcd_export = batch_acc_pcd_export.farthest_point_down_sample(nr_points_fps)

                        o3d.io.write_point_cloud(os.path.join(cur_ex_dir, str(scan_index) + "_acc_pc.pcd"), batch_acc_pcd_export, write_ascii=True)


                        #np.save(os.path.join(cur_ex_dir, str(scan_index) + "_acc_pc.npy"), acc_pc_points)

                        np.save(os.path.join(cur_ex_dir, str(scan_index) + "_target_value.npy"), target_value)  

                        # print(target_value.T)

                        print("Coverage current:" + str(cur_cov) + " Coverage max:" + str(max_view_cov))

                        print("choose view:" + str(max_view_index) + " add coverage:" + str(max_view_cov-cur_cov)) 

                        f.write("choose view:" + str(max_view_index) + " add coverage:" + str(max_view_cov-cur_cov) +'\n')  

                        

                        target_value_rel =  target_value - cur_cov

                        cur_view = max_view_index

                        #cur_cov = cur_cov + target_value[max_view_index, 0]
                        cur_cov =  target_value[max_view_index, 0]

                        view_state[cur_view] = 1
                        acc_pc_points = np.append(acc_pc_points, max_new_pc, axis=0)   

                        

                        batch_acc_pcd.points = o3d.utility.Vector3dVector(acc_pc_points)

                        batch_acc_pcd.paint_uniform_color([1, 0, 0])

                        # o3d.visualization.draw_geometries([batch_acc_pcd, gt_pcd])

                        print('scan %s done, time=%.4f sec' % (scan_index, time.time() - start))

                        f.write('scan %s done, time=%.4f sec' % (scan_index, time.time() - start) +'\n')


            


