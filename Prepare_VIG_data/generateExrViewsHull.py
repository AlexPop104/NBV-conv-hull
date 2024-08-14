# Author: Wentao Yuan (wyuan1@cs.cmu.edu) 05/31/2018
# Modified by Rui Zeng 07/12/2020

import bpy
import mathutils
import numpy as np
import os
import sys
import time
import pdb
import argparse

import copy
# Usage: blender -b -P render_depth_shapenet.py

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


def setup_blender(width, height, focal_length, output_dir):
    # camera
    camera = bpy.data.objects['Camera']
    camera.data.angle = np.arctan(width / 2 / focal_length) * 2
    # camera.data.clip_end = 1.2
    # camera.data.clip_start = 0.2

    # render layer
    scene = bpy.context.scene
    scene.render.filepath = 'buffer'
    scene.render.image_settings.color_depth = '16'
    scene.render.resolution_percentage = 100
    scene.render.resolution_x = width
    scene.render.resolution_y = height

    # compositor nodes
    scene.use_nodes = True
    tree = scene.node_tree
    rl = tree.nodes.new('CompositorNodeRLayers')
    output = tree.nodes.new('CompositorNodeOutputFile')
    output.base_path = ''
    output.format.file_format = 'OPEN_EXR'
    #tree.links.new(rl.outputs['Z'], output.inputs[0])
    tree.links.new(rl.outputs['Depth'], output.inputs[0]) #-Original

    # remove default cube
    bpy.data.objects['Camera'].select=False
    bpy.data.objects['Cube'].select = True
    bpy.ops.object.delete()
    bpy.data.objects['Camera'].select=True

    return scene, camera, output





if __name__ == '__main__':

    

   

    #nr_views=4
    print("Desired number of views:")
    nr_views = int(input())

    angle = 360 /nr_views

    print("Random viewspace? (y/n)")
    random_viewspace_sel = input()

    

    
    #folder_dataset = "3d_Objects_full"
    folder_dataset = "3d_Shape_small"


    root_dataset = "/mnt/ssd1/Alex_data/Test_PCNBV_again_2"

    path_scale = os.path.join(root_dataset,"0_gt_meshes_translated_2",folder_dataset)

    
    if random_viewspace_sel == "y":
        viewspace_path = "/home/alex/Alex_documents/Alex_work_new2/Alex_work/GeoA3/Extra_code_Neurocomputing/Cod_PC_ALEX/New_version_NBVnet_p4/viewspace_shapenet_33_rand.txt"
        output_dir_intrinsics = os.path.join(root_dataset,'8_blender_views_otherviews_exr',folder_dataset)
    else:
        
        viewspace_path = "/home/alex/Alex_documents/Alex_work_new2/Alex_work/GeoA3/Extra_code_Neurocomputing/Cod_PC_ALEX/New_version_PCNBV/viewspace.txt"
        output_dir_intrinsics = os.path.join(root_dataset,'8_blender_views_hull_exr',folder_dataset)

    
    print(viewspace_path)
    

    path_meshes = os.path.join(root_dataset,"7_NBV_hull_selected")
    ShapeNetv1_dir_path =  os.path.join(path_meshes,folder_dataset,"valid")

    
    
    category_list_categs = os.listdir(ShapeNetv1_dir_path)

    output_extra = os.path.join(output_dir_intrinsics,"extra")


    if not os.path.exists(output_extra):
            os.makedirs(output_extra)

    
    width = 640
    height = 480
    focal = 238 * 2

    scene, camera, output = setup_blender(width, height, focal, output_dir_intrinsics)
    intrinsics = np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]])

    open('blender.log', 'w+').close()
    
    if not os.path.exists(output_dir_intrinsics):
        os.makedirs(output_dir_intrinsics)

    np.savetxt(os.path.join(output_dir_intrinsics, 'intrinsics.txt'), intrinsics, '%f')

    datatype_list = ["test"]
    

    for data_type in datatype_list:

        ShapeNetv1_dir = os.path.join(path_meshes,folder_dataset)
       
        output_dir = os.path.join(output_dir_intrinsics, data_type) 

        viewspace = np.loadtxt(viewspace_path)
       
        category_list = os.listdir(os.path.join(ShapeNetv1_dir, data_type))

        index_transf=0

        file_path = os.path.join(output_extra,data_type+"_classes.txt")
        file= open(file_path, "w")
        for categ in category_list:
            file.write(categ+"\n")
        file.close()


        for category_id in category_list:

        
            model_list = os.listdir(os.path.join(ShapeNetv1_dir, data_type,category_id))

            for model_id in model_list:

                scale_model = np.load(os.path.join(path_scale,data_type,category_id,model_id,"scale.npy"))
                # Selected_rotation = T[index_transf,:]
                

                start = time.time()
                exr_dir = os.path.join(output_dir, 'exr',category_id, model_id)
                pose_dir = os.path.join(output_dir, 'pose',category_id, model_id)
                
                if os.path.exists(os.path.join(exr_dir, '32.exr')):
                    print("skip " + exr_dir)
                    continue

                os.makedirs(exr_dir, exist_ok=True)
                os.makedirs(pose_dir, exist_ok=True) 

                

                # Redirect output to log file
                old_os_out = os.dup(1)
                os.close(1)
                os.open('blender.log', os.O_WRONLY) 

                # Import mesh model
                model_path = os.path.join(ShapeNetv1_dir, data_type,category_id, model_id, 'model_hull.obj')
                bpy.ops.import_scene.obj(filepath=model_path)  

                # for ob in scene.objects:
                #     if ob.type == 'MESH':
                #         ob.scale[0] *= scale_model[0]
                #         ob.scale[1] *= scale_model[1]
                #         ob.scale[2] *= scale_model[2]
                        
                #         ob.select = True
                    
                #         bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
                #         ob.select = False  

                #         ob.rotation_euler[0] = ob.rotation_euler[0] + rot_x
                #         ob.rotation_euler[1] = ob.rotation_euler[1] + rot_y
                #         ob.rotation_euler[2] = ob.rotation_euler[2] + rot_z

                

                print(bpy.context.window)
                print(bpy.context.mode)
                print(bpy.context.area)
                print(bpy.context.region)
                print(bpy.context.scene)

                # Rotate model by 90 degrees around x-axis (z-up => y-up) to match ShapeNet's coordinates
                #bpy.ops.transform.rotate(value=-np.pi / 2, axis=(1, 0, 0))  # ->original code

                points_pcd_viewspace = copy.deepcopy(viewspace)

                points_pcd_viewspace = scalePoints(points_pcd_viewspace, scale_model[0], scale_model[1], scale_model[2])
                

                # Render
                #for i in range(viewspace.shape[0]):
                for i in range(nr_views):    # added just for 16 views. (Above ones)
                    scene.frame_set(i)

                    print("Scale" ,scale_model)
                    print(viewspace[i])
                    print(points_pcd_viewspace[i])

                    cam_pose = mathutils.Vector((points_pcd_viewspace[i][0], points_pcd_viewspace[i][1], points_pcd_viewspace[i][2]))
                    center_pose = mathutils.Vector((0,0,0))
                    direct = center_pose - cam_pose
                    rot_quat = direct.to_track_quat('-Z', 'Y')
                    camera.rotation_euler = rot_quat.to_euler()
                    # translated_cam_pose = mathutils.Vector((translated_points_pcd_viewspace[i][0], translated_points_pcd_viewspace[i][1], translated_points_pcd_viewspace[i][2]))
                    # camera.location = translated_cam_pose
                    camera.location = cam_pose
                    pose_matrix = camera.matrix_world
                    output.file_slots[0].path = os.path.join(exr_dir, '#.exr')
                    bpy.ops.render.render(write_still=True)
                    np.savetxt(os.path.join(pose_dir, '%d.txt' % i), pose_matrix, '%f') 

                # Clean up
                bpy.ops.object.delete()
                for m in bpy.data.meshes:
                    bpy.data.meshes.remove(m)
                for m in bpy.data.materials:
                    m.user_clear()
                    bpy.data.materials.remove(m)    

                # Show time
                os.close(1)
                os.dup(old_os_out)
                os.close(old_os_out)
                print('%s done, time=%.4f sec' % (model_id, time.time() - start))

            #index_transf+=1
