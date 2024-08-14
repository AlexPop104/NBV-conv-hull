import numpy as np
import open3d as o3d
import os
import shutil



def remove(path):
    """ param <path> could either be relative or absolute. """
    if os.path.isfile(path) or os.path.islink(path):
        os.remove(path)  # remove the file
    elif os.path.isdir(path):
        shutil.rmtree(path)  # remove dir and all contains
    else:
        raise ValueError("file {} is not a file or dir.".format(path))





root_folder="/mnt/ssd1/Alex_data/Test_PCNBV_again/Shapenet_without_image/"

dataset_type_list = ["train" ,"valid","test"]
#dataset_type_list = ["test"]

for dataset_type in dataset_type_list:

    path_folder = os.path.join(root_folder,dataset_type)

    root_dir_list = os.listdir(path_folder)

    for category in root_dir_list:
        model_list= os.listdir(os.path.join(path_folder, category))
        
        


        for model_id in model_list:
            sub_folder_list= os.listdir(os.path.join(path_folder,category, model_id))
            
            for sub_fold_id in sub_folder_list:
                
                if ("imag"  in sub_fold_id):
                    
                    path_image= os.path.join(path_folder,category, model_id,sub_fold_id)
                    print(str(path_image))
                    remove(path_image)

print("Done removing image folders")
            
            
            
        
    
        
        
        
        