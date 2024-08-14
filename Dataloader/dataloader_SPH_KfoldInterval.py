from nis import cat
from unicodedata import category
import numpy as np
import os

import torch
# from torch.utils.data import DataLoader
from torchvision import transforms, utils

from torch_geometric.loader import DenseDataLoader, DataLoader
from torch_geometric.data.dataset import Dataset

import torch_geometric

from path import Path




import open3d as o3d



from random import choice


import time

def normalize_point_batch(pc, NCHW=True):
    """
    normalize a batch of point clouds
    :param
        pc      [B, N, 3] or [B, 3, N]
        NCHW    if True, treat the second dimension as channel dimension
    :return
        pc      normalized point clouds, same shape as input
        centroid [B, 1, 3] or [B, 3, 1] center of point clouds
        furthest_distance [B, 1, 1] scale of point clouds
    """
    point_axis = 2 if NCHW else 1
    dim_axis = 1 if NCHW else 2
    centroid = torch.mean(pc, dim=point_axis, keepdim=True)
    pc = pc - centroid
    furthest_distance, _ = torch.max(
        torch.sqrt(torch.sum(pc ** 2, dim=dim_axis, keepdim=True)), dim=point_axis, keepdim=True)
    pc = pc / furthest_distance
    return pc



def normalize_pc(points):
    centroid =np.mean(points)
    points -=centroid
    furthest_distance = np.max(np.sqrt(np.sum(abs(points)**2,axis=-1)))
    points /= furthest_distance
    
    return points

def gen_random_number(low, high, exclude):
    return choice(
        [number for number in range(low, high)
         if number not in exclude]
    )

def default_transforms():
    return transforms.Compose([
        # transforms.PointSampler(512),
        # transforms.Normalize(),
        # transforms.ToTensor()
    ])


# SKIP_STRINGS = ["b51032670accbd25d11b9a498c2f9ed5/"+"2/0_2_acc_pc.pcd",
#                 "ab35aa631852d30685dfb1711fe4ff6d/3/0_3_acc_pc.pcd",
#                 "a19a5a459d234919c1ccec171a275967/1/0_1_acc_pc.pcd",
#                 "30bfb515f027be9a4642ec4b6f68a/1/0_1_acc_pc.pcd"
#                 ]

SKIP_STRINGS =[]

class PcdDataset(Dataset):
    def __init__(self, root_dir,root_spherical, valid=False, folder="train", transform=default_transforms(), with_normals=False,nr_views=5,nr_points=512,list_categ=[]):
        self.root_dir = root_dir
        self.nr_points = nr_points

        self.root_spherical = root_spherical
       

        folders = [dir for dir in sorted(os.listdir(self.root_dir/folder)) if os.path.isdir(self.root_dir/folder/dir)]
        self.classes = {folder: i for i, folder in enumerate(folders)}
        self.nr_views=nr_views

        self.transforms = transform
        self.valid = valid
        self.files = []
        self.with_normals = with_normals
        self.folder=folder

        selection_classes=list(self.classes.keys())


        if (list_categ != []):
            selection_classes = [selection_classes[i] for i in list_categ]
        
        
        for category in selection_classes:
                model_list = os.listdir(self.root_spherical/folder/Path(category))
                for model in model_list:
                            view_list = os.listdir(self.root_spherical/folder/Path(category)/Path(model))
                            for i in view_list:
                                new_dir = self.root_spherical/folder/Path(category)/Path(model)/Path(i)
                                if os.path.exists(new_dir):
                                    for file in os.listdir(new_dir):
                                        if file.endswith('_acc_pc_spherical.pcd'):
                                            sample = {}
                                            sample['pcd_path'] = new_dir/file
                                            sample['category'] = category
                                            self.files.append(sample)
            
                
    def __len__(self):
        return len(self.files)

    def __preproc__(self, file, idx):
        start_time = time.time()

        nr_points_desired = self.nr_points

        pcd_sph = o3d.io.read_point_cloud(file)
        points_sph = torch.tensor(np.asarray(pcd_sph.points))
        points_sph = points_sph.unsqueeze(0)


        

        path_pcd = file.replace(self.root_spherical,self.root_dir)
        
        path_pcd = path_pcd.replace("_acc_pc_spherical.pcd","_acc_pc.pcd")

        

        # pcd_orig = o3d.io.read_point_cloud(path_pcd)
        # points = np.asarray(pcd_orig.points)

        # if(points.shape[0]<nr_points_desired):
        #     nr_repetitions = nr_points_desired // points.shape[0]
        #     nr_point_to_add = nr_points_desired - nr_repetitions*points.shape[0]

        #     points_repeat = np.tile(points,(nr_repetitions,1))

        #     nr_points_remaining = nr_points_desired - nr_repetitions*points.shape[0]

        #     points_repeat = np.concatenate((points_repeat,points[0:nr_points_remaining]),axis=0)

        #     points = points_repeat

        # else:
        #     pcd_orig = pcd_orig.farthest_point_down_sample(nr_points_desired)
        #     points = np.asarray(pcd_orig.points)
 
        
        path_viewstate = path_pcd.replace(".pcd",".npy")
        path_viewstate = path_viewstate.replace("_acc_pc","_viewstate")
        

        viewstate = torch.from_numpy(np.load(path_viewstate))
        
        viewstate = viewstate.unsqueeze(0)
        
        path_target_values = path_pcd.replace(".pcd",".npy")
        path_target_values = path_target_values.replace("_acc_pc","_target_value")
        
        
        
        
        
        target_values = torch.from_numpy(np.load(path_target_values))
        target_values = target_values.unsqueeze(0)
        
        
        unvisited_target_values =  target_values - torch.mul(viewstate,target_values)
        
        next_position=torch.argmax(unvisited_target_values)
        
                   
        
        # points = torch.tensor(points)
        # points = points.unsqueeze(0)

        # normals=np.asarray(pcd_orig.normals)

        pointcloud = torch_geometric.data.Data(pos_sph=points_sph, viewstate=viewstate, target_values = target_values,next_position =  next_position, obj_class = self.classes[self.files[idx]['category']])

        if self.transforms:
            pointcloud = self.transforms(pointcloud)

        

        return pointcloud

    def __getitem__(self, idx):
        pcd_path = self.files[idx]['pcd_path']
        with open(pcd_path, 'r') as f:
            pointcloud = self.__preproc__(f.name.strip(), idx)

        return pointcloud


if __name__ == '__main__':
    nr_views=33
    
    folder="valid/"

    nr_points = 1024
    l_max  = 3

    sel_classes = list(range(7))

    #root = Path("/mnt/ssd1/Alex_data/Test_PCNBV_again_2/4_Export_NBV_gt_pcd_SCO/3d_Shape_small/views/")
    
    root = Path("/mnt/ssd1/Alex_data/Test_PCNBV_again_2/5_NBV_coverage/3d_Shape_small/")

    path_spherical = os.path.join("/mnt/ssd1/Alex_data/Test_PCNBV_again_2/6_NBV_spherical/",str(l_max),"3d_Shape_small/")
    root_spherical = Path(path_spherical)

    test_dataset = PcdDataset(root,valid=True,folder=folder,nr_views=nr_views,nr_points=nr_points,root_spherical=root_spherical,list_categ=sel_classes)
    loader_test = DataLoader(test_dataset, batch_size=1,shuffle = False)
    
    
    i=0

    print("Length of dataset:"+str(len(test_dataset)))

    for i in range(len(test_dataset)):

        data = test_dataset[i]

        print(data.pos_sph.shape)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(data.pos_sph[0].numpy())
        pcd.paint_uniform_color([0.1, 0.1, 0.7])
        o3d.visualization.draw_geometries([pcd])
       

        

