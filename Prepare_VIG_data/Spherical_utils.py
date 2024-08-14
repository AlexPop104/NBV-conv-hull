import numpy as np
import open3d as o3d
import pyshtools as pysh
import copy
import torch
# from cv2 import getGaussianKernel

# from torch_geometric.nn import fps


# import spherical_harmonics_defense

def convert_pc_to_radial(pc):

    pc = torch.from_numpy(pc)
    origin = torch.mean(pc, axis=0)  # the center of the unit sphere
    pc -= origin  # for looking from the perspective of the origin
    npc = len(pc)
    origin = origin.to("cpu").numpy()

    pc_x, pc_y, pc_z = pc[:, 0], pc[:, 1], pc[:, 2]

    pc_r = torch.sqrt(pc_x * pc_x + pc_y * pc_y + pc_z * pc_z)
    pc_lat = torch.arcsin(pc_z / pc_r)
    pc_lon = torch.atan2(pc_y, pc_x)
    pc_r = pc_r.view(npc, 1)
    pc_lat = pc_lat.view(npc, 1)
    pc_lon = pc_lon.view(npc, 1)

    return pc_r, pc_lat, pc_lon, origin

def convert_pc_to_grid(pc, lmax, device="cuda"):
    """ Function for projecting a point cloud onto the surface of the unit sphere centered at its centroid. """

    #pc = torch.from_numpy(pc).to(device)

    new_pc = copy.deepcopy(pc)

    new_pc = torch.from_numpy(new_pc)

    #pc = torch.from_numpy(pc)

    #grid = pysh.SHGrid.from_zeros(lmax, grid='DH')
    grid = pysh.SHGrid.from_zeros(lmax, grid='DH',sampling=1)
    nlon = grid.nlon
    nlat = grid.nlat
    ngrid = nlon * nlat

    # grid_lon = torch.from_numpy(np.linspace(0, nlon * np.pi * 2 / (nlon - 1), num=nlon, endpoint=False)).to(device)
    # grid_lat = torch.from_numpy(np.linspace(nlat // 2 * np.pi / -(nlat - 1), np.pi / 2, num=nlat, endpoint=True)).to(device)

    grid_lon = torch.from_numpy(np.linspace(0, nlon * np.pi * 2 / (nlon - 1), num=nlon, endpoint=False))
    grid_lat = torch.from_numpy(np.linspace(nlat // 2 * np.pi / -(nlat - 1), np.pi / 2, num=nlat, endpoint=True))

    grid_lon = torch.broadcast_to(grid_lon, (nlat, nlon))
    grid_lat = torch.broadcast_to(grid_lat.view(nlat, 1), (nlat, nlon))
    grid_lon = grid_lon.reshape(1, ngrid)
    grid_lat = grid_lat.reshape(1, ngrid)

    origin = torch.mean(new_pc, axis=0)  # the center of the unit sphere
    #pc -= origin  # for looking from the perspective of the origin
    new_pc -= origin
    npc = len(new_pc)
    origin = origin.to("cpu").numpy()

    pc_x, pc_y, pc_z = new_pc[:, 0], new_pc[:, 1], new_pc[:, 2]

    pc_r = torch.sqrt(pc_x * pc_x + pc_y * pc_y + pc_z * pc_z)
    pc_lat = torch.arcsin(pc_z / pc_r)
    pc_lon = torch.atan2(pc_y, pc_x)
    pc_r = pc_r.view(npc, 1)
    pc_lat = pc_lat.view(npc, 1)
    pc_lon = pc_lon.view(npc, 1)

    cilm, chi2 = pysh.expand.SHExpandLSQ(pc_r, pc_lat, pc_lon, lmax)

    dist = -torch.cos(grid_lat) * torch.cos(pc_lat) * torch.cos(grid_lon - pc_lon) + torch.sin(grid_lat) * torch.sin(pc_lat)

    # value_threshold = 0.5
    # dist[dist>value_threshold]=0
    # dist[dist<-value_threshold]=0

    argmin = torch.argmin(dist, axis=0)
    grid_r = pc_r[argmin].view(nlat, nlon)
    grid.data = grid_r.to("cpu").numpy()  # data of the projection onto the unit sphere

    argmin = torch.argmin(dist, axis=1)  # argmin on a different axis
    
    
    flag = torch.zeros(ngrid, dtype=bool)
    
    
    flag[argmin] = True  # indicates the polar angles for which the grid data can be interpreted as a point
    flag = flag.to("cpu").numpy()


    ## Add the origin back to the point cloud
    
    

    return grid, flag, origin,cilm ,pc_lat,pc_lon,pc_r
    
def convert_grid_to_pc_reconstruct(topo):
    """ Function for reconstructing a point cloud from its projection onto the unit sphere. """

    nlon = topo.shape[0]
    nlat = topo.shape[1]
    lon = np.linspace(0, nlon * np.pi * 2 / (nlon - 1), num=nlon, endpoint=False)
    lat = np.linspace(nlat // 2 * np.pi / -(nlat - 1), np.pi / 2, num=nlat, endpoint=True)
    lon = np.broadcast_to(lon.reshape((1, nlon)), (nlat, nlon))
    lat = np.broadcast_to(lat.reshape((nlat, 1)), (nlat, nlon))


    r = topo

    z = np.sin(lat) * r
    t = np.cos(lat) * r
    x = t * np.cos(lon)
    y = t * np.sin(lon)

    pc = np.zeros(topo.shape + (3, ))
    
    pc[:, :, 0] = x
    pc[:, :, 1] = y
    pc[:, :, 2] = -z  # must have the minus
    pc = pc.reshape((-1, 3))
    #pc = pc[flag, :]  # only the flagged polar angles must be used in the point cloud reconstruction
    #pc += origin

    return pc
    
def convert_grid_to_pc(grid, flag, origin):
    """ Function for reconstructing a point cloud from its projection onto the unit sphere. """

    nlon = grid.nlon
    nlat = grid.nlat
    lon = np.linspace(0, nlon * np.pi * 2 / (nlon - 1), num=nlon, endpoint=False)
    lat = np.linspace(nlat // 2 * np.pi / -(nlat - 1), np.pi / 2, num=nlat, endpoint=True)
    lon = np.broadcast_to(lon.reshape((1, nlon)), (nlat, nlon))
    lat = np.broadcast_to(lat.reshape((nlat, 1)), (nlat, nlon))


    r = grid.data

    z = np.sin(lat) * r
    t = np.cos(lat) * r
    x = t * np.cos(lon)
    y = t * np.sin(lon)

    pc = np.zeros(grid.data.shape + (3, ))
    
    pc[:, :, 0] = x
    pc[:, :, 1] = y
    pc[:, :, 2] = -z  # must have the minus
    pc = pc.reshape((-1, 3))
    #pc = pc[flag, :]  # only the flagged polar angles must be used in the point cloud reconstruction
    pc += origin  # translate to the original origin

    return pc
    
# def low_pass_filter(grid, sigma):
#     ''' Function for diminishing high frequency components in the spherical harmonics representation. '''

#     # transform to the frequency domain
#     clm = grid.expand()

#     # create filter weights
#     weights = getGaussianKernel(clm.coeffs.shape[1] * 2 - 1, sigma)[clm.coeffs.shape[1] - 1:]
#     weights /= weights[0]

#     # low-pass filtering
#     clm.coeffs *= weights

#     # transform back into spatial domain
#     low_passed_grid = clm.expand()

#     return low_passed_grid
    
def duplicate_randomly(pc, size):  
    """ Make up for the point loss due to conflictions in the projection process. """
    loss_cnt = size - pc.shape[0]
    if loss_cnt <= 0:
        return pc
    rand_indices = np.random.randint(0, pc.shape[0], size=loss_cnt)
    dup = pc[rand_indices]
    return np.concatenate((pc, dup))


def reconstruct_pc_from_clim(clim_coeffs,max_lat=0,min_lat=0,max_lon=0,min_lon=0,choice_filtering=1):
    topo = pysh.expand.MakeGridDH(clim_coeffs, sampling=1)
    smooth_pc = convert_grid_to_pc_reconstruct(topo)

    if choice_filtering == 0:
        selected_pc = smooth_pc
    else:
        selected_pc = filter_pc_lat_lon(smooth_pc=smooth_pc,max_lat=max_lat,min_lat=min_lat,max_lon=max_lon,min_lon=min_lon)

    

    return selected_pc

def reconstruct_pc_from_grid(grid,max_lat,min_lat,max_lon,min_lon,flag,origin,choice_filtering=1):
    smooth_pc = convert_grid_to_pc(grid, flag, origin)
    
    if choice_filtering == 0:
        selected_pc = smooth_pc
    else:
        selected_pc = filter_pc_lat_lon(smooth_pc=smooth_pc,max_lat=max_lat,min_lat=min_lat,max_lon=max_lon,min_lon=min_lon)
    
    return selected_pc

def filter_pc_lat_lon(smooth_pc,max_lat,min_lat,max_lon,min_lon):
    
    smooth_pc_r, smooth_pc_lat, smooth_pc_lon, smooth_origin= convert_pc_to_radial(smooth_pc)
    all_smooth_pc = np.concatenate((smooth_pc, smooth_pc_r,smooth_pc_lat ,smooth_pc_lon), axis=1)
    selected_pc = all_smooth_pc[all_smooth_pc[:,5]<max_lon]
    selected_pc = selected_pc[selected_pc[:,5]>min_lon]
    selected_pc = selected_pc[selected_pc[:,4]<max_lat]
    selected_pc = selected_pc[selected_pc[:,4]>min_lat]
    

    selected_pc = selected_pc[:,0:3]

    return selected_pc

def our_method(pc, lmax, sigma, pc_size=1024, device="cuda",choice=0,choice_filtering=1):
    grid, flag, origin,cilm,pc_lat,pc_lon,pc_r = convert_pc_to_grid(pc, lmax, device)

    max_lat = pc_lat.max().item()
    min_lat = pc_lat.min().item()
    max_lon = pc_lon.max().item()
    min_lon = pc_lon.min().item()
    
    

    clim = grid.expand()


    if( choice == 0):
        smooth_pc = reconstruct_pc_from_clim(clim.coeffs,max_lat,min_lat,max_lon,min_lon,choice_filtering=choice_filtering)
    else:
        smooth_pc = reconstruct_pc_from_grid(grid,max_lat,min_lat,max_lon,min_lon,flag,origin,choice_filtering=choice_filtering)

    lat_lon = np.array([[min_lat],
                        [max_lat],
                        [min_lon],
                        [max_lon]])

    return smooth_pc , clim.coeffs ,lat_lon 

def reconstruct_pcd_from_points(points,nr_points_fps=1024,lmax=100,sigma=25,choice=0,choice_filtering=0,choice_fps=0):
    
    
    try:
        smooth_pc, coeffs,lat_lon = our_method(pc=points,lmax= lmax,sigma=sigma,pc_size=points.shape[0],choice=choice,choice_filtering=choice_filtering)

        #print("All good here")
        smooth_pc =torch.from_numpy(smooth_pc)
        nr_points=smooth_pc.shape[0]
    
    except:
        lat_lon= torch.empty((1), dtype=torch.int64)
        smooth_pc= torch.empty((1), dtype=torch.int64)
        coeffs= torch.empty((1), dtype=torch.int64)
        nr_points=0

    if(nr_points==0):

        return smooth_pc ,coeffs,lat_lon
    
    
        
    if (not (choice_fps==0)):
        index_fps = fps(smooth_pc, ratio=float(nr_points_fps/nr_points) , random_start=True)

        smooth_pc=smooth_pc[index_fps]

    return smooth_pc , coeffs ,lat_lon

def reconstruct_pcd_from_points_resampled(points,nr_points_fps=1024,lmax=100,sigma=25,choice=0,choice_filtering=0,choice_fps=0):
    smooth_pc, coeffs,lat_lon=our_method(pc=points,lmax= lmax,sigma=sigma,pc_size=points.shape[0],choice=choice,choice_filtering=choice_filtering)

    if (smooth_pc.shape[0] % nr_points_fps != 0):
            modulo_nr = (smooth_pc.shape[0] // nr_points_fps) +1
    else:
            modulo_nr = (smooth_pc.shape[0] // nr_points_fps)

    indices_pc = np.asarray(range(smooth_pc.shape[0]))
    
    #smooth_pc = smooth_pc[0:1024]

    smooth_pc = smooth_pc[indices_pc % modulo_nr ==0]

    remaining_points = nr_points_fps - smooth_pc.shape[0]

    

    if (remaining_points > 0):
        smooth_pc = np.concatenate((smooth_pc,smooth_pc[0:remaining_points]),axis=0)

    smooth_pc =torch.from_numpy(smooth_pc)


    return smooth_pc , coeffs ,lat_lon

if __name__ == '__main__':

    sigma=25 # Don't care at the moment


    print("lmax=")
    lmax=int(input())
    print("num_points=")
    num_points=int(input())
    print("nr_points_fps=")
    nr_points_fps=int(input())

    print("choice=")
    choice=int(input())
    print("choice_filtering=")
    choice_filtering=int(input())

    print("choice_fps=")
    choice_fps=int(input())




    #path_obj="/mnt/ssd1/Alex_data/PC-NBV/Shapenet_selected_data/train/02691156/1a74b169a76e651ebc0909d98a1ff2b4/model.obj"
    #path_obj="/mnt/ssd1/Alex_data/PC-NBV/Shapenet_selected_data/train/02958343/1abeca7159db7ed9f200a72c9245aee7/model.obj"
    #path_obj="/mnt/ssd1/Alex_data/PC-NBV/Shapenet_selected_data/valid/02691156/2c9e063352a538a4af7dd1bfd65143a9/model.obj"
    #path_obj="/mnt/ssd1/Alex_data/PC-NBV/Shapenet_datasets/4_category/train/02933112/1a4d4980bbe2dcf24da5feafe6f1c8fc/model.obj"
    path_obj="/mnt/ssd1/Alex_data/PC-NBV/Shapenet_selected_data/valid/04379243/ebc897217df591d73542091189dc62b5/model.obj"


    model_obj = o3d.io.read_triangle_mesh(path_obj)

    pcd=model_obj.sample_points_uniformly(number_of_points=num_points)

    nr_moves = 3

    for i in range(nr_moves):
        
        #path_pcd="/mnt/ssd1/Alex_data/PC-NBV/Coverage_scores/4_categories/train/1a4d4980bbe2dcf24da5feafe6f1c8fc/0/"+str(i)+"_acc_pc.pcd"

        path_pcd = "/mnt/ssd1/Alex_data/PC-NBV/Output_Blender/other_4_models/valid/pcd/2c9e063352a538a4af7dd1bfd65143a9/"+str(i)+".pcd"

        # path_pcd = "/mnt/ssd1/Alex_data/PC-NBV/Output_Blender/other_4_models/valid/pcd/ebc897217df591d73542091189dc62b5/"+str(i)+".pcd"

        # pcd = o3d.io.read_point_cloud(path_pcd)

        center = pcd.get_center()
        #center = np.asarray([0.,0.,0.])

        pcd.paint_uniform_color([1, 0, 0])
        points=np.asarray(pcd.points)

        # smooth_pc=our_method(pc=points,lmax= lmax,sigma=sigma,pc_size=num_points)

        # nr_points_fps = 1024

        # smooth_pc =torch.from_numpy(smooth_pc)
        # nr_points=smooth_pc.shape[0]

        # index_fps = fps(smooth_pc, ratio=float(nr_points_fps/nr_points) , random_start=True)

        # smooth_pc=smooth_pc[index_fps]

        smooth_pc,coeffs,lat_lon =reconstruct_pcd_from_points(points=points,nr_points_fps=nr_points_fps,lmax=lmax,sigma=sigma,choice=choice,choice_filtering=choice_filtering,choice_fps=choice_fps)

        smooth_pc2,coeffs2,lat_lon2 =reconstruct_pcd_from_points(points=points,nr_points_fps=nr_points_fps,lmax=lmax,sigma=sigma,choice=choice,choice_filtering=choice_filtering,choice_fps=choice_fps)



        pcd_smooth=o3d.geometry.PointCloud()
        pcd_smooth.points= o3d.utility.Vector3dVector(np.asarray(smooth_pc))
        pcd_smooth.paint_uniform_color([0, 1, 0])

        pcd_smooth2=o3d.geometry.PointCloud()
        pcd_smooth2.points= o3d.utility.Vector3dVector(np.asarray(smooth_pc2))
        pcd_smooth2.paint_uniform_color([0, 0, 1])


        center_smooth = pcd_smooth.get_center()

        

        print(smooth_pc.shape)
        o3d.visualization.draw_geometries([pcd_smooth,pcd])

        #o3d.visualization.draw_geometries([pcd_smooth2,pcd_smooth])





