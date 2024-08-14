import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model_utils import Feature_Extraction, SelfAttention, normalize_point_batch

from torch_geometric.loader import DataLoader

from Dataloader.dataloader_pcd_v2 import PcdDataset

from torch_geometric.data.dataset import Dataset
from torch_geometric.loader import DenseDataLoader, DataLoader

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

from torch.optim import lr_scheduler

from path import Path

from pc_nbv import AutoEncoder, Encoder, Decoder

import time

import os

from datetime import datetime

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train(model, optimizer,criterion, loader,nr_points):
    model.train()
    
    total_loss = 0
    total_correct = 0
    total = 0
    #for data in loader:
    for i, data in enumerate(loader):
        
        optimizer.zero_grad()
        
        batch_size=int(data.next_position.shape[0])

        eval_values_gt=data.target_values

        eval_values_gt = eval_values_gt.squeeze(2)

        #eval_values_gt=torch.reshape(eval_values_gt,(batch_size,int(eval_values_gt.shape[0]/batch_size)))
        
        pcd_data = data.pos
        pcd_data = pcd_data.permute(0,2,1)
       
        viewstate=data.viewstate
        #x=torch.reshape(x,(batch_size,nr_points,x.shape[1]))
        #viewstate=torch.reshape(viewstate,(batch_size,int(viewstate.shape[0]/batch_size)))

        pcd_data=pcd_data.float()
        viewstate=viewstate.float()


        pcd_data=pcd_data.to(device)
        
        viewstate=viewstate.to(device)
        eval_values_gt=eval_values_gt.to(device)
        labels=data.next_position.to(device)
       

        feature, outputs = model(pcd_data,viewstate)

        #outputs = outputs - torch.mul(outputs,viewstate)


        _, predicted = torch.max(outputs.data, 1)
        total_correct += (predicted.detach() == labels.detach()).sum().item()

        loss_nbv = sum([torch.sum(param**2) for param in model.encoder.parameters()]) * 0.0001

        loss = criterion(outputs, eval_values_gt.float())
        
        loss=loss+loss_nbv
        
        # # print("Loss total:"+str(loss.item()))
        
        loss.backward()  # Backward pass.

        loss = loss.detach().cpu().numpy()
        optimizer.step()  # Update model parameters.
        total_loss += loss.item() / len(loader.dataset)

        
    return 1. *total_correct/len(loader.dataset), total_loss


@torch.no_grad()
def test(model, loader,nr_points):

    optimizer.zero_grad()
    model.eval()

    total_loss = 0
    total_correct = 0
    total = 0
    #for data in loader:
    for i, data in enumerate(loader):
        
        batch_size=int(data.next_position.shape[0])

        eval_values_gt=data.target_values

        eval_values_gt = eval_values_gt.squeeze(2)

        
        x = data.pos
        
        viewstate=data.viewstate
       

        x=x.float()
        viewstate=viewstate.float()


        x=x.to(device)
        x=x.permute(0,2,1)
        viewstate=viewstate.to(device)
        eval_values_gt=eval_values_gt.to(device)
        
        labels=data.next_position.to(device)
       

        outputs = model(x,viewstate)

        #outputs = outputs - torch.mul(outputs,viewstate)

        _, predicted = torch.max(outputs.data, 1)
    
        
        total_correct += (predicted == labels).sum().item()
        
        trainable_variables=dict(model.named_parameters())
        
        loss_nbv = sum([torch.sum(param**2) for param in model.encoder.parameters()]) * 0.0001

        loss = criterion(outputs, eval_values_gt.float())
        
        loss=loss+loss_nbv


        total_loss += loss.item()  / len(loader.dataset)

        

        

    val_acc = 1. *total_correct / len(loader.dataset)
    

    #print(val_acc)
   
        

    #return total_correct / len(loader.dataset)
    return val_acc , total_loss




if __name__ == "__main__":

    torch.backends.cudnn.enabled = False
    now = datetime.now()
    directory = now.strftime("%d_%m_%y_%H:%M:%S")
    directory="PCNBV_remake_33_"+directory
    parent_directory = "/mnt/ssd1/Alex_data/PC-NBV/Trained_models"
    path = os.path.join(parent_directory, directory)
    os.mkdir(path)

    num_classes = 33 
    num_points= 1024
    batch_size=32
    num_epochs=200
    nr_features=3

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Training on {device}")



    #model = PointNet(num_classes=num_classes,nr_features=nr_features)
    model = AutoEncoder(views=num_classes)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    #criterion = torch.nn.CrossEntropyLoss()  # Define loss criterion.

    #criterion = torch.nn.CrossEntropyLoss(reduce=None, reduction='sum')  # Define loss criterion.

    criterion = torch.nn.MSELoss(reduce=None, reduction='sum')  # Define loss criterion.

    my_lr_scheduler = lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.95)



    torch.manual_seed(0)

    #################################

    # root = Path("/mnt/ssd1/Alex_data/Test_PCNBV_again_2/4_Export_NBV_gt_pcd_SCO/3d_Shape_small/views/")

    root = Path("/mnt/ssd1/Alex_data/Test_PCNBV_again_2/5_NBV_coverage/3d_Shape_small/")
    

    nr_views = 33


    train_dataset = PcdDataset(root,valid=True,folder="train",nr_views=nr_views,nr_points=512)    
    valid_dataset = PcdDataset(root,valid=True,folder="valid",nr_views=nr_views,nr_points=1024)   


    train_loader_0 = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader_0 = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=True)

    print("Done loading data")


    torch.backends.cudnn.enabled = False

    for epoch in range(1, num_epochs+1):


        train_start_time = time.time()
        train_acc,train_loss = train(model, optimizer,criterion, train_loader_0,nr_points=num_points)
        train_stop_time = time.time()


        # test_start_time = time.time()
        # test_acc,test_loss = test(model, test_loader_0,nr_points=num_points)
        # test_stop_time = time.time()

        
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Acc/train", train_acc, epoch)
        # writer.add_scalar("Loss/test", test_loss, epoch)
        # writer.add_scalar("Acc/test", test_acc, epoch)


        print(f'Epoch: {epoch:02d}, Loss: {train_loss:.4f}, Train accuracy: {train_acc:.4f}')
        print(f'\tTrain Time: \t{train_stop_time - train_start_time} \n \
        Test Time: \t{train_start_time - train_stop_time }')

        # print(f'Epoch: {epoch:02d}, Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}')
        # print(f'\tTrain Time: \t{train_stop_time - train_start_time} \n \
        # Test Time: \t{test_stop_time - test_start_time }')

        if(epoch%3==0):
            torch.save(model.state_dict(), path + '/model' + str(epoch) + '.pt')


        my_lr_scheduler.step()


    torch.save(model.state_dict(), path + '/model' + str(epoch) + '.pt')