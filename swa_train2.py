import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torch.optim as optim
import argparse
import os
from torch.optim.swa_utils import AveragedModel, SWALR
import numpy as np
from skimage.io import imread, imsave

from Unet import Unet, Unet_softmax
from semantic_set import multiclass_onehot
import config as cfg

print("="*100)
##############################################################################################################################
parser = argparse.ArgumentParser()
parser.add_argument("--exp_name", type=str, default="Drawing", help="Experiment index")
parser.add_argument("--dataset", type=str, default="Bird", help="Dataset")
parser.add_argument("--init_test", type=int, default=1, help="Initial test index")
parser.add_argument("--repeat", type=int, default=10, help="How many repeat")

parser.add_argument("--n_epoch", type=int, default=60, help="Number of epoch")
parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
parser.add_argument("--lr", type=float, default=0.1, help="Learning rate")

parser.add_argument("--path1", type=str, default="/home/compu/ymh/drawing/dataset/num_001/frames", help="Main path")
parser.add_argument("--path2", type=str, default="/home/compu/ymh/drawing/dataset/num_001/mask_npy", help="Main path")
parser.add_argument("--save_path", type=str, default="/home/compu/ymh/drawing/save/num_001", help="save path")

opt = parser.parse_args()

##############################################################################################################################
save_path = os.path.join(opt.save_path, opt.exp_name, opt.dataset)
if os.path.isdir(save_path):
    print("Save path exists: ",save_path)
else:
    os.makedirs(save_path)
    print("Save path is created: ",save_path)

##############################################################################################################################
dataset_train = multiclass_onehot(opt.path1, opt.path2)
train_loader = DataLoader(dataset=dataset_train, batch_size=opt.batch_size, pin_memory=True)


##############################################################################################################################
device = torch.device("cuda:0")
inference = Unet_softmax().to(device)

def softXEnt (input, target):
    logprobs = nn.functional.log_softmax (input, dim = 1)
    return -(target * logprobs).sum(dim=1).mean()

for test_idx in range(opt.init_test,(opt.init_test + opt.repeat)):
    print("="*40, end=" ")
    print("Test #%d" % test_idx, end=" ")
    print("="*40)
    print(opt)
    print("="*100)
    
    model = Unet(12).to(device)
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)
    
    swa_model = AveragedModel(model)
    swa_start = 20
    swa_scheduler = SWALR(optimizer, swa_lr=0.05)
    
    best_f1 = 0
    for epoch in range(opt.n_epoch):
        running_loss=0
        model.train()
        for x, y in train_loader:
            x = x.float().to(device)
            y = y.float().to(device)
            
            optimizer.zero_grad()
            output = model(x)
            loss = softXEnt(output, y)
            loss.backward()
            optimizer.step()
        
            running_loss += loss.item()
            #print(loss.item())
            
        if epoch > swa_start:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            scheduler.step()
            
        running_loss /= len(train_loader)
        
        print("[Epoch:%d] [Loss:%f]" % ((epoch+1), running_loss))
    
    swa_model.cpu()
    torch.optim.swa_utils.update_bn(train_loader, swa_model)
    
    model_name = os.path.join(save_path, "test_%02d.pth" % test_idx)
    torch.save({'model_state_dict': swa_model.state_dict()}, model_name)
    
    swa_model.to(device)
    swa_model.eval()
    
    for index in range(dataset_train.len):
        x, y = dataset_train.__getitem__(index)
        x = x.unsqueeze(0).float().to(device)
        
        output = inference(swa_model(x))
        output = output.squeeze().detach().clone().cpu()
        
        new = 0.9 * y + 0.1 * output
        new = new.numpy()
    
        np.save(os.path.join(opt.path2, "onehot_%07d" % index), new)

            