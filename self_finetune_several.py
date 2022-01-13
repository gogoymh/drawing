import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms as tf
from torch.utils.data.sampler import SubsetRandomSampler
import torch.optim as optim
import argparse
import os

from Unet import Unet_part, Unet_last
from semantic_set import multiclass_mask
#import config as cfg

print("="*100)
##############################################################################################################################
parser = argparse.ArgumentParser()
parser.add_argument("--exp_name", type=str, default="Drawing", help="Experiment index")
parser.add_argument("--dataset", type=str, default="bird", help="Dataset")

parser.add_argument("--n_epoch", type=int, default=50, help="Number of epoch")
parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")

parser.add_argument("--path1", type=str, default="/home/compu/ymh/drawing/dataset/num_001/frames", help="Main path")
parser.add_argument("--path2", type=str, default="/home/compu/ymh/drawing/dataset/num_001/mask_npy", help="Main path")
parser.add_argument("--save_path", type=str, default="/home/compu/ymh/drawing/save/num_003", help="save path")

opt = parser.parse_args()

##############################################################################################################################
save_path = os.path.join(opt.save_path, opt.exp_name, opt.dataset)
if os.path.isdir(save_path):
    print("Save path exists: ",save_path)
else:
    os.makedirs(save_path)
    print("Save path is created: ",save_path)

##############################################################################################################################
dataset_train = multiclass_mask(opt.path1, opt.path2)
train_loader = DataLoader(dataset=dataset_train, batch_size=opt.batch_size, pin_memory=True, num_workers=4)

##############################################################################################################################
device = torch.device("cuda:0")

'''
def criterion (input, target):
    logprobs = torch.nn.functional.log_softmax (input, dim = 1)
    return  -(target * logprobs).sum(dim=1).mean()
'''
criterion = nn.CrossEntropyLoss()



for index in range(100, 550, 50):
    print("="*40, end=" ")
    print("Test %d" %index, end=" ")
    print("="*40)
    
    model_part1 = Unet_part().to(device)
    model_part2 = Unet_last(12).to(device)
    
    model_name = os.path.join("/home/compu/ymh/drawing/save/num_001/Drawing/Bird", "self_sup_%04d.pth" % index)
    checkpoint = torch.load(model_name)

    '''
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in checkpoint['model_state_dict'].items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
        
    model.load_state_dict(new_state_dict, strict=False)
    '''
    model_part1.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model_part1.eval()

    params = list(model_part1.parameters()) + list(model_part2.parameters())
    optimizer_1 = optim.Adam(params, lr=opt.lr)

    optimizer_2 = optim.Adam(model_part2.parameters(), lr=0.001)

    for epoch in range(opt.n_epoch):
        running_loss=0
        if epoch < 10:
            model_part1.eval()
            for x, y in train_loader:
                x = x.float().to(device)
                y = y.long().to(device)
            
                #print(x.shape, y.shape)
                
                optimizer_2.zero_grad()
                with torch.no_grad():
                    out = model_part1(x)
                output = model_part2(out)
                loss = criterion(output, y)
                loss.backward()
                optimizer_2.step()
        
                running_loss += loss.item()
            
        else:
            model_part1.train()
            for x, y in train_loader:
                x = x.float().to(device)
                y = y.long().to(device)
            
                optimizer_1.zero_grad()
                out = model_part1(x)
                output = model_part2(out)
                loss = criterion(output, y)
                loss.backward()
                optimizer_1.step()
        
                running_loss += loss.item()
        
        running_loss /= len(train_loader)    
        print("[Epoch:%d] [Loss:%f]" % ((epoch+1), running_loss))
    
    model_name_1 = os.path.join(save_path, "base_part1_%04d.pth" % index)
    torch.save({'model_state_dict': model_part1.state_dict()}, model_name_1)
    
    model_name_2 = os.path.join(save_path, "base_part2_%04d.pth" % index)
    torch.save({'model_state_dict': model_part2.state_dict()}, model_name_2)
    

