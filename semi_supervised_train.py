import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch.optim as optim
import argparse
import os
from skimage.io import imread, imsave
from torchvision import transforms
import numpy as np

from Unet import Unet as net
from semantic_set import semantic_set
import config as cfg

print("="*100)
##############################################################################################################################
parser = argparse.ArgumentParser()
parser.add_argument("--exp_name", type=str, default="Drawing", help="Experiment index")
parser.add_argument("--dataset", type=str, default="Bird", help="Dataset")
parser.add_argument("--init_test", type=int, default=1, help="Initial test index")
parser.add_argument("--repeat", type=int, default=10, help="How many repeat")

parser.add_argument("--n_epoch", type=int, default=80, help="Number of epoch")
parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")

parser.add_argument("--path1", type=str, default="/home/compu/ymh/drawing/dataset/num_001/frames", help="Main path")
parser.add_argument("--path2", type=str, default="/home/compu/ymh/drawing/dataset/num_001/bird_mask_gen", help="Main path")
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
dataset_train = semantic_set(opt.path1, opt.path2)
train_loader = DataLoader(dataset=dataset_train, batch_size=opt.batch_size, pin_memory=True)

dataset_valid = semantic_set(opt.path1, opt.path2, False)

##############################################################################################################################
device = torch.device("cuda:0")
criterion = cfg.DiceLoss()

for test_idx in range(opt.init_test,(opt.init_test + opt.repeat)):
    print("="*40, end=" ")
    print("Test #%d" % test_idx, end=" ")
    print("="*40)
    print(opt)
    print("="*100)
    
    model = net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    steps = 250 // opt.batch_size + 1
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)

    inference = net().to(device)
    inference.load_state_dict(model.state_dict())
    
    best_f1 = 0
    for epoch in range(opt.n_epoch):
        running_loss=0
        model.train()
        for x, y in train_loader:
            x = x.float().to(device)
            y = y.float().to(device)
            
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
        
            running_loss += loss.item()
            #print(loss.item())
            
            scheduler.step()
            #print(scheduler.get_lr())
            
        running_loss /= len(train_loader)
        
        print("[Epoch:%d] [Loss:%f]" % ((epoch+1), running_loss))
    
        for target_param, param in zip(inference.parameters(), model.parameters()):
            target_param.data.copy_(target_param.data * steps/(steps+1) + param.data * 1/(steps+1))
            
        model_name = os.path.join(save_path, "test_%02d.pth" % test_idx)
        torch.save({'model_state_dict': inference.state_dict()}, model_name)
    
    inference.eval()
    for index in range(dataset_valid.len):
        img, mask = dataset_valid.__getitem__(index)
        img = img.unsqueeze(0).float().to(device)
        
        output = inference(img)
        output = output.squeeze().detach().clone().cpu()
        
        mask = mask.squeeze()
        mask = 0.9 * mask + 0.1 * output
        mask = mask.numpy()
        mask = mask * 255
        mask = mask.astype(np.uint8)
    
        imsave(os.path.join(opt.path2, "mask_%07d.png" % index), mask)
    
        '''
        if (epoch+1) % 1 == 0:
            accuracy = 0
            TP = 0
            FP = 0
            TN = 0
            FN = 0
            with torch.no_grad():
                model.eval()
                correct = 0
                for x, y in valid_loader:
                    x = x.float().to(device)
                    y = y >= 0.5
                    y = y.to(device)
            
                    output = model(x)
                    pred = output >= 0.5
                    correct += (y == pred).sum().item()
                
                    pred = pred.view(-1)
                
                    Trues = pred[pred == y.view(-1)]
                    Falses = pred[pred != y.view(-1)]
            
                    TP += (Trues == 1).sum().item()
                    TN += (Trues == 0).sum().item()
                    FP += (Falses == 1).sum().item()
                    FN += (Falses == 0).sum().item()
                
            accuracy = correct / (n_val_sample*442*565)
        
            if TP == 0:
                pass
            else:
                precision = TP / (TP + FP)
                recall = TP / (TP + FN)
                f1 = 2 * (precision * recall) / (precision + recall)
                if f1 >= best_f1:
                    print("[Epoch:%d] [Loss:%f]" % ((epoch+1), running_loss), end=" ")
                    print("[Accuracy:%f]" % accuracy, end=" ")
                    print("[Precision:%f]" % precision, end=" ")
                    print("[Recall:%f]" % recall, end=" ")
                    print("[F1 score:%f] **Best**" % f1)
                
                    best_f1 = f1
                    model_name = save_path + "/test_%02d.pth" % test_idx
                    
                    torch.save({'model_state_dict': model.state_dict()}, model_name)
                '''

            