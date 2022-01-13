import torch
import numpy as np
from skimage.io import imread, imsave
import os
import matplotlib.pyplot as plt

input_path = "C://유민형//기타//영상//mp4//main_frame//"
output_path = "C://유민형//기타//영상//mp4//main_frame//"


colors = np.array([[  0,   0,   0],
                   [  0,   0, 128],
                   [  0, 128,   0],
                   [  0, 128, 128],
                   [ 64,   0,   0],
                   [ 64, 128,   0],
                   [128,   0,   0],
                   [128,   0, 128],
                   [128, 128,   0],
                   [128, 128, 128],
                   [192,   0,   0],
                   [192, 128,   0]], dtype=np.uint8)

files = os.listdir(input_path)
files.sort()

for name in files:
    #index = int(name.split(".")[0])
    
    index = int(name.split("_")[1].split(".")[0])
    
    target = imread(os.path.join(input_path, "pic_%07d.png" % index))
    
    plt.imshow(target)
    plt.show()
    plt.close()
    
    target = torch.from_numpy(target[:,:,:3])
    target = target.permute(2,0,1).contiguous()

    mapping = {tuple(c): t for c, t in zip(colors.tolist(), range(len(colors)))}

    mask = torch.zeros(426, 240, dtype=torch.long)
    for k in mapping:
        idx = (target==torch.tensor(k, dtype=torch.uint8).unsqueeze(1).unsqueeze(2))
        validx = (idx.sum(0) == 3)
        mask[validx] = torch.tensor(mapping[k], dtype=torch.long)
    
    #print(mask)
    mask = mask.unsqueeze(0).long()
        
    onehot = torch.zeros(12,426,240)
    onehot = onehot.scatter_(0,mask,1)
    onehot = onehot.numpy()
    
    np.save(os.path.join(output_path, "onehot_%07d" % index), onehot)
    
    print(index)
    
'''
mask = mask.type(torch.uint8)

from torchvision import transforms
transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((426,240)),
                transforms.RandomAffine(0, shear=[-15, 15, -15, 15])
            ])

mask = transform(mask)
mask = np.asarray(mask)


pred = torch.empty(3, 426, 240, dtype=torch.long)
for i, k in enumerate(mapping):
    idx = mask == i
    pred[0][idx] = k[0]
    pred[1][idx] = k[1]
    pred[2][idx] = k[2]

plt.imshow(pred.numpy().transpose(1,2,0))
plt.show()
plt.close()
'''


'''
mask = np.zeros((852,480))
mask[label[:,:,0] != 0] = 1
mask = mask * 255
mask = mask.astype(np.uint8)

plt.imshow(mask)
plt.show()
plt.close()

imsave(os.path.join(output_path, "mask_%07d.png"), mask)
'''
'''
files = os.listdir(input_path)
for name in files:
    label = imread(os.path.join(input_path, name))

    #plt.imshow(label)
    #plt.show()
    #plt.close()

    #label = torch.from_numpy(label)
    #colors = torch.unique(label.view(-1, label.size(2)), dim=0).numpy()

    mask = np.zeros((852,480))
    mask[label[:,:,0] != 0] = 1
    mask = mask * 255
    mask = mask.astype(np.uint8)

    #plt.imshow(mask)
    #plt.show()
    #plt.close()

    imsave(os.path.join(output_path, "mask_" + name), mask)
    
    print(name)
'''