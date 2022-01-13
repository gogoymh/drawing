import torch
import numpy as np
from skimage.io import imread, imsave
from skimage.transform import resize
import os
import matplotlib.pyplot as plt

input_path = "C://results//video//background//"
output_path = "C://results//video//background_npy//"

width, height = 240, 426
ifresize = True

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
    index = int(name.split(".")[0])
    
    #index = int(name.split("_")[1].split("-")[0])
    
    target = imread(os.path.join(input_path, name))
    plt.imshow(target)
    plt.show()
    plt.close()
    
    if ifresize:
        target = resize(target,
                    (height,width),
                    mode='edge',
                    anti_aliasing=False,
                    anti_aliasing_sigma=None,
                    preserve_range=True,
                    order=0)
        target = target.astype('uint8')
    
        plt.imshow(target)
        plt.show()
        plt.close()
    
    target = torch.from_numpy(target[:,:,:3])
    a = target.reshape(-1,3)
    
    target = target.permute(2,0,1).contiguous()

    mapping = {tuple(c): t for c, t in zip(colors.tolist(), range(len(colors)))}

    mask = torch.zeros(height, width, dtype=torch.long)
    for k in mapping:
        #print(k)
        idx = (target==torch.tensor(k, dtype=torch.uint8).unsqueeze(1).unsqueeze(2))
        validx = (idx.sum(0) == 3)
        mask[validx] = torch.tensor(mapping[k], dtype=torch.long)
    
    #print(mask)
    mask = mask.unsqueeze(0).long()#.numpy()
    np.save(os.path.join(output_path, "mask_%07d" % index), mask.numpy())
    
    onehot = torch.zeros(12,height,width)
    onehot = onehot.scatter_(0,mask,1)
    onehot = onehot.numpy()
    
    #np.save(os.path.join(output_path, "onehot_%07d" % index), onehot)
    
    print(index)
    
    onehot = torch.from_numpy(onehot)
    pred = onehot.argmax(dim=0).numpy().astype(np.uint8)
    
    pic = torch.zeros(3, height, width, dtype=torch.uint8)
    for i, k in enumerate(mapping):
        idx = pred == i
        pic[0][idx] = k[0]
        pic[1][idx] = k[1]
        pic[2][idx] = k[2]
        
    pic = pic.numpy().transpose(1,2,0)
    plt.imshow(pic)
    plt.show()
    plt.close()
    
    b = pic[:,:,:3].reshape(-1,3)
    print(np.unique(a,axis=0).shape[0]-np.unique(b,axis=0).shape[0])

