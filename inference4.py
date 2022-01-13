import os
import torch
import numpy as np
from skimage.io import imread, imsave
import matplotlib.pyplot as plt


input_path = "/home/compu/ymh/drawing/dataset/num_003/frame"
output_path = "/home/compu/ymh/drawing/dataset/num_003/mask_gen"

colors = np.array([[0,0,0],
                   [199,69,98],
                   [179,102,255],
                   [0,0,160],
                   [255,55,155],
                   [128,0,255],
                   [255,0,255],
                   [0,0,64],
                   [15,183,255],
                   [255,128,255],
                   [255,245,210],
                   [255,201,14]], dtype=np.uint8)

mapping = {tuple(c): t for c, t in zip(colors.tolist(), range(len(colors)))}

files = os.listdir(input_path)
files.sort()

for name in files:    
    onehot = np.load(os.path.join(input_path, name))
    onehot = torch.from_numpy(onehot)
    
    pred = onehot.argmax(dim=0).numpy().astype(np.uint8)
    
    pic = torch.zeros(3, 426, 240, dtype=torch.uint8)
    for i, k in enumerate(mapping):
        idx = pred == i
        pic[0][idx] = k[0]
        pic[1][idx] = k[1]
        pic[2][idx] = k[2]
    
    pic = pic.numpy().transpose(1,2,0)
    plt.imshow(pic)
    plt.show()
    plt.close()
    
    imsave(os.path.join(output_path, "pic_%07d.png" % int(name.split("_")[1].split(".")[0])), pic)
    
    print(name)