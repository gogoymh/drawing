import os
import torch
from torchvision import transforms
from skimage.io import imread, imsave
import numpy as np

from Unet import Unet as net


device = torch.device("cuda:0")
model = net().to(device)


save_path = "/home/compu/ymh/drawing/save/num_001/Drawing/Bird"
index = 0
model_name = os.path.join(save_path, "test_%02d.pth" % index)

checkpoint = torch.load(model_name)

'''
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in checkpoint['model_state_dict'].items():
    name = k[7:] # remove `module.`
    new_state_dict[name] = v

model.load_state_dict(new_state_dict, strict=False)
'''
model.load_state_dict(checkpoint["model_state_dict"])

model.eval()

print("Loaded.")

transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.ToPILImage(),
                transforms.Resize((426,240)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])


img_path = "/home/compu/ymh/drawing/dataset/num_001/frames"
result_path = "/home/compu/ymh/drawing/dataset/num_001/bird_mask_gen"

images = os.listdir(img_path)
images.sort()
print(images)
for name in images:
    img = imread(os.path.join(img_path, name))
    img = transform(img)
    img = img.unsqueeze(0).float().to(device)
    
    mask = model(img)
    mask = mask.squeeze().detach().clone().cpu().numpy()
    mask = mask * 255
    mask = mask.astype(np.uint8)
    
    imsave(os.path.join(result_path, "mask_" + name.split("_")[1]), mask)
    
    print(name)
    
    
    