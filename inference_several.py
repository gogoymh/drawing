import os
import torch
from torchvision import transforms
from skimage.io import imread, imsave
import numpy as np
import matplotlib.pyplot as plt

from Unet import Unet_part, Unet_last


device = torch.device("cuda:0")
model_part1 = Unet_part().to(device)
model_part2 = Unet_last(12).to(device)


save_path = "/home/compu/ymh/drawing/save/num_003/Drawing/bird"

for index in range(100,550,50):
    print("="*40, end=" ")
    print("Test %d" %index, end=" ")
    print("="*40)
    model_name_1 = os.path.join(save_path, "base_part1_%04d.pth" % index)
    model_name_2 = os.path.join(save_path, "base_part2_%04d.pth" % index)

    checkpoint1 = torch.load(model_name_1)
    checkpoint2 = torch.load(model_name_2)

    '''
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in checkpoint['model_state_dict'].items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict, strict=False)
    '''

    model_part1.load_state_dict(checkpoint1["model_state_dict"])
    model_part2.load_state_dict(checkpoint2["model_state_dict"])

    model_part1.eval()
    model_part2.eval()

    print("Loaded.")

    transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.ToPILImage(),
                    transforms.Resize((426,240)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])


    img_path = "/home/compu/ymh/drawing/dataset/num_001/frames"
    result_path = os.path.join("/home/compu/ymh/drawing/save/num_003/Drawing/bird_result/", "result_%04d" % index)
    if os.path.isdir(result_path):
        print("Save path exists: ",result_path)
    else:
        os.makedirs(result_path)
        print("Save path is created: ",result_path)


    images = os.listdir(img_path)
    images.sort()
    #print(images)

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
    
    mapping = {tuple(c): t for c, t in zip(colors.tolist(), range(len(colors)))}

    for name in images:
        img = imread(os.path.join(img_path, name))
        img = transform(img)
        img = img.unsqueeze(0).float().to(device)
    
        output = model_part2(model_part1(img))
        pred = output.argmax(dim=1).squeeze().detach().clone().cpu().numpy().astype(np.uint8)
    
        #np.save(os.path.join(result_path, "mask_%07d" % int(name.split("_")[1].split(".")[0])), pred)
    
        pic = torch.zeros(3, 426, 240, dtype=torch.uint8)
        for i, k in enumerate(mapping):
            idx = pred == i
            pic[0][idx] = k[0]
            pic[1][idx] = k[1]
            pic[2][idx] = k[2]
    
        pic = pic.numpy().transpose(1,2,0)
        #plt.imshow()
        #plt.show()
        #plt.close()
    
        imsave(os.path.join(result_path, "pic_%07d.png" % int(name.split("_")[1].split(".")[0])), pic)
    
        print(name)
    
    
    