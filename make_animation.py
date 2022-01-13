import cv2
import numpy as np
import os
from os.path import isfile, join

org_path = "/home/compu/ymh/drawing/dataset/num_002/"
pathOut = '/home/compu/ymh/drawing/save/num_002/scene_00.mp4'
fps = 30
repeat = 10

frame_array = []

files = [f for f in os.listdir(org_path) if isfile(join(org_path, f))]
files.sort()


x, y = 539, 959
h, w = 2160, 3840

time = 10 # Length(seconds) you desire.


cum = 0 
while cum < time:
    for i in range(len(files)):
        filename = org_path + files[i]
        img = cv2.imread(filename)
        #print(img.shape)
        img = img[x:x+h,y:y+w]
    
        for _ in range(repeat):
            frame_array.append(img)
        
        cum += repeat/fps
        
        print("%f is finished." % cum)
        if cum >= time:
            break
        
out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'FMP4'), fps, (w, h))

for i in range(len(frame_array)):
    out.write(frame_array[i])

out.release()


#savename = org_path + "smp.png"
#cv2.imwrite(savename, img)

'''
#for i in range(1):
for i in range(len(files)):
    print(i)
    filename = org_path + files[i]
    #index = int(files[i].split("_")[1].split(".")[0])
    #filename = os.path.join(pathIn, "pic_%07d.png" % index)
    #print(filename)
    #reading each files
    #org = cv2.imread(filename2)
    #org = cv2.resize(org, (240,426))
    #print(org)
    #org = org * 255
    #org = org.astype('uint8')
    
    img = cv2.imread(filename)
    img = cv2.resize(img, (720, 1278), fx=0, fy=0, interpolation=cv2.INTER_NEAREST)
    #print(img)
    #img = img * 255
    #img = img.astype('uint8')
    #print(img.shape)
    #height, width, layers = img.shape
    #size = (width,height)
    
    #inserting the frames into an image array
    
    #print(org.shape, img.shape)
    #frame = np.concatenate((org, img), axis=1)
    #frame = (org - img)
    #frame = np.abs(frame)
    #print(frame.shape)
    
    frame_array.append(img)
    #frame_array.append(frame)

out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'FMP4'), fps, (720, 1278))

for i in range(len(frame_array)):
    # writing to a image array
    out.write(frame_array[i])

out.release()
'''