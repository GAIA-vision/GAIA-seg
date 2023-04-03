# 这个脚本需要运行两次，第一次获取mean，第二次根据第一次获取的mean求std
# 只计算train集

import os
from os import path
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from imageio import imread,imsave

import copy
import pdb

RGB = [0., 0., 0.]
RGB_square = [0., 0., 0.]
RGB_mean = [0., 0., 0.]
RGB_count = 0

# 因为VOC通过split集判断train，新增白名单
with open('/data1/yuqi_wang/VOC2012/ImageSets/Segmentation/train.txt','r') as f:
    train_set = f.readlines()
for idx, each_line in enumerate(train_set):
    train_set[idx] = train_set[idx].strip() + '.jpg'

def osPath(url,mode='mean'):
    global RGB,RGB_square,RGB_count
    files = os.listdir(url)
    for f in files:
        real_path = path.join(url, f) #绝对路径
        if path.isfile(real_path):  #如果是文件，读取
            # if("_leftImg8bit.png" not in f):  #不知道22 23原来是干啥的，备注掉
            #     continue
            #img_copy = cv2.imread(real_path) BGR
            # print(real_path) # 查看每张图片的路径
            #img = imread(real_path).astype(np.float64) # RGB
            if f not in train_set:
                continue
            
            img = cv2.imread(real_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            if mode == 'mean':
                for i in range(3):
                    RGB[i] += np.sum(img[:,:,i])
                RGB_count += img.shape[0]*img.shape[1]
            elif mode == 'std':
                for i in range(3):
                    #pdb.set_trace()
                    img[:,:,i] = (img[:,:,i] - RGB_mean[i])**2

                    RGB_square[i] += np.sum(img[:,:,i])

        elif path.isdir(real_path): # 如果是目录。打印，并进入下一级
            print(real_path)
            osPath(real_path, mode='std') # 注意这里也要修改
        else:
            print("Other Situations")
            pass


#osPath('/data/qing_chang/Data/cityscapes/leftImg8bit/train')
#print("RGB:", RGB)
#print("RGB_count: ",RGB_count)
#RGB_count = 6239027200q
#RGB_mean = [73.15835921071157, 82.90891754262587, 72.3923987619416]
#RGB_std = [47.675755341815155, 48.49421436881456, 47.736546325441566]
#pdb.set_trace()


# 第一次执行需要注释，第二次将第一次求得的平均值输入
RGB_mean = [116.47453561037983, 112.8488434018078, 103.88464188067795]  

osPath('/data1/yuqi_wang/VOC2012/JPEGImages',mode='std')
#pdb.set_trace()

# # 第一次执行正常，第二次执行注释
# print("RGB mean: ",RGB[0]/RGB_count, RGB[1]/RGB_count, RGB[2]/RGB_count)
# print("RGB_count: ",RGB_count)

# 第一次执行注释，第二次执行脚本，将count输入，
RGB_count = 262462800
print("RGB std: ",(RGB_square[0]/RGB_count)**0.5, (RGB_square[1]/RGB_count)**0.5, (RGB_square[2]/RGB_count)**0.5)