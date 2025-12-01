import os
from PIL import Image
import numpy as np
from tqdm import tqdm # 用于显示进度条
org_path=r"train_dataset/SegmentationClass_org/"#Only 0,255 value
dir_path=r"train_dataset/SegmentationClass/"#Only 0,1 value

if not os.path.exists(dir_path):
    os.makedirs(dir_path)


dirlist=os.listdir(org_path)
for img_id in tqdm(dirlist):
    img_path=os.path.join(org_path,img_id)
    img=Image.open(img_path).convert('L')  # 读取为灰度图(L模式)
    img=np.array(img)                      # 转为 numpy数组
    img_cls=np.array(img)                  # 复制一份
    img_cls[img>=127]=1
    img_cls[img < 127]=0
    img_cls=Image.fromarray(img_cls)
    img_cls.save(os.path.join(dir_path,img_id)[:-4]+".png")
