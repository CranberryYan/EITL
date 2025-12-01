import os
import random

# 设置图片文件夹路径
image_folder = '/media/yst/Elements SE/jwj/my_data/JPEGImages'  # ← 请替换为你的实际路径

# 支持的图片扩展名
image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.gif')

# 获取所有图片的文件名(不带扩展名)
image_files = [
    os.path.splitext(f)[0] for f in os.listdir(image_folder)
    if f.lower().endswith(image_extensions)
]

# 打乱顺序
random.shuffle(image_files)

# 分成 70% 和 30%
split_idx = int(len(image_files) * 0.7)
train_files = image_files[:split_idx]
val_files = image_files[split_idx:]

# 写入 train.txt
with open('/media/yst/Elements SE/jwj/my_data/ImageSets/Segmentation/train.txt', 'w') as train_f:
    for name in train_files:
        train_f.write(name + '\n')

# 写入 val.txt
with open('/media/yst/Elements SE/jwj/my_data/ImageSets/Segmentation/val.txt', 'w') as val_f:
    for name in val_files:
        val_f.write(name + '\n')

print(f"✅ 已写入 {len(train_files)} 条到 train.txt, {len(val_files)} 条到 val.txt")