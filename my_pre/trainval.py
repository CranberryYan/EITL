import os
import random

# 图片所在的文件夹路径
image_folder = '/media/yst/Elements SE/jwj/my_data/JPEGImages'  # 替换为你的图片路径
# 输出的txt文件路径
output_txt = '/media/yst/Elements SE/jwj/my_data/ImageSets/Segmentation/trainval.txt'


# 支持的图片扩展名
image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.gif')

# 获取所有图片文件名
image_files = [
    f for f in os.listdir(image_folder)
    if f.lower().endswith(image_extensions)
]

# 打乱顺序
random.shuffle(image_files)

# 在文件名前加上 img/ 并写入 txt 文件
with open(output_txt, 'w') as f:
    for img_name in image_files:
        f.write(f"img/{img_name}\n")

print(f"✅ 共写入 {len(image_files)} 张图片到 {output_txt}")