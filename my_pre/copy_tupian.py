import os
import random
import shutil

# 原图文件夹(.jpg)和处理图文件夹(.png)
image_folder = '/media/yst/Elements SE/jwj/EITL_datasets/JPEGImages'        # 原图路径, 例如：'data/original'
label_folder = '/media/yst/Elements SE/jwj/EITL_datasets/SegmentationClass' # 标签图路径, 例如：'data/processed'
label_folder0255 = '/media/yst/Elements SE/jwj/EITL_datasets/SegmentationClass0255' # 标签图路径, 例如：'data/processed'

# 输出目标文件夹
output_image_folder = '/media/yst/Elements SE/jwj/test_data/Images'  
output_label_folder = '/media/yst/Elements SE/jwj/test_data/Mask'
output_label_folder0255 = '/media/yst/Elements SE/jwj/test_data/Mask0255'

# 创建输出文件夹(如果不存在)
os.makedirs(output_image_folder, exist_ok=True)
os.makedirs(output_label_folder, exist_ok=True)
os.makedirs(output_label_folder0255, exist_ok=True)

# 获取所有原图的文件名(不包含扩展名)
image_files = [
    os.path.splitext(f)[0] for f in os.listdir(image_folder)
    if f.lower().endswith('.jpg')
]

# 打乱顺序并取更多一点, 防止跳过时不够
random.shuffle(image_files)

# 用于记录成功复制的图片数
copied_count = 0
target_count = 100

for name in image_files:
    if copied_count >= target_count:
        break

    src_img = os.path.join(image_folder, name + '.jpg')
    src_lbl = os.path.join(label_folder, name + '.png')
    src_lbl0255 = os.path.join(label_folder0255, name + '.png')

    # 检查两个文件是否都存在
    if not os.path.exists(src_img):
        print(f"[跳过] 缺失原图：{src_img}")
        continue
    if not os.path.exists(src_lbl):
        print(f"[跳过] 缺失处理图：{src_lbl}")
        continue
    if not os.path.exists(src_lbl0255):
        print(f"[跳过] 缺失处理图：{src_lbl0255}")
        continue

    # 目标路径
    dst_img = os.path.join(output_image_folder, name + '.jpg')
    dst_lbl = os.path.join(output_label_folder, name + '.png')
    dst_lbl0255 = os.path.join(output_label_folder0255, name + '.png')

    # 复制文件
    shutil.copy(src_img, dst_img)
    shutil.copy(src_lbl, dst_lbl)
    shutil.copy(src_lbl0255, dst_lbl0255)
    copied_count += 1

print(f"\n✅ 已成功随机复制 {copied_count} 对图片到：")
print(f"📁 原图目录：{output_image_folder}")
print(f"📁 标签目录：{output_label_folder}")