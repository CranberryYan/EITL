import os
import shutil
import random
from PIL import Image

# ========== 路径配置 ==========
# # 源数据：未篡改、已 resize 到 256×256 的真图
# src_img_dir = r"E:\datasets\PSCC\Training Dataset\authentic\train2014resize_256"

# # 目标：你的分割数据集 JPEGImages
# dst_img_dir = r"E:\jwj\my_data\JPEGImages"

# # 目标：两种 mask 存放目录
# mask_dir1 = r"E:\jwj\my_data\SegmentationClass"
# mask_dir2 = r"E:\jwj\my_data\SegmentationClass0255"

# 如果你在 WSL 下跑, 把上面改成：
src_img_dir = r"/mnt/e/datasets/PSCC/Training Dataset/authentic/train2014resize_256"
dst_img_dir = r"/mnt/e/jwj/my_data/JPEGImages"
mask_dir1   = r"/mnt/e/jwj/my_data/SegmentationClass"
mask_dir2   = r"/mnt/e/jwj/my_data/SegmentationClass0255"

# 支持的图片后缀
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# 随机种子(可选：为了每次运行结果一致)
random.seed(2025)

# ========== 创建目标文件夹 ==========
os.makedirs(dst_img_dir, exist_ok=True)
os.makedirs(mask_dir1, exist_ok=True)
os.makedirs(mask_dir2, exist_ok=True)

# ========== 收集所有源图片 ==========
all_img_names = [
    name for name in os.listdir(src_img_dir)
    if os.path.splitext(name)[1].lower() in IMG_EXTS
]
all_img_names.sort()

print(f"源文件夹中共有图片：{len(all_img_names)} 张")

# ========== 随机挑选最多 2000 张 ==========
num_select = 2000
if len(all_img_names) > num_select:
    selected_names = random.sample(all_img_names, num_select)
    print(f"随机抽取 {num_select} 张用于复制。")
else:
    selected_names = all_img_names
    print(f"图片不足 {num_select} 张, 使用全部 {len(selected_names)} 张。")

selected_names.sort()  # 只是为了日志好看, 可选

copied = 0
skipped = 0

for name in selected_names:
    src_path = os.path.join(src_img_dir, name)
    dst_path = os.path.join(dst_img_dir, name)

    # 1. 复制图片到 JPEGImages
    if os.path.exists(dst_path):
        print(f"[跳过] 目标已存在图片：{dst_path}")
        skipped += 1
    else:
        print(f"[复制] {src_path} -> {dst_path}")
        shutil.copy(src_path, dst_path)
        copied += 1

    # 2. 为该图片创建全黑 mask(同名 .png)
    base, _ = os.path.splitext(name)
    mask_name = base + ".png"
    mask_path1 = os.path.join(mask_dir1, mask_name)
    mask_path2 = os.path.join(mask_dir2, mask_name)

    # 读取图片尺寸, 确保 mask 尺寸一致
    with Image.open(src_path) as img:
        w, h = img.size

    # 单通道 L, 像素值全 0(全黑 = 全背景)
    black_mask = Image.new("L", (w, h), 0)

    # 这里采用覆盖写入(因为这些真图本来就不应该有之前的 mask)
    black_mask.save(mask_path1)
    black_mask.save(mask_path2)

print("\n========== 处理完成 ==========")
print(f"计划使用图片数：{len(selected_names)}")
print(f"新复制图片：{copied}")
print(f"已存在而跳过的图片：{skipped}")
print(f"Mask 已写入：\n  {mask_dir1}\n  {mask_dir2}")
