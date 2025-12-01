import os
import random
from pathlib import Path

import numpy as np
from PIL import Image

# ========== 0. 基本配置 ==========

# 固定随机种子，保证可复现
# random.seed(42)
random.seed(24)

# 支持的图像后缀
valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".gif"}


def get_ext(path, default_ext=".jpg"):
    """从路径中取出合法后缀，否则返回一个默认后缀"""
    ext = os.path.splitext(path)[1].lower()
    if ext in valid_exts:
        return ext
    return default_ext


# ========== 1. 路径设置 ==========

# 源数据根目录（WSL 下）
src_root = r"/mnt/e/datasets/PSCC/Training Dataset"
# 如果在 Windows 原生环境运行，改成：
# src_root = r"E:\datasets\PSCC\Training Dataset"

# 目标根目录
dst_root = r"/mnt/e/jwj/new_datasets"
# dst_root = r"E:\jwj\new_datasets"  # Windows 版本

image_dst_dir = os.path.join(dst_root, "Image")
mask_dst_dir  = os.path.join(dst_root, "Mask")

os.makedirs(image_dst_dir, exist_ok=True)
os.makedirs(mask_dst_dir,  exist_ok=True)

# train_txt_path = os.path.join(dst_root, "train.txt")
train_txt_path = os.path.join(dst_root, "val.txt")

# 五个子文件夹（类别名） & 抽样比例
class_names = ["authentic", "splice", "splice_randmask", "copymove", "removal"]
train_ratio = [0.20,       0.10,     0.10,             0.45,       0.15]

# 每个样本的信息: (abs_image_path, abs_mask_path, class_name)
samples_info = []


# ========== 2. 工具函数 ==========

def collect_authentic_images(auth_dir):
    """
    收集 authentic/train2014resize_256 下的所有图片
    返回：列表 [("文件名", "完整路径"), ...]
    """
    if not os.path.isdir(auth_dir):
        print(f"[Warn] authentic 目录不存在: {auth_dir}")
        return []

    pairs = []
    for fname in os.listdir(auth_dir):
        ext = os.path.splitext(fname)[1].lower()
        if ext in valid_exts:
            pairs.append((fname, os.path.join(auth_dir, fname)))
    return pairs


def collect_fake_images_and_masks(img_dir, mask_dir):
    """
    收集 splice / splice_randmask / copymove / removal 下的
    (图像, mask) 配对。

    目录结构：
      img_dir  = <cls>/fake
      mask_dir = <cls>/mask

    规则：
      - 图像文件在 fake 目录，mask 文件在 mask 目录
      - 二者文件名的 stem 相同（例如：fake/xxx.jpg 和 mask/xxx.png）
    返回：列表 [(img_fname, img_path, mask_path), ...]
    """
    if not os.path.isdir(img_dir):
        print(f"[Warn] fake 目录不存在: {img_dir}")
        return []
    if not os.path.isdir(mask_dir):
        print(f"[Warn] mask 目录不存在: {mask_dir}")
        return []

    # 1) 收集 fake 目录中的所有图像文件
    img_dict = {}
    for fname in os.listdir(img_dir):
        ext = os.path.splitext(fname)[1].lower()
        if ext in valid_exts:
            stem, _ = os.path.splitext(fname)
            img_dict[stem] = fname

    # 2) 收集 mask 目录中的所有 mask 文件
    mask_dict = {}
    for fname in os.listdir(mask_dir):
        ext = os.path.splitext(fname)[1].lower()
        if ext in valid_exts:
            stem, _ = os.path.splitext(fname)
            mask_dict[stem] = fname

    # 3) 用 stem 做交集配对
    common_stems = sorted(set(img_dict.keys()) & set(mask_dict.keys()))
    if not common_stems:
        print(f"[Warn] 在 {img_dir} 和 {mask_dir} 中没有找到同名 (img, mask) 对")
        return []

    pairs = []
    for stem in common_stems:
        img_fname  = img_dict[stem]
        mask_fname = mask_dict[stem]
        img_path   = os.path.join(img_dir,  img_fname)
        mask_path  = os.path.join(mask_dir, mask_fname)
        pairs.append((img_fname, img_path, mask_path))

    return pairs


# ========== 3. 按比例抽样并复制/转换 ==========

for cls_name, ratio in zip(class_names, train_ratio):
    if cls_name == "authentic":
        # -------- authentic: 图片在 authentic/train2014resize_256 --------
        img_dir = os.path.join(src_root, "authentic", "train2014resize_256")
        img_list = collect_authentic_images(img_dir)

        # n_total = len(img_list)
        # n_total = min(10000, len(img_list))
        n_total = min(500, len(img_list))
        if n_total == 0:
            print(f"[Warn] 类别 {cls_name} 下没有图片文件")
            continue

        n_select = max(1, int(n_total * ratio))
        n_select = min(n_select, n_total)

        print(f"[Info] 类别 {cls_name}: 总数 = {n_total}, 取 {n_select} 张 (ratio={ratio})")

        sampled = random.sample(img_list, n_select)

        for fname, src_img_path in sampled:
            stem, ext = os.path.splitext(fname)

            # 为防止不同类重名：前面加上类别前缀
            sample_id = f"{cls_name}_{stem}"

            # 目标图像统一用 .jpg
            dst_img_path  = os.path.join(image_dst_dir, sample_id + ".jpg")
            # 目标 mask 统一用 .png（纯黑图）
            dst_mask_path = os.path.join(mask_dst_dir,  sample_id + ".png")

            # 1) 处理“原图”：转为 RGB + JPEG 保存
            try:
                with Image.open(src_img_path) as img:
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                    img.save(dst_img_path, format="JPEG", quality=95)
            except Exception as e:
                print(f"[Error] 处理 authentic 图片失败: {src_img_path}, err={e}")
                continue

            # 2) 生成“纯黑 mask”，尺寸与图像一致
            try:
                with Image.open(src_img_path) as img:
                    w, h = img.size
                mask_np = np.zeros((h, w), dtype=np.uint8)  # 全 0
                mask_img = Image.fromarray(mask_np, mode="L")
                mask_img.save(dst_mask_path, format="PNG")
            except Exception as e:
                print(f"[Error] 生成纯黑 mask 失败: {src_img_path}, err={e}")
                continue

            # 记录绝对路径 + 类别名
            img_abs  = os.path.abspath(dst_img_path)
            mask_abs = os.path.abspath(dst_mask_path)
            samples_info.append((img_abs, mask_abs, cls_name))

    else:
        # -------- 其他四类：图片在 <cls>/fake，下；mask 在 <cls>/mask 下 --------
        fake_dir = os.path.join(src_root, cls_name, "fake")
        mask_dir = os.path.join(src_root, cls_name, "mask")

        pairs = collect_fake_images_and_masks(fake_dir, mask_dir)

        # n_total = len(pairs)
        # n_total = min(10000, len(pairs))
        n_total = min(500, len(pairs))
        if n_total == 0:
            print(f"[Warn] 类别 {cls_name} 下没有 (img,mask) 有效配对")
            continue

        n_select = max(1, int(n_total * ratio))
        n_select = min(n_select, n_total)

        print(f"[Info] 类别 {cls_name}: 总数 = {n_total}, 取 {n_select} 张 (ratio={ratio})")

        sampled = random.sample(pairs, n_select)

        for img_fname, src_img_path, src_mask_path in sampled:
            stem, _ = os.path.splitext(img_fname)
            sample_id = f"{cls_name}_{stem}"

            # 目标图像统一用 .jpg，避免后续混乱
            dst_img_path  = os.path.join(image_dst_dir, sample_id + ".jpg")
            # 目标 mask 统一用 .png
            dst_mask_path = os.path.join(mask_dst_dir,  sample_id + ".png")

            # 1) 处理图像：转 RGB + JPEG 保存
            try:
                with Image.open(src_img_path) as img:
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                    img.save(dst_img_path, format="JPEG", quality=95)
            except Exception as e:
                print(f"[Error] 处理 fake 图片失败: {src_img_path}, err={e}")
                continue

            # 2) 处理 mask：转 L + 二值化后 PNG 保存
            try:
                with Image.open(src_mask_path) as m:
                    m = m.convert("L")
                    m_np = np.array(m, dtype=np.uint8)
                    # 二值化：>127 为 255，其余为 0
                    # m_np = np.where(m_np > 127, 255, 0).astype(np.uint8)
                    m_np = np.where(m_np > 127, 1, 0).astype(np.uint8) # 01分布
                    m_bin = Image.fromarray(m_np, mode="L")
                    m_bin.save(dst_mask_path, format="PNG")
            except Exception as e:
                print(f"[Error] 处理 fake mask 失败: {src_mask_path}, err={e}")
                continue

            img_abs  = os.path.abspath(dst_img_path)
            mask_abs = os.path.abspath(dst_mask_path)
            samples_info.append((img_abs, mask_abs, cls_name))


# ========== 4. 写 train.txt ==========

with open(train_txt_path, "w", encoding="utf-8") as f:
    for img_abs, mask_abs, cls_name in samples_info:
        # 每行记录：图像绝对路径、mask 绝对路径、所属类别
        f.write(f"{img_abs} {mask_abs} {cls_name}\n")

print(f"\n[Done] 共写入 {len(samples_info)} 个样本到 {train_txt_path}")
print(f"所有图像位于: {image_dst_dir}")
print(f"所有 mask 位于: {mask_dst_dir}")
print("train.txt 每行格式: <image_abs_path> <mask_abs_path> <class_name>")
