# import os
# import random

# # ================== 配置区域 ==================
# # JPEGImages 路径(WSL 下)
# jpeg_dir = "/mnt/e/jwj/my_data/JPEGImages"

# # ImageSets/Segmentation 路径
# seg_dir = "/mnt/e/jwj/my_data/ImageSets/Segmentation"
# os.makedirs(seg_dir, exist_ok=True)

# train_small_path = os.path.join(seg_dir, "train_small.txt")
# val_small_path = os.path.join(seg_dir, "val_small.txt")

# # 想抽取的数量 n
# n_select = 20   # 如果想改别的数量, 改这里即可

# # 支持的图片后缀
# IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

# # 随机种子(可选, 为了复现)
# random.seed(2025)

# # ================== 收集 COCO* 图片 ==================
# all_files = os.listdir(jpeg_dir)

# coco_imgs = [
#     f for f in all_files
#     if f.startswith("COCO") and os.path.splitext(f)[1].lower() in IMG_EXTS
# ]

# coco_imgs.sort()
# print(f"在 JPEGImages 中找到以 'COCO' 开头的图片: {len(coco_imgs)} 张")

# if not coco_imgs:
#     raise RuntimeError("没有找到任何以 'COCO' 开头的图片, 请检查路径和文件名。")

# # ================== 计算候选 ID(去掉后缀) ==================
# coco_ids = [os.path.splitext(f)[0] for f in coco_imgs]

# # ================== 读取已有的 train_small.txt(如果有的话) ==================
# existing_ids = set()
# if os.path.exists(train_small_path):
#     with open(train_small_path, "r") as f:
#         for line in f:
#             line = line.strip()
#             if line:
#                 existing_ids.add(line)
#     print(f"已有 train_small.txt, 当前包含样本数: {len(existing_ids)}")

# # ================== 过滤掉已经存在的 ID ==================
# candidate_ids = [id_ for id_ in coco_ids if id_ not in existing_ids]
# print(f"可用于新增的 COCO 图像数: {len(candidate_ids)}")

# if not candidate_ids:
#     print("没有可新增的 COCO 图像, 全部都已经在 train_small.txt 里了。")
# else:
#     # 如果数量不足 n_select, 就全用
#     if len(candidate_ids) <= n_select:
#         selected_ids = candidate_ids
#         print(f"候选数少于 {n_select}, 将全部 {len(selected_ids)} 个样本加入 train_small.txt。")
#     else:
#         selected_ids = random.sample(candidate_ids, n_select)
#         print(f"从候选中随机抽取 {n_select} 个样本。")

#     selected_ids.sort()  # 仅为了文件内容美观, 可选

#     # ================== 追加写入 train_small.txt ==================
#     with open(train_small_path, "a") as f:
#         for id_ in selected_ids:
#             f.write(id_ + "\n")

#     print("\n===== 写入完成 =====")
#     print(f"新增写入样本数: {len(selected_ids)}")
#     print(f"文件路径: {train_small_path}")

# # ================== 读取已有的 train_small.txt(如果有的话) ==================
# existing_ids = set()
# if os.path.exists(val_small_path):
#     with open(val_small_path, "r") as f:
#         for line in f:
#             line = line.strip()
#             if line:
#                 existing_ids.add(line)
#     print(f"已有 train_small.txt, 当前包含样本数: {len(existing_ids)}")

# # ================== 过滤掉已经存在的 ID ==================
# candidate_ids = [id_ for id_ in coco_ids if id_ not in existing_ids]
# print(f"可用于新增的 COCO 图像数: {len(candidate_ids)}")

# if not candidate_ids:
#     print("没有可新增的 COCO 图像, 全部都已经在 train_small.txt 里了。")
# else:
#     # 如果数量不足 n_select, 就全用
#     if len(candidate_ids) <= n_select:
#         selected_ids = candidate_ids
#         print(f"候选数少于 {n_select}, 将全部 {len(selected_ids)} 个样本加入 train_small.txt。")
#     else:
#         selected_ids = random.sample(candidate_ids, n_select)
#         print(f"从候选中随机抽取 {n_select} 个样本。")

#     selected_ids.sort()  # 仅为了文件内容美观, 可选

#     # ================== 追加写入 train_small.txt ==================
#     with open(val_small_path, "a") as f:
#         for id_ in selected_ids:
#             f.write(id_ + "\n")

#     print("\n===== 写入完成 =====")
#     print(f"新增写入样本数: {len(selected_ids)}")
#     print(f"文件路径: {val_small_path}")


import os
import random

# 配置部分 -------------------------------------------------
JPEG_DIR   = r"/mnt/e/jwj/my_data/JPEGImages"   # 存原图的文件夹
# TRAIN_TXT  = r"/mnt/e/jwj/my_data/ImageSets/Segmentation/train.txt"    # 你的 train.txt 路径
TRAIN_TXT  = r"/mnt/e/jwj/my_data/ImageSets/Segmentation/val.txt"    # 你的 val.txt 路径
NUM_NEW    = 200                            # 想随机加多少张，自己改
IMG_EXTS   = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
# -----------------------------------------------------------

# 1. 读现有 train.txt 里的 ID
if os.path.exists(TRAIN_TXT):
    with open(TRAIN_TXT, 'r', encoding='utf-8') as f:
        existing_ids = {line.strip() for line in f if line.strip() != ''}
else:
    existing_ids = set()

print(f"已有样本数: {len(existing_ids)}")

# 2. 扫描 JPEGImages 下所有图片，取 basename 作为 ID
all_ids = []
for fname in os.listdir(JPEG_DIR):
    fname_lower = fname.lower()
    if fname_lower.endswith(IMG_EXTS):
        base = os.path.splitext(fname)[0]
        all_ids.append(base)

all_ids = list(set(all_ids))  # 去重一下
print(f"JPEGImages 中图片总数: {len(all_ids)}")

# 3. 找出还没在 train.txt 里的候选 ID
candidates = [iid for iid in all_ids if iid not in existing_ids]
print(f"可新增的候选图片数: {len(candidates)}")

if len(candidates) == 0:
    print("没有可以新增的图片了，全部都在 train.txt 里。")
else:
    k = min(NUM_NEW, len(candidates))
    new_ids = random.sample(candidates, k)
    print(f"本次随机选取 {k} 个新样本。")

    # 4. 追加写入 train.txt
    with open(TRAIN_TXT, 'a', encoding='utf-8') as f:
        for iid in new_ids:
            f.write(iid + '\n')

    print("写入完成，新加入的 ID：")
    for iid in new_ids:
        print(iid)
