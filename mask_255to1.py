import os
from PIL import Image
import numpy as np

# ===== 路径按你自己的环境改 =====
# Windows 原始写法（如果在 Windows 直接跑）：
# src_dir = r"E:\jwj\new_datasets\Mask_255"
# dst_dir = r"E:\jwj\new_datasets\Mask"

# 如果在 WSL 里跑，对应路径一般是这样：
src_dir = r"/mnt/e/jwj/new_datasets/Mask_255"
dst_dir = r"/mnt/e/jwj/new_datasets/Mask"

os.makedirs(dst_dir, exist_ok=True)

valid_exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

count = 0
for fname in os.listdir(src_dir):
    ext = os.path.splitext(fname)[1].lower()
    if ext not in valid_exts:
        continue

    src_path = os.path.join(src_dir, fname)
    dst_path = os.path.join(dst_dir, fname)

    try:
        img = Image.open(src_path).convert("L")
        arr = np.array(img, dtype=np.uint8)

        # 将 0 / 255 映射到 0 / 1（按阈值二值化更稳一点）
        arr_bin = (arr > 127).astype(np.uint8)  # 背景=0，前景=1

        out = Image.fromarray(arr_bin, mode="L")
        out.save(dst_path)

        count += 1
    except Exception as e:
        print(f"[Error] 处理失败: {src_path}, err={e}")

print(f"完成转换，共处理 {count} 张 mask。")
print(f"0-1 掩码已保存到: {dst_dir}")
