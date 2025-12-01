import os
from PIL import Image

# === 路径根据你现在的数据来 ===
# Windows 路径写法
# src_dir = r"E:\jwj\test_data\Images"
# dst_dir = r"E:\jwj\test_data\Mask0255"

# 如果你在 WSL 里跑，把上面换成：
src_dir = r"/mnt/e/jwj/test_data/Images"
dst_dir = r"/mnt/e/jwj/test_data/Mask0255"

os.makedirs(dst_dir, exist_ok=True)

# 支持的图像后缀
valid_exts = (".bmp", ".dib", ".png", ".jpg", ".jpeg", ".pbm", ".pgm", ".ppm", ".tif", ".tiff")

count = 0

for fname in os.listdir(src_dir):
    fname_lower = fname.lower()

    # 1) 只处理以 coco 开头的文件（不区分大小写）
    if not fname_lower.startswith("coco"):
        continue

    # 2) 只处理图像类型
    if not fname_lower.endswith(valid_exts):
        continue

    img_path = os.path.join(src_dir, fname)

    # 打开图像，获取尺寸
    with Image.open(img_path) as im:
        w, h = im.size

    # 创建同尺寸的全黑 mask（单通道 L，像素值 0）
    mask = Image.new("L", (w, h), 0)

    # GT 命名：base + "_gt.png"，和你的 evaluate 完全一致
    base = os.path.splitext(fname)[0]
    out_name = base + "_gt.png"
    out_path = os.path.join(dst_dir, out_name)

    mask.save(out_path)
    count += 1

print(f"Done! 生成了 {count} 张全黑 GT mask 放在: {dst_dir}")
