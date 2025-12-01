import os
import argparse
from pathlib import Path

import numpy as np
from PIL import Image


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def is_image_file(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMG_EXTS


def make_black_mask(image_path: Path, mask_path: Path) -> None:
    # 只读取尺寸，不做任何图像处理
    with Image.open(image_path) as im:
        im = im.convert("RGB")  # 确保能正确拿到尺寸
        w, h = im.size

    mask = np.zeros((h, w), dtype=np.uint8)  # 全黑：0
    mask_img = Image.fromarray(mask, mode="L")

    mask_path.parent.mkdir(parents=True, exist_ok=True)
    mask_img.save(mask_path, format="PNG")


def main():
    parser = argparse.ArgumentParser(
        description="Scan Image folder; for missing masks in Mask_255, create all-black masks."
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default="/mnt/e/jwj/datasets_with_C1_C2_Cov_Colu/Image",
        help=r'Image folder, e.g. "jwj\datasets_with_C1_C2_Cov_Colu\Image"',
    )
    parser.add_argument(
        "--mask_dir",
        type=str,
        default="/mnt/e/jwj/datasets_with_C1_C2_Cov_Colu/Mask_255",
        help=r'Mask folder, e.g. "jwj\datasets_with_C1_C2_Cov_Colu\Mask_255"',
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only print what would be created, do not write files.",
    )
    args = parser.parse_args()

    image_root = Path(args.image_dir)
    mask_root = Path(args.mask_dir)

    if not image_root.exists():
        raise FileNotFoundError(f"Image dir not found: {image_root}")
    if not mask_root.exists():
        print(f"[WARN] Mask dir not found, will create: {mask_root}")
        mask_root.mkdir(parents=True, exist_ok=True)

    total_imgs = 0
    created = 0
    skipped = 0

    # 递归扫描 Image 下的所有图片
    for img_path in image_root.rglob("*"):
        if not is_image_file(img_path):
            continue

        total_imgs += 1

        rel = img_path.relative_to(image_root)          # 保留相对路径
        rel_png = rel.with_suffix(".png")               # mask 统一用 .png
        mask_path = mask_root / rel_png

        if mask_path.exists():
            skipped += 1
            continue

        if args.dry_run:
            print(f"[DRY] create: {mask_path}  (from {img_path})")
        else:
            make_black_mask(img_path, mask_path)
            print(f"[OK ] created: {mask_path}")
        created += 1

    print("\n===== Summary =====")
    print(f"Image scanned : {total_imgs}")
    print(f"Mask existed  : {skipped}")
    print(f"Mask created  : {created}")
    print("===================")


if __name__ == "__main__":
    main()
