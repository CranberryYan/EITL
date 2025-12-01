#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成“新加入数据集”的 txt（只包含新增样本），每行格式：
/mnt/e/jwj/new_datasets/Image/xxx.jpg /mnt/e/jwj/new_datasets/Mask/xxx.png hard

“新增”的判定方式（鲁棒）：
- 读取你已有的旧 txt（里面每行第 1 列是 image 路径）
- 扫描当前 Image 目录下所有图片
- 若某个样本的 key（相对路径+stem，大小写不敏感）不在旧 txt 中，则认为它是新加入样本

mask 匹配规则（鲁棒）：
- 优先匹配：Mask/<相对路径>/<stem>.png
- 若不存在，则 Mask/<相对路径>/<stem>.* 任意后缀
- 若仍不存在，则 Mask/<stem>.png
- 若仍不存在，则 Mask/<stem>.* 任意后缀
"""

import os
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

IMG_EXTS  = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
MASK_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def norm_posix(p: Path) -> str:
    return p.as_posix()


def make_key(root: Path, p: Path) -> str:
    """
    key = 相对目录 + stem（小写），用于稳定匹配
    例如：Image/a/b/001.jpg  ->  "a/b/001"
    """
    try:
        rel = p.relative_to(root)
        rel_parent = rel.parent.as_posix().lower()
        stem = rel.stem.lower()
        if rel_parent == ".":
            return stem
        return f"{rel_parent}/{stem}"
    except Exception:
        return p.stem.lower()


def read_old_keys(old_txt: Path, image_root: Path) -> set:
    """
    旧 txt 每行格式：
      img_path mask_path label
    我们只取第1列 img_path
    """
    keys = set()
    if not old_txt.exists():
        print(f"[WARN] old_txt not found: {old_txt} (treat as empty)")
        return keys

    with old_txt.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 1:
                continue
            img_path = Path(parts[0])
            # 有些 txt 可能是相对路径，或路径不在 image_root 下，都做兜底
            keys.add(make_key(image_root, img_path))
    return keys


def iter_images(image_root: Path, recursive: bool):
    if recursive:
        for p in image_root.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                yield p
    else:
        for p in image_root.iterdir():
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                yield p


def find_mask_for_image(mask_root: Path, image_root: Path, img_path: Path) -> Optional[Path]:
    """
    依次尝试（越靠前优先级越高）：
      1) Mask/<rel_parent>/<stem>.png
      2) Mask/<rel_parent>/<stem>.*
      3) Mask/<stem>.png
      4) Mask/<stem>.*
    """
    stem = img_path.stem
    try:
        rel = img_path.relative_to(image_root)
        rel_parent = rel.parent  # 可能是 "."
    except Exception:
        rel_parent = Path(".")

    # 1) 同相对目录 + .png
    cand = mask_root / rel_parent / f"{stem}.png"
    if cand.exists():
        return cand

    # 2) 同相对目录 + 任意后缀
    cand_list = sorted((mask_root / rel_parent).glob(f"{stem}.*"))
    for c in cand_list:
        if c.is_file() and c.suffix.lower() in MASK_EXTS:
            return c

    # 3) 根目录下同 stem + .png
    cand = mask_root / f"{stem}.png"
    if cand.exists():
        return cand

    # 4) 根目录下同 stem + 任意后缀
    cand_list = sorted(mask_root.glob(f"{stem}.*"))
    for c in cand_list:
        if c.is_file() and c.suffix.lower() in MASK_EXTS:
            return c

    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image_root", type=str, default="/mnt/e/jwj/datasets_with_C1_C2_Cov_Colu/Image")
    ap.add_argument("--mask_root", type=str, default="/mnt/e/jwj/datasets_with_C1_C2_Cov_Colu/Mask")
    ap.add_argument("--old_txt", type=str, default="./train.txt", help="你已有的旧数据集 txt（用于判断哪些是新加入）")
    ap.add_argument("--out_txt", type=str, default="/mnt/e/jwj/datasets_with_C1_C2_Cov_Colu/new_added_hard.txt")
    ap.add_argument("--recursive", action="store_true", help="递归扫描子目录（默认不递归）")
    ap.add_argument("--suffix_label", type=str, default="hard", help="行尾标签（默认 hard）")
    args = ap.parse_args()

    image_root = Path(args.image_root)
    mask_root  = Path(args.mask_root)
    old_txt    = Path(args.old_txt)
    out_txt    = Path(args.out_txt)

    assert image_root.exists(), f"Not found: {image_root}"
    assert mask_root.exists(), f"Not found: {mask_root}"

    old_keys = read_old_keys(old_txt, image_root)
    print(f"[INFO] old keys loaded: {len(old_keys)}")

    new_lines: List[str] = []
    miss_masks: List[str] = []

    total_imgs = 0
    new_imgs = 0

    for img_path in iter_images(image_root, args.recursive):
        total_imgs += 1
        k = make_key(image_root, img_path)
        if k in old_keys:
            continue  # 旧样本，跳过

        # 新样本
        new_imgs += 1
        mask_path = find_mask_for_image(mask_root, image_root, img_path)
        if mask_path is None:
            miss_masks.append(norm_posix(img_path))
            continue

        line = f"{norm_posix(img_path)} {norm_posix(mask_path)} {args.suffix_label}"
        new_lines.append(line)

    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_txt.write_text("\n".join(new_lines) + ("\n" if new_lines else ""), encoding="utf-8")

    # 额外输出缺 mask 的列表，方便你清洗
    miss_txt = out_txt.with_suffix(".missing_mask.txt")
    if miss_masks:
        miss_txt.write_text("\n".join(miss_masks) + "\n", encoding="utf-8")

    print("======== DONE ========")
    print(f"Scanned images: {total_imgs}")
    print(f"New images detected: {new_imgs}")
    print(f"New pairs written: {len(new_lines)}")
    print(f"Missing masks: {len(miss_masks)}")
    print(f"[OK] out_txt: {out_txt.as_posix()}")
    if miss_masks:
        print(f"[WARN] missing mask list: {miss_txt.as_posix()}")


if __name__ == "__main__":
    main()
