#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".gif"}


def convert_one(in_path: Path, out_path: Path, force_binary: bool = True):
    """
    读入 mask -> 转 L -> 输出 0/255
    - 如果是 0/1：直接 *255
    - 否则：>0 视为 255（可选 force_binary）
    """
    img = Image.open(in_path)
    if img.mode != "L":
        img = img.convert("L")
    arr = np.array(img)

    # 输出数组
    if force_binary:
        # 任何非零都当 1
        out = (arr > 0).astype(np.uint8) * 255
    else:
        # 只在确实是 0/1 时才 *255，否则原样
        uniq = np.unique(arr)
        if set(uniq.tolist()).issubset({0, 1}):
            out = (arr.astype(np.uint8) * 255)
        else:
            out = arr.astype(np.uint8)

    out_img = Image.fromarray(out, mode="L")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_img.save(out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", default="/mnt/e/jwj/new_datasets/Mask", help="输入目录（递归处理）")
    ap.add_argument("--out_dir", default="/mnt/e/jwj/new_datasets/Mask_255", help="输出目录（为空则与输入同目录）")
    ap.add_argument("--suffix", default="", help="输出文件名后缀，如 _255（为空则不加）")
    ap.add_argument("--overwrite", action="store_true", help="覆盖原文件（忽略 out_dir/suffix）")
    ap.add_argument("--force_binary", action="store_true", help="强制 >0=>255（推荐，最鲁棒）")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    if not in_dir.is_dir():
        raise FileNotFoundError(f"in_dir not found: {in_dir}")

    out_dir = Path(args.out_dir) if args.out_dir else None

    files = [p for p in in_dir.rglob("*") if p.is_file() and p.suffix.lower() in VALID_EXTS]
    print(f"[Info] Found {len(files)} files under {in_dir}")

    ok = 0
    fail = 0
    for p in files:
        try:
            if args.overwrite:
                out_path = p
            else:
                rel = p.relative_to(in_dir)
                stem = p.stem + (args.suffix or "")
                out_name = stem + p.suffix
                if out_dir is None:
                    out_path = p.with_name(out_name)
                else:
                    out_path = out_dir / rel
                    out_path = out_path.with_name(out_name)

            convert_one(p, out_path, force_binary=args.force_binary)
            ok += 1
        except Exception as e:
            print(f"[Warn] Failed: {p} | {e}")
            fail += 1

    print(f"[Done] ok={ok}, fail={fail}")
    if args.overwrite:
        print("Mode: overwrite original files")
    else:
        print(f"Mode: write new files (suffix='{args.suffix}', out_dir='{args.out_dir or 'same as input'}')")


if __name__ == "__main__":
    main()
