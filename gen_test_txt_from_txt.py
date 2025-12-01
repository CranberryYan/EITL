#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import argparse
from typing import List


def replace_path_component(p: str, src: str, dst: str) -> str:
    """
    只替换路径中的某个“目录名组件”，例如 .../fake/xxx -> .../mask_01/xxx
    """
    p = p.strip().strip('"').strip("'")
    p = p.replace("\\", "/")  # 兼容 win 风格输入
    parts = p.split("/")
    parts = [dst if x == src else x for x in parts]
    return "/".join(parts)


def map_filename(stem: str, from_prefix: str, to_prefix: str) -> str:
    """
    canonxt_02_sub_01 -> canong3_02_sub_01
    只替换第一个下划线之前的前缀；若不匹配则保持原样。
    """
    if "_" in stem:
        head, tail = stem.split("_", 1)
        if head == from_prefix:
            return f"{to_prefix}_{tail}"
        return stem
    else:
        # 没有下划线：若整体等于 from_prefix 就替换，否则原样
        return to_prefix if stem == from_prefix else stem


def read_lines(txt_path: str) -> List[str]:
    if not os.path.isfile(txt_path):
        raise FileNotFoundError(f"txt not found: {txt_path}")
    out = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            out.append(s)
    return out


def main():
    ap = argparse.ArgumentParser(description="Rewrite txt lines into: img mask hard")
    ap.add_argument("--in_txt", default="./sampled_files.txt", help="输入txt（每行一个图片路径）")
    ap.add_argument("--out_txt", default="./test_2000.txt", help="输出txt（每行三列：img mask hard）")

    ap.add_argument("--img_dirname", default="fake", help="图片目录名组件（默认 fake）")
    ap.add_argument("--mask_dirname", default="mask_01", help="mask目录名组件（默认 mask_01）")

    ap.add_argument("--from_prefix", default="canonxt", help="文件名前缀：待替换（默认 canonxt）")
    ap.add_argument("--to_prefix", default="canong3", help="文件名前缀：替换为（默认 canong3）")

    ap.add_argument("--img_ext", default=".jpg", help="输出图片后缀（默认 .jpg）")
    ap.add_argument("--mask_ext", default=".png", help="输出mask后缀（默认 .png）")

    ap.add_argument("--label", default="hard", help="每行最后的label（默认 hard）")

    ap.add_argument("--check_exists", action="store_true",
                    help="可选：检查输出img/mask是否存在，不存在则记录到 .missing.txt")

    args = ap.parse_args()

    lines = read_lines(args.in_txt)

    out_lines = []
    missing = []

    for p in lines:
        p = p.replace("\\", "/").strip().strip('"').strip("'")

        # 取文件名 stem
        base = os.path.basename(p)
        stem, _ext = os.path.splitext(base)

        new_stem = map_filename(stem, args.from_prefix, args.to_prefix)

        # 输出 img：仍在 fake 目录下，改名 + 强制后缀
        img_dir = os.path.dirname(p)
        img_out = f"{img_dir}/{new_stem}{args.img_ext}"

        # 输出 mask：把路径组件 fake -> mask_01，强制 .png
        mask_path_base = replace_path_component(p, args.img_dirname, args.mask_dirname)
        mask_dir = os.path.dirname(mask_path_base)
        mask_out = f"{mask_dir}/{new_stem}{args.mask_ext}"

        out_line = f"{img_out} {mask_out} {args.label}"
        out_lines.append(out_line)

        if args.check_exists:
            if not os.path.isfile(img_out) or not os.path.isfile(mask_out):
                missing.append(out_line)

    os.makedirs(os.path.dirname(args.out_txt) or ".", exist_ok=True)
    with open(args.out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines) + ("\n" if out_lines else ""))

    print("DONE")
    print(f"in_txt : {args.in_txt}")
    print(f"out_txt: {args.out_txt}")
    print(f"count  : {len(out_lines)}")

    if args.check_exists:
        miss_path = args.out_txt + ".missing.txt"
        with open(miss_path, "w", encoding="utf-8") as f:
            f.write("\n".join(missing) + ("\n" if missing else ""))
        print(f"missing: {len(missing)} -> {miss_path}")


if __name__ == "__main__":
    main()
