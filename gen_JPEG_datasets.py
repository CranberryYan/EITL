#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import shlex
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Tuple, Optional

from PIL import Image


def parse_line(line: str) -> Optional[Tuple[str, str, str]]:
    """
    解析一行：img_path mask_path label
    允许路径带空格（shlex.split）
    label 可能是一个词（如 hard / normal），也可能有多个词（就全拼回去）
    """
    line = line.strip()
    if not line:
        return None
    try:
        parts = shlex.split(line)
    except Exception:
        parts = line.split()

    if len(parts) < 3:
        return None

    img_path = parts[0].strip('"').strip("'")
    mask_path = parts[1].strip('"').strip("'")
    label = " ".join(parts[2:]).strip()
    return img_path, mask_path, label


def recompress_to_jpeg(
    in_img_path: str,
    out_img_path: str,
    quality: int,
    subsampling: str = "keep",
    optimize: bool = True,
    progressive: bool = False,
    overwrite: bool = False,
) -> Tuple[bool, str]:
    """
    把 in_img_path 重压缩保存为 JPEG 到 out_img_path
    subsampling:
      - "keep": 尽量沿用原图(若原图是JPEG可尝试使用其 subsampling 信息；PIL不一定总能取到)
      - "444": 4:4:4  (subsampling=0)
      - "422": 4:2:2  (subsampling=1)
      - "420": 4:2:0  (subsampling=2)  常用默认
    """
    if (not overwrite) and os.path.exists(out_img_path):
        return True, "skip_exists"

    os.makedirs(os.path.dirname(out_img_path), exist_ok=True)

    try:
        with Image.open(in_img_path) as im:
            # 转 RGB（JPEG 不支持 alpha；L/CMYK 等也统一转）
            if im.mode != "RGB":
                im = im.convert("RGB")

            save_kwargs = dict(
                format="JPEG",
                quality=int(quality),
                optimize=bool(optimize),
                progressive=bool(progressive),
            )

            # subsampling 设置
            if subsampling.lower() == "444":
                save_kwargs["subsampling"] = 0
            elif subsampling.lower() == "422":
                save_kwargs["subsampling"] = 1
            elif subsampling.lower() == "420":
                save_kwargs["subsampling"] = 2
            # "keep"：不显式指定，让 PIL 用默认策略

            # 尽量保留 exif（若有）
            exif = im.info.get("exif", None)
            if exif is not None:
                save_kwargs["exif"] = exif

            # 保存
            im.save(out_img_path, **save_kwargs)

        return True, "ok"
    except Exception as e:
        return False, f"err:{e}"


def main():
    parser = argparse.ArgumentParser(
        description="Make JPEG recompressed dataset & new txt for robustness experiments."
    )
    parser.add_argument("--in_txt", type=str, default="./test_txt/fused_with_allTemp.txt", help="输入txt（每行：img mask label）")
    parser.add_argument("--out_root", type=str, default="/mnt/e/jwj/new_datasets/", help="输出根目录（例如 /mnt/e/jwj/new_datasets）")

    # 你说的“压缩因子”一般就对应这里的 quality
    parser.add_argument("--quality", type=int, default=95, help="JPEG quality (1~95/100), e.g. 90/80/70")

    # 输出子目录名：你可以写 JPEG_80 或 JPEG_压缩因子 之类
    parser.add_argument("--out_dir_name", type=str, default=None,
                        help="输出子目录名（默认 JPEG_<quality>），例如 JPEG_80 或 JPEG_压缩因子")

    parser.add_argument("--out_txt", type=str, default=None,
                        help="输出txt路径（默认在 out_root 下自动生成）")

    parser.add_argument("--workers", type=int, default=16, help="并行线程数（建议 4~16）")
    parser.add_argument("--overwrite", action="store_true", help="若输出图已存在则覆盖")
    parser.add_argument("--subsampling", type=str, default="420", choices=["keep", "444", "422", "420"],
                        help="JPEG chroma subsampling（默认 420，常用）")
    parser.add_argument("--no_optimize", action="store_true", help="关闭 optimize")
    parser.add_argument("--progressive", action="store_true", help="使用 progressive JPEG")

    args = parser.parse_args()

    in_txt = args.in_txt
    out_root = args.out_root
    quality = int(args.quality)

    if args.out_dir_name is None:
        out_dir_name = f"JPEG_{quality}"
    else:
        out_dir_name = args.out_dir_name

    out_img_dir = os.path.join(out_root, out_dir_name)

    if args.out_txt is None:
        base = os.path.splitext(os.path.basename(in_txt))[0]
        out_txt = os.path.join(out_root, f"{base}_{out_dir_name}.txt")
    else:
        out_txt = args.out_txt

    # 读取输入
    with open(in_txt, "r", encoding="utf-8") as f:
        lines = f.readlines()

    items = []
    for ln in lines:
        parsed = parse_line(ln)
        if parsed is None:
            continue
        items.append(parsed)

    if len(items) == 0:
        print(f"[ERROR] No valid lines parsed from: {in_txt}")
        sys.exit(1)

    # 任务：每个样本输出到 out_img_dir/原文件名.jpg
    # 并生成新 txt：out_img_path mask_path label
    results = []
    ok_cnt = 0
    skip_cnt = 0
    err_cnt = 0

    optimize = (not args.no_optimize)

    def _job(img_path, mask_path, label):
        base = os.path.splitext(os.path.basename(img_path))[0]
        out_img_path = os.path.join(out_img_dir, base + ".jpg")
        ok, msg = recompress_to_jpeg(
            img_path,
            out_img_path,
            quality=quality,
            subsampling=args.subsampling,
            optimize=optimize,
            progressive=args.progressive,
            overwrite=args.overwrite,
        )
        return ok, msg, out_img_path, mask_path, label, img_path

    with ThreadPoolExecutor(max_workers=int(args.workers)) as ex:
        futs = [ex.submit(_job, *it) for it in items]
        for fut in as_completed(futs):
            ok, msg, out_img_path, mask_path, label, src = fut.result()
            if ok:
                if msg == "skip_exists":
                    skip_cnt += 1
                else:
                    ok_cnt += 1
                results.append((out_img_path, mask_path, label))
            else:
                err_cnt += 1
                print(f"[FAIL] {src} -> {msg}", file=sys.stderr)

    # 写新 txt
    os.makedirs(os.path.dirname(out_txt), exist_ok=True)
    with open(out_txt, "w", encoding="utf-8") as f:
        for out_img_path, mask_path, label in results:
            f.write(f"{out_img_path} {mask_path} {label}\n")

    print("========== DONE ==========")
    print(f"in_txt      : {in_txt}")
    print(f"out_img_dir : {out_img_dir}")
    print(f"out_txt     : {out_txt}")
    print(f"quality     : {quality}")
    print(f"subsampling : {args.subsampling}")
    print(f"optimize    : {optimize}")
    print(f"progressive : {args.progressive}")
    print(f"workers     : {args.workers}")
    print(f"ok          : {ok_cnt}")
    print(f"skip_exists : {skip_cnt}")
    print(f"errors      : {err_cnt}")
    print("==========================")

    if err_cnt > 0:
        sys.exit(2)


if __name__ == "__main__":
    main()
