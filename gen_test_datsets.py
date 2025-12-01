#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把 src_txt 里每行的 fake 图片路径，转换/拷贝到新数据集目录，并生成新格式 txt：

输出每行格式：
/mnt/e/jwj/new_datasets/Image/<label>_<id>.jpg /mnt/e/jwj/new_datasets/Mask/<label>_<id>.png <label>

要求：
- 自动从路径里识别 label（一般是 .../<label>/fake/xxx.png）
- 自动寻找对应 mask（优先同级的 mask/gt/labels 等目录；找不到会报 missing）
- 保存的 mask 保证是 0/1 分布（0/255 或灰度都会二值化为 0/1，再写回 png）

用法示例：
python3 pscc_to_newtxt.py \
  --src_txt "/mnt/e/datasets/PSCC/src.txt" \
  --dst_root "/mnt/e/jwj/new_datasets" \
  --out_txt "/mnt/e/jwj/new_datasets/pscc_copymove.txt" \
  --workers 16
"""

import os
import re
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np


IMG_EXTS  = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
MASK_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

MASK_DIR_CANDIDATES = [
    "mask", "Mask", "masks", "Masks",
    "gt", "GT", "gts", "GTS",
    "label", "Label", "labels", "Labels",
    "groundtruth", "GroundTruth", "ground_truth", "gt_mask", "GT_MASK",
    "tamper", "Tamper", "tampered", "Tampered",
]

KEYWORDS = ("mask", "gt", "label", "groundtruth", "tamper")


def infer_label_from_path(img_path: Path) -> str:
    """
    从路径推断 label：优先找 .../<label>/fake/<file> 或 .../<label>/Fake/<file>
    找不到就返回 "hard"
    """
    parts = [p for p in img_path.parts]
    parts_low = [p.lower() for p in parts]
    if "fake" in parts_low:
        idx = parts_low.index("fake")
        if idx - 1 >= 0:
            return parts[idx - 1]
    return "hard"


def find_category_dir(img_path: Path) -> Path:
    """
    尝试定位 category_dir（label 目录）：.../<label>/fake/xxx.png -> .../<label>
    """
    parts = list(img_path.parts)
    parts_low = [p.lower() for p in parts]
    if "fake" in parts_low:
        idx = parts_low.index("fake")
        if idx - 1 >= 0:
            return Path(*parts[:idx])  # 到 fake 之前
    return img_path.parent


def locate_mask(img_path: Path) -> Path | None:
    """
    给定 fake 图片路径，尽量鲁棒地找到 mask：
    1) .../<label>/fake/xxx.png -> .../<label>/<cand>/xxx.png
    2) 或 .../<label>/<cand>/... 子目录里找同 stem
    """
    stem = img_path.stem
    suffix = img_path.suffix.lower()

    category_dir = find_category_dir(img_path)

    # 1) 直接同级候选目录
    for d in MASK_DIR_CANDIDATES:
        cand = category_dir / d / f"{stem}.png"
        if cand.exists():
            return cand
        cand2 = category_dir / d / f"{stem}{suffix}"
        if cand2.exists():
            return cand2
        # stem 任意后缀
        for ext in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"):
            cand3 = category_dir / d / f"{stem}{ext}"
            if cand3.exists():
                return cand3

    # 2) 在 category_dir 内 rglob 搜索同 stem 的文件，优先路径含关键词的
    matches = []
    for ext in MASK_EXTS:
        matches.extend(category_dir.rglob(f"{stem}{ext}"))
    if not matches:
        # 再兜底：stem.* 但要过滤后缀
        for p in category_dir.rglob(f"{stem}.*"):
            if p.is_file() and p.suffix.lower() in MASK_EXTS:
                matches.append(p)

    if not matches:
        return None

    def score(p: Path) -> tuple:
        s = p.as_posix().lower()
        kw_hit = any(k in s for k in KEYWORDS)
        # 更偏好：含关键词、路径更短、离 category_dir 更近
        return (0 if kw_hit else 1, len(p.parts), len(s))

    matches.sort(key=score)
    return matches[0]


def ensure_unique_path(p: Path) -> Path:
    """若目标文件已存在，自动加 _1/_2... 避免覆盖"""
    if not p.exists():
        return p
    stem = p.stem
    suf = p.suffix
    parent = p.parent
    i = 1
    while True:
        cand = parent / f"{stem}_{i}{suf}"
        if not cand.exists():
            return cand
        i += 1


def save_image_as_jpg(src_img_path: Path, dst_jpg_path: Path, quality: int = 95) -> None:
    img = cv2.imread(str(src_img_path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"imread failed: {src_img_path}")
    ok = cv2.imwrite(str(dst_jpg_path), img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError(f"imwrite jpg failed: {dst_jpg_path}")


def save_mask_as_01_png(src_mask_path: Path, dst_png_path: Path, thresh: int = 127) -> None:
    m = cv2.imread(str(src_mask_path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise RuntimeError(f"imread mask failed: {src_mask_path}")

    # 统一二值化为 0/1（不管原来是 0/255、0/1、还是灰度）
    bin01 = (m > thresh).astype(np.uint8)  # 0 or 1
    ok = cv2.imwrite(str(dst_png_path), bin01)
    if not ok:
        raise RuntimeError(f"imwrite mask failed: {dst_png_path}")


def process_one(line: str, dst_img_dir: Path, dst_mask_dir: Path, jpg_quality: int) -> tuple[str, str] | tuple[None, str]:
    """
    成功返回 (out_line, "")
    失败返回 (None, err_msg)
    """
    line = line.strip()
    if not line:
        return (None, "empty line")

    img_path = Path(line)
    if not img_path.exists():
        return (None, f"image not found: {img_path}")

    if img_path.suffix.lower() not in IMG_EXTS:
        return (None, f"not image ext: {img_path}")

    label = infer_label_from_path(img_path)
    stem = img_path.stem

    mask_path = locate_mask(img_path)
    if mask_path is None or (not mask_path.exists()):
        return (None, f"mask not found for: {img_path}")

    # 目标命名：<label>_<id>.jpg / <label>_<id>.png
    dst_img = ensure_unique_path(dst_img_dir / f"{label}_{stem}.jpg")
    dst_msk = ensure_unique_path(dst_mask_dir / f"{label}_{stem}.png")

    # 保存
    save_image_as_jpg(img_path, dst_img, quality=jpg_quality)
    save_mask_as_01_png(mask_path, dst_msk, thresh=127)

    out_line = f"{dst_img.as_posix()} {dst_msk.as_posix()} {label}"
    return (out_line, "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_txt", default="./sampled_files.txt", type=str, help="每行一个 fake 图片路径（可以包含空格目录，整行就是路径）")
    ap.add_argument("--dst_root", default="/mnt/e/jwj/new_datasets", type=str, help="新数据集根目录，里面会有 Image/Mask")
    ap.add_argument("--out_txt", default="./test_2000.txt", type=str, help="输出的新 txt")
    ap.add_argument("--workers", default=16, type=int, help="线程数")
    ap.add_argument("--jpg_quality", default=95, type=int, help="jpg 质量 1~100")
    args = ap.parse_args()

    src_txt = Path(args.src_txt)
    assert src_txt.exists(), f"src_txt not found: {src_txt}"

    dst_root = Path(args.dst_root)
    dst_img_dir = dst_root / "Image"
    dst_mask_dir = dst_root / "Mask"
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_mask_dir.mkdir(parents=True, exist_ok=True)

    lines = src_txt.read_text(encoding="utf-8").splitlines()
    lines = [x.strip() for x in lines if x.strip()]
    if not lines:
        print("src_txt 为空")
        return

    out_lines = []
    bad = []

    workers = args.workers if args.workers > 0 else min(32, (os.cpu_count() or 8) * 2)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(process_one, ln, dst_img_dir, dst_mask_dir, args.jpg_quality) for ln in lines]
        for fut in as_completed(futs):
            out_line, err = fut.result()
            if out_line is not None:
                out_lines.append(out_line)
            else:
                bad.append(err)

    out_lines.sort()
    out_txt = Path(args.out_txt)
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_txt.write_text("\n".join(out_lines) + ("\n" if out_lines else ""), encoding="utf-8")

    # 失败记录
    bad_txt = out_txt.with_suffix(".bad.txt")
    if bad:
        bad_txt.write_text("\n".join(bad) + "\n", encoding="utf-8")

    print("======== DONE ========")
    print(f"src lines: {len(lines)}")
    print(f"ok pairs : {len(out_lines)}")
    print(f"bad      : {len(bad)}")
    print(f"out_txt  : {out_txt.as_posix()}")
    if bad:
        print(f"bad_txt  : {bad_txt.as_posix()}")


if __name__ == "__main__":
    main()
