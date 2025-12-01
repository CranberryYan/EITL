#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import random
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Tuple, List

import cv2
import numpy as np

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
MASK_EXTS = {".png", ".bmp", ".tif", ".tiff", ".webp", ".jpg", ".jpeg"}


def norm_path(p: str) -> Path:
    return Path(p.replace("\\", "/")).expanduser().resolve()


def iter_images(root: Path, recursive: bool) -> List[Path]:
    if recursive:
        return [p for p in root.rglob("*")
                if p.is_file() and p.suffix.lower() in IMG_EXTS and not p.name.startswith(".")]
    else:
        return [p for p in root.iterdir()
                if p.is_file() and p.suffix.lower() in IMG_EXTS and not p.name.startswith(".")]


def ensure_dir(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)


def find_mask(mask_root: Path, rel_no_ext: Path) -> Optional[Path]:
    """
    优先找 rel_no_ext + ".png"
    否则尝试其他后缀（同目录同 stem）
    """
    p_png = mask_root / (str(rel_no_ext) + ".png")
    if p_png.exists():
        return p_png

    cand_dir = (mask_root / rel_no_ext).parent
    stem = rel_no_ext.name
    if cand_dir.exists():
        for ext in MASK_EXTS:
            c = cand_dir / (stem + ext)
            if c.exists():
                return c

    return None


def mask_is_all_black(mask_path: Path) -> bool:
    m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return False
    return cv2.countNonZero(m) == 0


def save_mask_png(dst_mask: Path, mask_arr_u8: np.ndarray) -> None:
    ensure_dir(dst_mask)
    ok = cv2.imwrite(str(dst_mask), mask_arr_u8)
    if not ok:
        raise RuntimeError(f"imwrite failed: {dst_mask}")


def save_image_jpg(dst_img: Path, img_bgr: np.ndarray, quality: int = 95) -> None:
    ensure_dir(dst_img)
    params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    ok = cv2.imwrite(str(dst_img), img_bgr, params)
    if not ok:
        raise RuntimeError(f"imwrite failed: {dst_img}")


def read_image_bgr(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"imread failed: {path}")
    return img


def read_mask_gray(path: Path) -> np.ndarray:
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise RuntimeError(f"imread failed: {path}")
    return m


def process_one(
    img_path: Path,
    image_root: Path,
    mask_root: Path,
    out_root: Path,
    overwrite: bool,
) -> Tuple[str, str, str]:
    """
    返回： (dst_img_posix, dst_mask_posix, label)
    """
    rel = img_path.relative_to(image_root)
    rel_no_ext = rel.with_suffix("")

    dst_img = out_root / "Image" / (str(rel_no_ext) + ".jpg")
    dst_mask = out_root / "Mask" / (str(rel_no_ext) + ".png")

    src_mask = find_mask(mask_root, rel_no_ext)

    img_bgr = read_image_bgr(img_path)
    h, w = img_bgr.shape[:2]

    if src_mask is None or (not src_mask.exists()):
        label = "authentic"
        black = np.zeros((h, w), dtype=np.uint8)
        if overwrite or (not dst_mask.exists()):
            save_mask_png(dst_mask, black)
    else:
        is_black = mask_is_all_black(src_mask)
        label = "authentic" if is_black else "hard"

        if overwrite or (not dst_mask.exists()):
            m = read_mask_gray(src_mask).astype(np.uint8)
            save_mask_png(dst_mask, m)

    if overwrite or (not dst_img.exists()):
        save_image_jpg(dst_img, img_bgr, quality=95)

    return (dst_img.as_posix(), dst_mask.as_posix(), label)


def main():
    # ====== 路径 ======
    image_root = norm_path(r"/mnt/e/jwj/new_datasets/Image")
    mask_root  = norm_path(r"/mnt/e/jwj/new_datasets/Mask")
    out_root   = norm_path(r"/mnt/e/jwj/new_datasets")
    out_list   = out_root / "train.txt"

    # ====== 抽样设置 ======
    SAMPLE_N = 20000
    SEED = 2026

    # True：只处理抽到的 20000 个（推荐，省时间/省空间）
    # False：全量都复制到 new_datasets，但 txt 只写 20000 行
    COPY_ONLY_SAMPLED = True

    # ====== 其他设置 ======
    RECURSIVE = True
    WORKERS = 0
    OVERWRITE = False

    if not image_root.exists():
        raise FileNotFoundError(f"Image root not found: {image_root}")
    if not mask_root.exists():
        print(f"[WARN] Mask root not found: {mask_root} (将把缺失全部视为 authentic 并补全黑 mask)")

    (out_root / "Image").mkdir(parents=True, exist_ok=True)
    (out_root / "Mask").mkdir(parents=True, exist_ok=True)

    imgs_all = iter_images(image_root, RECURSIVE)
    if not imgs_all:
        print("未找到任何图像文件，请检查 Image 路径与后缀。")
        return

    # ====== 随机抽样 ======
    random.seed(SEED)
    if SAMPLE_N > len(imgs_all):
        print(f"[WARN] 你要求抽 {SAMPLE_N}，但总共只有 {len(imgs_all)}，将使用全部。")
        imgs_sampled = imgs_all
    else:
        imgs_sampled = random.sample(imgs_all, SAMPLE_N)

    # 是否全量复制
    imgs_to_process = imgs_sampled if COPY_ONLY_SAMPLED else imgs_all

    workers = WORKERS if WORKERS and WORKERS > 0 else min(32, (os.cpu_count() or 8) * 2)

    results: List[Tuple[str, str, str]] = []
    bad: List[Tuple[str, str]] = []

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [
            ex.submit(process_one, p, image_root, mask_root, out_root, OVERWRITE)
            for p in imgs_to_process
        ]
        for fu in as_completed(futs):
            try:
                results.append(fu.result())
            except Exception as e:
                bad.append(("<unknown>", str(e)))

    # results 里可能是全量或抽样处理结果；我们只写抽样的 20000 行
    # 所以再过滤一下：只保留抽样对应的 dst 路径
    sampled_set = set()
    for p in imgs_sampled:
        rel = p.relative_to(image_root).with_suffix("")
        sampled_set.add((out_root / "Image" / (str(rel) + ".jpg")).as_posix())

    results_sampled = [r for r in results if r[0] in sampled_set]
    results_sampled.sort(key=lambda x: x[0])

    # 写 txt（只写抽样）
    with open(out_list, "w", encoding="utf-8") as f:
        for img_p, mask_p, label in results_sampled:
            f.write(f"{img_p} {mask_p} {label}\n")

    n_auth = sum(1 for _, _, lab in results_sampled if lab == "authentic")
    n_hard = sum(1 for _, _, lab in results_sampled if lab == "hard")

    print("====== DONE ======")
    print(f"Total images (all)   : {len(imgs_all)}")
    print(f"Sampled for txt      : {len(imgs_sampled)}")
    print(f"Actually processed   : {len(imgs_to_process)}  (COPY_ONLY_SAMPLED={COPY_ONLY_SAMPLED})")
    print(f"List file            : {out_list}")
    print(f"Authentic in sample  : {n_auth}")
    print(f"Hard in sample       : {n_hard}")

    if bad:
        print(f"[WARN] Failed samples: {len(bad)} (示例前10条)")
        for i, (p, err) in enumerate(bad[:10]):
            print(i, p, "->", err)


if __name__ == "__main__":
    main()
