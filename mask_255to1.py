#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from typing import Tuple, Optional, List, Dict, Any

import cv2
import numpy as np

# ===== 路径 =====
src_dir = r"/mnt/e/jwj/datasets_with_C1_C2_Cov_Colu/Mask_255"
dst_dir = r"/mnt/e/jwj/datasets_with_C1_C2_Cov_Colu/Mask"   # ✅ 输出目录（不覆盖原始）

RECURSIVE = False          # True 则递归子目录
WORKERS = 0                # 0=自动
THRESH = 127               # 用于 0/255 -> 0/1 的阈值

# ✅ 输出策略：把所有 mask 都“镜像”到 dst_dir（0255转01、01原样、other原样）
MIRROR_ALL_TO_DST = True

valid_exts = {".png", ".bmp", ".tif", ".tiff", ".webp", ".jpg", ".jpeg"}


def iter_files(root: str, recursive: bool):
    root = os.path.abspath(root)
    if recursive:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if not d.startswith(".")]
            for fn in filenames:
                if fn.startswith("."):
                    continue
                ext = os.path.splitext(fn)[1].lower()
                if ext in valid_exts:
                    yield os.path.join(dirpath, fn)
    else:
        with os.scandir(root) as it:
            for e in it:
                if not e.is_file() or e.name.startswith("."):
                    continue
                ext = os.path.splitext(e.name)[1].lower()
                if ext in valid_exts:
                    yield e.path


def classify_mask(img: np.ndarray) -> str:
    """
    返回:
      - "01"    : 值域只含 {0,1}
      - "0255"  : 值域只含 {0,255}
      - "other" : 其他情况（含多值/灰度/异常）
    """
    is_01 = np.all((img == 0) | (img == 1))
    if is_01:
        return "01"

    is_0255 = np.all((img == 0) | (img == 255))
    if is_0255:
        return "0255"

    return "other"


def short_unique_preview(img: np.ndarray, max_k: int = 10) -> List[int]:
    """给 other 类型做个简短的 unique 预览（最多 max_k 个），避免日志爆炸"""
    u = np.unique(img)
    if u.size > max_k:
        return [int(x) for x in u[:max_k]] + ["..."]  # type: ignore
    return [int(x) for x in u]


def ensure_parent_dir(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def map_to_dst(src_path: str, src_root: str, dst_root: str) -> str:
    """
    将 src_path 映射到 dst_root 下，并统一后缀为 .png，保持相对路径一致。
    """
    rel = os.path.relpath(src_path, start=src_root)
    rel_no_ext, _ = os.path.splitext(rel)
    dst_path = os.path.join(dst_root, rel_no_ext + ".png")
    return dst_path


def process_one(path: str, src_root: str, dst_root: str) -> Tuple[str, str, str, bool, bool, str, Dict[str, Any]]:
    """
    返回:
      status: "ok"/"bad"
      path: src path
      kind: "01"/"0255"/"other"
      converted: 是否进行了 0255->01
      copied_or_saved: 是否向 dst 写出了文件（镜像输出）
      dst_path: 对应输出路径
      stats: 统计信息
    """
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return ("bad", path, "imread_failed", False, False, "", {})

    kind = classify_mask(img)

    total = int(img.size)
    c0 = int(np.count_nonzero(img == 0))
    c1 = int(np.count_nonzero(img == 1))
    c255 = int(np.count_nonzero(img == 255))
    mn = int(img.min())
    mx = int(img.max())

    stats: Dict[str, Any] = {
        "total": total,
        "c0": c0,
        "c1": c1,
        "c255": c255,
        "min": mn,
        "max": mx,
    }
    if kind == "other":
        stats["uniq_preview"] = short_unique_preview(img)

    # 输出路径（统一 .png）
    out_path = map_to_dst(path, src_root, dst_root)

    converted = False
    wrote = False

    if MIRROR_ALL_TO_DST:
        ensure_parent_dir(out_path)

        if kind == "0255":
            # 0/255 -> 0/1
            bin01 = (img > THRESH).astype(np.uint8)  # 0 or 1
            ok = cv2.imwrite(out_path, bin01)        # ✅ 写到 dst，不覆盖源
            if not ok:
                return ("bad", path, "imwrite_failed", False, False, out_path, stats)
            converted = True
            wrote = True

        elif kind == "01":
            # 原样写到 dst（保持 0/1）
            ok = cv2.imwrite(out_path, img.astype(np.uint8))
            if not ok:
                return ("bad", path, "imwrite_failed", False, False, out_path, stats)
            wrote = True

        else:
            # other：也写到 dst，便于后续统一使用 dst_dir
            ok = cv2.imwrite(out_path, img.astype(np.uint8))
            if not ok:
                # 如果写失败，退化为直接拷贝（有些格式 cv2 写不稳）
                try:
                    # 注意：拷贝会保留原后缀，这里我们仍然强制 .png，所以拷贝到 .png 不合适
                    # 因此这里仍然报错更直观
                    return ("bad", path, "imwrite_failed_other", False, False, out_path, stats)
                except Exception:
                    return ("bad", path, "copy_failed_other", False, False, out_path, stats)
            wrote = True

    else:
        # 只对 0255 输出转换结果；01/other 不写
        if kind == "0255":
            ensure_parent_dir(out_path)
            bin01 = (img > THRESH).astype(np.uint8)
            ok = cv2.imwrite(out_path, bin01)
            if not ok:
                return ("bad", path, "imwrite_failed", False, False, out_path, stats)
            converted = True
            wrote = True

    return ("ok", path, kind, converted, wrote, out_path, stats)


def main():
    src_root = os.path.abspath(src_dir)
    dst_root = os.path.abspath(dst_dir)

    files = list(iter_files(src_root, RECURSIVE))
    if not files:
        print("未找到任何 mask 文件，请检查路径/后缀/是否递归。")
        return

    os.makedirs(dst_root, exist_ok=True)

    workers = WORKERS if WORKERS and WORKERS > 0 else min(32, (os.cpu_count() or 8) * 2)
    chunksize = 64

    cnt_kind = Counter()
    ok_cnt = 0
    bad_cnt = 0
    converted_cnt = 0
    wrote_cnt = 0

    # 像素分布累计（全数据集）
    sum0 = 0
    sum1 = 0
    sum255 = 0
    sum_total = 0

    # 采样一些异常文件
    bad_samples = []
    other_samples = []  # (src_path, dst_path, min,max, uniq_preview)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for status, p, kind_or_info, converted, wrote, outp, st in ex.map(
            lambda x: process_one(x, src_root, dst_root),
            files,
            chunksize=chunksize,
        ):
            if status == "ok":
                ok_cnt += 1
                kind = kind_or_info
                cnt_kind[kind] += 1
                if converted:
                    converted_cnt += 1
                if wrote:
                    wrote_cnt += 1

                if st:
                    sum_total += st.get("total", 0)
                    sum0 += st.get("c0", 0)
                    sum1 += st.get("c1", 0)
                    sum255 += st.get("c255", 0)

                    if kind == "other" and len(other_samples) < 20:
                        other_samples.append((
                            p,
                            outp,
                            st.get("min"),
                            st.get("max"),
                            st.get("uniq_preview"),
                        ))
            else:
                bad_cnt += 1
                if len(bad_samples) < 20:
                    bad_samples.append((p, kind_or_info))

    print("======== Mask 值域分布统计（按文件数）========")
    print(f"src_dir: {src_root}")
    print(f"dst_dir: {dst_root}")
    print(f"总计文件: {len(files)}")
    print(f"读取成功: {ok_cnt} | 失败: {bad_cnt}")
    print(f"值域= {{0,1}}   的文件数: {cnt_kind.get('01', 0)}")
    print(f"值域= {{0,255}} 的文件数: {cnt_kind.get('0255', 0)}")
    print(f"其他值域(异常)  的文件数: {cnt_kind.get('other', 0)}")

    print("\n======== 像素级分布累计（全数据集）========")
    if sum_total > 0:
        print(f"总像素: {sum_total}")
        print(f"像素=0   : {sum0}  ({sum0/sum_total:.4%})")
        print(f"像素=1   : {sum1}  ({sum1/sum_total:.4%})")
        print(f"像素=255 : {sum255} ({sum255/sum_total:.4%})")
    else:
        print("像素统计为空（可能全部读取失败）。")

    print("\n======== 输出结果 ========")
    print(f"写入 dst_dir 的文件数（镜像输出）: {wrote_cnt}")
    print(f"其中 {{0,255}} -> {{0,1}} 的转换文件数: {converted_cnt}")

    if other_samples:
        print("\n======== other(异常值域) 样例（最多20条）========")
        for sp, dp, mn, mx, prev in other_samples:
            print(f" src: {sp}\n dst: {dp}\n   min={mn}, max={mx}, uniq_preview={prev}")

    if bad_samples:
        print("\n======== 失败样例（最多20条）========")
        for p, err in bad_samples:
            print(" ", p, "->", err)


if __name__ == "__main__":
    main()
