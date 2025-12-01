#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把源数据集(fake_256 + mask_256)生成你的 txt，每行格式：

/mnt/e/jwj/new_datasets/Image/<真实文件名>.jpg /mnt/e/jwj/new_datasets/Mask/<真实文件名>.png hard

要求满足：
- ✅ 不拷贝/不改名/不保存任何图片，只写 txt
- ✅ 不输出编号，输出“真实文件名”（源文件的 stem）
- ✅ mask 按 stem（以及相对目录）配对，找不到就记录到 missing
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional

IMG_EXTS  = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
MASK_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def norm_posix(p: Path) -> str:
    return p.as_posix()


def scan_files(root: Path, exts: set, recursive: bool) -> List[Path]:
    if recursive:
        files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    else:
        files = [p for p in root.iterdir() if p.is_file() and p.suffix.lower() in exts]
    files.sort()
    return files


def make_key(root: Path, p: Path) -> str:
    """
    key = 相对目录 + stem（小写），用于稳健配对
    例：fake_256/a/b/xxx.png -> a/b/xxx
    """
    rel = p.relative_to(root)
    rel_parent = rel.parent.as_posix().lower()
    stem = rel.stem.lower()
    return stem if rel_parent == "." else f"{rel_parent}/{stem}"


def infer_mask_dir(src_img_dir: Path) -> Path:
    """默认推断：与 fake_256 同级的 mask_256"""
    parent = src_img_dir.parent
    cand = parent / "mask_256"
    if cand.exists():
        return cand
    # 兜底：把 fake 替换为 mask
    if "fake" in src_img_dir.name.lower():
        cand2 = parent / src_img_dir.name.lower().replace("fake", "mask", 1)
        if cand2.exists():
            return cand2
    return cand


def build_mask_index(mask_root: Path, recursive: bool) -> Dict[str, Path]:
    masks = scan_files(mask_root, MASK_EXTS, recursive)
    idx: Dict[str, Path] = {}
    for m in masks:
        k = make_key(mask_root, m)
        # 同 key 多个 mask：保留第一个（你也可以改成报错）
        if k not in idx:
            idx[k] = m
    return idx


def find_mask(mask_index: Dict[str, Path], key: str) -> Optional[Path]:
    return mask_index.get(key, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_img_dir", type=str, default="/mnt/e/datasets/PSCC/Training Dataset/splice_columbia/fake/",
                    help="PSCC 的 fake_256 目录")
    ap.add_argument("--src_mask_dir", type=str, default="/mnt/e/datasets/PSCC/Training Dataset/splice_columbia/mask_01/",
                    help="PSCC 的 mask 目录（可选，默认自动推断同级 mask_256）")
    ap.add_argument("--dst_img_root", type=str, default="/mnt/e/datasets/PSCC/Training Dataset/splice_columbia/fake/",
                    help="输出 txt 里 Image 根路径")
    ap.add_argument("--dst_mask_root", type=str, default="/mnt/e/datasets/PSCC/Training Dataset/splice_columbia/mask_01/",
                    help="输出 txt 里 Mask 根路径")
    ap.add_argument("--label", type=str, default="hard",
                    help="每行最后的类别标签，同时也作为新文件名前缀")
    ap.add_argument("--out_txt", type=str, default="./test_Columbia.txt",
                    help="输出 txt 路径")
    ap.add_argument("--recursive", action="store_true",
                    help="递归扫描子目录（默认不递归）")
    ap.add_argument("--keep_rel_dir", action="store_true",
                    help="把源相对目录也带到输出路径里，避免重名冲突（默认仅用文件名）")
    args = ap.parse_args()

    src_img_dir = Path(args.src_img_dir)
    if not src_img_dir.exists():
        raise FileNotFoundError(f"src_img_dir not found: {src_img_dir}")

    src_mask_dir = Path(args.src_mask_dir) if args.src_mask_dir else infer_mask_dir(src_img_dir)
    if not src_mask_dir.exists():
        raise FileNotFoundError(
            f"src_mask_dir not found: {src_mask_dir}\n"
            f"你可以用 --src_mask_dir 手动指定正确的 mask 目录"
        )

    dst_img_root = Path(args.dst_img_root)
    dst_mask_root = Path(args.dst_mask_root)

    imgs = scan_files(src_img_dir, IMG_EXTS, args.recursive)
    mask_index = build_mask_index(src_mask_dir, args.recursive)

    out_lines: List[str] = []
    miss: List[str] = []

    for img_path in imgs:
        key = make_key(src_img_dir, img_path)
        m = find_mask(mask_index, key)
        if m is None:
            miss.append(norm_posix(img_path))
            continue

        # ✅ 真实文件名（不带后缀）
        stem = img_path.stem

        if args.keep_rel_dir:
            rel_parent = img_path.relative_to(src_img_dir).parent
            out_img = dst_img_root / rel_parent / (stem + ".jpg")
            out_msk = dst_mask_root / rel_parent / (stem + ".png")
        else:
            out_img = dst_img_root / (stem + ".jpg")
            out_msk = dst_mask_root / (stem + ".png")

        out_lines.append(f"{norm_posix(out_img)} {norm_posix(out_msk)} {args.label}")

    out_txt = Path(args.out_txt)
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_txt.write_text("\n".join(out_lines) + ("\n" if out_lines else ""), encoding="utf-8")

    miss_txt = out_txt.with_suffix(".missing_mask.txt")
    if miss:
        miss_txt.write_text("\n".join(miss) + "\n", encoding="utf-8")

    print("======== DONE ========")
    print(f"src_img_dir : {src_img_dir}")
    print(f"src_mask_dir: {src_mask_dir}")
    print(f"scanned imgs: {len(imgs)}")
    print(f"written pairs: {len(out_lines)} -> {out_txt}")
    print(f"missing mask: {len(miss)}")
    if miss:
        print(f"missing list: {miss_txt}")


if __name__ == "__main__":
    main()
