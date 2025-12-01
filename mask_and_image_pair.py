# check_pairs.py
import os
from pathlib import Path
from collections import defaultdict

IMG_EXTS  = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
MASK_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

IMAGE_DIR = Path(r"/mnt/e/jwj/new_datasets/Image")
MASK_DIR  = Path(r"/mnt/e/jwj/new_datasets/Mask")

REPORT_DIR = Path.cwd()
REPORT_TXT = REPORT_DIR / "pair_report.txt"


def make_key(base: Path, file_path: Path):
    """
    key = (相对父目录, stem)，都转小写，windows 下更稳
    """
    rel_parent = file_path.parent.relative_to(base).as_posix().lower()
    stem = file_path.stem.lower()
    return (rel_parent, stem)


def scan_dir(base: Path, exts: set):
    """
    返回：key -> [paths...]（用 list 是为了检查重名冲突）
    """
    mp = defaultdict(list)
    for p in base.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in exts:
            continue
        k = make_key(base, p)
        mp[k].append(p)
    return mp


def format_list(title, items, limit=50):
    lines = [f"\n==== {title} (count={len(items)}) ===="]
    for i, x in enumerate(items[:limit]):
        lines.append(str(x))
    if len(items) > limit:
        lines.append(f"... ({len(items)-limit} more)")
    return "\n".join(lines)


def main():
    assert IMAGE_DIR.exists(), f"Not found: {IMAGE_DIR}"
    assert MASK_DIR.exists(), f"Not found: {MASK_DIR}"

    img_map  = scan_dir(IMAGE_DIR, IMG_EXTS)
    mask_map = scan_dir(MASK_DIR, MASK_EXTS)

    img_keys  = set(img_map.keys())
    mask_keys = set(mask_map.keys())

    # 1) 缺失对应
    img_missing_mask_keys = sorted(list(img_keys - mask_keys))
    mask_missing_img_keys = sorted(list(mask_keys - img_keys))

    # 2) 重名冲突（同一个 key 找到多文件）
    img_dup  = sorted([k for k, v in img_map.items() if len(v) > 1])
    mask_dup = sorted([k for k, v in mask_map.items() if len(v) > 1])

    # 展开成路径列表（便于阅读）
    img_missing_mask_paths = []
    for k in img_missing_mask_keys:
        img_missing_mask_paths.extend(img_map[k])

    mask_missing_img_paths = []
    for k in mask_missing_img_keys:
        mask_missing_img_paths.extend(mask_map[k])

    img_dup_paths = []
    for k in img_dup:
        img_dup_paths.append(f"[KEY={k}]")
        img_dup_paths.extend([str(p) for p in img_map[k]])

    mask_dup_paths = []
    for k in mask_dup:
        mask_dup_paths.append(f"[KEY={k}]")
        mask_dup_paths.extend([str(p) for p in mask_map[k]])

    # 3) 输出汇总
    summary = []
    summary.append("======== Pair Check Report ========")
    summary.append(f"IMAGE_DIR: {IMAGE_DIR}")
    summary.append(f"MASK_DIR : {MASK_DIR}")
    summary.append("")
    summary.append(f"Total images (keys): {len(img_keys)}")
    summary.append(f"Total masks  (keys): {len(mask_keys)}")
    summary.append(f"Images without mask : {len(img_missing_mask_keys)}")
    summary.append(f"Masks without image : {len(mask_missing_img_keys)}")
    summary.append(f"Image duplicate keys: {len(img_dup)}")
    summary.append(f"Mask  duplicate keys: {len(mask_dup)}")

    report = "\n".join(summary)
    report += "\n" + format_list("Images WITHOUT mask (showing image paths)", img_missing_mask_paths, limit=200)
    report += "\n" + format_list("Masks WITHOUT image (showing mask paths)", mask_missing_img_paths, limit=200)
    report += "\n" + format_list("Image DUPLICATE keys (key + paths)", img_dup_paths, limit=400)
    report += "\n" + format_list("Mask  DUPLICATE keys (key + paths)", mask_dup_paths, limit=400)

    REPORT_TXT.write_text(report, encoding="utf-8")
    print(report)
    print(f"\n[OK] Report saved to: {REPORT_TXT.resolve()}")


if __name__ == "__main__":
    main()
