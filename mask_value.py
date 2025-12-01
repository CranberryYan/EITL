# check_mask_range_01_0255.py
import os
import argparse
from concurrent.futures import ThreadPoolExecutor
import numpy as np

try:
    import cv2
except ImportError:
    raise SystemExit("缺少 opencv-python: 请先 pip install opencv-python")

IMG_EXTS_DEFAULT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
NP_EXTS_DEFAULT = {".npy"}

def iter_files(root: str, recursive: bool, exts: set[str]):
    root = os.path.abspath(root)
    if recursive:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if not d.startswith(".")]
            for fn in filenames:
                if fn.startswith("."):
                    continue
                ext = os.path.splitext(fn)[1].lower()
                if ext in exts:
                    yield os.path.join(dirpath, fn)
    else:
        with os.scandir(root) as it:
            for e in it:
                if not e.is_file() or e.name.startswith("."):
                    continue
                ext = os.path.splitext(e.name)[1].lower()
                if ext in exts:
                    yield e.path

def _read_mask(path: str):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        arr = np.load(path, mmap_mode="r")
        if arr.ndim == 3:
            arr = arr[..., 0]
        return arr

    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 3:
        img = img[..., 0]
    return img

def _classify_mask(path: str, eps: float):
    arr = _read_mask(path)
    if arr is None:
        return ("bad", path, None, None, None)

    a = np.asarray(arr)
    dt = a.dtype

    # 统计 min/max(便于直观看到是不是 1 vs 255)
    mn = a.min()
    mx = a.max()

    if np.issubdtype(dt, np.floating):
        # float: 允许微小误差
        is01 = np.logical_or(np.isclose(a, 0.0, atol=eps), np.isclose(a, 1.0, atol=eps)).all()
        is0255 = np.logical_or(np.isclose(a, 0.0, atol=eps), np.isclose(a, 255.0, atol=eps)).all()
    else:
        # int/bool: 严格等于
        is01 = np.logical_or(a == 0, a == 1).all()
        is0255 = np.logical_or(a == 0, a == 255).all()

    if is01 and not is0255:
        cls = "01"
    elif is0255 and not is01:
        cls = "0255"
    elif is01 and is0255:
        # 只有全 0 时两者都成立(因为全 0 同时属于 {0,1} 和 {0,255})
        cls = "all_zero"
    else:
        cls = "other"

    return (cls, path, str(dt), float(mn), float(mx))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/mnt/e/jwj/new_datasets/Mask", help="mask 文件夹路径")
    ap.add_argument("--recursive", action="store_true", help="递归遍历子目录")
    ap.add_argument("--workers", type=int, default=0, help="线程数(0表示自动)")
    ap.add_argument("--exts", nargs="*", default=None,
                    help="只处理这些后缀，如: .png .npy(默认常见图像+.npy)")
    ap.add_argument("--eps", type=float, default=1e-6, help="float 判定容差(默认1e-6)")
    ap.add_argument("--show_examples", type=int, default=10, help="每类展示前N个样例路径")
    ap.add_argument("--csv", default="", help="可选: 导出 CSV 路径")
    args = ap.parse_args()

    if args.exts is None:
        exts = set(IMG_EXTS_DEFAULT) | set(NP_EXTS_DEFAULT)
    else:
        exts = {e.lower() if e.startswith(".") else "." + e.lower() for e in args.exts}

    paths = list(iter_files(args.root, args.recursive, exts))
    total = len(paths)
    if total == 0:
        print("未找到匹配文件。请检查 --root / --recursive / --exts")
        return

    workers = args.workers if args.workers > 0 else min(32, (os.cpu_count() or 8) * 2)

    counts = {"01": 0, "0255": 0, "all_zero": 0, "other": 0, "bad": 0}
    examples = {k: [] for k in counts.keys()}
    dtype_counter = {}
    rows = []

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for cls, path, dt, mn, mx in ex.map(lambda p: _classify_mask(p, args.eps), paths, chunksize=64):
            if cls == "bad":
                counts["bad"] += 1
                if len(examples["bad"]) < args.show_examples:
                    examples["bad"].append(path)
                continue

            counts[cls] += 1
            dtype_counter[dt] = dtype_counter.get(dt, 0) + 1

            if len(examples[cls]) < args.show_examples:
                examples[cls].append(path)

            if args.csv:
                rows.append((path, cls, dt, mn, mx))

    print("========== Mask 分布检查 ==========")
    print(f"Root: {os.path.abspath(args.root)}")
    print(f"Files matched: {total}")
    print(f"Read failed:   {counts['bad']}")
    print(f"Dtypes:        {dtype_counter}")
    print("----------------------------------")
    print(f"01 masks:      {counts['01']}")
    print(f"0255 masks:    {counts['0255']}")
    print(f"all-zero:      {counts['all_zero']}  (全0会同时满足01与0255)")
    print(f"other:         {counts['other']}  (存在除0/1/255之外的值)")
    print("----------------------------------")

    def dump_examples(tag: str):
        if examples[tag]:
            print(f"[{tag}] examples (up to {args.show_examples}):")
            for p in examples[tag]:
                print("  ", p)

    dump_examples("01")
    dump_examples("0255")
    dump_examples("all_zero")
    dump_examples("other")
    dump_examples("bad")

    if args.csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.csv)) or ".", exist_ok=True)
        import csv
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["path", "class", "dtype", "min", "max"])
            w.writerows(rows)
        print("----------------------------------")
        print(f"CSV saved to: {os.path.abspath(args.csv)}")

if __name__ == "__main__":
    main()
