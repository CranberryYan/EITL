#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import random


def read_lines(path: str):
    with open(path, "r", encoding="utf-8") as f:
        # 过滤空行，保留原始行内容（不 strip 末尾空格以外的信息）
        lines = [line.rstrip("\n") for line in f if line.strip()]
    return lines


def write_lines(path: str, lines):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


def main():
    parser = argparse.ArgumentParser(description="Randomly sample lines from train.txt to create val.txt")
    parser.add_argument("--train", type=str, default="/mnt/e/jwj/datasets_with_C1_C2_Cov_Colu/train.txt", help="path to train.txt")
    parser.add_argument("--val", type=str, default="/mnt/e/jwj/datasets_with_C1_C2_Cov_Colu/val.txt", help="path to val.txt")
    # group = parser.add_mutually_exclusive_group(required=True)
    parser.add_argument("--ratio", type=float, default=0.1, help="val ratio in (0,1), e.g., 0.1")
    parser.add_argument("--num", type=int, default=1500, help="number of val samples, e.g., 1000")
    parser.add_argument("--seed", type=int, default=2026, help="random seed for reproducibility")
    parser.add_argument("--remove_from_train", action="store_true",
                        help="if set, sampled lines are removed from train and train.txt is overwritten")
    parser.add_argument("--train_out", type=str, default="/mnt/e/jwj/new_datasets/train_remain.txt",
                        help="optional output path for remaining train lines (if not set and --remove_from_train, overwrite --train)")
    args = parser.parse_args()

    lines = read_lines(args.train)
    if not lines:
        raise RuntimeError(f"Empty train file: {args.train}")

    n_total = len(lines)

    if args.ratio is not None:
        if not (0.0 < args.ratio < 1.0):
            raise ValueError("--ratio must be in (0,1)")
        n_val = int(round(n_total * args.ratio))
    else:
        n_val = args.num
        if n_val <= 0:
            raise ValueError("--num must be > 0")

    if n_val >= n_total:
        raise ValueError(f"val size must be smaller than train size: val={n_val}, total={n_total}")

    random.seed(args.seed)
    idx = list(range(n_total))
    random.shuffle(idx)

    val_idx = set(idx[:n_val])
    val_lines = [lines[i] for i in range(n_total) if i in val_idx]
    train_remain = [lines[i] for i in range(n_total) if i not in val_idx]

    # 写 val
    write_lines(args.val, val_lines)
    print(f"[OK] val saved: {args.val}  (n={len(val_lines)})")

    # 是否处理 train
    if args.remove_from_train:
        out_train = args.train_out.strip() if args.train_out.strip() else args.train
        write_lines(out_train, train_remain)
        if out_train == args.train:
            print(f"[OK] train overwritten (removed val): {out_train}  (n={len(train_remain)})")
        else:
            print(f"[OK] remaining train saved: {out_train}  (n={len(train_remain)})")
    else:
        print(f"[INFO] train not changed. Remaining train would be n={len(train_remain)}")
        if args.train_out.strip():
            write_lines(args.train_out, train_remain)
            print(f"[OK] remaining train also saved: {args.train_out}  (n={len(train_remain)})")

    print("[DONE]")


if __name__ == "__main__":
    main()
