import argparse
import random

def shuffle_txt(input_path, output_path=None, seed=42):
    # 1. 读入所有行
    with open(input_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # 2. 打乱顺序（可复现）
    random.seed(seed)
    random.shuffle(lines)

    # 3. 写回文件（默认覆盖原文件）
    if output_path is None:
        output_path = input_path

    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(lines)

    print(f"Shuffled lines written to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Shuffle lines in a txt file.")
    parser.add_argument("input", help="Path to input txt file, e.g. val.txt")
    parser.add_argument(
        "-o", "--output",
        help="Path to output txt file (default: overwrite input)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    args = parser.parse_args()

    shuffle_txt(args.input, args.output, args.seed)
