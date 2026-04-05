"""
balance_dataset.py
==================
查看 metadata.csv 各类别数量，
然后每类随机抽取 min_count 条（即最小类别的数量），
保存为 metadata_balanced.csv 作为新数据集。

用法：
    python src/balance_dataset.py
    python src/balance_dataset.py --input data/metadata_new.csv --output data/metadata_balanced.csv
"""

import argparse
import pandas as pd

# ── 默认路径 ──────────────────────────────────────────────────────────────────
DEFAULT_INPUT  = "data/metadata.csv"
DEFAULT_OUTPUT = "data/metadata_balanced.csv"
RANDOM_STATE   = 42
# ─────────────────────────────────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main():
    args = parse_args()

    df = pd.read_csv(args.input)

    # ── 查看原始分布 ───────────────────────────────────────────────────────
    print("=" * 45)
    print("原始数据集分布：")
    counts = df["label"].value_counts()
    for label, cnt in counts.items():
        print(f"  {label:<25} {cnt} 条")
    print(f"  {'总计':<25} {len(df)} 条")

    # ── 确定目标数量（最小类别） ───────────────────────────────────────────
    min_count = counts.min()
    min_class = counts.idxmin()
    print(f"\n最小类别：{min_class}（{min_count} 条）")
    print(f"每类均抽取：{min_count} 条")

    # ── 每类随机抽取 ───────────────────────────────────────────────────────
    balanced_parts = []
    for label in counts.index:
        subset = df[df["label"] == label].sample(
            n=min_count, random_state=RANDOM_STATE
        )
        balanced_parts.append(subset)

    balanced_df = pd.concat(balanced_parts).sample(
        frac=1, random_state=RANDOM_STATE
    ).reset_index(drop=True)

    # ── 保存 ───────────────────────────────────────────────────────────────
    balanced_df.to_csv(args.output, index=False)

    # ── 验证结果 ───────────────────────────────────────────────────────────
    print("\n均衡后数据集分布：")
    new_counts = balanced_df["label"].value_counts()
    for label, cnt in new_counts.items():
        print(f"  {label:<25} {cnt} 条")
    print(f"  {'总计':<25} {len(balanced_df)} 条")
    print(f"\n已保存至：{args.output}")
    print("=" * 45)


if __name__ == "__main__":
    main()