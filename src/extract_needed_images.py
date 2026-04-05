"""
extract_needed_images.py
========================
从已解压的 part_113 ~ part_116 中，只提取 metadata.csv 里需要的图片，
复制到 data/images/ 对应路径下。

使用方法
--------
把四个 part 解压到任意位置（SOURCE_ROOT），然后运行：

    python src/extract_needed_images.py

也可以通过命令行参数指定路径：

    python src/extract_needed_images.py \
        --source D:/qilin_raw \
        --dest   E:/xiaohongshu-multimodal-ablation/data/images \
        --meta   E:/xiaohongshu-multimodal-ablation/data/metadata.csv

目录结构假设（SOURCE_ROOT 下）：
    SOURCE_ROOT/
        part_113/
            1119356/
                5012641.jpg
        part_114/
        part_115/
        part_116/

输出结构（DEST_ROOT 下，与 metadata.csv image_path 对齐）：
    DEST_ROOT/
        part_113/
            1119356/
                5012641.jpg
        ...
"""

import os
import shutil
import argparse
import pandas as pd
from tqdm import tqdm


# ── 默认路径（按需修改） ────────────────────────────────────────────────────
# SOURCE_ROOT: 你解压四个 tar.gz 的目录（里面有 part_113 ~ part_116 文件夹）
DEFAULT_SOURCE = r"E:\xiaohongshu-multimodal-ablation\data\images_raw"

# DEST_ROOT: 项目的 data/images 目录
DEFAULT_DEST   = r"E:\xiaohongshu-multimodal-ablation\data\images"

# METADATA: 项目的 metadata.csv
DEFAULT_META   = r"E:\xiaohongshu-multimodal-ablation\data\metadata_balanced.csv"
# ────────────────────────────────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(description="Extract needed images from Qilin parts")
    parser.add_argument("--source", default=DEFAULT_SOURCE,
                        help="Root directory containing part_113 ~ part_116")
    parser.add_argument("--dest",   default=DEFAULT_DEST,
                        help="Destination: project data/images directory")
    parser.add_argument("--meta",   default=DEFAULT_META,
                        help="Path to metadata.csv")
    return parser.parse_args()


def resolve_paths(image_path_csv: str, source_root: str, dest_root: str):
    """
    metadata.csv 里的 image_path 格式：  image/part_113/1119356/5012641.jpg
    去掉开头的 'image/' 得到相对路径：   part_113/1119356/5012641.jpg
    source 绝对路径：  SOURCE_ROOT/part_113/1119356/5012641.jpg
    dest   绝对路径：  DEST_ROOT/part_113/1119356/5012641.jpg
    """
    rel       = image_path_csv.replace("image/", "", 1)   # part_113/...
    src_path  = os.path.join(source_root, rel)
    dest_path = os.path.join(dest_root,   rel)
    return src_path, dest_path


def main():
    args = parse_args()

    source_root = args.source
    dest_root   = args.dest
    meta_path   = args.meta

    print(f"Source : {source_root}")
    print(f"Dest   : {dest_root}")
    print(f"Meta   : {meta_path}")

    # ── 读取 metadata ───────────────────────────────────────────────────────
    df = pd.read_csv(meta_path)
    print(f"\nTotal samples in metadata.csv : {len(df)}")

    # ── 去重（同一图片可能被多条记录引用） ───────────────────────────────────
    unique_paths = df["image_path"].dropna().unique()
    print(f"Unique image paths needed     : {len(unique_paths)}")

    # ── 复制 ────────────────────────────────────────────────────────────────
    found     = 0
    not_found = []

    for img_csv_path in tqdm(unique_paths, desc="Copying images"):
        src, dst = resolve_paths(img_csv_path, source_root, dest_root)

        if not os.path.exists(src):
            not_found.append(src)
            continue

        os.makedirs(os.path.dirname(dst), exist_ok=True)

        # 如果目标已存在则跳过（支持断点续传）
        if os.path.exists(dst):
            found += 1
            continue

        shutil.copy2(src, dst)
        found += 1

    # ── 报告 ────────────────────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"Copied / already exists : {found} / {len(unique_paths)}")
    print(f"Not found in source     : {len(not_found)}")

    if not_found:
        log_path = os.path.join(os.path.dirname(meta_path), "missing_images.txt")
        with open(log_path, "w") as f:
            f.write("\n".join(not_found))
        print(f"Missing paths saved to  : {log_path}")
        print("\nPossible reasons:")
        print("  1. The tar.gz for that part is not yet extracted")
        print("  2. SOURCE_ROOT is set to the wrong directory")
        print("  3. The image simply does not exist in the Qilin dataset")
    else:
        print("All images found successfully!")

    print("Done.")


if __name__ == "__main__":
    main()