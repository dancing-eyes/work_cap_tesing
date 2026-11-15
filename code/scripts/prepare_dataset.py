"""Build an independent, clean YOLO dataset for work-cap wearing detection.

Source : d:/hat/data/考核任务数据/安全帽（无test）数据 (read-only)
Target : d:/hat/dataset/helmet

Steps
1. collect every train/valid image + label from the source dataset
2. remap class ids to with_work_cap(0) / without_work_cap(1)
3. merge offline-augmented copies of the same source frame into one cluster
4. split clusters by source-video family and time order (70/15/15),
   so that all copies of one frame always stay in the same split
5. write data.yaml, split_manifest.csv and qc.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
from collections import defaultdict

import numpy as np
from PIL import Image

CLASS_NAMES = ("with_work_cap", "without_work_cap") # 两种类别
FEAT_SIZE = (128, 72) # 缩略图特征尺寸
DUPLICATE_NCC = 0.90 # 判定重复的归一化相关系数阈值，判定把两张图片缩小成灰度图后，是否属于同一张原图的增广副本
BOX_TOLERANCE = 0.02 # 两个yolo框超过百分之二就当做不是同一个框
SPLITS = ("train", "val", "test") # 数据划分
RATIO = (0.70, 0.15, 0.15) # 数据划分的比例
SEED = 42
# 分解出一个文件名里面的family和idx部分
STEM_RE = re.compile(r"^(?P<family>.*_)(?P<idx>\d{5,6})_(?:jpg|png)$", re.IGNORECASE)

# 这个函数就是用来返回label的，(class_id,cx,cy,w,h)类别，框中心的x坐标，框中心的y坐标，框的宽度，框的高度
def read_boxes(path):
    boxes = []
    if not os.path.exists(path):
        return tuple(boxes)
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) >= 5:
                boxes.append((int(parts[0]), *(float(v) for v in parts[1:5])))
    return tuple(boxes)

# 把每张图片变成一个可以互相比较的“指纹”向量

def image_feature(path):
    with Image.open(path) as img:
        # 把图片转成灰度，缩小到128*72（减少因为颜色导致对是否为同一帧的判断，图片缩小忽略细节噪声）
        arr = np.asarray(img.convert("L").resize(FEAT_SIZE, Image.BILINEAR), dtype=np.float32)
    arr -= arr.mean() # 减去均值
    norm = np.linalg.norm(arr) # 除以范数
    return (arr / (norm + 1e-6)).ravel()

# 判断两张图是不是同一原始帧的不同增广副本
def same_source(a, b):
    if len(a["boxes"]) != len(b["boxes"]):
        return False
    for box_a, box_b in zip(a["boxes"], b["boxes"]):
        if box_a[0] != box_b[0]:
            return False
        if any(abs(p - q) > BOX_TOLERANCE for p, q in zip(box_a[1:], box_b[1:])):
            return False
    return float(np.dot(a["feat"], b["feat"])) >= DUPLICATE_NCC # 0.90
# 保证同一帧永远落在同一个划分，不要训练集和验证集同时出现，这样会导致作弊

# 把所有数据打包成records一个列表嵌套字典[来源+stem+解析好的框]
def collect(src):
    records = []
    for split in ("train", "valid"):
        img_dir = os.path.join(src, split, "images")
        lbl_dir = os.path.join(src, split, "labels")
        for name in sorted(os.listdir(img_dir)):
            stem = name.split(".rf.")[0]
            label_path = os.path.join(lbl_dir, os.path.splitext(name)[0] + ".txt")
            records.append(
                {
                    "name": name,
                    "stem": stem,
                    "source_split": "val" if split == "valid" else "train",
                    "path": os.path.join(img_dir, name),
                    "boxes": read_boxes(label_path), # read_boxes把标签结构化成
                }
            )
    return records # 列表嵌套字典


def cluster_copies(records):
    """Group offline-augmented copies that share the same stem, class and box."""
    # 访问一个不存在的key时，会自动创建一个空列表作为默认值
    by_stem = defaultdict(list) # 建一个字典，用按stem分组来存放记录
    for rec in records:
        by_stem[rec["stem"]].append(rec) # 记录对于stem有多少个副本（按stem分组）

    clusters = []
    for stem in sorted(by_stem):
        groups = []
        for rec in by_stem[stem]:
            rec["feat"] = image_feature(rec["path"])
            for group in groups:
                if same_source(group[0], rec):
                    group.append(rec)
                    break
            else:
                groups.append([rec])
        for group in groups:
            rep = max(group, key=lambda r: (r["source_split"] == "train", r["name"]))
            clusters.append(
                {
                    "stem": stem,
                    "rep": rep,
                    "copies": [r["name"] for r in group],
                    "source_split": rep["source_split"],
                }
            )
    return clusters


def family_of(stem):
    match = STEM_RE.match(stem)
    if match:
        return match.group("family"), int(match.group("idx"))
    return stem, None


def split_clusters(clusters, ratio=RATIO, seed=SEED):
    families = defaultdict(list)
    for cluster in clusters:
        cluster["family"], cluster["idx"] = family_of(cluster["stem"])
        families[cluster["family"]].append(cluster)

    for family, members in sorted(families.items()):
        numeric = [c for c in members if c["idx"] is not None]
        if len(numeric) >= 10:
            numeric.sort(key=lambda c: (c["idx"], c["stem"]))
            n = len(numeric)
            n_train = max(1, int(round(ratio[0] * n)))
            n_val = max(1, int(round(ratio[1] * n)))
            for pos, cluster in enumerate(numeric):
                if pos < n_train:
                    cluster["split"] = "train"
                elif pos < n_train + n_val:
                    cluster["split"] = "val"
                else:
                    cluster["split"] = "test"
        for cluster in members:
            if "split" in cluster:
                continue
            digest = hashlib.md5((str(seed) + cluster["stem"]).encode("utf-8")).hexdigest()
            bucket = (int(digest[:8], 16) % 1000) / 1000.0
            if bucket < ratio[0]:
                cluster_split = "train"
            elif bucket < ratio[0] + ratio[1]:
                cluster_split = "val"
            else:
                cluster_split = "test"
            cluster["split"] = cluster_split
    return clusters


def remove_cross_split_leakage(clusters, ncc_limit=0.99, box_limit=0.02, max_rounds=10):
    """Drop frames that still have a near-duplicate twin in another split.

    Temporal block boundaries leave a couple of almost identical frames on both
    sides; removing them makes train/val/test truly disjoint.
    """
    if not clusters:
        return [], []
    feats = np.stack([c["rep"]["feat"] for c in clusters])
    classes = [tuple(sorted({b[0] for b in c["rep"]["boxes"]})) for c in clusters]
    coords = []
    for c in clusters:
        box = c["rep"]["boxes"][0] if c["rep"]["boxes"] else (0, -1.0, -1.0, 0.0, 0.0)
        coords.append(box[1:])
    coords = np.array(coords, dtype=np.float32)
    splits = np.array([c["split"] for c in clusters])

    dropped = []
    for _ in range(max_rounds):
        sim = feats @ feats.T
        delta = np.abs(coords[:, None, :] - coords[None, :, :]).max(axis=2)
        mask = (sim >= ncc_limit) & (delta <= box_limit)
        same = np.zeros_like(mask)
        for i, ci in enumerate(classes):
            if ci:
                same[i] = np.array([ci == cj for cj in classes], dtype=bool)
        cross = mask & same & (splits[:, None] != splits[None, :])
        np.fill_diagonal(cross, False)
        if not cross.any():
            break
        remove = set()
        for i, j in zip(*np.where(np.triu(cross, 1))):
            for idx in (i, j):
                if splits[idx] != "train":
                    remove.add(idx)
        if not remove:
            break
        dropped.extend(clusters[idx]["rep"]["name"] for idx in sorted(remove))
        keep = [c for idx, c in enumerate(clusters) if idx not in remove]
        clusters = keep
        feats = np.stack([c["rep"]["feat"] for c in clusters])
        classes = [tuple(sorted({b[0] for b in c["rep"]["boxes"]})) for c in clusters]
        coords = np.array(
            [c["rep"]["boxes"][0][1:] if c["rep"]["boxes"] else (-1.0, -1.0, 0.0, 0.0) for c in clusters],
            dtype=np.float32,
        )
        splits = np.array([c["split"] for c in clusters])
    return clusters, sorted(set(dropped))


def balance_report(clusters):
    stats = {s: {"images": 0, "with_work_cap": 0, "without_work_cap": 0, "empty": 0} for s in SPLITS}
    for cluster in clusters:
        entry = stats[cluster["split"]]
        entry["images"] += 1
        if not cluster["rep"]["boxes"]:
            entry["empty"] += 1
        for box in cluster["rep"]["boxes"]:
            entry[CLASS_NAMES[box[0]]] += 1
    return stats


def cross_split_similarity(clusters):
    """Report the strongest visual similarity across splits (honesty metric)."""
    feats, labels, splits = [], [], []
    for cluster in clusters:
        feats.append(cluster["rep"]["feat"])
        labels.append(tuple(sorted({b[0] for b in cluster["rep"]["boxes"]})))
        splits.append(cluster["split"])
    feats = np.stack(feats)
    sim = feats @ feats.T
    result = {}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        ia = [i for i, s in enumerate(splits) if s == a]
        ib = [j for j, s in enumerate(splits) if s == b]
        if not ia or not ib:
            continue
        sub = sim[np.ix_(ia, ib)]
        mask_same_class = np.array([[labels[i] == labels[j] for j in ib] for i in ia], dtype=bool)
        if mask_same_class.any():
            values = sub[mask_same_class]
            result[f"{a}-{b}"] = {
                "pairs": int(values.size),
                "median_ncc": round(float(np.median(values)), 4),
                "p99_ncc": round(float(np.percentile(values, 99)), 4),
                "max_ncc": round(float(values.max()), 4),
                "share_ncc_ge_0.99": round(float((values >= 0.99).mean()), 4),
            }
    return result


def write_dataset(clusters, dst):
    for split in SPLITS:
        os.makedirs(os.path.join(dst, "images", split), exist_ok=True)
        os.makedirs(os.path.join(dst, "labels", split), exist_ok=True)

    rows = []
    ordered = sorted(clusters, key=lambda c: (c["split"], c["family"], c["idx"] or 0, c["stem"]))
    for i, cluster in enumerate(ordered):
        rep = cluster["rep"]
        split = cluster["split"]
        new_name = f"c{i:05d}_{rep['name']}"
        shutil.copy2(rep["path"], os.path.join(dst, "images", split, new_name))
        with open(os.path.join(dst, "labels", split, os.path.splitext(new_name)[0] + ".txt"), "w", encoding="utf-8") as fh:
            for box in rep["boxes"]:
                fh.write("%d %.6f %.6f %.6f %.6f\n" % box)
        rows.append(
            {
                "image": f"images/{split}/{new_name}",
                "split": split,
                "family": cluster["family"],
                "frame_index": "" if cluster["idx"] is None else cluster["idx"],
                "class": ",".join(CLASS_NAMES[b[0]] for b in rep["boxes"]),
                "copies_merged": len(cluster["copies"]),
                "original_split": cluster["source_split"],
                "original_image": rep["name"],
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description="build the helmet dataset")
    parser.add_argument(
        "--src",
        default="d:/hat/data/考核任务数据/安全帽（无test）数据",
        help="source dataset directory",
    )
    parser.add_argument("--dst", default="d:/hat/dataset/helmet", help="output dataset directory")
    args = parser.parse_args()

    if os.path.isdir(args.dst):
        shutil.rmtree(args.dst)
    os.makedirs(args.dst, exist_ok=True)

    records = collect(args.src)
    unknown = sorted({b[0] for r in records for b in r["boxes"]} - {0, 1})
    if unknown:
        raise SystemExit(f"unexpected class ids: {unknown}")

    clusters = cluster_copies(records)
    unique_frames = len(clusters)
    clusters, dropped = remove_cross_split_leakage(split_clusters(clusters))
    stats = balance_report(clusters)
    similarity = cross_split_similarity(clusters)
    rows = write_dataset(clusters, args.dst)

    with open(os.path.join(args.dst, "data.yaml"), "w", encoding="utf-8") as fh:
        fh.write("# work-cap wearing-state detection dataset (rebuilt)\n")
        fh.write(f"path: {args.dst}\n")
        fh.write("train: images/train\n")
        fh.write("val: images/val\n")
        fh.write("test: images/test\n\n")
        fh.write("nc: 2\n")
        fh.write("names: ['with_work_cap', 'without_work_cap']\n")

    with open(os.path.join(args.dst, "split_manifest.csv"), "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    qc = {
        "source_images": len(records),
        "unique_source_frames": unique_frames,
        "images_in_dataset": len(clusters),
        "augmented_copies_removed": len(records) - unique_frames,
        "cross_split_duplicates_removed": len(dropped),
        "split_stats": stats,
        "cross_split_similarity": similarity,
        "duplicate_ncc_threshold": DUPLICATE_NCC,
        "box_tolerance": BOX_TOLERANCE,
        "ratio": RATIO,
        "seed": SEED,
        "class_names": list(CLASS_NAMES),
    }
    with open(os.path.join(args.dst, "qc.json"), "w", encoding="utf-8") as fh:
        json.dump(qc, fh, ensure_ascii=False, indent=2)

    print(json.dumps(qc, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
