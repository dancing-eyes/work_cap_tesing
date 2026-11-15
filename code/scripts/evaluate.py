"""Evaluate trained YOLO11 weights on the independent val / test splits.

Example
    python evaluate.py --run baseline=d:/hat/code/weights/baseline_yolo11n_best.pt
    python evaluate.py --run baseline=... --run strong=...
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import time

import numpy as np
from ultralytics import YOLO

THRESHOLDS = (0.25, 0.4, 0.5, 0.6)


def parse_args():
    parser = argparse.ArgumentParser(description="evaluate YOLO weights on val/test")
    parser.add_argument("--data", default="d:/hat/dataset/helmet/data.yaml")
    parser.add_argument("--run", action="append", required=True, help="name=weights_path")
    parser.add_argument("--splits", nargs="+", default=["val", "test"])
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--project", default="d:/hat/code/runs/eval")
    parser.add_argument("--out", default="d:/hat/code/report")
    return parser.parse_args()


def read_split_size(data_yaml, split):
    """Number of images in a split, read from data.yaml + image folder."""
    import yaml

    with open(data_yaml, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    root = cfg.get("path", os.path.dirname(data_yaml))
    entry = cfg.get(split)
    if not entry:
        return ""
    folder = entry if os.path.isabs(entry) else os.path.join(root, entry)
    if not os.path.isdir(folder):
        return ""
    return sum(1 for n in os.listdir(folder) if os.path.splitext(n)[1].lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"})


def curve_at(px, curve, conf):
    if px is None or curve is None:
        return None
    px = np.asarray(px).ravel()
    curve = np.asarray(curve)
    idx = int(np.argmin(np.abs(px - conf)))
    values = curve[:, idx] if curve.ndim > 1 else curve[idx]
    return float(np.mean(values))


def evaluate_run(name, weights, args):
    rows, threshold_rows = [], []
    speeds = {}
    for split in args.splits:
        model = YOLO(weights)
        started = time.time()
        result = model.val(
            data=args.data,
            split=split,
            imgsz=args.imgsz,
            batch=args.batch,
            workers=args.workers,
            conf=0.001,
            iou=0.7,
            plots=True,
            project=args.project,
            name=f"{name}_{split}",
            exist_ok=True,
            verbose=False,
        )
        elapsed = time.time() - started
        box = result.box
        names = result.names
        n_images = read_split_size(args.data, split)
        per_class_counts = np.asarray(result.nt_per_class).ravel()
        for idx in range(len(names)):
            p, r, ap50, ap = box.class_result(idx)
            rows.append(
                {
                    "run": name,
                    "split": split,
                    "class": names[idx],
                    "images": n_images,
                    "instances": int(per_class_counts[idx]) if idx < len(per_class_counts) else "",
                    "precision": round(float(p), 4),
                    "recall": round(float(r), 4),
                    "mAP50": round(float(ap50), 4),
                    "mAP50-95": round(float(ap), 4),
                }
            )
        rows.append(
            {
                "run": name,
                "split": split,
                "class": "all",
                "images": n_images,
                "instances": int(per_class_counts.sum()) if len(per_class_counts) else "",
                "precision": round(float(box.mp), 4),
                "recall": round(float(box.mr), 4),
                "mAP50": round(float(box.map50), 4),
                "mAP50-95": round(float(box.map), 4),
            }
        )
        speed = result.speed or {}
        speeds[split] = {
            "preprocess_ms": round(float(speed.get("preprocess", 0.0)), 2),
            "inference_ms": round(float(speed.get("inference", 0.0)), 2),
            "postprocess_ms": round(float(speed.get("postprocess", 0.0)), 2),
            "fps": round(1000.0 / max(float(speed.get("inference", 1.0)), 1e-6), 1),
            "val_seconds": round(elapsed, 1),
        }
        for conf in THRESHOLDS:
            threshold_rows.append(
                {
                    "run": name,
                    "split": split,
                    "conf": conf,
                    "precision": round(curve_at(result.box.px, result.box.p_curve, conf), 4),
                    "recall": round(curve_at(result.box.px, result.box.r_curve, conf), 4),
                    "f1": round(curve_at(result.box.px, result.box.f1_curve, conf), 4),
                }
            )

        asset_dir = os.path.join(args.out, "assets", name, split)
        os.makedirs(asset_dir, exist_ok=True)
        for fname in (
            "confusion_matrix.png",
            "confusion_matrix_normalized.png",
            "BoxPR_curve.png",
            "BoxF1_curve.png",
            "BoxP_curve.png",
            "BoxR_curve.png",
            "val_batch0_pred.jpg",
            "val_batch0_labels.jpg",
        ):
            src = os.path.join(result.save_dir, fname)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(asset_dir, fname))
    return rows, threshold_rows, speeds


def write_outputs(all_rows, all_thresholds, all_speeds, args):
    os.makedirs(args.out, exist_ok=True)
    csv_path = os.path.join(args.out, "metrics.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    with open(os.path.join(args.out, "threshold_sweep.csv"), "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(all_thresholds[0].keys()))
        writer.writeheader()
        writer.writerows(all_thresholds)

    with open(os.path.join(args.out, "metrics.json"), "w", encoding="utf-8") as fh:
        json.dump({"metrics": all_rows, "thresholds": all_thresholds, "speeds": all_speeds}, fh, ensure_ascii=False, indent=2)

    lines = ["| run | split | class | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |", "|---|---|---|---|---|---|---|"]
    for row in all_rows:
        lines.append(
            "| {run} | {split} | {class} | {precision} | {recall} | {mAP50} | {mAP50-95} |".format(**row)
        )
    table = "\n".join(lines)
    with open(os.path.join(args.out, "metrics_table.md"), "w", encoding="utf-8") as fh:
        fh.write(table + "\n")
    print(table)
    for name, speed in all_speeds.items():
        print(name, speed)


def main():
    args = parse_args()
    runs = []
    for item in args.run:
        name, _, path = item.partition("=")
        if not path:
            raise SystemExit("--run must look like name=weights_path")
        runs.append((name, path))

    all_rows, all_thresholds, all_speeds = [], [], {}
    for name, path in runs:
        rows, thresholds, speeds = evaluate_run(name, path, args)
        all_rows.extend(rows)
        all_thresholds.extend(thresholds)
        all_speeds[name] = speeds
    write_outputs(all_rows, all_thresholds, all_speeds, args)


if __name__ == "__main__":
    main()
