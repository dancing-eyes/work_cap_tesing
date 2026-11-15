"""Assemble the comparison table (baseline vs strong augmentation) for the report.

Reads code/runs/<run>/config.json and report/metrics.json, writes
report/compare_table.md and report/compare_table.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
import os


def parse_args():
    parser = argparse.ArgumentParser(description="build comparison table")
    parser.add_argument("--report", default="d:/hat/code/report")
    parser.add_argument("--runs-dir", default="d:/hat/code/runs")
    parser.add_argument("--runs", nargs="+", default=["baseline=baseline_yolo11n", "strongaug=strongaug_yolo11n"],
                        help="entries as metrics_name=run_directory")
    parser.add_argument("--primary-split", default="test")
    return parser.parse_args()


def load_metrics(report_dir):
    with open(os.path.join(report_dir, "metrics.json"), "r", encoding="utf-8") as fh:
        return json.load(fh)


def main():
    args = parse_args()
    data = load_metrics(args.report)
    rows = []
    for item in args.runs:
        metrics_name, _, run_dir = item.partition("=")
        if not run_dir:
            run_dir = metrics_name
        config_path = os.path.join(args.runs_dir, run_dir, "config.json")
        if not os.path.exists(config_path):
            print(f"[warn] missing config for {run_dir}")
            continue
        with open(config_path, "r", encoding="utf-8") as fh:
            config = json.load(fh)
        for split in ("val", "test"):
            metrics = [
                m for m in data["metrics"] if m["run"] == metrics_name and m["split"] == split and m["class"] == "all"
            ]
            if not metrics:
                continue
            m = metrics[0]
            speed = data["speeds"].get(metrics_name, {}).get(split, {})
            rows.append(
                {
                    "run": metrics_name,
                    "augmentation": config["augmentation"],
                    "model": config["model"],
                    "epochs": config["epochs"],
                    "imgsz": config["imgsz"],
                    "batch": config["batch"],
                    "split": split,
                    "precision": m["precision"],
                    "recall": m["recall"],
                    "mAP50": m["mAP50"],
                    "mAP50-95": m["mAP50-95"],
                    "params": config["parameters"],
                    "gflops": config["gflops"],
                    "train_min": round(config["train_seconds"] / 60.0, 1),
                    "inference_ms": speed.get("inference_ms", ""),
                    "fps": speed.get("fps", ""),
                }
            )

    if not rows:
        raise SystemExit("no metrics matched the requested runs")

    with open(os.path.join(args.report, "compare_table.csv"), "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    header = (
        "| 模型 | 增强 | split | P | R | mAP@0.5 | mAP@0.5:0.95 | 参数量 | GFLOPs | 训练耗时(min) | 推理(ms/图) | FPS |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|---|"
    )
    lines = [header]
    for row in rows:
        lines.append(
            "| {model} | {augmentation} | {split} | {precision} | {recall} | {mAP50} | {mAP50-95} | "
            "{params} | {gflops} | {train_min} | {inference_ms} | {fps} |".format(**row)
        )
    table = "\n".join(lines)
    with open(os.path.join(args.report, "compare_table.md"), "w", encoding="utf-8") as fh:
        fh.write(table + "\n")
    print(table)


if __name__ == "__main__":
    main()
