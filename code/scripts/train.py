"""Train a YOLO11 detector for work-cap wearing state detection.

Examples
    python train.py --name baseline_yolo11n
    python train.py --name strongaug_yolo11n --aug strong --epochs 100
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import time

import torch
import ultralytics
from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_flops

STRONG_AUG = {
    "hsv_h": 0.03,
    "hsv_s": 0.9,
    "hsv_v": 0.5,
    "degrees": 10.0,
    "translate": 0.2,
    "scale": 0.7,
    "shear": 2.0,
    "perspective": 0.0005,
    "fliplr": 0.5,
    "flipud": 0.0,
    "mosaic": 1.0,
    "mixup": 0.15,
    "erasing": 0.5,
}


def parse_args():
    parser = argparse.ArgumentParser(description="train YOLO11 for cap detection")
    parser.add_argument("--data", default="d:/hat/dataset/helmet/data.yaml")
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--name", required=True, help="run name under code/runs")
    parser.add_argument("--aug", choices=("default", "strong"), default="default")
    parser.add_argument("--project", default="d:/hat/code/runs")
    parser.add_argument("--weights-dir", default="d:/hat/code/weights")
    return parser.parse_args()


def main():
    args = parse_args()
    extra = STRONG_AUG if args.aug == "strong" else {}
    started = time.time()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        seed=args.seed,
        deterministic=True,
        patience=args.patience,
        project=args.project,
        name=args.name,
        exist_ok=False,
        pretrained=True,
        optimizer="auto",
        plots=True,
        val=True,
        **extra,
    )
    elapsed = round(time.time() - started, 1)

    run_dir = os.path.join(args.project, args.name)
    best = os.path.join(run_dir, "weights", "best.pt")
    os.makedirs(args.weights_dir, exist_ok=True)
    final_weight = os.path.join(args.weights_dir, f"{args.name}_best.pt")
    shutil.copy2(best, final_weight)

    trainer = model.trainer
    metrics = {
        "precision": round(float(trainer.metrics.get("metrics/precision(B)", 0.0)), 4),
        "recall": round(float(trainer.metrics.get("metrics/recall(B)", 0.0)), 4),
        "mAP50": round(float(trainer.metrics.get("metrics/mAP50(B)", 0.0)), 4),
        "mAP50-95": round(float(trainer.metrics.get("metrics/mAP50-95(B)", 0.0)), 4),
    }
    config = {
        "run": args.name,
        "weights": final_weight,
        "model": args.model,
        "data": args.data,
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "workers": args.workers,
        "device": args.device,
        "seed": args.seed,
        "patience": args.patience,
        "augmentation": args.aug,
        "augmentation_params": extra or "ultralytics defaults",
        "optimizer": "auto",
        "lr0": trainer.args.lr0,
        "lrf": trainer.args.lrf,
        "momentum": trainer.args.momentum,
        "weight_decay": trainer.args.weight_decay,
        "warmup_epochs": trainer.args.warmup_epochs,
        "pretrained": args.model,
        "parameters": sum(p.numel() for p in model.model.parameters()),
        "gflops": round(float(get_flops(model.model, imgsz=args.imgsz)), 2),
        "train_seconds": elapsed,
        "epochs_completed": int(trainer.epoch) + 1,
        "val_metrics": metrics,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as fh:
        json.dump(config, fh, ensure_ascii=False, indent=2)
    print(json.dumps(config, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
