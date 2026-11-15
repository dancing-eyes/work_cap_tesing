"""Run inference on images or videos and save annotated results.

Examples
    python predict.py --weights d:/hat/code/weights/baseline_yolo11n_best.pt --source d:/hat/dataset/helmet/images/test
    python predict.py --weights ... --source d:/hat/code/outputs/demo/test_clip.mp4
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import tempfile

import cv2
import numpy as np
from ultralytics import YOLO

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv"}


def parse_args():
    parser = argparse.ArgumentParser(description="image / video inference for cap detection")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--source", required=True, help="image, image folder, video file or webcam id")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--out", default="d:/hat/code/outputs")
    parser.add_argument("--name", default="")
    return parser.parse_args()


def imread_unicode(path):
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_unicode(path, image):
    ok, buffer = cv2.imencode(os.path.splitext(path)[1], image)
    if not ok:
        raise OSError(f"cannot encode image for {path}")
    buffer.tofile(path)


def ascii_path(path):
    """OpenCV cannot open non-ASCII paths on Windows; copy to a temp file."""
    try:
        path.encode("ascii")
        return path, None
    except UnicodeEncodeError:
        suffix = os.path.splitext(path)[1]
        handle, tmp = tempfile.mkstemp(suffix=suffix)
        os.close(handle)
        shutil.copy2(path, tmp)
        return tmp, tmp


def collect_detections(result, source_name, frame_index):
    names = result.names
    boxes = result.boxes
    records = []
    if boxes is None or len(boxes) == 0:
        return records
    xyxy = boxes.xyxy.cpu().numpy()
    conf = boxes.conf.cpu().numpy()
    cls = boxes.cls.cpu().numpy().astype(int)
    for (x1, y1, x2, y2), score, label in zip(xyxy, conf, cls):
        records.append(
            {
                "source": source_name,
                "frame": frame_index,
                "class": names[label],
                "confidence": round(float(score), 4),
                "x1": round(float(x1), 1),
                "y1": round(float(y1), 1),
                "x2": round(float(x2), 1),
                "y2": round(float(y2), 1),
            }
        )
    return records


def frame_status(records, names):
    counts = {name: 0 for name in names}
    for record in records:
        counts[record["class"]] += 1
    if counts.get("with_work_cap", 0) and not counts.get("without_work_cap", 0):
        status = "wearing"
    elif counts.get("without_work_cap", 0) and not counts.get("with_work_cap", 0):
        status = "not_wearing"
    elif counts.get("without_work_cap", 0):
        status = "mixed"
    else:
        status = "no_detection"
    return counts, status


def write_csv(path, rows, fieldnames):
    with open(path, "w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_images(model, source, args, out_dir):
    if os.path.isdir(source):
        names = sorted(n for n in os.listdir(source) if os.path.splitext(n)[1].lower() in IMAGE_EXT)
        paths = [os.path.join(source, n) for n in names]
    else:
        paths = [source]
    image_dir = os.path.join(out_dir, "images")
    os.makedirs(image_dir, exist_ok=True)

    all_records, summary = [], []
    for path in paths:
        frame = imread_unicode(path)
        if frame is None:
            print(f"[skip] cannot read {path}")
            continue
        result = model.predict(frame, conf=args.conf, iou=args.iou, imgsz=args.imgsz, device=args.device, verbose=False)[0]
        name = os.path.basename(path)
        records = collect_detections(result, name, 0)
        all_records.extend(records)
        counts, status = frame_status(records, model.names.values())
        summary.append(
            {
                "source": name,
                "with_work_cap": counts.get("with_work_cap", 0),
                "without_work_cap": counts.get("without_work_cap", 0),
                "status": status,
            }
        )
        out_path = os.path.join(image_dir, os.path.splitext(name)[0] + ".jpg")
        imwrite_unicode(out_path, result.plot())
    return all_records, summary, len(paths)


def run_video(model, source, args, out_dir):
    video_dir = os.path.join(out_dir, "video")
    os.makedirs(video_dir, exist_ok=True)
    real_source, tmp_source = ascii_path(source)
    cap = cv2.VideoCapture(real_source)
    if not cap.isOpened():
        raise SystemExit(f"cannot open video {source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out_path = os.path.join(video_dir, os.path.splitext(os.path.basename(source))[0] + "_pred.mp4")
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    all_records, summary = [], []
    index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        result = model.predict(frame, conf=args.conf, iou=args.iou, imgsz=args.imgsz, device=args.device, verbose=False)[0]
        records = collect_detections(result, os.path.basename(source), index)
        all_records.extend(records)
        counts, status = frame_status(records, model.names.values())
        summary.append(
            {
                "frame": index,
                "time_s": round(index / fps, 3),
                "with_work_cap": counts.get("with_work_cap", 0),
                "without_work_cap": counts.get("without_work_cap", 0),
                "status": status,
            }
        )
        writer.write(result.plot())
        index += 1
        if index % 50 == 0:
            print(f"  processed {index}/{total} frames")
    cap.release()
    writer.release()
    if tmp_source:
        os.remove(tmp_source)
    print(f"[video] saved {out_path} ({index} frames, {fps:.1f} fps)")
    return all_records, summary, index


def main():
    args = parse_args()
    model = YOLO(args.weights)
    source = args.source
    stem = os.path.splitext(os.path.basename(source.rstrip("\\/")))[0] or "webcam"
    name = args.name or stem
    out_dir = os.path.join(args.out, name)
    os.makedirs(out_dir, exist_ok=True)

    ext = os.path.splitext(source)[1].lower()
    if os.path.isdir(source) or ext in IMAGE_EXT:
        records, summary, count = run_images(model, source, args, out_dir)
        summary_fields = ["source", "with_work_cap", "without_work_cap", "status"]
        print(f"[images] {count} files -> {os.path.join(out_dir, 'images')}")
    elif ext in VIDEO_EXT:
        records, summary, count = run_video(model, source, args, out_dir)
        summary_fields = ["frame", "time_s", "with_work_cap", "without_work_cap", "status"]
    else:
        raise SystemExit(f"unsupported source: {source}")

    write_csv(
        os.path.join(out_dir, "detections.csv"),
        records,
        ["source", "frame", "class", "confidence", "x1", "y1", "x2", "y2"],
    )
    write_csv(os.path.join(out_dir, "summary.csv"), summary, summary_fields)

    counts = {}
    for record in records:
        counts[record["class"]] = counts.get(record["class"], 0) + 1
    print(f"[done] {count} frames/images, detections by class: {counts}")


if __name__ == "__main__":
    main()
