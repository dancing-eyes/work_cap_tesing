"""Build a short demo clip from the independent test split (no real video available).

Example
    python make_demo_video.py            # -> d:/hat/code/outputs/demo/test_clip.mp4
"""

from __future__ import annotations

import argparse
import csv
import os

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="compose a demo video from test images")
    parser.add_argument("--dataset", default="d:/hat/dataset/helmet")
    parser.add_argument("--split", default="test")
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--out", default="d:/hat/code/outputs/demo/test_clip.mp4")
    parser.add_argument("--max-frames", type=int, default=160)
    return parser.parse_args()


def imread_unicode(path):
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)


def main():
    args = parse_args()
    manifest = os.path.join(args.dataset, "split_manifest.csv")
    rows = []
    with open(manifest, "r", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["split"] != args.split:
                continue
            rows.append(row)

    def sort_key(row):
        index = row["frame_index"]
        return (row["family"], int(index) if index else 0, row["image"])

    rows.sort(key=sort_key)
    rows = rows[: args.max_frames]
    if not rows:
        raise SystemExit(f"no images for split={args.split}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    first = imread_unicode(os.path.join(args.dataset, rows[0]["image"]))
    height, width = first.shape[:2]
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (width, height))

    written = 0
    for row in rows:
        frame = imread_unicode(os.path.join(args.dataset, row["image"]))
        if frame is None:
            continue
        frame = cv2.resize(frame, (width, height))
        writer.write(frame)
        written += 1
    writer.release()
    print(f"[demo] {written} frames -> {args.out} ({written / args.fps:.1f}s @ {args.fps} fps)")


if __name__ == "__main__":
    main()
