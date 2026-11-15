"""Verify the four claimed hard flaws in dataset/helmet/split_manifest.csv.

Reads only the manifest + qc.json (no image decoding needed for the structural claims).
"""
import csv
import json
from collections import defaultdict, Counter

MANIFEST = "d:/hat/dataset/helmet/split_manifest.csv"
QC = "d:/hat/dataset/helmet/qc.json"

rows = []
with open(MANIFEST, newline="", encoding="utf-8") as fh:
    for r in csv.DictReader(fh):
        rows.append(r)

print(f"total rows (images in dataset): {len(rows)}")

# Source frame identity = (family, frame_index) as written by the script.
# frame_index may be empty string for non-matching stems.
frame_key = lambda r: (r["family"], r["frame_index"])

by_frame = defaultdict(list)
for r in rows:
    by_frame[frame_key(r)].append(r)

n_frames = len(by_frame)
print(f"unique (family, frame_index) source frames: {n_frames}")
print(f"redundancy: {len(rows)/n_frames:.3f} copies/frame")

# --- Claim 1: same frame carrying BOTH classes ---
def classes_of(r):
    # class column is comma-joined class names of the rep's boxes
    return frozenset(c for c in r["class"].split(",") if c)

conflict = []   # (frame, set_of_images, classes)
for key, recs in by_frame.items():
    allc = set()
    for r in recs:
        allc |= classes_of(r)
    if len(allc) > 1:
        conflict.append((key, recs, allc))

print("\n=== CLAIM 1: same frame with two opposite labels ===")
print(f"frames with BOTH with_work_cap AND without_work_cap: {len(conflict)} "
      f"({len(conflict)/n_frames*100:.1f}% of {n_frames} frames)")

# train subset
train_conflict = [c for c in conflict if any(r["split"] == "train" for r in c[1])]
train_frames = set()
for r in rows:
    if r["split"] == "train":
        train_frames.add(frame_key(r))
print(f"train frames: {len(train_frames)}, train frames in conflict: {len(train_conflict)} "
      f"({len(train_conflict)/len(train_frames)*100:.1f}%)")

# show examples
print("\nexample conflict frames:")
for key, recs, allc in conflict[:5]:
    print(f"  frame {key}: classes={sorted(allc)}, copies:")
    for r in recs:
        print(f"    split={r['split']:5s} class={r['class']:15s} copies_merged={r['copies_merged']:3s} "
              f"orig={r['original_image']}")

# --- Claim 1 quantitative: drop conflict frames -> how many remain ---
# user claims: with 73 / without 170 / empty 31 -> 274
def label_of_frame(recs):
    # "visible hat => with" heuristic is not applied here; just tally per-frame unique class
    cs = set()
    empty = True
    for r in recs:
        c = classes_of(r)
        if c:
            empty = False
            cs |= c
    if empty:
        return "empty"
    if cs == {"with_work_cap", "without_work_cap"}:
        return "conflict"
    if cs == {"with_work_cap"}:
        return "with"
    if cs == {"without_work_cap"}:
        return "without"
    return f"other:{cs}"

label_tally = Counter(label_of_frame(recs) for recs in by_frame.values())
print(f"\nframe-level label tally (after treating conflict as its own bucket): {dict(label_tally)}")
unambiguous = {k: v for k, v in label_tally.items() if k in ("with", "without", "empty")}
print(f"unambiguous frames (with/without/empty): {sum(unambiguous.values())} = {unambiguous}")

# --- Claim 2: same frame crossing splits ---
print("\n=== CLAIM 2: same source frame across train/val/test ===")
cross = []
for key, recs in by_frame.items():
    splits = {r["split"] for r in recs}
    if len(splits) > 1:
        cross.append((key, splits, recs))
print(f"frames present in more than one split: {len(cross)}")
for key, splits, recs in cross:
    print(f"  frame {key}: splits={splits}")
    for r in recs:
        print(f"    split={r['split']:5s} class={r['class']:15s} copies_merged={r['copies_merged']} "
              f"orig={r['original_image']}")

# --- Claim 3 & 4 support: class distribution per split, empty-frame handling ---
print("\n=== CLAIM 4 support: per-split images vs unique frames ===")
for split in ("train", "val", "test"):
    sub = [r for r in rows if r["split"] == split]
    sub_frames = {frame_key(r) for r in sub}
    print(f"  {split:5s}: images={len(sub):4d}  unique_frames={len(sub_frames):4d}  "
          f"copies/frame={len(sub)/len(sub_frames):.3f}")

# empty-frame rows
empty_rows = [r for r in rows if not classes_of(r)]
print(f"\nempty-label rows in dataset: {len(empty_rows)}")
empty_frames = {frame_key(r) for r in empty_rows}
print(f"empty-label unique frames: {len(empty_frames)}")

# how many frames have empty label as their ONLY label (never paired with a label)
only_empty = sum(1 for recs in by_frame.values() if label_of_frame(recs) == "empty")
print(f"frames whose every copy is empty: {only_empty}")
