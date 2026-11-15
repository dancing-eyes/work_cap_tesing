"""Verify claim 3: cross-split NCC>=0.99 residual pairs, and whether empty-label
frames are excluded from dedup (they are, by code). Recompute features the same
way prepare_dataset.image_feature does (128x72 grayscale, zero-mean, L2-normalized).
"""
import csv
import os
import numpy as np
from PIL import Image

ROOT = "d:/hat/dataset/helmet"
FEAT = (128, 72)

rows = list(csv.DictReader(open(os.path.join(ROOT, "split_manifest.csv"), encoding="utf-8")))

def feat(path):
    with Image.open(path) as im:
        arr = np.asarray(im.convert("L").resize(FEAT, Image.BILINEAR), dtype=np.float32)
    arr -= arr.mean()
    n = np.linalg.norm(arr)
    return (arr / (n + 1e-6)).ravel()

def classes(r):
    return frozenset(c for c in r["class"].split(",") if c)

paths = []
splits = []
labels = []
for r in rows:
    paths.append(os.path.join(ROOT, r["image"]))
    splits.append(r["split"])
    labels.append(tuple(sorted(classes(r))))

F = np.stack([feat(p) for p in paths])
S = F @ F.T

# cross-split NCC>=0.99 pairs
pairs = []
n = len(rows)
for i in range(n):
    for j in range(i + 1, n):
        if splits[i] == splits[j]:
            continue
        if S[i, j] >= 0.99:
            pairs.append((i, j, S[i, j], labels[i], labels[j], splits[i], splits[j]))

print(f"cross-split NCC>=0.99 pairs: {len(pairs)}")

# classify by label
empty_empty = [p for p in pairs if not p[3] and not p[4]]
empty_label = [p for p in pairs if (not p[3]) != (not p[4])]
both_label = [p for p in pairs if p[3] and p[4]]
print(f"  empty-empty pairs: {len(empty_empty)}")
print(f"  empty-vs-labeled pairs: {len(empty_label)}")
print(f"  labeled-labeled pairs: {len(both_label)}")

# group empty-empty into connected components (a 'group' = a frame linked to >=1 others)
from collections import defaultdict
adj = defaultdict(set)
for i, j, *_ in empty_empty:
    adj[i].add(j); adj[j].add(i)
seen = set()
comps = []
for i in adj:
    if i in seen:
        continue
    stack = [i]; seen.add(i); comp = []
    while stack:
        x = stack.pop(); comp.append(x)
        for y in adj[x]:
            if y not in seen:
                seen.add(y); stack.append(y)
    comps.append(comp)
print(f"  empty-empty connected components (groups): {len(comps)}")
for comp in comps:
    splits_in = {splits[x] for x in comp}
    print(f"    comp size={len(comp)} splits={splits_in} ncc={max(S[i,j] for i in comp for j in comp if i!=j):.4f}")
    for x in comp:
        print(f"       {splits[x]:5s} {rows[x]['original_image']}")
