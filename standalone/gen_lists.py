#!/usr/bin/env python3
"""Generate cell lists for the batch solver from reference_timings.csv:
   lists/random.txt (shuffled, fixed seed) and lists/sorted.txt (by cost)."""
import csv, random, os
D = "/storage1/fs1/rvmartin/Active/y.zhuge/optimize_gc/run_H_harvest"
rows = []
with open(os.path.join(D, "reference_timings.csv")) as f:
    for r in csv.DictReader(f):
        rows.append((r["file"], float(r["ms"])))
os.makedirs(os.path.join(D, "lists"), exist_ok=True)
rnd = rows[:]
random.seed(42)
random.shuffle(rnd)
with open(os.path.join(D, "lists/random.txt"), "w") as f:
    for name, _ in rnd:
        f.write(os.path.join(D, "KppSaOutput", name) + "\n")
srt = sorted(rows, key=lambda x: x[1])
with open(os.path.join(D, "lists/sorted.txt"), "w") as f:
    for name, _ in srt:
        f.write(os.path.join(D, "KppSaOutput", name) + "\n")
print("random:", len(rnd), "sorted:", len(srt))
