#!/usr/bin/env python3
"""Compare two GEOS-Chem restart files variable by variable.

Usage: compare_restarts.py fileA fileB

Reports, for every variable present in both files:
  - bit-identical (byte-wise equal arrays), or
  - max absolute diff, max relative diff, count of differing elements.
Exit code 0 if all variables bit-identical, 1 otherwise.
"""
import sys
import numpy as np
import netCDF4 as nc

fa, fb = sys.argv[1], sys.argv[2]
da, db = nc.Dataset(fa), nc.Dataset(fb)

va, vb = set(da.variables), set(db.variables)
only_a, only_b = va - vb, vb - va
if only_a:
    print("Only in A:", sorted(only_a))
if only_b:
    print("Only in B:", sorted(only_b))

n_ident = 0
n_diff = 0
worst = []
for name in sorted(va & vb):
    xa = da.variables[name][:]
    xb = db.variables[name][:]
    if xa.shape != xb.shape:
        print(f"SHAPE MISMATCH {name}: {xa.shape} vs {xb.shape}")
        n_diff += 1
        continue
    a = np.ma.filled(xa, np.nan).astype(np.float64, copy=False)
    b = np.ma.filled(xb, np.nan).astype(np.float64, copy=False)
    if np.array_equal(a, b, equal_nan=True):
        n_ident += 1
        continue
    n_diff += 1
    d = np.abs(a - b)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = d / np.maximum(np.abs(a), np.abs(b))
    r = np.where(d == 0, 0.0, r)
    nd = int(np.sum(d > 0))
    maxd = float(np.nanmax(d))
    maxr = float(np.nanmax(r))
    worst.append((maxr, name, maxd, nd, a.size))
    print(f"DIFF {name}: ndiff={nd}/{a.size} maxabs={maxd:.6e} maxrel={maxr:.6e}")

print(f"\nSummary: {n_ident} variables bit-identical, {n_diff} with differences")
if worst:
    worst.sort(reverse=True)
    print("Worst 10 by max relative diff:")
    for maxr, name, maxd, nd, tot in worst[:10]:
        print(f"  {name}: maxrel={maxr:.3e} maxabs={maxd:.3e} ndiff={nd}/{tot}")
sys.exit(0 if n_diff == 0 else 1)
