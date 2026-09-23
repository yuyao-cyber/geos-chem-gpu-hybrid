#!/usr/bin/env python3
"""cmp_week.py <rundir> <caseA> <caseB>
Per-day drift of case B vs case A over a multi-day run: for each daily restart,
global-mean relative difference and p99 (cells with conc >= 1e-6 x global mean)
for key species, plus the count of bit-identical species variables.
Run in the ncio conda env.
"""
import sys, glob, os, gc
import numpy as np, netCDF4 as nc
R, A, B = sys.argv[1:4]
SP = ["O3", "OH", "NO2", "CO", "HNO3", "SO4", "ISOP", "BCPI"]

def arr(v):
    x = v[:]
    if np.ma.isMaskedArray(x): x = x.filled(0.0)
    return np.asarray(x, dtype=np.float64).ravel()

def stats(a, b):
    m = np.abs(a).mean()
    if m == 0: return 0.0, 0.0
    sel = np.abs(a) >= 1e-6 * m
    if sel.sum() == 0: return 0.0, 0.0
    rel = np.abs(a[sel] - b[sel]) / np.maximum(np.abs(a[sel]), np.abs(b[sel]))
    rel[np.isnan(rel)] = 0.0
    return float(np.abs(a - b).sum() / np.abs(a).sum()), float(np.percentile(rel, 99))

files = sorted(glob.glob(f"{R}/case_{A}/GEOSChem.Restart.*.nc4"))
hdr1 = f"{'day':>10s} {'identical':>10s} " + " ".join(f"{s:>19s}" for s in SP)
hdr2 = f"{'':>10s} {'vars':>10s} " + " ".join(f"{'gmean':>9s} {'p99':>9s}" for s in SP)
print(hdr1); print(hdr2)
for fa in files:
    day = os.path.basename(fa).split(".")[2][:8]
    fb = f"{R}/case_{B}/" + os.path.basename(fa)
    if not os.path.exists(fb):
        print(f"{day:>10s}  (missing in {B})"); continue
    da, db = nc.Dataset(fa), nc.Dataset(fb)
    names = [v for v in da.variables if v.startswith("SpeciesRst_")]
    ident, unread = 0, 0
    for v in names:
        try: ident += int(np.array_equal(arr(da[v]), arr(db[v])))
        except Exception: unread += 1
    row = []
    for s in SP:
        v = f"SpeciesRst_{s}"
        try:
            g, p = stats(arr(da[v]), arr(db[v])) if v in da.variables else (float("nan"), float("nan"))
            row.append(f"{g:9.2e} {p:9.2e}")
        except Exception:
            row.append(f"{'unread':>9s} {'unread':>9s}")
    tag = f"  ({unread} unreadable vars)" if unread else ""
    print(f"{day:>10s} {ident:>4d}/{len(names):<5d} " + " ".join(row) + tag, flush=True)
    da.close(); db.close(); gc.collect()
