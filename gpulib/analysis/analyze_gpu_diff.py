#!/usr/bin/env python3
"""analyze_gpu_diff.py A.nc4 B.nc4 -- robust CPU-vs-GPU difference statistics.

The raw "max relative difference" over all cells is dominated by zero-crossing
cells (one value ~1e-24, the other ~0 -> rel = 1) and says nothing useful.  This
reports, per species: the global-mean relative difference, and relative-difference
percentiles restricted to cells carrying a meaningful concentration
(>= 1e-6 of that species' global mean), which is the number that reflects
actual solver agreement.
"""
import sys
import numpy as np
import netCDF4 as nc

fa, fb = sys.argv[1], sys.argv[2]
da, db = nc.Dataset(fa), nc.Dataset(fb)

SPECIES = ["OH", "HO2", "O3", "NO", "NO2", "CO", "HNO3", "ISOP", "CH2O", "SO2",
           "SO4", "PAN", "N2O5", "BrO", "ALD2", "MO2",
           "BCPI", "OCPI", "SALA", "SALC", "DST1", "BCPO"]
NONKPP = {"BCPI", "OCPI", "SALA", "SALC", "DST1", "BCPO"}

print(f"A = {fa}\nB = {fb}\n")
print("Relative difference restricted to cells with conc >= 1e-6 x global mean")
print(f"{'species':8s} {'class':9s} {'gmean_rel':>11s} {'p50':>10s} {'p99':>10s} "
      f"{'p99.9':>10s} {'max':>10s} {'ncells':>9s}")
for sp in SPECIES:
    v = f"SpeciesRst_{sp}"
    if v not in da.variables or v not in db.variables:
        continue
    a = np.ma.filled(da.variables[v][:], np.nan).astype(np.float64).ravel()
    b = np.ma.filled(db.variables[v][:], np.nan).astype(np.float64).ravel()
    ma = np.nanmean(a)
    gmean = abs(np.nanmean(a) - np.nanmean(b)) / abs(ma) if ma else 0.0
    thr = 1e-6 * abs(ma)
    m = (np.abs(a) >= thr) & (np.abs(b) >= thr) & np.isfinite(a) & np.isfinite(b)
    if m.sum() == 0:
        continue
    r = np.abs(a[m] - b[m]) / np.maximum(np.abs(a[m]), np.abs(b[m]))
    cls = "nonKPP" if sp in NONKPP else "KPP-var"
    print(f"{sp:8s} {cls:9s} {gmean:11.3e} {np.percentile(r,50):10.3e} "
          f"{np.percentile(r,99):10.3e} {np.percentile(r,99.9):10.3e} "
          f"{r.max():10.3e} {m.sum():9d}")

print()
print("Zero-crossing check -- for the variables reporting max_rel = 1.0, the")
print("largest ABSOLUTE difference anywhere in the field:")
worst = []
for v in sorted(set(da.variables) & set(db.variables)):
    if not v.startswith("SpeciesRst_"):
        continue
    a = np.ma.filled(da.variables[v][:], np.nan).astype(np.float64)
    b = np.ma.filled(db.variables[v][:], np.nan).astype(np.float64)
    d = np.abs(a - b)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(d == 0, 0.0, d / np.maximum(np.abs(a), np.abs(b)))
    if np.nanmax(r) > 0.99:
        worst.append((v, float(np.nanmax(d)), float(np.nanmax(np.abs(a)))))
worst.sort(key=lambda t: -t[1])
print(f"  {len(worst)} variables have some cell with rel diff > 0.99")
for v, mad, peak in worst[:10]:
    print(f"    {v:26s} max_abs_diff={mad:.3e}  field_peak={peak:.3e}  "
          f"ratio={mad/peak if peak else 0:.3e}")

print()
print("Chem_* (solver bookkeeping) fields:")
for v in sorted(set(da.variables) & set(db.variables)):
    if not v.startswith("Chem_"):
        continue
    a = np.ma.filled(da.variables[v][:], np.nan).astype(np.float64)
    b = np.ma.filled(db.variables[v][:], np.nan).astype(np.float64)
    if np.array_equal(a, b, equal_nan=True):
        print(f"  {v:26s} bit-identical")
    else:
        d = np.abs(a - b)
        ma = np.nanmean(np.abs(a))
        print(f"  {v:26s} max_abs={np.nanmax(d):.3e} mean_abs={np.nanmean(d):.3e} "
              f"field_mean={ma:.3e}")
