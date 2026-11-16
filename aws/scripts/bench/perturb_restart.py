#!/usr/bin/env python3
"""perturb_restart.py FILE -- in place: move every SPC_* value one unit in the last place (ulp), up or down at random.
Second, independent rounding control for the GCHP benchmark: stock code, initial state changed only in the last digit.
If the 15 July isoprene event also shows up here, it is ordinary rounding-level chaos and not something the GPU causes."""
import sys, numpy as np, netCDF4 as nc
rng = np.random.default_rng(20190701)
ds = nc.Dataset(sys.argv[1], "r+")
n = 0
for name, v in ds.variables.items():
    if not name.startswith("SPC_"):
        continue
    a = v[:]
    a = np.ma.filled(a, 0) if np.ma.isMaskedArray(a) else np.asarray(a)
    up = rng.random(a.shape) < 0.5
    b = np.where(up, np.nextafter(a, np.inf), np.nextafter(a, -np.inf)).astype(a.dtype)
    b = np.where(a == 0, a, b)  # leave exact zeros alone (no new negative or tiny values)
    v[:] = b
    n += 1
    if n == 1:
        rel = np.abs(b - a)[a != 0] / np.abs(a[a != 0])
        print(f"{name}: dtype {a.dtype}, median relative change {np.median(rel):.2e}")
ds.close()
print(f"perturbed {n} SPC_ variables in {sys.argv[1]}")
