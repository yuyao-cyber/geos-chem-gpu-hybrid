#!/usr/bin/env python3
"""drift_gchp_week.py -- day-by-day accuracy of the GCHP week runs.
For each end-of-day checkpoint: GPU hybrid vs stock, judged by the rounding control vs stock
(same rule as envelope.py: a variable is OUTSIDE if d_gpu > 3 x d_ctl and d_gpu >= 1e-6)."""
import numpy as np, netCDF4 as nc
R = "/shared/gc/runs"
ST, CT, GP = f"{R}/gchp_c24_cpu/case_off_week", f"{R}/gchp_c24_fma/case_fma_week", f"{R}/gchp_c24/case_gpu_week"
DAYS = [(d, f"gcchem_internal_checkpoint.201407{d+1:02d}_0000z.nc4") for d in range(1, 7)] + [(7, "gcchem_internal_checkpoint")]
KEY = ["SPC_O3", "SPC_OH", "SPC_NO2", "SPC_CO", "SPC_HNO3", "SPC_SO4", "SPC_SALAAL"]

def gm(a, b):
    if np.array_equal(a, b): return 0.0
    den = np.abs(a).sum()
    return float(np.abs(a - b).sum() / den) if den > 0 else float(np.abs(a - b).max())

rows = []
print(f"{'day':>3} {'ident':>5} {'outside':>7} {'ratio_med':>9} {'ratio_p90':>9} {'ratio_max':>9}   " +
      "  ".join(f"{k[4:]:>17s}" for k in KEY))
print(" " * 49 + "  ".join(f"{'ctl / gpu':>17s}" for k in KEY))
for day, fn in DAYS:
    S, C, G = nc.Dataset(f"{ST}/{fn}"), nc.Dataset(f"{CT}/{fn}"), nc.Dataset(f"{GP}/{fn}")
    names = [v for v in S.variables if S[v].ndim >= 2 and v in C.variables and v in G.variables]
    ratios, ident, out, keyv = [], 0, 0, {}
    for v in names:
        a = np.asarray(S[v][:], dtype=np.float64).ravel()
        dc = gm(a, np.asarray(C[v][:], dtype=np.float64).ravel())
        dg = gm(a, np.asarray(G[v][:], dtype=np.float64).ravel())
        if dg == 0.0: ident += 1
        elif dg > 3 * dc and dg >= 1e-6: out += 1
        if dc > 1e-12 and dg > 0: ratios.append(dg / dc)
        if v in KEY: keyv[v] = (dc, dg)
    r = np.array(ratios)
    print(f"{day:>3} {ident:>5} {out:>7} {np.median(r):>9.2f} {np.percentile(r,90):>9.2f} {r.max():>9.2f}   " +
          "  ".join(f"{keyv[k][0]:8.2e}/{keyv[k][1]:8.2e}" for k in KEY), flush=True)
    for d in (S, C, G): d.close()
print("WEEK_DRIFT_DONE")
