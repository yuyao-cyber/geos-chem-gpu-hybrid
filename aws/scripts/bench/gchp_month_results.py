#!/usr/bin/env python3
"""gchp_month_results.py -- July 2019 GCHP benchmark accuracy from the daily checkpoints.
Monthly-mean HISTORY files were never written (multi-node MPI deadlock at the end-of-month write),
so accuracy uses the 30 end-of-day checkpoints (2-31 July, 00 UTC) of stock, rounding control and hybrid.
Rule as envelope.py: a variable is OUTSIDE if d_gpu > 3 x d_ctl and d_gpu >= 1e-6,
d = global-mean |run - stock| / global-mean |stock| over all cells and levels."""
import numpy as np, netCDF4 as nc
B = "/shared/bench/gchp"
ST, CT, GP = (f"{B}/{x}/Restarts" for x in ("stock", "control", "hybrid"))
FN = lambda d: f"gcchem_internal_checkpoint.201907{d:02d}_0000z.nc4"
DAYS = list(range(2, 32))  # end of 1 July ... end of 30 July
KEY = ["SPC_O3", "SPC_OH", "SPC_CO", "SPC_NO2", "SPC_HNO3", "SPC_NIT", "SPC_SO4", "SPC_CH4", "SPC_MTPO", "SPC_ISOP"]

def gm(a, b):
    if np.array_equal(a, b): return 0.0
    den = np.abs(a).sum()
    return float(np.abs(a - b).sum() / den) if den > 0 else float(np.abs(a - b).max())

def rd(ds, v):
    return np.asarray(ds[v][:], dtype=np.float64).ravel()

S0 = nc.Dataset(f"{ST}/{FN(2)}")
names = [v for v in S0.variables if S0[v].ndim >= 2]
S0.close()
acc = {k: {v: None for v in names} for k in ("S", "C", "G")}

print("########## DAILY CHECKPOINTS vs STOCK (end of day, 00 UTC)")
print(f"{'date':>10} {'ident':>5} {'outside':>7} {'med':>6} {'p90':>6} {'max':>7}  outside variables (ratio)")
for d in DAYS:
    S, C, G = (nc.Dataset(f"{p}/{FN(d)}") for p in (ST, CT, GP))
    ratios, ident, outs = [], 0, []
    for v in names:
        if v not in C.variables or v not in G.variables: continue
        a, c, g = rd(S, v), rd(C, v), rd(G, v)
        for k, x in (("S", a), ("C", c), ("G", g)):
            acc[k][v] = x.copy() if acc[k][v] is None else acc[k][v] + x
        dc, dg = gm(a, c), gm(a, g)
        if dg == 0.0: ident += 1
        elif dg > 3 * dc and dg >= 1e-6: outs.append(f"{v[4:]}({dg/dc if dc>0 else np.inf:.1f})")
        if dc > 1e-12 and dg > 0: ratios.append(dg / dc)
    r = np.array(ratios)
    print(f"2019-07-{d:02d} {ident:>5} {len(outs):>7} {np.median(r):>6.2f} {np.percentile(r,90):>6.2f} {r.max():>7.2f}  {' '.join(outs)}", flush=True)
    for x in (S, C, G): x.close()

print("\n########## MEAN OF THE 30 DAILY SNAPSHOTS (proxy for the monthly mean)")
ratios, outs, ident, n = [], [], 0, 0
rows = {}
for v in names:
    if acc["C"][v] is None: continue
    a, c, g = acc["S"][v] / len(DAYS), acc["C"][v] / len(DAYS), acc["G"][v] / len(DAYS)
    dc, dg = gm(a, c), gm(a, g)
    n += 1
    if dg == 0.0: ident += 1
    elif dg > 3 * dc and dg >= 1e-6: outs.append(f"{v[4:]}({dg/dc if dc>0 else np.inf:.1f})")
    if dc > 1e-12 and dg > 0: ratios.append((dg / dc, v))
    if v in KEY:
        sa = a.sum()
        rows[v] = ((g.sum() - sa) / sa * 100 if sa else 0, (c.sum() - sa) / sa * 100 if sa else 0, dg, dc)
r = np.array([x[0] for x in ratios])
print(f"variables {n}  bit-identical {ident}  outside {len(outs)}  ratio median {np.median(r):.2f}  p90 {np.percentile(r,90):.2f}  max {r.max():.2f} ({max(ratios)[1][4:]})")
print("outside:", " ".join(outs) if outs else "none")
print(f"\n{'species':10s} {'hybrid-stock %':>15s} {'control-stock %':>16s} {'|d| hybrid':>11s} {'|d| control':>12s}")
for k in KEY:
    if k in rows:
        h, c, dg, dc = rows[k]
        print(f"{k[4:]:10s} {h:15.2e} {c:16.2e} {dg:11.2e} {dc:12.2e}")
print("GCHP_MONTH_RESULTS_DONE")
