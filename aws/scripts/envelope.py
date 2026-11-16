#!/usr/bin/env python3
"""envelope.py STOCK CONTROL TEST [label] [CTL_STOCK]
If CTL_STOCK is given, the control difference is measured as CTL_STOCK vs CONTROL (e.g. a 1-node
control pair) while the test difference is STOCK vs TEST (e.g. both in a 2-node layout).

Species-by-species accuracy check of a TEST run (GPU / hybrid) against the STOCK run,
judged by the rounding-only CONTROL (same code, FMA allowed in KPP only).

For each variable present in all three files:
  d_ctl  = global-mean relative difference |stock-control| summed / |stock| summed
  d_test = same for |stock-test|
  p99    = 99th percentile of the cell relative difference over cells >= 1e-6 x mean
Verdict per variable:
  IDENT   test bit-identical to stock
  INSIDE  d_test <= 3*d_ctl  (or both below 1e-12)
  SMALL   outside 3x control but d_test < 1e-6 (one part per million of the global burden)
  OUTSIDE everything else (listed)
Exit code 0 when no variable is OUTSIDE.
"""
import sys, numpy as np, netCDF4 as nc

fs, fc, ft = sys.argv[1:4]
label = sys.argv[4] if len(sys.argv) > 4 else ft
fcs = sys.argv[5] if len(sys.argv) > 5 else fs
S, C, T, CS = nc.Dataset(fs), nc.Dataset(fc), nc.Dataset(ft), nc.Dataset(fcs)
names = [v for v in S.variables if v in C.variables and v in T.variables and S[v].ndim >= 2]

def stats(a, b):
    if np.array_equal(a, b): return 0.0, 0.0, True
    den = np.abs(a).sum()
    g = float(np.abs(a - b).sum() / den) if den > 0 else float(np.abs(a - b).max())
    m = np.abs(a).mean()
    sel = np.abs(a) >= 1e-6 * m if m > 0 else np.ones(a.shape, bool)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.abs(a[sel] - b[sel]) / np.maximum(np.abs(a[sel]), np.abs(b[sel]))
    r = np.nan_to_num(r)
    return g, float(np.percentile(r, 99)) if r.size else 0.0, False

rows = []; cnt = {"IDENT": 0, "INSIDE": 0, "SMALL": 0, "OUTSIDE": 0}
for v in names:
    a = np.asarray(S[v][:], dtype=np.float64).ravel()
    gc_, pc, _ = stats(np.asarray(CS[v][:], dtype=np.float64).ravel(), np.asarray(C[v][:], dtype=np.float64).ravel())
    gt, pt, ident = stats(a, np.asarray(T[v][:], dtype=np.float64).ravel())
    if ident: verdict = "IDENT"
    elif gt <= 3 * gc_ or (gt < 1e-12 and gc_ < 1e-12): verdict = "INSIDE"
    elif gt < 1e-6: verdict = "SMALL"
    else: verdict = "OUTSIDE"
    cnt[verdict] += 1
    rows.append((v, gc_, pc, gt, pt, verdict))

print(f"### {label}")
print(f"variables: {len(names)}   " + "  ".join(f"{k}={n}" for k, n in cnt.items()))
ratios = np.array([r[3] / r[1] for r in rows if r[1] > 1e-12 and r[3] > 0])
if ratios.size:
    print("test/control gmean ratio: median %.2f  p90 %.2f  max %.2f" %
          (np.median(ratios), np.percentile(ratios, 90), ratios.max()))
key = ["O3", "OH", "NO2", "NO", "CO", "HNO3", "SO4", "ISOP", "CH2O", "PAN", "SALAAL", "BCPI"]
print(f"{'variable':22s} {'ctl_gmean':>10s} {'ctl_p99':>10s} {'test_gmean':>10s} {'test_p99':>10s}  verdict")
def short(v): return v.replace("SpeciesRst_", "").replace("SPC_", "")
for r in rows:
    if short(r[0]) in key or r[5] == "OUTSIDE":
        print(f"{r[0]:22s} {r[1]:10.2e} {r[2]:10.2e} {r[3]:10.2e} {r[4]:10.2e}  {r[5]}")
worst = sorted(rows, key=lambda r: -r[3])[:5]
print("largest test gmean:", ", ".join(f"{short(r[0])}={r[3]:.1e}(ctl {r[1]:.1e})" for r in worst))
sys.exit(1 if cnt["OUTSIDE"] else 0)
