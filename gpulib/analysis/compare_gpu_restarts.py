#!/usr/bin/env python3
"""
compare_gpu_restarts.py fileA fileB [gckpp_Monitor.F90]

Per-variable comparison of two GEOS-Chem restart files (like compare_restarts.py)
plus a species-class summary for the GPU validation:
  - KPP variable species (in the mechanism's SPC_NAMES, solved by the integrator)
  - KPP fixed species
  - non-KPP species (transport / aerosol-only tracers: expected bit-identical)
  - non-species restart variables (Chem_*, e.g. Chem_KPPHvalue)
and global-mean relative differences for OH, O3, NO2, CO, HO2, NO.
Relative diff of an element = |a-b| / max(|a|,|b|) (0 where a==b).
"""
import re
import sys
import numpy as np
import netCDF4 as nc

fa, fb = sys.argv[1], sys.argv[2]
monitor = sys.argv[3] if len(sys.argv) > 3 else None

kpp_var, kpp_fix = set(), set()
if monitor:
    txt = open(monitor).read()
    # SPC_NAMES data statements: 'NAME    ' strings, first NVAR are variable
    m = re.search(r"SPC_NAMES_0\s*=\s*\(/(.*?)/\)", txt, re.S)
    names = re.findall(r"'([A-Za-z0-9_]+)\s*'", txt.split("SPC_NAMES_0")[1].split("EQN_NAMES")[0]) \
        if "SPC_NAMES_0" in txt else re.findall(r"'([A-Za-z0-9_]+)\s*'", txt)
    # NVAR from parameters file next to monitor
    try:
        ptxt = open(monitor.replace("gckpp_Monitor", "gckpp_Parameters")).read()
        nvar = int(re.search(r"NVAR\s*=\s*(\d+)", ptxt).group(1))
    except Exception:
        nvar = len(names)
    kpp_var = set(names[:nvar])
    kpp_fix = set(names[nvar:])

da, db = nc.Dataset(fa), nc.Dataset(fb)
va, vb = set(da.variables), set(db.variables)
rows = []
for name in sorted(va & vb):
    xa, xb = da.variables[name][:], db.variables[name][:]
    if xa.shape != xb.shape or xa.dtype.kind not in "fi":
        continue
    a = np.ma.filled(xa, np.nan).astype(np.float64, copy=False)
    b = np.ma.filled(xb, np.nan).astype(np.float64, copy=False)
    if name.startswith("SpeciesRst_"):
        sp = name[len("SpeciesRst_"):]
        cls = "KPP-var" if sp in kpp_var else ("KPP-fix" if sp in kpp_fix else "nonKPP")
    elif name.startswith("Chem_"):
        cls = "Chem_*"
    else:
        cls = "other"
    ident = np.array_equal(a, b, equal_nan=True)
    if ident:
        rows.append((cls, name, True, 0.0, 0.0, 0, a.size, 0.0))
        continue
    d = np.abs(a - b)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = d / np.maximum(np.abs(a), np.abs(b))
    r = np.where(d == 0, 0.0, r)
    ma, mb = np.nanmean(a), np.nanmean(b)
    gm = abs(ma - mb) / abs(ma) if ma != 0 else 0.0
    rows.append((cls, name, False, float(np.nanmax(r)), float(np.nanmax(d)),
                 int(np.sum(d > 0)), a.size, float(gm)))

n_ident = sum(1 for r in rows if r[2])
print(f"Summary: {n_ident} variables bit-identical, {len(rows)-n_ident} with differences "
      f"(of {len(rows)} compared)")
print()
print("Per class:")
print(f"{'class':8s} {'nvars':>5s} {'ident':>5s} {'diff':>5s} {'max_rel':>10s} {'median_maxrel':>13s} {'max_gmean_rel':>13s}")
for cls in ["KPP-var", "KPP-fix", "nonKPP", "Chem_*", "other"]:
    rr = [r for r in rows if r[0] == cls]
    if not rr:
        continue
    diff = [r for r in rr if not r[2]]
    mx = max((r[3] for r in rr), default=0.0)
    med = float(np.median([r[3] for r in diff])) if diff else 0.0
    gmx = max((r[7] for r in rr), default=0.0)
    print(f"{cls:8s} {len(rr):5d} {len(rr)-len(diff):5d} {len(diff):5d} {mx:10.3e} {med:13.3e} {gmx:13.3e}")
print()
print("Key species (global-mean relative diff, max elementwise relative diff):")
for sp in ["OH", "O3", "NO2", "CO", "HO2", "NO", "HNO3", "ISOP", "SO2", "SO4", "BCPI", "OCPI", "SALA", "DST1"]:
    rr = [r for r in rows if r[1] == f"SpeciesRst_{sp}"]
    if rr:
        r = rr[0]
        print(f"  {sp:6s} identical={r[2]!s:5s} gmean_rel={r[7]:.3e} max_rel={r[3]:.3e} "
              f"ndiff={r[5]}/{r[6]}")
print()
diff = sorted([r for r in rows if not r[2]], key=lambda r: -r[3])
print("Worst 15 by max relative diff:")
for r in diff[:15]:
    print(f"  {r[1]:28s} [{r[0]}] maxrel={r[3]:.3e} maxabs={r[4]:.3e} gmean_rel={r[7]:.3e} ndiff={r[5]}/{r[6]}")
print()
print("Non-identical variables outside KPP-var:")
for r in diff:
    if r[0] != "KPP-var":
        print(f"  {r[1]:28s} [{r[0]}] maxrel={r[3]:.3e} ndiff={r[5]}/{r[6]}")
sys.exit(0 if len(rows) == n_ident else 1)
