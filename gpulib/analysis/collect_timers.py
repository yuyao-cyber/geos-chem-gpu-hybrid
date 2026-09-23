#!/usr/bin/env python3
import json, sys
R = "/storage1/fs1/rvmartin/Active/y.zhuge/optimize_gc/run_J_gpu"
keys = ["GEOS-Chem", "HEMCO", "All chemistry", "=> Gas-phase chem", "=> Photolysis",
        "=> Aerosol chem", "Transport", "Convection", "Boundary layer mixing",
        "Diagnostics"]
cases = sys.argv[1:] or ["off", "cpu", "gpu"]
data = {}
for c in cases:
    with open(f"{R}/case_{c}/gcclassic_timers.json") as f:
        data[c] = json.load(f)["GEOS-Chem Classic timers"]
hdr = "timer".ljust(24) + "".join(c.rjust(12) for c in cases)
print(hdr); print("-" * len(hdr))
for k in keys:
    if all(k in data[c] for c in cases):
        print(k.ljust(24) + "".join(f"{data[c][k]['seconds']:12.2f}" for c in cases))
