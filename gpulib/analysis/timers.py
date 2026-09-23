import json, sys
R = sys.argv[1]; cases = sys.argv[2:]
keys = ["GEOS-Chem","HEMCO","All chemistry","=> Gas-phase chem","=> Photolysis","=> Aerosol chem","Transport","Convection","Boundary layer mixing","Diagnostics"]
d = {c: json.load(open(f"{R}/case_{c}/gcclassic_timers.json"))["GEOS-Chem Classic timers"] for c in cases}
print("timer".ljust(24) + "".join(c.rjust(12) for c in cases))
for k in keys:
    print(k.ljust(24) + "".join(f"{d[c][k]['seconds']:12.2f}" for c in cases))
