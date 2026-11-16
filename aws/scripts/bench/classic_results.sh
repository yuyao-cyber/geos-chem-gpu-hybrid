#!/bin/bash
#SBATCH -J cl_res -p cpu -N 1 -o /shared/logs/classic_results.%j.log
# Classic July-2019 benchmark summary: timers, benchmark metrics, monthly-mean differences, MTPO bias.
P=/shared/pyenv/bin/python; C=/shared/bench/classic
$P -m pip install -q xarray pyyaml 2>&1 | tail -1
echo "########## TIMERS (seconds)"
$P - <<"PY"
import json
R="/shared/bench/classic"; cases=["stock","hybrid","control"]
d={c: json.load(open(f"{R}/{c}/gcclassic_timers.json"))["GEOS-Chem Classic timers"] for c in cases}
keys=["GEOS-Chem","HEMCO","All chemistry","=> Gas-phase chem","=> Photolysis","=> Aerosol chem","Transport","Convection","Boundary layer mixing","Dry deposition","Wet deposition","Diagnostics","Input","Output"]
print(f"{'timer':26s}{'stock':>10s}{'hybrid':>10s}{'change':>9s}{'control':>10s}")
for k in keys:
    if k in d["stock"]:
        s,h,c=[d[x][k]["seconds"] for x in cases]
        print(f"{k:26s}{s:10.0f}{h:10.0f}{(h-s)/s*100 if s else 0:8.0f}%{c:10.0f}")
PY
echo "########## BENCHMARK METRICS (metrics.py)"
for x in stock hybrid control; do echo "== $x"; (cd $C/$x && $P metrics.py 2>&1 | grep -vE "^\s*$|^=+$" | head -14); done
echo "########## MONTHLY-MEAN SPECIES: global mean (all levels), % difference from stock"
$P - <<"PY"
import netCDF4 as nc, numpy as np
R="/shared/bench/classic"; f="OutputDir/GEOSChem.SpeciesConc.20190701_0000z.nc4"
S,H,C=[nc.Dataset(f"{R}/{x}/{f}") for x in ["stock","hybrid","control"]]
names=[v for v in S.variables if v.startswith("SpeciesConcVV_")]
ratios=[]; rows=[]
for v in names:
    a=np.asarray(S[v][:],float); h=np.asarray(H[v][:],float); c=np.asarray(C[v][:],float)
    den=np.abs(a).sum()
    if den==0: continue
    dh=np.abs(h-a).sum()/den; dc=np.abs(c-a).sum()/den
    rows.append((v[14:],a.mean(),(h.mean()-a.mean())/a.mean()*100,(c.mean()-a.mean())/a.mean()*100,dh,dc))
    if dc>1e-12 and dh>0: ratios.append(dh/dc)
r=np.array(ratios)
print(f"species compared: {len(rows)}; |diff| ratio hybrid/control: median {np.median(r):.2f} p90 {np.percentile(r,90):.2f} max {r.max():.2f}; species with ratio>3: {int((r>3).sum())}")
print(f"{'species':10s}{'stock mean (mol/mol)':>22s}{'hybrid %':>12s}{'control %':>12s}")
for k in ["O3","OH","CO","NO","NO2","HNO3","CH2O","ISOP","MTPA","MTPO","LIMO","SO2","SO4","NH4","NIT","BCPI","OCPI","SALA","DST1","PAN","CH4"]:
    for row in rows:
        if row[0]==k: print(f"{k:10s}{row[1]:22.4e}{row[2]:+12.2e}{row[3]:+12.2e}")
print("largest hybrid/control ratios:", ", ".join(f"{x[0]}={x[4]/x[5]:.1f}" for x in sorted(rows,key=lambda x:-(x[4]/max(x[5],1e-30)))[:6]))
PY
echo "########## MTPO signed bias over the month (restart files)"
$P - <<"PY"
import netCDF4 as nc, numpy as np
R="/shared/bench/classic"; v="SpeciesRst_MTPO"
for D in ["0702","0708","0715","0722","0725","0729","0801"]:
    f=f"Restarts/GEOSChem.Restart.2019{D}_0000z.nc4"
    a,h,c=[np.asarray(nc.Dataset(f"{R}/{x}/{f}")[v][:],float).squeeze() for x in ["stock","hybrid","control"]]
    m=a.mean(); fh=(h>a).mean()*100; fc=(c>a).mean()*100
    print(f"{D}: mean bias hybrid {(h.mean()-m)/m*100:+.2e}%  control {(c.mean()-m)/m*100:+.2e}%  | cells hybrid>stock {fh:.0f}%  control>stock {fc:.0f}%")
PY
echo CLASSIC_RESULTS_DONE
