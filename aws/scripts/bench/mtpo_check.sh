#!/bin/bash
#SBATCH -J mtpo -p cpu -N 1 -o /shared/logs/mtpo_check.%j.log
# Localize the Classic July-2019 MTPO difference (hybrid vs stock, against the rounding control). Runs on a compute node.
cd /shared/bench/classic
/shared/pyenv/bin/python - <<"PY"
import netCDF4 as nc, numpy as np
v="SpeciesRst_MTPO"
lat=nc.Dataset("stock/Restarts/GEOSChem.Restart.20190701_0000z.nc4")["lat"][:]; lon=nc.Dataset("stock/Restarts/GEOSChem.Restart.20190701_0000z.nc4")["lon"][:]
for D in ["0708","0715","0722","0725","0727","0728","0729","0730","0731","0801"]:
    f=f"stock/Restarts/GEOSChem.Restart.2019{D}_0000z.nc4"
    a,c,h=[np.asarray(nc.Dataset(f.replace("stock",x))[v][:],float).squeeze() for x in ["stock","control","hybrid"]]
    dh=np.abs(h-a); dc=np.abs(c-a); k=np.unravel_index(np.argmax(dh),a.shape)
    share=np.sort(dh.ravel())[::-1][:5].sum()/dh.sum()
    print(f"{D} ratio {dh.sum()/max(dc.sum(),1e-300):6.2f} gpu_gmean {dh.sum()/np.abs(a).sum():.2e} ctl_gmean {dc.sum()/np.abs(a).sum():.2e} top5 {100*share:3.0f}% | worst lev {k[0]} lat {lat[k[1]]:.0f} lon {lon[k[2]]:.0f}: stock {a[k]:.3e} gpu {h[k]:.3e} (rel {dh[k]/a[k]:.1e}); ctl worst rel {np.max(dc/np.maximum(a,1e-30)*(a>1e-6*a.mean())):.1e}")
PY
echo MTPO_DONE
