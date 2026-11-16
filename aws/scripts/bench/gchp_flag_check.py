import numpy as np, netCDF4 as nc
B="/shared/bench/gchp"; FN=lambda d: f"gcchem_internal_checkpoint.201907{d:02d}_0000z.nc4"
cases=[("SPC_HMS",d) for d in (5,6,9,12,13)]+[(v,15) for v in ("SPC_LISOPOH","SPC_IHOO4","SPC_ISOP")]+[("SPC_LISOPOH",14),("SPC_LISOPOH",16)]
print(f"{'var':12s} {'date':>6s} {'gmean':>9s} {'d_ctl':>9s} {'d_gpu':>9s} {'ratio':>6s} {'top10 share gpu/ctl':>20s} {'max cell rel gpu/ctl':>22s} {'gpu>stock %cells':>16s}")
for v,d in cases:
    S,C,G=[nc.Dataset(f"{B}/{x}/Restarts/{FN(d)}") for x in ("stock","control","hybrid")]
    a,c,g=[np.asarray(x[v][:],float).ravel() for x in (S,C,G)]
    den=np.abs(a).sum(); ec=np.abs(c-a); eg=np.abs(g-a)
    dc,dg=ec.sum()/den,eg.sum()/den
    t=lambda e: np.sort(e)[-10:].sum()/e.sum() if e.sum()>0 else 0
    m=a>1e-3*a.max()
    rel=lambda x: (np.abs(x-a)[m]/a[m]).max()
    nz=(g!=a); up=(g>a)[nz].mean()*100 if nz.any() else 0
    print(f"{v[4:]:12s} 07-{d:02d} {a.mean():9.2e} {dc:9.2e} {dg:9.2e} {dg/dc:6.1f} {t(eg):9.2f}/{t(ec):<9.2f} {rel(g):10.2e}/{rel(c):<10.2e} {up:16.0f}")
    for x in (S,C,G): x.close()
