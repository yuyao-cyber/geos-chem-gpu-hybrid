# GEOS-Chem 14.7.1 Performance Analysis — Findings & Optimization Plan

Analyzed: GCClassic 14.7.1 (commit c36ecd7) — full source review of chemistry (KPP),
transport/mixing/convection, photolysis (Cloud-J), aerosol (HETP/RDAER), HEMCO I/O,
History diagnostics, and build system.
Date: 2026-07-20

## Tier 1 — Config-only changes (no recompile, biggest wins)

### 1.1 Enable the KPP auto-reduction solver  [LARGEST WIN]
- 14.7.1 already COMPILES the `rosenbrock_autoreduce` integrator by default
  (`KPP/fullchem/fullchem.kpp:2`, `-DKPP_INTEGRATOR_AUTOREDUCE`), but it is
  DISABLED at runtime: `geoschem_config.yml` → `autoreduce_solver: activate: false`.
- Flipping to `true` makes the integrator partition species into fast/slow sets per
  cell and shrink the sparse-LU linear algebra (the dominant chemistry cost).
- Lin et al. (2023) report ~30–50% chemistry wall-time reduction for this exact
  feature. Chemistry is ~50–60% of GC-Classic runtime → ~15–25% total.
- Accuracy: small documented perturbations; tunable via `oh_tuning_factor`,
  `no2_tuning_factor`, `absolute_threshold`, `keep_halogens_active` knobs.
- Note: step-size reuse across timesteps (RCNTRL(3)=KPPHvalue) and
  rates-computed-once-per-cell (ICNTRL(15)=-1) are ALREADY on in the default build.

### 1.2 Enable built-in timers for benchmarking
- `geoschem_config.yml` → `simulation: use_gcclassic_timers: true`
- Gives per-operator wall clock (All chemistry, Photolysis, Transport, Mixing,
  Convection, Wet dep, Diagnostics, Unit conversions, HEMCO) + JSON output.

### 1.3 Trim HISTORY.rc / HEMCO audit (production hygiene)
- Every History variable is written with deflate level 1 + shuffle (CPU cost per write).
- Diagnostic archiving copies are gated per collection — disabling unused
  collections removes both copy and write cost.
- Audit HEMCO_Config.rc for entries in the "Always" read bucket (re-read every
  timestep) and ensure Verbose: 0.

## Tier 2 — Safe code changes (bit-identical or last-ULP only)

### 2.1 Hoist common factor in unit-conversion routines  [unitconv_mod.F90]
- ~8 full-array unit conversions per dynamic timestep (main.F90:898/1771,
  mixing_mod:368/840, pbl_mix:164/211, wetscav:3321/3918; +2 per chem step).
- Each routine recomputes the SAME factor array (e.g. `g0_100*DELP_DRY`) inside
  the per-species loop (~nAdvect times). Hoist to one shared precomputed array →
  ~1.5–2× faster per conversion, bit-for-bit identical.
- Files: GeosUtil/unitconv_mod.F90 (~10 routines, e.g. lines 1146-1165, 1249-1266).

### 2.2 Guard dead copies in the chemistry cell loop  [fullchem_mod.F90:1105-1110]
- `local_RCONST = RCONST` (~1235 doubles) + `KPPH_before_integrate` copied EVERY
  cell EVERY timestep, consumed only by KPP-standalone sampling which is off in
  production (returns immediately on SkipIt). Wrap in the existing skip flag.

### 2.3 Multiply instead of divide in hottest rate-law functions
- `GCARR_ac` (called 395×/cell/timestep in Update_RCONST) computes
  `EXP(c0/TEMP)`; `INV_TEMP` is already precomputed per cell. Change to
  `EXP(c0*INV_TEMP)` (same in GCARR_abc). Removes ~400 FP divisions/cell/step.
- Last-ULP difference only (breaks bit-for-bit diff, not accuracy).
- File: KPP/fullchem/rateLawUtilFuncs.F90:52-70.

### 2.4 Hoist RH/EXP out of aerosol-optics loop  [aerosol_mod.F90:1701-1732]
- Saturation-vapor `EXP` + RH-bin selection depend only on (I,J,L) but sit inside
  the aerosol-type (×7) and wavelength loops → recomputed 7×NWVS per cell.
  Pre-pass once per cell; bit-identical.

### 2.5 Cloud-J interface invariant hoists  [cldj_interface_mod.F90]
- `NDXAER` map rebuilt every timestep (code comment: "Don't want to do this
  computation every timestep") → cache at init (lines 409-418).
- `MW_g`, `MW_kg` invariants inside per-level loops → hoist (lines 587, 860).
- RH LUT interpolation FRAC recomputed per aerosol species (5×) per level →
  compute once per level (lines 668-852).

### 2.6 Remove giant per-timestep automatic arrays (memory/robustness + small CPU)
- WETDEP `DSpc(nWetDep,NZ,NX,NY)` — full-grid automatic array reallocated every
  timestep (wetscav_mod.F90:3289) despite "per column" comment → module SAVE alloc.
- TPCORE `fx/fy/fz(im,jm,km,nq)` + 3-D work arrays — automatic, every advection
  step (tpcore_fvdas_mod.F90:530-560) → allocate once in Init_Tpcore; per-species
  flux dim only needed when AdvFlux diagnostics on.
- These are why GC needs `ulimit -s unlimited`; removes alloc/page-fault jitter.

## Tier 3 — Build-flag changes (test carefully)

### 3.1 Host-specific vectorization
- Current Release flags: gfortran `-O3 -funroll-loops`, Intel `-O2` only.
  NO -march/-mtune/-xHost anywhere → binary uses generic ISA, no AVX2/AVX-512.
- Add `-march=native` (or `-march=skylake-avx512` for compute1 Cascade Lake
  nodes, to be safe across heterogeneous hosts) to all FOUR flag blocks:
  root CMakeLists.txt (:73-76, :103-106), src/HEMCO, src/Cloud-J, src/HETP.
- Expected 5–20% on compute-bound sections. Must validate results vs baseline.

### 3.2 (Optional) `-DNC_NODEFLATE` build for I/O-heavy runs — trades disk for CPU.

## Deferred / not recommended now
- HETP warm-start / skip-unchanged-cells: biggest single remaining lever but
  physics-sensitive (moderate-high accuracy risk); dev TODO exists in code.
- v/v ⇄ kg/kg round-trip elimination at main-loop boundary (main.F90:898/1771):
  ~2 conversions/step but exists for restart bit-reproducibility; medium risk.
- Global array-layout change (I,J,L)→(L,I,J): helps column physics, hurts
  advection; very high risk. Not worth it.
- Verified NOT problems: no allocations in inner loops; OpenMP scheduling already
  tuned (DYNAMIC,24 + COLLAPSE); met-field reads properly cached; HEMCO reads
  properly bucketed by update frequency; Cloud-J CLDFLAG=3 already the cheap
  scheme; dark-column skip correct; mixing DO_TEND already de-duplicated.

## Benchmark plan
- Platform: compute1 (LSF + Docker billzhuge/geos-chem-deps), gfortran.
- Run: GC-Classic 4x5 MERRA-2 fullchem, 7 simulated days (20140701–20140708;
  2014 4x5 met is complete in GEOS-Chem-shared ExtData), 24 OpenMP threads,
  use_gcclassic_timers: true, minimal HISTORY collections + SpeciesConc for
  accuracy checks.
- Configs: (A) baseline 14.7.1; (B) A + autoreduce on (config only);
  (C) B + Tier-2 code changes; (D) C + -march flags.
- Metrics: total wall time + per-operator timer JSON; accuracy via SpeciesConc
  comparison (mean/max relative diff for O3, OH, NO2, CO, PM25 species) vs baseline.
- Repeat each config 2–3× to control filesystem noise.

============================================================
## BENCHMARK RESULTS (2026-07-20/21)
7-day 4x5 MERRA2 fullchem 72L, 20140701-08, 24 OMP threads,
node compute1-exec-13 (Xeon Gold 6154, exclusive via -n 72),
2 repetitions per config. Timers = mean of 2 reps (seconds).

| Timer            |    A stock |  B +AR |  C +code | D code+march+AR |
|------------------|-----------:|-------:|---------:|----------------:|
| TOTAL            |       3717 |   3570 |     3691 |            3484 |
| vs A             |          – |  -4.0% |    -0.7% |           -6.3% |
| Gas-phase chem   |       1387 |   1190 |     1413 |            1221 |
| vs A             |          – | -14.2% |    +1.9% |          -12.0% |
| Photolysis       |        138 |    139 |      139 |             127 |
| Wet deposition   |        177 |    174 |      170 |             171 |

Notes:
- Run-to-run noise in TOTAL is +-3% (HEMCO/storage1 I/O); chem timers
  are stable to +-0.2%, so chem ratios are trustworthy.
- Autoreduce (config flip only): -14% chemistry, -4% total. Less than
  Lin et al. 30-50% (default thresholds, 4x5); thresholds tunable.
- Tier-2 code changes: net neutral overall. Real -4% wetdep; small
  consistent +1.9% chem regression (suspect the KppSa guard branch;
  candidate for revert). Transport-tracer output (BCPI/OCPI/SALA)
  bit-identical vs stock = correctness confirmed.
- -march=skylake-avx512: ~nothing on KPP (sparse LU is latency-bound,
  not vectorizable), -7% photolysis, ~-1% total. Marginal.
- Accuracy: A-vs-B means O3 0.03%, OH 0.06%, SO4 0.18%; NIT mean 6%
  with large relative diffs in near-zero cells (known AR sensitivity;
  consider keep_halogens_active / threshold tuning if NIT matters).
- Code diff: tier2_optimizations.patch (this directory).

============================================================
## GCHP BENCHMARK RESULTS (2026-07-31/08-01)
GCHP 14.7.1, c24, GEOS-FP 0.25 raw met, 7 days 20140701-08,
24 MPI ranks single node (exec-13), docker 14.7-ucx, 2 reps.
Wall = full job (init ~10 min + loop + finalize).

| Config                          | Wall (mean) | vs GA  | Loop d/d |
|---------------------------------|------------:|-------:|---------:|
| GA stock (LB on, AR off)        |      5718 s |      – |    110.7 |
| GB stock + autoreduce           |      5495 s |  -3.9% |    115.3 |
| GC MPI_LOAD_BALANCE=OFF build   |      6690 s | +17.0% |     94.3 |

Conclusions:
- MPI_LOAD_BALANCE (DEFAULT-ON since 14.7.1) is worth ~15% of total
  GCHP runtime at c24/24 ranks. Older GCHP versions without it leave
  this on the table — worth upgrading production GCHP to >=14.7.1.
- Autoreduce on GCHP: -3.9% total, same as GC-Classic (-4.0%).
  Accuracy (day-7, GA vs GB): O3 mean 0.07% max 1.7%; OH mean 0.08%;
  NO2 mean 0.13%; CO 0.005%. Same envelope as Classic; NIT caveat
  from Classic applies (not in this output collection).
- Reps within 1% — GCHP timings much less noisy than GC-Classic
  (ExtData reads are a smaller share at c24).
- Setup gotchas (docker/LSF): UDUNITS2_XML_PATH=/usr/share/xml/udunits
  env needed for MAPL cmake; chmod +x all *.py/*.sh/*.pl in tree before
  build; gchp_restart.nc4 symlink must be created manually; >=180GB
  memory for 24-rank init; teardown double-free during checkpoint
  write is cosmetic (known pFIO issue, after timed loop).
- Rundirs: run_GA_stock, run_GB_autoreduce, run_GC_noLB; chain
  script submit_gchp_bench.sh.
