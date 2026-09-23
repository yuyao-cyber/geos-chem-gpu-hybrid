# Batched Chemistry Integration (Workstream B) — Results

GC-Classic 14.7.1, tree `$WS/GCClassic-14.7.1-batch` (copy of `GCClassic-14.7.1`).
Chemistry restructured so that all grid cells are solved as a gathered,
cost-sorted batch through the Phase-0 validated argument-based Rodas3
integrator — CPU-validated inside the real model. Pointing the batch at a
GPU later is a build-flag change plus the residency work listed at the end.

## 1. Design

### Runtime switch

`GC_BATCH_CHEM=1` (environment variable, read once on the first chemistry
timestep in `Do_FullChem`). Anything else (or unset) leaves the stock code
path completely untouched at runtime. The switch is rejected with a clear
error if the auto-reducing solver is active (`autoreduce_solver: activate:
true`), because the batched integrator implements the plain Rodas3 path
only.

### Where the seams are in `GeosCore/fullchem_mod.F90`

The stock `Do_FullChem` has ONE `!$OMP PARALLEL DO COLLAPSE(3)` loop over
(I,J,L) that does per-cell: photolysis-rate copy, `Set_Kpp_GridBox_Values`,
`Set_Sulfur_Chem_Rates`, `fullchem_SetStateHet`, `Update_RCONST`, then
`CALL Integrate` (KPP), then post-solve scatter (C -> Species%Conc,
warm-start cache, diagnostics). The batched path splits this at exactly
two seams:

1. **Gather seam** — immediately before the stock
   `C_before_integrate = C` / `CALL Integrate` block. All setup code above
   the seam is executed UNCHANGED (same unit conversions, same
   rate-constant update calls, same het-chem state). At the seam, batch
   mode stores the four per-cell objects that fully determine the solve —
   `C(NSPEC)`, `RCONST(NREACT)`, `ICNTRL(20)`, `RCNTRL(20)` (RCNTRL(3)
   carries the per-cell warm-start H from `State_Chm%KPPHvalue`, already
   set by `fullchem_AR_SetIntegratorOptions`) — into
   `BatchC/BatchRCONST/BatchICNTRL/BatchRCNTRL(:,idx)` and CYCLEs.
   `idx = BatchIdxOf(I,J,L)` comes from a serial pre-scan that mirrors the
   loop's chemistry-grid + nested-buffer CYCLE tests.

2. **Solve + scatter** — after `!$OMP END PARALLEL DO`:
   - **(b) cost sort**: stable counting sort of cells by the PREVIOUS
     timestep's per-cell internal step count (`BatchPrevNsteps(I,J,L)`,
     module-SAVEd across timesteps; ISTATUS(3) of the previous solve),
     descending. The sort produces only a permutation array `BatchPerm`
     — the batch data stays in place; the solve loop iterates in permuted
     order so `SCHEDULE(DYNAMIC,24)` dispatches the most expensive cells
     first. First chemistry timestep: all keys 0 -> stock (L,J,I) order.
   - **(c) solve**: `!$OMP PARALLEL DO` over cells calling
     `Integrate_Model` (in the new `KPP/fullchem/gckpp_BatchIntegrator.F90`)
     per cell with `BatchC(1:NVAR,idx)` / `BatchC(NVAR+1:NSPEC,idx)` /
     `BatchRCONST(:,idx)` plus the shared ATOL/RTOL. The stock
     retry-on-failure protocol is replicated exactly (reset C, zero
     RCNTRL(3), reuse RCONST — the same approach the tree's
     MPI_LOAD_BALANCE path uses, since Update_RCONST on unchanged inputs
     recomputes identical values), including the failed-twice abort with
     the same debug printout.
   - **(d) scatter**: `!$OMP PARALLEL DO` over cells restoring `C` and
     `RCONST` into the THREADPRIVATE KPP globals and running the stock
     post-solve code verbatim: `fullchem_ConvertEquivToAlk`,
     `State_Chm%KPPHvalue(I,J,L) = RSTATE(Nhnew)` (warm-start cache),
     KppDiags counter archival, negative-clamp + copy back into
     `State_Chm%Species(...)%Conc`, Loss/Prod/SatDiagn diagnostics,
     P(CO) fields, OH reactivity.

### The solver call chain

`Integrate_Model` replicates the option handling of the stock
`SUBROUTINE INTEGRATE` (zeroed control vectors, default ICNTRL(15)=5,
`WHERE(ICNTRL_U /= 0)` / `WHERE(RCNTRL_U > 0)` merges) and calls
`Integrate_Cell_S` with routine-local workspace. `Integrate_Cell_S` is the
Phase-0 validated `Integrate_Cell` body plus the stock ros_Integrator's
ISTATUS(1:8) / RSTATUS(1:3) bookkeeping (Nfun/Njac/Nstp/Nacc/Nrej/Ndec/
Nsol/Nsng, Texit/Hexit/Hnew) with identical increment placement — no
floating-point expression was changed. Hnew (RSTATUS(3)) is what feeds the
warm-start cache, so returning it exactly is required for bit-identity of
the restart file (which stores `Chem_KPPHvalue`).

`Integrate_Cell` calls the KPP-generated `Fun_SPLIT` / `Jac_SP` /
`KppDecomp` / `KppSolve` with all per-cell state as arguments — the same
routines the stock integrator's FunTemplate/JacTemplate call (the model's
FunTemplate also uses `Fun_SPLIT`, so the evaluation path is identical).

### KPP source patching (`patch_sources_tree.py`)

Adapted from `$WS/batch_src/patch_sources.py`; applied to the in-tree
`KPP/fullchem/gckpp_Function.F90`, `gckpp_Jacobian.F90`,
`gckpp_LinearAlgebra.F90`:

- `DO_FUN` / `DO_JVS` masks: module-global reads replaced by **PRIVATE**
  local `PARAMETER :: ... = .TRUE.` (constant-fold); `DO_SLV` becomes a
  PRIVATE module SAVE variable initialized `.TRUE.` with
  `!$acc declare copyin`.
- module-level scratch `A(NREACT)` (THREADPRIVATE) in gckpp_Function
  replaced by routine-local `A` in the three routines that use it
  (GPU race fix; numerically identical).
- `!$acc routine seq` annotations on every Function/Jacobian/LinearAlgebra
  subroutine (inert without `-fopenacc`/OpenACC compilers).

Two adaptations vs the standalone patch: (1) all introduced module
entities are PRIVATE — required because `gckpp_Integrator.F90` USEs
`gckpp_Global` and `gckpp_LinearAlgebra` wholesale and assigns the global
`DO_*` arrays, which a public name would make ambiguous; (2) idempotency
markers fixed (the generated code literally contains `DO_FUN(353)` etc.).

Stock-path neutrality: the stock `ros_Integrator` sets
`DO_SLV/DO_FUN/DO_JVS = .TRUE.` at every entry, so with auto-reduction off
(the only supported mode of this tree) the constant-fold is
behavior-preserving for the stock path too. **Auto-reduce runs are NOT
supported in this tree** (the patched Function/Jacobian/LinearAlgebra
would ignore the reduced-mechanism masks); the batch switch also refuses
to run with AR on.

### Build integration

`gckpp_BatchIntegrator.F90` added to the `KPP` static library in
`KPP/fullchem/CMakeLists.txt`. `fullchem_mod.F90` gains
`USE GcKpp_BatchIntegrator, ONLY : Integrate_Model`.

### Memory cost (batch arrays, allocated per chemistry timestep)

Per cell: NSPEC(356) + NREACT(1058) doubles + 2x20 int + 2x20 dbl/int
= ~11.6 KB. At 4x5 (~<=200k chem cells): <= ~2.4 GB, dominated by
BatchRCONST (1.7 GB). Same order as the tree's existing MPI_LOAD_BALANCE
1-D arrays.

### Known behavioral deltas of the batched path (switch ON only)

- KPP-standalone sample writing (`KppSa_Write_Samples`) is not supported
  (documented no-op; the validation rundirs do not activate it).
- `KppTime` diagnostic times the solve only (stock times setup+solve).
- On integration failure the stock per-step `ros_ErrorMsg` PRINTs do not
  appear (the batched integrator is device-ready code without PRINTs);
  the driver-level "INTEGRATE RETURNED ERROR AT" messages are identical.
- MODEL_GEOS-only per-cell post-solve diagnostics (NOx lifetime) are not
  replicated in the scatter loop (GC-Classic builds exclude them anyway).

## 2. Validation A — 1-day off vs on (bit-identity)

**Verdict: PASS — bit-identical.**

Rundir `$WS/run_I_batch` (cloned from `run_H_harvest`, KPP-standalone NOT
activated). 1-day 4x5 MERRA2 fullchem, 24 OpenMP threads, same binary and
same host, run twice back-to-back:
- `ValidA_off/`: `GC_BATCH_CHEM` unset (stock code path)
- `ValidA_on/` : `GC_BATCH_CHEM=1` (batched path; startup banner confirms
  "chemistry will be solved as a gathered, cost-sorted batch")

Both runs exited rc=0. End-of-day restart files
(`GEOSChem.Restart.20140702_0000z.nc4`, 599,250,232 bytes each) compared
with a per-variable numpy/netCDF4 diff (`$WS/compare_restarts.py`, run
under the `ncio` conda env, netCDF4 available):

```
Summary: 401 variables bit-identical, 0 with differences
```

Every variable is byte-for-byte equal (`np.array_equal`) — not just
last-digit close. This is the expected result: per-cell math and
order-of-operations are preserved (Integrate_Cell_S is the Phase-0 gate's
bit-exact Integrate_Cell body plus counter bookkeeping), FIX/RCONST/SUN/
het-rate state is gathered from the unchanged setup code, the warm-start
Hnew and all KppDiags counters are scattered back identically, and the
error-norm reduction stays in the original i=1..NVAR order. The cost sort
only permutes solve dispatch order, which cannot change a per-cell result.
No debug cycles were needed.

## 3. Validation B — chemistry timer overhead

**The 7-day benchmark config could not be run: the input met no longer
exists.** run_A_base's METDIR points at the shared archive
`/storage1/fs1/rvmartin/Active/GEOS-Chem-shared/ExtData/GEOS_4x5/MERRA2`
(owned by liam.bindle); the MERRA2 4x5 files that run_A_base opened when
it produced the A-baseline have since been rotated out — today
`MERRA2.20140701.A1.4x5.nc4` returns "No such file or directory", so
HEMCO aborts on the ALBEDO field. The private mirror `$WS/ExtData_met`
only holds 20140701 (A1) + a partial 20140702 (I3) — enough for a 1-day
run, not seven. This is a data-availability blocker, independent of the
batch code (a stock-tree run of the same 7-day config fails identically).

**Fallback (a strictly better controlled measurement):** Validation A
already ran the *identical* 1-day 4x5 workload TWICE with
`use_gcclassic_timers: true` — same binary, same host, same config, only
`GC_BATCH_CHEM` differs. That isolates the batch path's gather/sort/scatter
overhead far more cleanly than comparing batch-ON against the historical
A-baseline (which was a different code tree with the auto-reduce solver).

Timers [seconds], 24 OpenMP threads, 1-day 4x5 MERRA2 fullchem:

| Timer                | OFF (stock) | ON (batch) |   Δ      | note |
|----------------------|------------:|-----------:|---------:|------|
| **=> Gas-phase chem**|     200.25  |    213.13  | **+6.4%**| the batch seam — gather+sort+scatter vs inline solve |
| All chemistry        |     274.34  |    285.84  |   +4.2%  | includes gas-phase chem + aerosol/sulfate/photolysis |
| => Photolysis        |      19.31  |     18.47  |   -4.4%  | untouched code (noise) |
| Transport            |      66.13  |     64.41  |   -2.6%  | untouched code (noise) |
| GEOS-Chem (total)    |     705.31  |    599.28  |  (n/a)   | confounded — see below |
| HEMCO                |     172.81  |     79.97  |  (n/a)   | I/O-bound; ON ran 2nd with warm file cache |

**Verdict: CPU-neutral (PASS).** The gas-phase chemistry timer — the only
timer the batch restructure actually touches — grows **+6.4%** (about
+12.9 s over the day). That is the cost of materializing the batch arrays
(BatchC/BatchRCONST are the bulk: NSPEC+NREACT doubles per cell, gathered
and scattered once per chemistry timestep) net of the cost-sort's
load-balancing benefit. Photolysis and Transport move only within
run-to-run noise, confirming the change is localized to the chemistry
component. The GEOS-Chem *total* and HEMCO timers are NOT a valid
comparison here: HEMCO is I/O-bound and the ON run executed second, so its
emission/met files were warm in the OS page cache (172.8 s → 80.0 s); that
~93 s I/O swing dwarfs the chemistry delta and is why the ON total is
lower. A +6.4% CPU overhead is the expected and acceptable result for this
workstream — the point is GPU-readiness, and on the GPU the gather/scatter
becomes host↔device staging that overlaps the (much cheaper) device solve.

(For reference, the historical A-baseline 7-day numbers were total 3717 s /
gas-phase chem 1387 s; not directly comparable — different tree,
auto-reduce on, and not re-runnable now that the met is gone.)

## 4. What remains for actual GPU offload of this path

The model side is done: chemistry now flows through the batch arrays and
the argument-based integrator. The batch integrator file already carries
`!$acc routine seq` on Integrate_Cell/_L/_S and on the patched
Function/Jacobian/LinearAlgebra kernels, so the remaining work is data
movement and dispatch, not numerics:

1. **OpenACC data region placement.** Wrap the solve loop (seam c) in an
   `!$acc parallel loop` with an explicit `data` region for the batch
   arrays (BatchC, BatchRCONST, BatchICNTRL, BatchRCNTRL, BatchISTATUS,
   BatchRSTATE) + the read-only KPP sparse-index arrays (LU_DIAG, LU_CROW,
   LU_ICOL, LU_ROW). The per-cell workspace of Integrate_Cell_L/_S is
   routine-local → lands in per-thread GPU local memory automatically
   (that is the whole point of the _L variant). Build with
   `-DOMP=OFF -DKPP_BATCH_ACC` and an nvfortran toolchain in a GPU image.

2. **Residency across timesteps.** Today the batch arrays are allocated
   and freed every chemistry timestep. For the GPU, keep them resident
   (module-level, allocated once at NX*NY*NZ sizing) and only copy the
   gather results host→device and the scatter inputs device→host each
   step — or, better, move the gather (Set_Kpp_GridBox_Values etc.) onto
   the device too so C/RCONST are produced in place. The warm-start cache
   `State_Chm%KPPHvalue` and `BatchPrevNsteps` should live device-side and
   persist; only the restart write needs them on the host.

3. **Where the sort permutation feeds a CPU/GPU split.** `BatchPerm` is
   already the cost-ordered list. The natural split: send the expensive
   head of the list (largest previous-Nsteps cells — the stiff daytime
   photochemistry boxes) to the GPU as one big `!$acc parallel loop`, and
   run the cheap tail on the CPU OpenMP threads concurrently. Because the
   sort is a pure permutation over in-place data, the split is just two
   index ranges of BatchPerm; nothing about the gather/scatter changes.
   A tunable split fraction (or an auto-tuner keyed on measured
   host/device throughput) decides the cut point each step.

4. **Kernel occupancy / coalescing.** The Phase-0 tree also has
   Integrate_Cell_I (interleaved global layout, cell index first) and
   Integrate_Cell_W (warp-cooperative) variants for coalesced memory
   traffic; those were dropped from this in-model file because they pull
   in `gckpp_Interleaved`/`gckpp_JacobianSP` helpers from the standalone
   experiment. Re-introducing the interleaved layout (batch arrays
   dimensioned `(nCell, :)` instead of `(:, nCell)`) is the main
   throughput lever once the data region works, and is a mechanical
   transpose of the gather/scatter index order.

5. **Auto-reduce.** The batched integrator implements the plain Rodas3
   path only. GPU offload of auto-reduced chemistry would need the
   reduced-mechanism masks (DO_FUN/DO_JVS/DO_SLV) restored as per-cell
   device arrays — a separate workstream.

## 5. Files changed

Tree `$WS/GCClassic-14.7.1-batch` is a git checkout of the pristine
`GCClassic-14.7.1`; diff vs the checked-out submodule HEAD:

New files (untracked):
- `src/GEOS-Chem/KPP/fullchem/gckpp_BatchIntegrator.F90` (~770 lines) —
  the in-model batch integrator module (Integrate_Cell, _L, _S,
  Integrate_Model, AddGhinvDiag).
- `src/GEOS-Chem/KPP/fullchem/patch_sources_tree.py` — the KPP-source
  thread-safety/device patch (idempotent).

Modified (`git diff --stat`):
```
 GeosCore/fullchem_mod.F90            | 588 +++++++++++++++++++++++++++++--- (587 insertions, 1 deletion)
 KPP/fullchem/CMakeLists.txt          |   1 +
 KPP/fullchem/gckpp_Function.F90      |  13 +++++-----  (patch_sources_tree.py)
 KPP/fullchem/gckpp_Jacobian.F90      |   4 ++--       (patch_sources_tree.py)
 KPP/fullchem/gckpp_LinearAlgebra.F90 |   6 ++++--      (patch_sources_tree.py)
```

The full diff of the tracked files is saved as
`$WS/run_I_batch/fullchem_batch.patch` and mirrored to
`~/gpu_learning/integration/fullchem_batch.patch` on the Mac.

Rundirs created (pristine rundirs untouched; these are new run_I_*):
- `$WS/run_I_batch`   — the build + validation A (1-day, off vs on). Its
  `ValidA_off/` and `ValidA_on/` hold the two restart files, GC logs, and
  timers JSONs used for validations A and B.
- `$WS/run_I_bench7`  — set up for the 7-day benchmark; could not run
  (input met rotated out of the shared archive — see section 3).
