# GEOS-Chem Classic 14.7.1 with the chemistry solver on an NVIDIA A100 (Workstream C)

Full-model result: **a 1-day 4x5 MERRA2 fullchem simulation whose entire
gas-phase chemistry was solved on an A100 GPU, inside the real model, with
zero solver failures — validated against the CPU path and timed.**

Tree `$WS/GCClassic-14.7.1-batch` (extends workstream B), GPU library
`$WS/gpulib`, rundir `$WS/run_J_gpu`.
`$WS` = `/storage1/fs1/rvmartin/Active/y.zhuge/optimize_gc`.

---

## 1. Headline numbers

| | value |
|---|---|
| Simulation | 1 day, 4x5 MERRA2 fullchem, 2014-07-01, 72 chemistry timesteps |
| Cells solved on the GPU per timestep | **195,408** |
| GPU launches over the day | **72** (one per chemistry timestep) |
| Solver failures (IERR /= 1) | **0** — in all 72 launches, and 0 `INTEGRATE RETURNED ERROR` in every case |
| Device | NVIDIA A100-SXM4-80GB (compute1-exec-401) |
| GPU kernel time, cumulative | **233.8 s** (3.25 s/step) |
| GPU API time incl. transfers, cumulative | **249.4 s** (3.46 s/step) |
| Host↔device transfer overhead | 15.6 s total = **6.2 %** of API time |
| GPU kernel throughput | **60,100 cells/s** |
| 24-thread CPU solve, same host | 134.3 s (1.86 s/step) = 104,800 cells/s |
| **GPU vs CPU solve** | **1.86× slower** (kernel-only 1.74×) |
| Regression: stock vs CPU batch | **401/401 restart variables bit-identical** |
| CPU batch vs GPU | median relative difference 1e-9…5e-6 per species (FMA noise) |

The GPU result is within the "parity to 2×" success band, at its slow edge.
Section 6 explains what the remaining factor is and why it is addressable.

---

## 2. Design: a mixed-compiler shared library

GEOS-Chem is not compiled with nvfortran. Only the chemistry kernels are.

```
  gcclassic (gfortran 11.4, deps image /opt/view)
      |
      |  ONE call per chemistry timestep, through a Fortran BIND(C) INTERFACE
      |  (plain arrays only — no .mod files cross the compiler boundary)
      v
  libgcchemgpu.so (nvfortran 24.7, OpenACC, -acc -gpu=cc70,cc80,cuda12.5)
      |
      +-- gckpp_Precision / Parameters / JacobianSP / Function / Jacobian /
      |   LinearAlgebra          <- copied verbatim FROM THE MODEL TREE
      +-- gckpp_BatchIntegrator  <- workstream B's in-model integrator
      +-- gc_gpu_batch_api.F90   <- !$acc data + !$acc parallel loop over cells
      +-- gc_gpu_devinfo.c       <- device query via the CUDA driver API
```

The KPP sources are copied out of
`$WS/GCClassic-14.7.1-batch/src/GEOS-Chem/KPP/fullchem` at build time and
their md5sums are recorded in `$WS/gpulib/build/SOURCES.md5`, so the
mechanism running on the GPU is provably the model's own mechanism, not a
separately maintained copy.

### The API

```fortran
gc_gpu_integrate_batch( nCell, Tstart, Tend, C, RCONST, ATOL, RTOL,
                        ICNTRL, RCNTRL, Perm,          &  ! in
                        ISTAT, RSTAT, IERR,            &  ! out
                        tTotal, tKernel )                 ! out (timing)
```

* `C(NSPEC,nCell)` — VAR in `1:NVAR`, FIX in `NVAR+1:NSPEC`, in/out.
* `RCNTRL(3,:)` carries the per-cell warm-start H from `State_Chm%KPPHvalue`,
  exactly as the CPU path passes it.
* `Perm(nCell)` — the cost-sort permutation. Cell `ic` of the launch solves
  column `Perm(ic)`, so the sorted dispatch order is honoured **without**
  repacking the host arrays into a second buffer.
* `ISTAT(1:8)` = Nfun, Njac, Nstp, Nacc, Nrej, Ndec, Nsol, Nsng and
  `RSTAT(1:3)` = Texit, Hexit, Hnew — precisely the entries the model's
  scatter consumes (`ISTATUS(1:8)` for KppDiags, `RSTATE(3)` = Hnew for the
  warm-start cache written to the restart file).

Inside, one `!$acc data` region (copyin/copy/copyout) wraps one
`!$acc parallel loop gang vector` whose body calls `Integrate_Cell_S` with
**routine-local** workspace and routine-local copies of the per-cell state —
the `Integrate_Cell_L` pattern that the Phase-1 standalone benchmark showed
is what makes the kernel fast (per-thread local memory is hardware
interleaved, so accesses coalesce without transposing the generated kernels).
nvfortran maps it to `!$acc loop gang, vector(128)`.

`gc_gpu_selftest` solves one dummy cell and `gc_gpu_info` returns the device
count and name, so the link and the device can be verified in isolation from
the model.

### Model side: `GC_BATCH_CHEM=2`

Same gather → cost-sort → scatter as workstream B's `=1`; only the solve loop
changes. The whole GPU path is inside `#ifdef GC_GPU_CHEM`, defined only when
CMake is configured with `-DGC_GPU_LIB=<path>`; **with the option absent the
tree builds and behaves exactly as it did after workstream B**, and the
`GC_BATCH_CHEM` switch-off path is untouched (proved in section 4).

* `=2` without a GPU-configured executable → clean `GC_Error`, not a crash.
* At the first chemistry timestep the model calls `gc_gpu_info` +
  `gc_gpu_selftest` and prints the device and `GPU selftest OK` before any
  chemistry runs; a failure aborts with a clear message.
* The retry-on-failure protocol is replicated exactly: failed cells are
  gathered into a small second batch, `RCNTRL(3)` zeroed, C reset to its
  pre-solve values, re-launched; counters accumulate over both attempts and a
  second failure aborts with the same debug printout as stock.
* `GC_BATCH_SORT=0` (new) disables the cost sort for either batch mode.
* Diagnostic line per timestep: cell count, failures, API/kernel/wall seconds
  and running totals.

---

## 3. Runtime environment: which option worked

**Option (a) worked — the model runs in the deps image on a GPU host, and
LSF injects the NVIDIA driver libraries.** No `/opt/view` copy-out and no
nvhpc image at run time was needed.

Probed inside `billzhuge/geos-chem-deps:14.7-lsf` on a GPU host: `libcuda.so.1`
→ `libcuda.so.550.54.15`, `libnvidia-ml.so.1`, `libnvidia-ptxjitcompiler`,
`libnvidia-nvvm`, a working `nvidia-smi`, and `/dev/nvidia*` present. The
eight nvhpc runtime `.so`s the library itself needs (libacchost, libaccdevice,
libaccdevaux, libcudadevice, libnvf, libnvomp, libnvcpumath, libnvc) were
copied once out of the nvhpc image into `$WS/gpulib/nvrt/` and are found via
RPATH.

### The one non-obvious requirement: do NOT request an exclusive GPU

| LSF `-gpu` request | result |
|---|---|
| `num=1:j_exclusive=yes:gmodel=NVIDIAA100_SXM4_80GB` | `cuInit` → **error 3, CUDA_ERROR_NOT_INITIALIZED** |
| `num=1:j_exclusive=no:gmodel=NVIDIAA100_SXM4_80GB` | `cuInit` rc=0, devices visible, kernels run |

This was verified with a bare C program calling the CUDA driver API through
`dlopen` (`$WS/gpulib/src/cuinit_test.c`), so it is a property of the LSF/driver
environment, not of our library: with `j_exclusive=yes` the container gets the
device files but the driver refuses to initialise. Every GPU job here therefore
uses `j_exclusive=no`. The working submission is in `$WS/gpulib/submit.sh`
(`run_shared` / `linktest_shared`).

---

## 4. The gfortran-main + nvfortran-library link (4 debug cycles)

Both runtimes export the same public symbol names; the dynamic linker binds
them to whichever object comes first, which is libgomp. Two real collisions
had to be resolved:

1. **`libgomp: TODO`, immediate abort.** The NVIDIA OpenACC runtime
   (libacchost) references `acc_register_library` **weakly** and calls it at
   init if anything defines it. libgomp exports a stub of that name that
   aborts. **Fix:** define a no-op `acc_register_library` as a module
   procedure of `fullchem_mod` (inside `#ifdef GC_GPU_CHEM`) and link with
   `-Wl,--export-dynamic-symbol=acc_register_library`, so the weak reference
   binds into the executable.
2. **`libgomp: no device found`.** `gc_gpu_info` originally used the OpenACC
   API (`acc_get_num_devices`, `acc_init`, `acc_get_property_string`); those
   are also exported by libgomp and bound there. **Fix:** implement the device
   query in C over the CUDA driver API via `dlopen("libcuda.so.1")`
   (`gc_gpu_devinfo.c`). The compute kernels never had this problem —
   nvfortran lowers `!$acc` regions to private `__pgi_uacc_*` entry points
   that libgomp does not define.

Two mechanical problems: `dp`/`c_char` not in scope in `fullchem_mod`
(fixed with explicit `REAL(KIND=8)` and a `USE ISO_C_BINDING` import), and
CMake stripping the RPATH at install time (fixed with `CMAKE_INSTALL_RPATH`).

**Rejected approach**, recorded because it looks attractive: loading the
library with `dlopen(..., RTLD_DEEPBIND)` through a C shim
(`$WS/gpulib/src/gc_gpu_shim.c`, kept for the record). DEEPBIND does make the
library prefer its own symbols, but it also puts `libcuda` in a private
namespace and `cuInit` then fails with `CUDA_ERROR_NOT_INITIALIZED`. It also
cannot fix problem 1, which is a *weak* reference resolved outward. The direct
link plus the two fixes above is correct and simpler.

Verified in isolation before touching the model
(`$WS/gpulib/test_link_job.sh`, log `logs/gpu_linktest.584292.log`):

```
gc_gpu_info: nDev=1 name=NVIDIA A100-SXM4-80GB
gc_gpu_selftest: rc=1 nsteps=12 t=  3.2400
GPU selftest OK
omp_then_gpu: n=2000 rc=1 nsteps=12 max_tid=7
OMP+GPU coexistence OK          <- 8 OpenMP threads alive before and after the GPU call
```

`LD_DEBUG=bindings` confirms the only symbol the executable resolves into the
library is `gc_gpu_*`, and the only symbols the nvhpc libraries resolve into
libgomp are the inert `acc_prof_register` / `acc_prof_unregister`.

---

## 5. Validation

Three 1-day runs, **same binary, same host (compute1-exec-401), same rundir**,
back to back: `case_gpu` (`=2`), `case_off` (unset), `case_cpu` (`=1`).
All exited rc=0. Restarts compared per variable with netCDF4/numpy.

### 5a. Regression — the CPU paths are untouched

```
case_off  vs  case_cpu :  Summary: 401 variables bit-identical, 0 with differences
```

Adding the GPU path changed nothing about the existing code: stock and CPU
batch still agree bit-for-bit, reproducing workstream B's result with the new
binary.

### 5b. CPU batch vs GPU — 21 variables bit-identical, 380 differ

Differences are at the floating-point-contraction level. The A100 kernel
contracts `a*b+c` into FMA where the CPU does not, so individual arithmetic
differs in the last bits; a day of stiff photochemistry amplifies that.

Relative difference restricted to cells carrying a meaningful concentration
(≥ 1e-6 of the species' global mean) — the raw whole-field maximum is
meaningless because it is dominated by cells where one value is ~1e-24 and the
other is ~0:

| species | global-mean rel | median rel | p99 | p99.9 | cells |
|---|---|---|---|---|---|
| OH   | 9.4e-07 | 1.1e-06 | 1.7e-03 | 9.7e-03 | 162,252 |
| HO2  | 5.9e-07 | 6.5e-07 | 1.6e-03 | 8.5e-03 | 191,386 |
| O3   | **1.4e-08** | 4.8e-09 | 5.3e-06 | 2.2e-05 | 238,464 |
| NO   | 1.7e-07 | 4.4e-07 | 6.5e-03 | 3.6e-02 | 171,556 |
| NO2  | 1.4e-07 | 4.5e-07 | 7.4e-04 | 3.8e-03 | 234,046 |
| CO   | **7.4e-09** | 5.0e-10 | 3.3e-06 | 3.1e-05 | 238,464 |
| HNO3 | 1.2e-06 | 9.7e-08 | 6.3e-03 | 8.0e-02 | 227,390 |
| ISOP | 1.7e-06 | 5.0e-06 | 1.5e-03 | 8.0e-03 | 72,422 |
| CH2O | 1.8e-07 | 6.7e-08 | 2.5e-04 | 2.1e-03 | 238,464 |
| SO4  | 1.1e-08 | 8.9e-10 | 1.3e-05 | 5.6e-05 | 238,464 |
| PAN  | 2.6e-08 | 2.0e-08 | 5.0e-05 | 2.4e-04 | 207,525 |

Transport-only / aerosol tracers, which the solver never touches directly:

| species | global-mean rel | median rel | max rel |
|---|---|---|---|
| BCPI | 0.0 | 1.5e-15 | 5.1e-12 |
| OCPI | 3.2e-16 | 1.5e-15 | 1.4e-12 |
| BCPO | 0.0 | 6.3e-16 | 1.6e-11 |
| SALA | 1.2e-16 | 1.2e-15 | 5.9e-10 |
| SALC | 0.0 | 1.0e-15 | 1.4e-08 |

These are at double-precision roundoff (1e-15) in the median, i.e. effectively
identical. They are not *exactly* identical because heterogeneous and sulfate
chemistry read gas-phase concentrations, so the solver's last-bit differences
leak into aerosol partitioning — a real physical coupling, not a bug.

**Honest caveat on the tails.** A handful of species have a p99.9 in the
percent range (HNO3 8e-2, NO 3.6e-2). These are stiff, near-bistable
partitioning systems (HNO3/NIT, NO/NO2 at low light) where a last-bit
difference can flip a cell across a partitioning threshold. Global means stay
at 1e-6, so integrated mass is preserved. Worst absolute difference anywhere in
the HNO3 field is 7.9e-10 against a field peak of 1.6e-08.

`Chem_KPPHvalue` (the warm-start step size) differs by up to 1200 s against a
field mean of 840 s. That is expected and harmless: H is a step-size heuristic,
not a conserved quantity — a different rounding leads the controller to accept a
different next step. Every cell still converged.

**Zero solver failures.** 72/72 launches reported `nFail=0`; no case logged a
single `INTEGRATE RETURNED ERROR`. The retry path was therefore never exercised
in this run (it is implemented and compiled, but untested against a real
failure — noted as a caveat).

---

## 6. Timing

Same host, same binary, 24 OpenMP threads, A100-SXM4-80GB. GC-Classic timers,
seconds:

| timer | off (stock) | =1 (CPU batch) | =2 (GPU) |
|---|---:|---:|---:|
| **=> Gas-phase chem** | **153.88** | **157.50** | **272.62** |
| All chemistry | 204.12 | 209.62 | 327.25 |
| => Photolysis | 10.50 | 10.00 | 10.62 |
| => Aerosol chem | 30.88 | 30.12 | 28.50 |
| Transport | 33.25 | 32.50 | 34.62 |
| Convection | 28.00 | 25.75 | 29.25 |
| Diagnostics | 40.75 | 55.62 | 48.12 |
| HEMCO | 362.00 | 405.00 | 607.12 |
| GEOS-Chem (total) | 757.00 | 826.38 | 1155.88 |
| wall clock | 12m38s | 13m47s | 19m18s |

GPU-internal instrumentation (timed around the API call, 72 launches):

| quantity | value |
|---|---:|
| cumulative API time (transfers + kernel) | 249.38 s |
| cumulative kernel time | 233.82 s |
| implied host↔device transfer | 15.56 s (6.2 %) |
| per timestep: API / kernel | 3.46 s / 3.25 s |
| cells per timestep | 195,408 |
| kernel throughput | 60,100 cells/s |

**Reading the numbers.** Gas-phase chem is the only timer the change touches,
and it is the honest comparison. Of the GPU case's 272.6 s, 249.4 s is the API
call, leaving 23.2 s for the setup/gather/sort/scatter that every batch run
pays. Subtracting the same 23.2 s from the CPU batch's 157.5 s puts the
24-thread CPU solve at **134.3 s** against the GPU's **249.4 s** — the GPU is
**1.86× slower** than 24 EPYC 7513 cores (1.74× counting kernel only).

Do **not** read the totals or HEMCO as a GPU effect: the GPU case ran first
with a cold file cache (HEMCO 607 s vs 362/405 s for the later runs); that
~200 s I/O swing, not chemistry, is most of the wall-clock gap.

**Consistency with the standalone benchmark.** 60,100 cells/s sits between the
Phase-1 standalone's ~29k cells/s (random order) and ~74k cells/s
(perfectly cost-sorted), which is what a real model batch should look like: the
sort key is the *previous* timestep's step count, so ordering is good but not
oracle-perfect.

**Where the 1.86× goes.** Transfers are only 6.2 %, so this is not a PCIe
problem — the kernel itself is the cost. The batch is one flat
`gang vector(128)` loop of 195k independent stiff ODE solves, each carrying
~1,400 doubles of routine-local workspace, so occupancy is limited by local
memory and the warps diverge whenever neighbouring cells need different step
counts. The levers, in order of expected value, are listed in section 8.

### Effect of the cost sort — NOT MEASURED (run still queued)

A fourth 1-day run with `GC_BATCH_SORT=0` (`=2`, identity permutation, stock
L,J,I dispatch order) was submitted as LSF job **591354** to quantify the sort's
effect on the GPU. It was still PENDING after 37 minutes — the cluster's 80 GB
A100 nodes were saturated, and a 24-slot + GPU request does not schedule
quickly. **This number is therefore missing from this report.**

When the job runs it writes to `$WS/run_J_gpu/case_gpu_nosort/` and
`$WS/logs/run_J_nosort.591354.log`; the comparison is then

```
grep "GC_GPU_CHEM: call" $WS/run_J_gpu/case_gpu_nosort/GC.log | tail -1
python3 $WS/gpulib/collect_timers.py gpu gpu_nosort
```

against the sorted case's 249.38 s cumulative API / 233.82 s kernel. The
standalone Phase-1 benchmark measured ~29k cells/s unsorted vs ~74k cells/s
cost-sorted (2.5×), so a substantial regression is expected; the in-model
number is what would confirm it. Nothing else in this report depends on it.

---

## 7. Files

| path | what |
|---|---|
| `$WS/gpulib/src/gc_gpu_batch_api.F90` | the OpenACC batch entry point + selftest |
| `$WS/gpulib/src/gc_gpu_devinfo.c` | CUDA-driver-API device query (avoids libgomp) |
| `$WS/gpulib/src/gc_gpu_shim.c` | rejected RTLD_DEEPBIND loader, kept for the record |
| `$WS/gpulib/src/test_link.F90`, `cuinit_test.c` | isolation tests |
| `$WS/gpulib/build_gpulib.sh` | full nvfortran build (~35 min; Jacobian alone 22 min) |
| `$WS/gpulib/relink_gpulib.sh` | fast relink reusing the KPP objects (~25 s) |
| `$WS/gpulib/build_model_job.sh`, `submit.sh` | model build + all LSF submissions |
| `$WS/gpulib/run_J_cases.sh` | the three (four) validation runs |
| `$WS/gpulib/compare_gpu_restarts.py`, `analyze_gpu_diff.py`, `collect_timers.py` | analysis |
| `$WS/gpulib/libgcchemgpu.so`, `nvrt/` | the library (23 MB) + 8 nvhpc runtime libs |
| `$WS/GCClassic-14.7.1-batch/gpu_in_model.patch` | model-side diff vs workstream B |
| `$WS/run_J_gpu/case_{off,cpu,gpu}/` | logs, restarts, timers of each case |

Model-side diff is confined to two files:
`src/GEOS-Chem/GeosCore/fullchem_mod.F90` (the `=2` path, the selftest, the
`GC_BATCH_SORT` switch, the `acc_register_library` hook) and `CMakeLists.txt`
(the `GC_GPU_LIB` option). `KPP/fullchem/CMakeLists.txt` and
`CMakeScripts/GC-ConfigureClassic.cmake` are unchanged.

---

## 8. Caveats and next steps

**Caveats.**
* Data is staged host→device **every timestep**; nothing is resident. At 195k
  cells that is ~2.4 GB of RCONST + ~0.6 GB of C per step. It costs only 6.2 %
  today because the kernel is slow — it becomes the bottleneck the moment the
  kernel gets faster.
* The GPU solve is launched serially from the master thread; the 24 OpenMP
  threads idle during it. There is no CPU/GPU overlap yet.
* The GPU retry path is implemented but was never exercised (no failures
  occurred). It is untested against a real failure.
* `KppTime` on the GPU path is the launch wall time divided evenly over the
  cells (there is no per-cell device timer).
* Auto-reduce is unsupported on both batch paths, as in workstream B.
* One run per configuration; timer differences of a few seconds are noise.
* `j_exclusive=no` is mandatory on this cluster (section 3).

**Next steps, highest value first.**
1. **Residency.** Keep the batch arrays in device memory across timesteps and
   move `Update_RCONST` onto the device, so RCONST is produced in place instead
   of being staged. Removes the 2.4 GB/step transfer and most of the gather.
2. **Occupancy.** ~1,400 doubles of per-thread local workspace is what caps
   occupancy. The Phase-1 `Integrate_Cell_I` (interleaved global layout) and
   `_W` (warp-cooperative) variants exist and target exactly this; re-introducing
   the interleaved layout is a mechanical transpose of the gather/scatter index
   order.
3. **CPU/GPU split.** `BatchPerm` is already the cost-ordered list, so sending
   its expensive head to the GPU and running the cheap tail on the idle OpenMP
   threads concurrently is two index ranges — the natural next experiment, and
   it should beat both pure paths.
4. **Divergence.** Bucket cells by predicted step count so a warp's 32 lanes
   take a similar number of internal steps.
