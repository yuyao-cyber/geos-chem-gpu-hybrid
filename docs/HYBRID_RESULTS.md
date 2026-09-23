# GPU chemistry in GEOS-Chem Classic 14.7.1: correctness closure + hybrid CPU/GPU + multi-GPU (Workstream D)

Date: 2026-09-09.  Tree `$WS/GCClassic-14.7.1-hybrid` (copy of the workstream-C
tree; only `GeosCore/fullchem_mod.F90` changed), library `$WS/gpulib`
(`gc_gpu_batch_api.F90`, `gc_gpu_devinfo.c`), rundirs `$WS/run_K_fma`,
`$WS/run_L_hybrid`, `$WS/run_L_hybrid4`.  `$WS` = /storage1/fs1/rvmartin/Active/y.zhuge/optimize_gc.
All model runs: 1 day 4x5 MERRA2 fullchem (2014-07-01, 72 chemistry steps).

## 1. Is the CPU-vs-GPU discrepancy a bug?  No.

Workstream C left the GPU result agreeing with the CPU in the global mean
(O3 1e-8, OH 1e-6) but with percent-level tails in stiff bistable species
(HNO3 p99.9 8e-2, NO 3.6e-2).  Two experiments settle where that comes from.

**Control: CPU-only, FMA allowed in the KPP library only.**  New CMake option
`KPP_FMA` (KPP/fullchem/CMakeLists.txt, `-mfma -ffp-contract=fast` on target
KPP only; default OFF).  Build + run in `run_K_fma` (GC_BATCH_CHEM=1), compared
with `run_J_gpu/case_cpu` (same code, no FMA).  Nothing but last-bit rounding
inside the solver differs between these two CPU runs.

| species | CPU-FMA vs CPU: gmean / p99 / p99.9 | GPU vs CPU (workstream C): gmean / p99 / p99.9 |
|---|---|---|
| OH   | 5.0e-7 / 1.8e-3 / 9.5e-3 | 9.4e-7 / 1.7e-3 / 9.7e-3 |
| O3   | 1.1e-8 / 5.2e-6 / 2.1e-5 | 1.4e-8 / 5.3e-6 / 2.2e-5 |
| NO   | 8.0e-8 / 6.0e-3 / 3.6e-2 | 1.7e-7 / 6.5e-3 / 3.6e-2 |
| HNO3 | 6.8e-6 / 6.2e-3 / 9.2e-2 | 1.2e-6 / 6.3e-3 / 8.0e-2 |
| CO   | 2.2e-10 / 3.4e-6 / 3.1e-5 | 7.4e-9 / 3.3e-6 / 3.1e-5 |
| BCPI (transport only) | 0 / 8.6e-15 / 9.1e-14 | 0 / 9.2e-15 / 1.5e-13 |

Same envelope, species by species, including the percent-level tails.  A pure
rounding perturbation of the solver produces exactly the GPU's signature after
one day: the stiff HNO3/NIT and NO/NO2 partitioning flips cells across
thresholds, while the global means stay at 1e-6 and non-chemistry tracers at
1e-15.  (Files: `gpulib/cmp_cpu_vs_cpufma.txt`, `gpulib/diff_cpu_vs_cpufma.txt`.)

**GPU built with `-gpu=nofma`** (`run_J_gpu/case_gpu_nofma`): still the same
envelope vs CPU (OH p99 1.8e-3, O3 gmean 1.5e-9) and 27% slower (315 s vs 249 s
API).  The GPU's last-bit differences are therefore not only FMA contraction
(device libm exp/pow etc. differ in the last bit as well); the CPU-FMA control
shows any such last-bit source gives this envelope.  Do not use nofma.

**Verdict:** the batched GPU solver is correct to rounding.  The tails are the
sensitivity of the chemistry, not of the port.  Any future GPU validation
should be judged against this control envelope, not against bit-identity.

## 2. Hybrid CPU + GPU solve (GC_BATCH_CHEM=3)

Workstream C's GPU-only path was 1.86x SLOWER than 24 CPU threads because the
GPU sat alone while the threads idled.  The hybrid path runs both.

Design (all in `fullchem_mod.F90`):
* The pre-scan now performs the stable counting sort (previous internal step
  count, descending) BEFORE the gather, so the batch arrays are filled directly
  in cost order and the dispatch permutation is the identity.  GPU head and
  CPU tail are then two contiguous index ranges - no repacking, no partial
  copy-out hazard.  Validated: GC_BATCH_CHEM=1 with this change is still
  **401/401 restart variables bit-identical to stock**.
* One OpenMP parallel region: thread 0 (or threads 0..nDev-1) issue the
  blocking GPU launch(es) on the head `[1..nGpu]`; all other threads immediately
  run the tail `[nGpu+1..nCell]` with `!$OMP DO SCHEDULE(DYNAMIC,24)`; GPU
  threads join the tail loop when their launch returns.
* Work model: cost(cell) = max(previous internal steps, 1); prefix sums give
  the cut for a target GPU work fraction; the fraction is re-estimated every
  step from the measured GPU and CPU work rates (EMA, alpha 0.5) so both sides
  finish together.  `GC_GPU_FRAC` initial value, `GC_GPU_FRAC_FIXED=1` freezes.
* GPU-side failures (none occurred) are retried on the CPU with the stock
  protocol.  Diagnostic line per step: `GC_HYB_CHEM: call N nGpu= nCpu= frac=
  ndev= t_gpu= t_cpu= t_solve= ...`.

Results, `run_L_hybrid`, **16 OpenMP threads** (the general queue's A100-80GB
hosts never had 24 free slots), A100-SXM4-80GB, same binary/host/job:

| | CPU batch (=1) | hybrid 1 GPU | hybrid 2 GPUs |
|---|---:|---:|---:|
| Gas-phase chem timer [s] | 262.9 | 167.1, repeat 158.5 (-36..-40%) | 132.8 (-50%) |
| All chemistry [s] | 333.8 | 240.1 | 199.9 |
| concurrent solve, 72 steps [s] | ~230 (est.) | 125.0, repeat 119.9 | 95.0 |
| converged GPU work fraction | - | 0.57 | 0.72 |
| solver failures | 0 | 0 | 0 |
| restart vs CPU | bit-identical | rounding envelope (sec. 1) | rounding envelope |

The split converges within ~5 steps and then tracks: t_gpu ~ t_cpu ~ 1.5 s
(1 GPU) / 1.15 s (2 GPUs) per step, both devices balanced to 3%.  Totals and
HEMCO timers are confounded by file-cache state on a shared node and are not
reported as GPU effects.

## 3. Multi-GPU from a gfortran model

New library entry `gc_gpu_integrate_batch_dev(dev, ...)`: the CALLING HOST
THREAD selects NVIDIA device `dev` and launches on it, so nDev OpenMP threads
drive nDev GPUs concurrently on disjoint contiguous slices (work-balanced by
the same prefix sums).  `GC_GPU_NDEV` caps the count.

Two dead ends, recorded because both look correct:
1. `!$acc set device_num(dev)` (no device type) -> the parallel loop silently
   ran on the HOST (and SIGILLed: nvfortran host code is `-tp` = build host,
   an AVX-512 Xeon, on the EPYC GPU nodes).
2. `!$acc set device_type(nvidia) device_num(dev)` -> `libgomp: no device
   found`: the directive lowers to the PUBLIC `acc_set_device_num`, which the
   gfortran model binds to libgomp - the same collision as workstream C's
   `acc_get_num_devices`.

Fix: `gc_gpu_set_device()` in `gc_gpu_devinfo.c` does
`dlopen("libacchost.so", RTLD_NOLOAD)` + `dlsym("acc_set_device_num")` and
calls NVIDIA's implementation directly with `acc_device_nvidia = 4`.
Functional test (`gpulib/src/mgpu_test.F90`, 100k cells, 2 A100s): 0 failures,
both devices at 100% utilization (nvidia-smi sampled), 1.70x over one device.
`$WS/gpulib/libgcchemgpu.so` is now this library (previous one kept as
`libgcchemgpu_v1.so`); it is built with `-tp=haswell` so its host code runs on
both Intel and AMD nodes (`nvrt/` gained `libnvcpumath-avx2.so`).

## 4. Files

| path | what |
|---|---|
| `GCClassic-14.7.1-hybrid/src/GEOS-Chem/GeosCore/fullchem_mod.F90` | cost-ordered pre-scan, GC_BATCH_CHEM=3, multi-GPU launch |
| `GCClassic-14.7.1-batch/src/GEOS-Chem/KPP/fullchem/CMakeLists.txt` | `KPP_FMA` control option |
| `gpulib/src/gc_gpu_batch_api.F90`, `gc_gpu_devinfo.c` | `_dev` entry point, `gc_gpu_set_device` |
| `gpulib/relink_mgpu.sh`, `relink_nofma.sh`, `mgpu_test_job.sh`, `src/mgpu_test.F90` | library relinks + multi-GPU test |
| `gpulib/cmp_cases.sh`, `timers.py` | restart comparison / timer table helpers |
| `run_K_fma/` | FMA control build + run |
| `run_L_hybrid/case_{cpu,hyb,hyb1,hyb2}`, `run_L_hybrid4/case_hyb4` | hybrid runs (logs, restarts, timers) |

## 5. Next steps
1. 4-GPU run (`run_L_hybrid4`, LSF job 825432, queued since 2026-09-09 16:24 waiting for a general-queue host with 4 free 80GB A100s; do not move it to 40GB cards - the 160 KB/thread CUDA stack reservation is ~35 GB) and a 24-thread
   repeat when a host frees up, for the headline table.
2. Residency: keep RCONST/C device-resident and move Update_RCONST onto the
   device; the gather/scatter + rate-constant setup (~30 s/day) is now a
   visible share of the chemistry timer.
3. Auto-reduce on the GPU (the CPU's -14% chemistry lever is unavailable on
   both batch paths today).
4. Higher resolution (2x2.5 / nested) where chemistry dominates the timer -
   the hybrid gain in TOTAL runtime at 4x5 is capped by HEMCO I/O and
   transport.

## 6. Fused hybrid (v4, 2026-09-10/11): gather only the GPU head, launch from inside the loop

The separate-region hybrid (sec. 2) still gathered and scattered every cell
and only started the GPU after the whole setup loop.  v4 changes the seam:

* The pre-scan plans the GPU head (nGpu, device slices) from the work model
  BEFORE the main loop; only head slots are gathered.  Tail cells fall
  through to the stock in-place solve: no gather, no scatter for ~75% of
  the cells.
* The main loop visits cells through a LoopMap: GPU head first, then the
  CPU tail in cost order, then the non-chemistry cells.  The first nDev
  threads that see the head complete (atomic counter) launch their device
  from inside the loop; the other threads keep solving the tail.  The GPU
  therefore overlaps the CPU setup AND solve.  Stock mode decodes L,J,I
  arithmetically in the same order as the original COLLAPSE(3) loop.
* Diagnostic `t_head` (time until the head is gathered) went from 1.2-1.6 s
  with stock visiting order (v3, GPU serialized after the CPU) to 0.1-0.2 s.

Results, 16 OpenMP threads, A100-SXM4-80GB x2 (exec-394), same binary:

| | stock | CPU batch (=1) | hybrid 1 GPU | hybrid 2 GPUs |
|---|---:|---:|---:|---:|
| Gas-phase chem timer [s] | 229.4 | 233.6 | **119.0 (-49%)** | **104.6 (-55%)** |
| All chemistry [s] | 296.9 | 294.8 | 182.4 | 168.0 |
| wall clock | 9m38 | 9m06 | **7m23** | (I/O-confounded) |
| converged GPU work fraction | - | - | 0.73 | 0.79 |
| solver failures | 0 | 0 | 0 | 0 |

Validation: v4 stock vs original stock **401/401 bit-identical**; v4 stock
vs CPU batch **401/401 bit-identical**; CPU vs 2-GPU hybrid within the
rounding envelope of sec. 1 (OH gmean 1.3e-7, p99 1.6e-3).
Files: gpulib/cmp_v4off_vs_off.txt, cmp_v4off_vs_cpu.txt, cmp_v4cpu_vs_hyb2.txt.

The 4-GPU run (`run_L_hybrid4`, job 825432) has been queued >30 h waiting
for a general-queue host with four free 80 GB A100s.

## 7. Week-long tests (2026-09-22/23): 4x5 and 2x2.5, 7 days, with a rounding-only control

Runs: 2014-07-01..08 (504 chemistry steps), 16 OpenMP threads, A100-SXM4-80GB
on shared general-queue nodes, same v4 binary for all cases, MERRA-2 met for
the week fetched from the public `geos-chem` S3 bucket (the shared archive no
longer holds it).  2x2.5 starts from the 4x5 restart regridded by HEMCO
(no 2x2.5 full-chemistry restart exists).  Zero solver failures in all runs.

### Timing

| | stock | hybrid 1 GPU | hybrid 2 GPUs |
|---|---:|---:|---:|
| **4x5** gas-phase chem [s] | 1566 | 1113 (-29%)* | 759 (-52%) |
| 4x5 all chemistry [s] | 2008 | 1558 | 1174 |
| 4x5 wall clock | 61 min | 54 min | 46 min |
| **2x2.5** gas-phase chem [s] (773k cells/step) | 6631 | 3359 (-49%) | 3122 (-53%)** |
| 2x2.5 all chemistry [s] | 8611 | 5375 | 5122 |
| 2x2.5 wall clock | 287 min | 226 min | 218 min |

\* slower than the 1-day result (159 vs 119 s/day): another user's job was
computing on the same shared device.  \*\* the second GPU added only 7%
because the two devices got equal work but one was 60% slower (shared);
fixed by v5 below.

### Correctness: drift over 7 days vs the rounding-only control

Per-day restart differences (global-mean relative difference / 99th
percentile over cells >= 1e-6 of the species mean), stock vs hybrid, and the
CONTROL = stock vs the CPU-only build with FMA enabled in the KPP library
(pure last-bit rounding change, no GPU).  4x5, day 7 (2014-07-08):

| species | hybrid 1 GPU | hybrid 2 GPUs | control (rounding only) |
|---|---|---|---|
| O3   | 5.2e-7 / 3.6e-5 | 5.1e-7 / 3.4e-5 | 5.3e-7 / 3.4e-5 |
| OH   | 6.3e-6 / 2.4e-3 | 4.9e-6 / 2.4e-3 | 6.2e-6 / 2.4e-3 |
| NO2  | 7.2e-6 / 1.8e-3 | 8.0e-6 / 1.8e-3 | 7.4e-6 / 1.8e-3 |
| CO   | 1.6e-6 / 3.0e-5 | 1.6e-6 / 2.9e-5 | 1.6e-6 / 3.0e-5 |
| HNO3 | 1.3e-4 / 1.4e-2 | 1.2e-4 / 1.3e-2 | 1.3e-4 / 1.4e-2 |
| SO4  | 1.4e-6 / 3.4e-5 | 1.3e-6 / 3.3e-5 | 5.2e-6 / 3.8e-5 |
| BCPI (transport only) | 5.4e-15 | 5.4e-15 | 5.9e-15 |

The hybrid drift is indistinguishable from the control on every day and
every species.  The drift SATURATES instead of growing: O3 global-mean
difference 1.2e-7, 2.8e-7, 3.7e-7, 4.4e-7, 4.8e-7, 4.8e-7, 5.2e-7 on days
1..7; HNO3 p99 flat at ~1.3e-2 from day 3; OH p99 flat at ~2.4e-3 from day 2.
The 2x2.5 hybrids show the same shape, slightly larger (O3 5-7e-7, HNO3 p99
1.7e-2 at day 7).  Full tables: `gpulib/drift_week.txt`,
`gpulib/drift_control.txt`.

**Conclusion:** GPU/hybrid results differ from CPU by a bounded,
non-accumulating rounding-level divergence identical to that of any
last-bit perturbation of the solver.

### v5: per-device adaptive work weighting

Each device's measured work rate (EMA) now sets its slice of the GPU head
(equal shares until every device has a measurement).  1-day, 2 GPUs: the two
devices finished within 1% of each other (1.087 / 1.076 s) with 28,493 vs
41,471 cells - i.e. one device was ~45% slower and got correspondingly less
work.  Gas-phase chem 101.6 s (v4: 104.6).  CPU path still 401/401
bit-identical; hybrid in the rounding envelope.  v5 is the current binary
(`run_L_hybrid/gcclassic`; v4 kept as `gcclassic.hyb_v4`).

### Operational lessons
* Shared GPU nodes: request `gmem=45G` in the `-gpu` spec (the per-thread
  solver stack reserves ~35 GB); otherwise another job's memory use causes
  `CUDA_ERROR_OUT_OF_MEMORY` at the first launch.
* Forcing more work onto the GPUs than the adaptive split chooses is slower
  (2 GPUs: 95% -> 181 s, 85% -> 123 s, adaptive 79% -> 105 s per day).
