# GEOS-Chem chemistry on GPUs: a hybrid CPU + GPU KPP integrator

This repository packages the changes that let **GEOS-Chem Classic 14.7.1**
solve its gas-phase chemistry on NVIDIA GPUs, concurrently with the CPU, using
the model's own KPP-generated Rosenbrock (Rodas3) solver. Nothing in the
chemistry was approximated or re-derived: the GPU runs the same generated
kernels as the CPU, and the CPU code path of the modified model is
**bit-identical** to stock GEOS-Chem.

Work by Yuyao (Bill) Zhuge, Washington University in St. Louis (ACAG),
July-September 2026, on the WashU RIS `compute1` cluster (NVIDIA A100 80 GB).

## Headline results

GEOS-Chem Classic 14.7.1, MERRA-2, full chemistry, 72 levels, 16 OpenMP threads,
one node, same executable for every case. "Gas-phase chem" is the model's own
timer for the KPP chemistry loop; zero solver failures in every run.

| Run | Stock (CPU) | Hybrid, 1 A100 | Hybrid, 2 A100 |
|---|---:|---:|---:|
| 1 day, 4x5: gas-phase chem | 229 s | 119 s (-49%) | 105 s (-55%) |
| 7 days, 4x5: gas-phase chem | 1566 s | 1113 s (-29%)\* | 759 s (-52%) |
| 7 days, 4x5: wall clock | 61 min | 54 min | 46 min |
| 7 days, 2x2.5: gas-phase chem | 6631 s | 3359 s (-49%) | 3122 s (-53%)\* |
| 7 days, 2x2.5: wall clock | 287 min | 226 min | 218 min |

\* degraded by other users' jobs sharing the GPU (see `docs/HYBRID_RESULTS.md`
section 7); the per-device adaptive weighting in the current code addresses the
2-GPU case.

**Correctness.** Stock vs. the modified model's CPU path: all 401 restart
variables byte-identical. CPU vs. GPU: differences are at the level of a
last-bit rounding change and do **not** accumulate. This was established with a
control experiment: a CPU-only build with fused multiply-add enabled in the KPP
library alone (no GPU at all) drifts from stock by exactly the same amount as
the GPU does, species by species and day by day over a 7-day run
(`results/drift_week_clean.txt` vs `results/drift_control_clean.txt`).

## What was changed (and what was not)

The whole model-side change is **7 files** against the `14.7.1` tags of
`geoschem/geos-chem` and `geoschem/GCClassic`; see `patches/` for the exact
diffs and `src/` for the modified files in tree layout.

| File | Change |
|---|---|
| `GeosCore/fullchem_mod.F90` | +1,450 lines. Runtime switch `GC_BATCH_CHEM`: gather cells in cost order, solve the expensive head on the GPU(s) while the OpenMP threads solve the tail in place, adaptive CPU/GPU split, per-device weighting, failure retry, diagnostics. Stock path untouched (bit-identical). |
| `KPP/fullchem/gckpp_BatchIntegrator.F90` | **New**, ~725 lines. The Rodas3 integrator (`ros_Integrator`) transcribed so every per-cell quantity is an argument instead of a module global; `!$acc routine seq`; identical arithmetic (bit-exact on CPU vs the original on 20,160 harvested cells). |
| `KPP/fullchem/gckpp_{Function,Jacobian,LinearAlgebra}.F90` | 23 lines: `!$acc routine seq` on the generated kernels, module scratch array made routine-local (GPU thread-safety), auto-reduce masks constant-folded. Applied by `patch_sources_tree.py`; the KPP generator patch in `kpp-codegen/` emits the same natively. |
| `KPP/fullchem/CMakeLists.txt` | Adds the batch integrator to the KPP library; optional `KPP_FMA` control build. |
| `GCClassic/CMakeLists.txt` | Optional `-DGC_GPU_LIB=<libgcchemgpu.so>`; defines `GC_GPU_CHEM` and links the GPU library. Absent = stock build. |

Not changed: the chemical mechanism, rate laws, photolysis, aerosols, transport,
emissions, tolerances, or the integrator's arithmetic.

## How it works

```
gcclassic (gfortran, OpenMP)                       libgcchemgpu.so (nvfortran, OpenACC)
----------------------------                       ------------------------------------
per chemistry timestep, in fullchem_mod:
  pre-scan: stable counting sort of cells by       gckpp_Function / Jacobian /
            last step's internal step count  --->  LinearAlgebra  (model's own KPP
  main loop visits GPU-head cells first:           output, copied at build time)
    setup (rates, photolysis, het chem) as stock   gckpp_BatchIntegrator: Rodas3,
    head cells: gathered into flat arrays          one GPU thread per cell
    first N threads: launch device 0..N-1 ------>  gc_gpu_integrate_batch_dev(dev, ...)
    other threads: solve tail cells in place            !$acc parallel loop gang vector
  scatter GPU results with the stock post-solve         (blocking; ~35 GB device stack)
  code; adapt GPU work fraction from measured rates
```

* The model stays a **gfortran** build. Only the solver library is built with
  NVIDIA's compiler, and it is reached through `BIND(C)` interfaces; three
  symbol collisions between the GNU and NVIDIA runtimes are documented and
  worked around in `gpulib/src/gc_gpu_devinfo.c` and `fullchem_mod.F90`.
* Runtime switch: `GC_BATCH_CHEM` unset = stock; `1` = CPU batch (bit-identical
  to stock); `2` = GPU only; `3` = hybrid. `GC_GPU_NDEV`, `GC_GPU_FRAC`,
  `GC_GPU_FRAC_FIXED`, `GC_BATCH_SORT` tune the hybrid.
* Cost sorting is what makes the GPU fast (2.5x in the standalone benchmark) and
  is free in the model because the previous step's step count is known.

## Repository layout

```
patches/       exact diffs vs the 14.7.1 tags (geos-chem, GCClassic) and vs KPP 3.3.0
src/           the modified/new files in GEOS-Chem tree layout
gpulib/        the GPU solver library: sources, build/relink/submit scripts, analysis tools
standalone/    the KPP-Standalone batch harness used for the bit-exact gate and GPU benchmarks
kpp-codegen/   (see patches/) the KPP generator "#ACCDIRECTIVES" mode
docs/          full write-ups: HYBRID_RESULTS.md (main), GPU_IN_MODEL_RESULTS.md,
               BATCH_INTEGRATION_RESULTS.md, KPP_GPU_CODEGEN_RESULTS.md, PHASE1_*.md, FINDINGS.md
results/       restart comparisons, drift tables, timer JSON of the reported runs
rundir/        the run configuration used (dates, HISTORY collections, job scripts)
```

## Building and running (compute1 recipe)

1. **GPU library** (NVIDIA HPC SDK 24.7 container):
   `bash gpulib/scripts/build_gpulib.sh` copies the KPP sources out of the model
   tree, compiles them with `-acc -gpu=cc70,cc80`, links `libgcchemgpu.so`, and
   stages the nvhpc runtime `.so` files next to it. ~35 min (the generated
   Jacobian dominates).
2. **Model**: apply `patches/geos-chem-14.7.1_gpu-hybrid-chemistry.patch` to the
   `src/GEOS-Chem` submodule and `patches/GCClassic-14.7.1_gpu-lib-option.patch`
   to GCClassic, then
   `cmake ../CodeDir -DRUNDIR=.. -DGC_GPU_LIB=/path/to/libgcchemgpu.so && make -j install`.
3. **Run**: `GC_BATCH_CHEM=3 GC_GPU_NDEV=2 ./gcclassic` with
   `NV_ACC_CUDA_STACKSIZE=160000` and, on shared nodes, an LSF GPU memory
   reservation (`-gpu "num=2:gmem=45G"`); see `gpulib/scripts/run_week_cases.sh`.
4. **Validate**: `gpulib/analysis/compare_gpu_restarts.py` (per-variable),
   `analyze_gpu_diff.py` (robust percentiles), `cmp_week.py` (per-day drift).
   Judge GPU results against the rounding-only control envelope, not bit-identity.

## Status and next steps

Validated and measured: 1-day and 7-day runs at 4x5 and 2x2.5, 1 and 2 GPUs.
Not yet done: 4-GPU measurement, 24-thread baseline, auto-reduce on the GPU,
GCHP, moving the rate-constant computation onto the device, and a build with a
vendor-neutral offload compiler (the kernels are plain Fortran + OpenACC; the
NVIDIA-specific pieces are confined to `gc_gpu_devinfo.c` and the stack-size
environment variable).

## License

GEOS-Chem-derived files keep the GEOS-Chem (MIT) license; new files are MIT,
see `LICENSE`.
