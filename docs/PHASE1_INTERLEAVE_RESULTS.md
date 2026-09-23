# Phase 1, Experiment 1: Interleaved (coalesced) memory layout — RESULTS

Date: 2026-08-29.  Cluster: WashU compute1 (LSF), GPU = NVIDIA A100-SXM4-80GB
(gmodel-pinned submits).  Solver: KPP Rodas3 batch integrator, GEOS-Chem
fullchem mechanism (NVAR=353, NREACT=1058, LU_NONZERO=5683).

## What was built

Goal: test the canonical KPP-GPU codegen transformation — lay every per-cell
array out as `(nCell, k)` (cell index FIRST) so thread `ic` touching element
`k` accesses an address adjacent to thread `ic+1` (coalesced), instead of the
cell-last layout where adjacent threads are ~2.8 KB apart.

New/changed files (all in `batch_src/`, additive — every existing mode still
builds and runs):

- `interleave_sources.py` (NEW): generates `gckpp_Interleaved.F90` inside
  `KPP-Standalone-batch/` from the patched KPP sources.  Produces `_I`
  variants of the four hot kernels — `Fun_SPLIT_I`, `Jac_SP_I`,
  `KppDecomp_I`, `KppSolve_I` — by mechanical rewrite: each routine gains
  `(ic, nP)` arguments, every per-cell array becomes 2D `(nP, :)`, every
  access `NAME(expr)` -> `NAME(ic, expr)`.  Routine-local scratch (A(1058)
  in Fun_SPLIT, B(1817) in Jac_SP, W(353) in KppDecomp) becomes
  caller-provided global interleaved scratch.  `KppSolve_I` takes an extra
  `noff` (stage offset into Kw).  Per-cell operation ORDER is untouched.
- `gckpp_BatchIntegrator.F90`: added `Integrate_Cell_I` (+`AddGhinvDiag_I`)
  — identical Rodas3 control flow to `Integrate_Cell`, all per-cell
  state/workspace in caller-provided `(nP, :)` global arrays indexed
  `(ic, k)`; scalars stay in registers; `!$acc routine seq`.
- `kpp_batch.F90`: new modes `cpui` (OpenMP correctness gate) and `gpui`
  (OpenACC benchmark).  Leading dim padded to a multiple of 128
  (20,160 -> 20,224).  Transposition into/out of the interleaved layout is
  timed separately and excluded from the solve time (0.1 s at 20k cells).
- `build_batch.sh`: `prep` (login node, python3) now also runs
  `interleave_sources.py`; compile lists include `gckpp_Interleaved`.

## Validation

- **CPU bit-exactness gate (mode `cpui`, gfortran -O2, OMP-24, 20,160
  cells): PASSED — 0 of 7,116,480 values differ vs the ORIGINAL integrator;
  max relative diff 0.0.**  The layout transform changes no math.
- GPU `gpui` (20,160 cells): 0 solver failures (all IERR=1); max relative
  diff vs CPU serial = 1.52e-6 (gpul in the same job: 1.05e-6).  Same
  order as the established FMA-rounding envelope (~1.1e-6); the small
  increase is different FMA contraction in the regenerated kernels, not a
  logic difference (the CPU path is bit-exact).

## Throughput (all on A100-SXM4-80GB)

| mode | layout | cells | cells/s | source |
|---|---|---|---|---|
| cpu1 (serial, gfortran)  | cell-last | 20,160 | ~1,900–2,750 | this + Phase 0 |
| omp (24 threads)         | cell-last | 20,160 | 32,189 | Phase 0 |
| cpui (24 threads)        | interleaved | 20,160 | 2,166 | job 354188 |
| gpu (module workspace)   | cell-last global | 20,160 | 18,440 | Phase 0 |
| gpul (local workspace)   | per-thread local | 20,160 | **14,872** | job 354269 |
| gpui (interleaved)       | cell-first global | 20,160 | **12,114** (default stack) / **12,208** (160 KB stack) | job 354269 |
| gpu (module workspace)   | cell-last global | 201,600 (saturated) | 22,557 | Phase 0 |
| gpul (local workspace)   | per-thread local | 201,600 (saturated) | 29,181 (29,513 w/ input-local copies) | Phase 0 plateau |
| gpui (interleaved)       | cell-first global | 201,600 (saturated) | **24,068** | job 355457 |

gpul and gpui numbers at 20,160 cells come from the SAME job on the SAME
A100, so the comparison is clean: **gpui is ~18% SLOWER than gpul at 20k
cells.**  Stack size is irrelevant for gpui (workspace is global), as
expected.

## Interpretation

**Coalescing did NOT break the plateau.**  The explicitly interleaved
(cell-first) global layout is ~18% slower than the routine-local-workspace
variant in BOTH regimes: 12.2k vs 14.9k c/s at 20k cells (undersaturated)
and 24.1k vs ~29.2–29.5k c/s at 201,600 cells (saturated).  The 20,160-cell
job was deliberately followed by one saturation run to rule out an
undersaturation false negative; the ranking is unchanged.  (Note gpui
saturated, 24.1k, does beat the module-workspace 'gpu' mode saturated,
22.6k, by ~7% — interleaving IS better than the naive uncoalesced global
layout — it just loses to the compiler/hardware local-memory arrangement
of gpul.)

Why explicit interleaving does not win here:

1. **`Integrate_Cell_L` already gets hardware coalescing.**  Its
   routine-local arrays live in CUDA *local memory*, which the hardware
   stores physically interleaved across the threads of a warp.  So gpul was
   never uncoalesced — the gpu->gpul jump (22.6k -> 29.2k saturated) already
   banked the coalescing win.  Manual `(nCell,k)` layout duplicates that
   mechanism, it does not add to it.
2. **The manual layout has a worse cache footprint.**  Local memory is
   allocated only for RESIDENT threads (~thread footprint x active
   threads), giving a compact, cache-friendly working set.  The interleaved
   global arrays are strided by the full padded pitch (nPad = 20,224
   cells x 8 B = ~158 KB between consecutive elements k of one cell), so
   every non-lockstep access pattern spreads across far more DRAM
   pages/L2 lines.
3. **Warp divergence breaks lockstep coalescing.**  Adjacent cells take
   different step counts and accept/reject paths through Rodas3.  Once the
   threads of a warp are at different statements, `(ic,k)` accesses are no
   longer adjacent-in-k across the warp — the coalescing premise decays as
   integration progresses, while the extra cache pressure (point 2)
   remains.
4. The CPU mirror of the same effect: cpui (2.2k c/s) vs omp (32.2k c/s)
   — interleaving is a 15x LOSS on CPU, where per-cell-contiguous layout is
   exactly what caches want.  Keep cell-last on CPU, always.

## Implications for the next experiment (warp-cooperative work)

The plateau at ~29.5k cells/s (saturated gpul) is NOT a global-memory
coalescing artifact — one-thread-per-cell with per-thread local workspace
is already the best memory arrangement for this mapping.  The remaining
limiters are per-thread serial latency, divergence, and the huge per-cell
state (~100 KB counting Jacobian+LU) vs on-chip memory.  Breaking the
plateau therefore requires changing the WORK mapping, not the data layout:

- warp-cooperative cells (a warp, or a few lanes, per cell) so the
  353-long species loops and the 5,683-entry LU sweeps are parallelized
  within a cell — this shrinks the per-thread state and re-converges
  control flow (all lanes of a warp follow ONE cell's accept/reject path,
  eliminating the divergence tax measured here);
- sorting/binning cells by expected step count (sorted.txt) to reduce
  inter-warp load imbalance is complementary but secondary.

## Reproduction

```
# login node
bash batch_src/build_batch.sh prep
# CPU gate:  bsub ... "docker(billzhuge/geos-chem-deps:14.7-lsf)" bash batch_src/job_cpui.sh
# GPU bench: bsub ... "docker(nvcr.io/nvidia/nvhpc:24.7-devel-cuda12.5-ubuntu22.04)"
#            -gpu "num=1:j_exclusive=yes:gmodel=NVIDIAA100_SXM4_80GB"
#            bash batch_src/job_gpui.sh        # 20k, gpul + gpui same GPU
#            bash batch_src/job_gpui_x10.sh    # 201,600-cell saturation
```
Logs: `logs/cpui_gate.354188.log`, `logs/gpui_20kA.354269.log`,
`logs/gpui_x10.355457.log`.
