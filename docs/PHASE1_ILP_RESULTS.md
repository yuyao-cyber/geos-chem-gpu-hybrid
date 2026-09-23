# Phase 1, Experiment 3: Per-thread ILP — NC cells in flight per GPU thread — RESULTS

Date: 2026-08-31.  Cluster: WashU compute1 (LSF).  GPU = NVIDIA
A100-SXM4-80GB (gmodel-pinned submits).  Solver: KPP Rodas3 batch
integrator, GEOS-Chem fullchem (NVAR=353, NREACT=1058, LU_NONZERO=5683).

**Verdict: NEGATIVE.  NC=2 is 0.58x of gpul (0.60x with cost-sorted
pairing).  NC=4 could not even be brought to the GPU: nvfortran did not
finish compiling the generated module within 7.2 hours, and the measured
NC=2 frame size shows the NC=4 stack reservation would exceed the A100's
80 GB anyway.  Third clean negative of Phase 1 — the structural
conclusion is now airtight.**

Side finding (measured, incidental, positive): simply ordering the gpul
cell loop by per-cell cost was worth **+33%** at 20,160 cells
(27,338 vs 20,538 c/s, same job, same GPU).  See "GPUL sorted".

## Hypothesis under test

Exp-2 showed that SPLITTING one cell across L lanes always loses (best
case parity by the `1/(f*L+(1-f))` argument; measured 0.45x).  The
mirror image: give each thread MORE independent work.  One thread
carries NC = 2 or 4 cells with their instruction streams interleaved, so
the in-order GPU core can overlap one cell's dependency stalls with
another cell's arithmetic.  No cross-thread synchronisation, so none of
Exp-2's work-efficiency penalty applies.

Two costs are paid instead, and both were measured:
1. **lockstep waste** — cells sharing a thread take Rosenbrock steps
   together; a group costs the `max` of its members' step counts, not
   the mean.  Mitigated by pairing cells of similar cost (both pairings
   measured; the delta is a number below).
2. **per-thread state x NC** — the local frame doubles and, more
   importantly, the live intermediate values double, and they no longer
   fit in registers (spill data below).

## What was built

All changes are additive; every pre-existing mode (`check`, `cpu1`,
`omp`, `gpu`, `gpul`, `cpui`, `gpui`, `cpuw`, `gpuw`) still builds and
runs.

- `batch_src/multicell_sources.py` (new) — generates
  `gckpp_MultiCell2.F90` / `gckpp_MultiCell4.F90`: multi-cell variants of
  the four hot kernels (`Fun_SPLIT_M`, `Jac_SP_M`, `KppDecomp_M`,
  `KppSolve_M`).  Mechanical transformation: every per-cell array
  `NAME(expr)` -> `NAME(mc, expr)` (cell index FIRST), and **every
  executable statement is emitted NC times, once per cell, with the NC
  copies ADJACENT in program order** — an explicit unroll that hands the
  scheduler NC mutually independent instances of every instruction.
  This is the whole mechanism: calling a scalar kernel NC times
  back-to-back gives zero overlap on an in-order GPU core, so the
  interleave has to happen at statement level inside the kernels.
  Logical statements interleaved: Fun_SPLIT 1,765, Jac_SP 7,488,
  KppSolve 505.  `KppDecomp_M` comes from a hand-written template with
  the identical loop structure and the NC cells unrolled at the
  innermost statement level (row gather / pivot divide / sub-diagonal
  update / row scatter each appear NC times in a row).  Module sizes:
  NC=2 = 24,576 lines (1.39 MB), NC=4 = 49,054 lines (2.77 MB).
- `batch_src/multicell_integrator.inc` (new) — `Integrate_Cell_M`,
  INCLUDEd by each generated module so NC is a compile-time PARAMETER
  (per the design guidance: no dynamic NC).  Lockstep-step variant: all
  NC cells take stages together, each keeps its OWN T, H, error, and
  accept/reject; finished/failed cells are masked out of the bookkeeping
  while their arithmetic is computed and discarded.
- `batch_src/kpp_batch.F90` — new modes `cpum` (bit-exact gate),
  `gpum` (gpul + gpul-sorted + NC=2 x both pairings, one job, one GPU,
  one `!$acc data` region) and `gpum4` (same for NC=4).
- `batch_src/build_batch.sh` — new knobs `BDIR` (private build dir so
  CPU/GPU builds don't clobber each other's .o/.mod), `WITH_M4`, `FOPT`;
  `-gpu=ptxinfo` now always on for GPU builds.
- `batch_src/job_cpum.sh`, `job_gpum.sh`, `sub_gpum.sh`,
  `job_gpum4.sh`, `sub_gpum4.sh`.

### Bit-exactness argument

For any single cell the sequence of floating-point operations is exactly
the sequence `Integrate_Cell` performs; the NC dimension is only ever an
extra array index, and the error-norm reduction keeps its serial
`i = 1..NVAR` order with one accumulator per cell.  Two places recompute
rather than skip, both value-identical: (1) `Fun_SPLIT_M`/`Jac_SP_M` are
re-evaluated for every lane whenever ANY lane starts a new step — a lane
mid-rejection has an unchanged Y (a rejected step never writes Y), so it
reproduces the bits it already held; (2) the singular-retry loop
refactorizes every lane — a non-singular lane has unchanged Jac0 and H.
`KppDecomp_M` records a singular pivot in `IER(mc)` and continues
instead of returning early; equivalent for the caller, which discards
the factorization on any IER/=0 (no cell in this set is ever singular).

## Validation

**CPU bit-exact gate (mode `cpum`, gfortran -O2, OMP-24, 20,160 cells,
job 406930): PASSED for all four configurations.**  This gated the
FINAL source — the exact `gckpp_MultiCell{2,4}.F90` +
`multicell_integrator.inc` the GPU job then built, no post-gate edits
(the Exp-2 gate caveat is not repeated).

```
Original serial:      9.39 s  (    2147.8 cells/s)
Cost sort: steps min=      1  max=     32  mean=     6.19
cpum M2 identity t=     0.90 s (   22314.1 cells/s)   differing values: 0 of 7116480   maxrel 0.0000E+00
cpum M2 sorted   t=     0.89 s (   22675.8 cells/s)   differing values: 0 of 7116480   maxrel 0.0000E+00
cpum M4 identity t=     2.10 s (    9601.6 cells/s)   differing values: 0 of 7116480   maxrel 0.0000E+00
cpum M4 sorted   t=     0.89 s (   22750.9 cells/s)   differing values: 0 of 7116480   maxrel 0.0000E+00
Cells with IERR/=1: 0
```

GPU (job 406931, 20,160 cells): **0 solver failures** in every mode.
Max rel diff vs the in-job CPU serial reference: gpul = 1.054E-06 (both
orderings — identical to the Exp-1/Exp-2 value), NC=2 = 5.934E-06 (both
pairings).  Same 1e-6 order; the difference against gpul's value is
nvfortran choosing different FMA contractions in the `_M` kernels, and
the bit-exact CPU gate bounds the arithmetic itself.

## Throughput — A100-SXM4-80GB, 20,160 cells, job 406931

All GPU numbers from the SAME job on the SAME GPU, same input state,
state reset and re-uploaded between timed runs.  In-job CPU serial:
2,670 c/s.

| mode | mapping | cell order | cells/s | vs gpul(identity) |
|---|---|---|---|---|
| **gpul** (baseline) | 1 thread = 1 cell | list (random) | **20,538** | 1.00x |
| gpul sorted | 1 thread = 1 cell | cost-sorted | **27,338** | **1.33x** |
| gpum NC=2 | 1 thread = 2 cells, interleaved | random pairs | 11,916 | **0.58x** |
| gpum NC=2 | 1 thread = 2 cells, interleaved | cost-sorted pairs | 16,454 | **0.80x** (0.60x vs gpul sorted) |

- The pairing delta the brief asked for: **cost-sorted pairing is +38%
  over random pairing** for NC=2 (16,454 vs 11,916).  That is the
  measured price of lockstep waste under random grouping at this
  step-count distribution (min 1 / max 32 / mean 6.19).  On the CPU the
  same delta at NC=4 is +137% (22,751 vs 9,602 c/s), consistent with
  E[max of NC] growing with NC.
- The fair like-for-like comparison (both sorted): 16,454 vs 27,338 =
  **0.60x**.  Pairing quality does not rescue the design.
- NC=4 GPU: not measured — see next section.
- The 201,600-cell saturation run was NOT run (protocol: only on a >10%
  win; this is a 40% loss).

Cross-job note: this job's gpul-identity (20,538) is higher than
Exp-2's in-job gpul (13,909) and Phase-0's 14,872 on the same list —
different physical node/GPU day; every conclusion above is drawn within
one job only.

## ptxinfo frame sizes and spills (measured, cc80, logs/build_gpum2.txt)

| kernel | stack frame | spill stores | spill loads |
|---|---|---|---|
| `Integrate_Cell_L` (gpul, NC=1) | 139,288 B | 452 B | 452 B |
| `Integrate_Cell_M` (NC=2) | 284,040 B | 452 B | 452 B |
| `Fun_SPLIT` (scalar) | 27,720 B | 27,916 B | 55,380 B |
| `Fun_SPLIT_M` (NC=2) | 27,808 B | 50,256 B | 119,232 B |
| `Jac_SP` (scalar) | 16,400 B | 32,220 B | 73,788 B |
| `Jac_SP_M` (NC=2) | 32,296 B | 68,708 B | 171,276 B |
| `KppDecomp` (scalar) | 2,856 B | 32 B | 32 B |
| `KppDecomp_M` (NC=2) | 5,688 B | 32 B | 32 B |
| `KppSolve` / `KppSolve_M` | 224 / 232 B | — | — |

ptxas register counts across the device functions run 32-128; the wide
straight-line kernels sit at the high end and spill heavily (above).
`NV_ACC_CUDA_STACKSIZE=365136` was required and worked (the job script
now sizes it from the ptxinfo frames: (284,040+33,008)x1.1+16K); at the
~221,184 pre-reserved thread slots that is a 75 GiB device-side stack
reservation on the 80 GB A100 — NC=2 is already at the ceiling.

## NC=4: infeasible on this stack, two independent ways (job 407492)

1. **Compile time.**  nvfortran 24.7 took 1h51m for the NC=2 module in
   this job, and had not finished the NC=4 module (49,054 lines of
   straight-line code) after **7h13m** on the exec host, at which point
   the job was killed to stop burning a j_exclusive A100.  Compile cost
   is strongly super-linear in NC.  (gfortran compiled the same file in
   ~9 minutes; the CPU gate and CPU throughput above are from that
   build.)
2. **Memory, from the measured NC=2 numbers.**  The integrator frame
   scales as ~NC x 142 KB (139,288 -> 284,040 measured).  At NC=4 that
   is ~568 KB/thread; with the CUDA stack pre-reserved for ~221,184
   thread slots the reservation alone would be ~125-140 GiB — over the
   80 GB A100 before any data arrays.  (Consistent with Phase-1 history:
   384,000 already OOMed.)  So even a finished binary could not launch
   under the current pre-reserved-stack execution model.

Both points are findings, not failures of protocol: the brief
anticipated that the NC-scaled frame "may blow up local-memory traffic,
which would itself be the finding."

## Interpretation

### Why NC=2 lost: the independent streams have nowhere to live

Compare per-thread time with equal-cost pairs (sorted): 2 interleaved
cells take 1.66x the time of 1 cell (1.225 s vs 0.737 s over the same
work with half the threads).  Pure back-to-back serialisation would be
2.0x.  So the interleave DID recover some overlap — about 17% — but
buying that overlap cost a full 2x reduction in thread-level
parallelism, for a net 0.60x.

The ptxinfo table shows why the recovered overlap is so small.  For the
interleave to hide cell A's dependency stall behind cell B's arithmetic,
cell B's operands must be *in registers* when the stall hits.  They are
not: doubling the live state doubled the register demand, ptxas hit its
budget, and the extra state went to local-memory spill — `Jac_SP_M`
does 171 KB of spill *loads* per call (2.3x the scalar's 74 KB, i.e.
~16% MORE per cell, plus double frame).  Both "independent" instruction
streams end up serialised on the very DRAM-backed local-memory latency
the design was supposed to hide.  Per-thread ILP through explicit
software interleaving is self-defeating here: the machine resource that
would make it work (registers) is precisely the resource the
transformation exhausts.

This is the same lesson as Exp-2 seen from the other side.  Exp-2:
lanes < 1 cell fails because the work has no usable intra-cell
parallelism.  Exp-3: threads > 1 cell fails because a thread has no
spare registers to keep a second cell's chain live.  The hardware's own
mechanism for overlapping independent cells — warp scheduling across
threads, with hardware-interleaved local memory — is `gpul` itself, and
it is better at it than software interleaving can be, because it holds
each cell's live state in a separate register set.

### The lockstep-waste numbers (bonus measurements)

- GPU NC=2: random pairs -> sorted pairs = +38%.
- CPU NC=4: random groups -> sorted groups = +137%.
Both consistent with group cost = max(member step counts), step counts
1..32, mean 6.19.  Useful for anyone considering batched-lockstep
designs (e.g. dense batched LU over cell blocks): pre-sorting by
expected cost is mandatory, and its benefit grows with block size.

### GPUL sorted: +33% for free (measured; mechanism inferred)

Running the UNCHANGED `Integrate_Cell_L` over a cost-sorted permutation
was worth 20,538 -> 27,338 c/s at 20,160 cells.  Mechanism (inferred):
at 20k cells the A100 holds every cell resident (~93-187 threads/SM);
wall time is set by warp-level imbalance, and sorting makes the 32 cells
in a warp finish together.  This is item 4 ("load balance") of the
Exp-2 what-could-still-move-it list, and it is the only positive number
Phase 1 has produced.  Caveat, stated plainly: at saturation (201,600
cells, ~10x oversubscription) idle SMs refill from the pool, so most of
this tail effect should shrink; the saturated benefit is NOT measured.
Worth one cheap job if the ~29.5k c/s plateau number is ever quoted as
a best-achievable: sorted-order gpul at 201,600 cells.

## Phase 1 status after three experiments

| experiment | change | result at 20,160 cells |
|---|---|---|
| Exp-1 interleaved | data layout: cell-first global arrays | -18% |
| Exp-2 warp-cooperative | work mapping: L lanes per cell, L=4..32 | -55%, flat in L |
| Exp-3 per-thread ILP | work mapping: NC cells per thread, NC=2 | **-40% (sorted-vs-sorted); NC=4 infeasible** |

Data layout, sub-thread mapping, and super-thread mapping are now all
measured negative.  One thread per cell with per-thread local workspace
(`gpul`) is the optimum of the entire mapping family for this kernel as
generated.  The ~29.5k c/s saturated plateau is the price of the
per-cell serial dependency structure itself, and no remapping of the
same instruction stream will move it.  What is left is what Exp-2
already ranked: (a) algorithmic work reduction — Jacobian/LU reuse
across steps, KPP auto-reduction, mixed precision; (b) a solver whose
per-cell work is lane-parallel by construction (dense/blocked batched
LU); (c) the measured +33% scheduling win, which composes with either.

## Measured vs inferred

Measured: all throughput numbers, both pairing deltas, frame/spill/
register figures, compile times, the step-count distribution, gate
exactness, GPU maxrel values.  Inferred: the register-spill explanation
of the small overlap fraction (from ptxinfo spill counts, not from a
profiler); the warp-imbalance mechanism of the sorted-gpul win; the
NC=4 stack-reservation arithmetic (extrapolated frame, measured
reservation model).  Not measured: achieved occupancy, NC=4 GPU
anything, saturated behaviour of any Exp-3 mode.

## Reproduction

```
# login node (needs python3)
bash batch_src/build_batch.sh prep
# CPU bit-exact gate (NC=2 and NC=4, both pairings; ~70 min, most of it
# gfortran compiling the generated modules and loading 20,160 samples)
export LSF_DOCKER_VOLUMES="/storage1/fs1/rvmartin/Active:/storage1/fs1/rvmartin/Active"
bsub -G compute-rvmartin -g /y.zhuge/bench -q general -n 24 \
     -R "span[hosts=1] rusage[mem=96GB]" \
     -a "docker(billzhuge/geos-chem-deps:14.7-lsf)" \
     -o logs/cpum_gate.%J.log bash batch_src/job_cpum.sh
# GPU: gpul(both orders) + NC=2(both pairings), same GPU, same job
bash batch_src/sub_gpum.sh 2 20160
# GPU incl. NC=4: bash batch_src/sub_gpum4.sh 20160  -- do NOT bother:
# see "NC=4: infeasible" (compile never finished; stack cannot fit)
```

Logs: `logs/cpum_gate.406930.log` (bit-exact gate),
`logs/gpum2_20160.406931.log` (GPU measurement),
`logs/build_gpum2.txt` (ptxinfo + -Minfo listing),
`logs/gpum4_20160.407492.log` (killed NC=4 build, for the compile-time
record).  Builds live in `build_cpum/` and `build_gpum2/` (private
BDIR build dirs; `KPP-Standalone-batch/` still holds the prepped
sources and generated modules).
