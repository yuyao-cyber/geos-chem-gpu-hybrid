# Phase 1, Experiment 2: Warp-cooperative cell solve — RESULTS

Date: 2026-08-30/31.  Cluster: WashU compute1 (LSF).  GPU = NVIDIA
A100-SXM4-80GB (gmodel-pinned submits).  Solver: KPP Rodas3 batch
integrator, GEOS-Chem fullchem (NVAR=353, NREACT=1058, LU_NONZERO=5683).

**Verdict: NEGATIVE.  Warp-cooperative is 2.2x SLOWER than one
thread per cell, and throughput is FLAT from 4 to 32 lanes per cell.**

## Hypothesis under test

Map one warp (or sub-warp) to ONE cell so that (a) all lanes share a
single accept/reject path through Rodas3 (control flow re-converges,
eliminating the divergence tax measured in Experiment 1), (b) the wide
per-cell loops (NVAR=353 element-wise ops, LU_NONZERO=5683 row sweeps)
are split across lanes, and (c) the per-thread working set shrinks.

## What was built

All changes are additive; every pre-existing mode (`check`, `cpu1`,
`omp`, `gpu`, `gpul`, `cpui`, `gpui`) still builds and runs.

`batch_src/gckpp_BatchIntegrator.F90`
- `Integrate_Cell_W` — `!$acc routine vector`.  Identical Rodas3 control
  flow and per-cell operation ORDER to `Integrate_Cell`.  The scalar
  spine (T, H, Err, accept/reject, stage loop) executes REDUNDANTLY on
  every lane, so all lanes of a gang are always at the same statement.
  Workspace is caller-provided per-cell slices in the cell-last layout
  (the same arrays mode `gpu` uses); with one warp per cell the lanes of
  a vector loop touch consecutive elements k, so those accesses are
  fully coalesced by construction.
- `KppDecomp_W` — `!$acc routine vector`.  Same sparse LU as
  `KppDecomp`.  The outer row loop `k` and the sub-diagonal loop `kk`
  carry true dependencies and remain serial (redundant on all lanes);
  the row gather, the row update, and the row scatter are lane-parallel.
- `AddGhinvDiag_W` — `!$acc routine vector`.
- Ten single-loop `!$acc routine vector` helper kernels — `VZERO`,
  `VCOPY`, `VCOPYPOS`, `VNEG`, `VAXPY`, `VDFDT`, `VERRTERM`,
  `VLU_GATHER`, `VLU_SCATTER`, `VLU_UPDATE` — see "Compiler mapping"
  below for why these exist.

`batch_src/kpp_batch.F90`
- new modes `cpuw` (bit-exact gate) and `gpuw` (benchmark).  `gpuw` runs
  `gpul` first in the SAME `!$acc data` region on the SAME GPU, then
  sweeps vector lengths 32 / 16 / 8 / 4, resetting and re-uploading the
  input state before each timed run and reading results back after it.

`batch_src/job_cpuw.sh`, `job_gpuw.sh`, `sub_gpuw.sh` (bsub wrapper —
the documented `-R "select[gpuhost] && hname!='...'"` form is rejected
by this LSF; two separate `-R "select[...]"` clauses work).
`job_gpuw_x10.sh` exists but was NOT run — see Verdict.

### Bit-exactness

Every lane-parallel loop is element-wise: each output element is written
by exactly one iteration, so no summation is reordered.  The one
reduction in the algorithm — the Rodas3 error norm — is computed in TWO
phases: `VERRTERM` forms the per-species terms `(Yerr(i)/Scal)**2` in a
vector loop, and the caller sums them SERIALLY in the original
`i = 1..NVAR` order.  **No `reduction` clause is used anywhere**, so the
cooperative path needs no CPU/GPU divergence in summation order and the
bit-exact gate applies to the same arithmetic the GPU executes.  (The
fallback allowed for in the experiment brief — cooperative reduction
under GPU only — was not needed.)

## Compiler mapping — what -Minfo=accel actually confirmed

The first two builds were MISMAPPED, silently.  Writing `!$acc loop
vector` inline inside the sequential control flow of a `routine vector`
produced, for every single loop:

```
kppdecomp_w:      738, !$acc loop seq   741, !$acc loop seq   746, !$acc loop seq
integrate_cell_w: 928, !$acc loop seq   941, !$acc loop seq   ... (21 loops)
addghinvdiag_w:   764, !$acc loop vector ! threadidx%x
```

Adding the `independent` clause and giving every loop a private
induction variable changed nothing — still all `loop seq`.  The one
loop that DID map was `AddGhinvDiag_W`, whose loop is the *entire body*
of its routine.  That is the rule for **nvfortran 24.7**: a `!$acc loop
vector` is mapped to `threadIdx%x` only when it is the whole body of a
`!$acc routine vector` procedure; a vector loop nested inside a DO
WHILE, a stage loop, or an IF block of a routine vector is silently
demoted to `loop seq`.  It reports this only in `-Minfo=accel`, and the
code still runs and still validates — it just uses one lane.

The fix was to hoist every element-wise loop into its own one-loop
`routine vector` helper, called from the serial control flow.  Final
build (job 383335):

```
vzero:       735, !$acc loop vector ! threadidx%x
vcopy:       747, !$acc loop vector ! threadidx%x
vcopypos:    759, !$acc loop vector ! threadidx%x
vneg:        771, !$acc loop vector ! threadidx%x
vaxpy:       783, !$acc loop vector ! threadidx%x
vdfdt:       795, !$acc loop vector ! threadidx%x
verrterm:    812, !$acc loop vector ! threadidx%x
vlu_gather:  831, !$acc loop vector ! threadidx%x
vlu_scatter: 844, !$acc loop vector ! threadidx%x
vlu_update:  857, !$acc loop vector ! threadidx%x
addghinvdiag_w: 905, !$acc loop vector ! threadidx%x
kppdecomp_w:    888, !$acc loop seq          <- outer row loop (intended)
integrate_cell_w: 1095/1101/1109/1124/1129/1135, !$acc loop seq
                                             <- TimeLoop, UntilAccepted,
                                                Stage, and the j-loops
                                                over previous stages
                                                (all intended)
```

and the gang loops in the driver:

```
496, !$acc loop gang ! blockidx%x     (vector_length 32)
522, !$acc loop gang ! blockidx%x     (vector_length 16)
548, !$acc loop gang ! blockidx%x     (vector_length  8)
574, !$acc loop gang ! blockidx%x     (vector_length  4)
```

So the intended mapping was achieved and confirmed: one gang per cell,
lanes on the wide loops, serial spine redundant.  **This is measured,
from the compiler listing, not assumed.**

`Fun_SPLIT`, `Jac_SP` and `KppSolve` remain `!$acc routine seq` and are
therefore executed redundantly by every lane.  This was the design (the
brief's "leave them serial first, measure"), and it matters — see
Interpretation.

## Validation

**CPU bit-exact gate (mode `cpuw`, gfortran -O2, OMP-24, 20,160 cells,
job 382514): PASSED.**

```
CPU-OMP warp-path:  0.68 s  (29,822.8 cells/s)
Original serial:   10.41 s  ( 1,935.8 cells/s)
CPUW CHECK: differing values:  0  of  7116480   (warp-path vs original)
CPUW CHECK: max relative diff: 0.0000E+00
Cells with IERR/=1: 0
```

GPU (job 383335, 20,160 cells): **0 solver failures** at every vector
length (all IERR=1), and max relative diff vs the CPU serial reference
= **1.054E-06 for gpuw at 32/16/8/4 lanes — identical, to all printed
digits, to gpul's 1.054E-06 in the same job.**  That the cooperative
and the one-thread-per-cell paths land on the same value is strong
evidence the arithmetic is unchanged.

*Caveat, stated plainly:* the CPU bit-exact gate was run against the
FIRST version of `Integrate_Cell_W` (inline `!$acc loop vector`
directives, which under gfortran are inert comments).  The final
version differs only by hoisting those identical element-wise loops
into helper subroutines — no expression, no operation order, and no
loop bound changed — and it passes `gfortran -fsyntax-only`, but the
`cpuw` gate was **not re-run** on the hoisted version.  The matching
GPU max-rel-diff is the evidence standing in for it.  If this code path
is ever taken further, re-run `cpuw` first.

## Throughput — all on A100-SXM4-80GB, 20,160 cells, job 383335

Same job, same GPU, same input state, so the comparison is clean.

| mode | mapping | cells/s | vs gpul |
|---|---|---|---|
| CPU serial (in-job reference) | 1 core | 2,436 | 0.18x |
| **gpul** (baseline) | 1 thread/cell, per-thread local workspace | **13,909** | 1.00x |
| gpuw vlen=32 | 1 gang/cell, 32 lanes | 6,195 | **0.45x** |
| gpuw vlen=16 | 1 gang/cell, 16 lanes | 6,535 | **0.47x** |
| gpuw vlen=8  | 1 gang/cell,  8 lanes | 6,527 | **0.47x** |
| gpuw vlen=4  | 1 gang/cell,  4 lanes | 6,494 | **0.47x** |

Reference points from earlier phases (A100, same 20,160-cell list):
gpul 14,872; gpui (interleaved) 12,208; gpu (module workspace) 18,440.
The in-job gpul here (13,909) is within 7% of the established 14,872,
so this GPU/job is representative.

An earlier build with every vector loop silently demoted to `loop seq`
(job 383040 — the accidental but useful control: warp-cooperative
*structure*, zero actual lane parallelism) gave 3,255 / 3,249 / 3,150 /
3,217 c/s at 32/16/8/4 lanes against gpul 14,776 in the same job.  So
correctly mapping the lanes bought 1.9x — the vectorization is real and
does work — and it still lands at less than half of gpul.

The saturation run (201,600 cells vs the 29,181–29,513 c/s gpul
plateau) was **not** run: at 0.45x with no lane-count sensitivity there
is no mechanism by which saturation flips the ranking, and the load
alone costs ~1 h of node time.

## Interpretation

### The flat lane sweep is the whole story

For a fixed hardware thread budget, splitting one cell across L lanes
gives, relative to one thread per cell,

```
throughput ratio = 1 / ( f*L + (1-f) )        f = serial fraction of the cell
```

because the cells resident per SM drop by L while only the parallel
(1-f) portion gets faster.  This is **<= 1 for all f >= 0 and L > 1**:
warp-cooperation can at best reach parity on work-efficiency grounds
and can only ever WIN through a memory-hierarchy or divergence effect.

The measurement does not even follow that curve.  Fitting the L=32
point (0.445) gives f = 0.040, which would predict 0.885 at L=4;
measured is 0.467.  The ratio is **flat within 5% across an 8x change
in lanes**.  Two conclusions, both measured:

1. **The extra lanes contribute essentially nothing.**  Going from 4 to
   32 lanes per cell neither helps (more parallel width) nor hurts
   (fewer resident cells).  Per-cell wall time and cells-in-flight must
   therefore both be governed by something that does not scale with L.
2. **The 2.2x deficit is a fixed per-gang overhead, not Amdahl.**  If it
   were Amdahl the sweep would have a strong slope.

### Where the time actually goes (inferred, consistent with the above)

- **The straight-line generated kernels were never parallelized, and
  they dominate.**  `Fun_SPLIT` (~2,900 source lines), `Jac_SP`
  (~20,800 source lines) and `KppSolve` stayed `routine seq` and run
  redundantly on all L lanes.  Per Rodas3 step that is 3 Fun + 1 Jac +
  4 Solve evaluations of pure scalar expression DAGs with no loop
  structure to distribute.  Only the element-wise vector work
  (WAXPY/WCOPY/error terms, ~10 x 353 elements per step) and the LU row
  sweeps were made cooperative.  Redundant execution costs no extra
  *time* in SIMT (all lanes issue the same instruction), which is
  exactly why the sweep is flat — but it means L-1 lanes are doing
  nothing useful for most of the cell, so the cooperative mapping buys
  nothing there while still paying L thread slots per cell.
- **The lane-parallel loops are far too short to amortize their
  barriers.**  Mean LU row length is 5683/353 = 16.1 nonzeros, and the
  sub-diagonal update loop is shorter still.  With 32 lanes, half the
  warp is idle on a 16-element gather and the implicit warp barrier at
  the end of each vector loop costs more than the 16 loads.  A single
  `KppDecomp_W` call issues 353 `VLU_GATHER` + 353 `VLU_SCATTER` calls
  plus one `VLU_UPDATE` per sub-diagonal nonzero — order 10^3 device
  function calls, each with a call/return and a lane barrier, per
  factorization, per step.  That is a per-gang cost independent of L,
  which is precisely the shape the data show.
- **Redundant scalar control flow adds a second L-independent cost.**
  Every lane evaluates the full option decoding, step-size logic and
  accept/reject test, and every lane holds its own copy of the
  `routine seq` locals — `A(NREACT)` = 8.5 KB and `B(1817)` = 14.5 KB —
  so the per-*lane* local frame is ~23 KB and the per-*gang* local
  footprint is L x 23 KB, i.e. 736 KB at 32 lanes versus 23 KB for one
  gpul thread's equivalent role.  Control flow did re-converge as
  hypothesised (all lanes share one accept/reject path), but there was
  no divergence tax left to recover that was worth 2.2x.
- **Occupancy: not measured.**  The build used `-Minfo=accel` only, so
  there is no ptxinfo register/local-memory/occupancy report in the log
  to cite.  The local-memory arithmetic above is computed from the
  declared array sizes, not read from the compiler.  Treat any
  occupancy claim here as inference.

### What the hypothesis got right, and why it still lost

Control-flow re-convergence and coalescing both worked as designed —
and neither was where the time was.  Experiment 1 had already shown the
memory layout was not the limiter; Experiment 2 shows the work mapping
is not either.  What is left is the per-cell serial arithmetic itself.

## Phase 1 status: what the ~29.5k cells/s plateau is

Two experiments, two negatives, and together they are more informative
than either alone:

| experiment | change | result at 20,160 cells |
|---|---|---|
| Exp-1 interleaved | data layout: cell-first `(nCell,k)` global arrays | **-18%** (12,208 vs 14,872) |
| Exp-2 warp-cooperative | work mapping: L lanes per cell, L = 4..32 | **-55%** (6,195 vs 13,909) |

Exp-1 ruled out *data layout* as the limiter: per-thread local memory
is already hardware-interleaved, so `gpul` was never uncoalesced, and
an explicit cell-first layout only adds cache-footprint cost.  Exp-2
now rules out *work mapping within the same algorithm*, in both
directions of lane count, and shows the ceiling of that whole family of
transformations is parity, not a win.

**The plateau is therefore the per-cell serial dependency structure of
the KPP Rodas3 cell solve as generated straight-line code.**  Each cell
is a long chain of scalar expression DAGs (`Fun_SPLIT`, `Jac_SP`) and
row-sequential sparse triangular work (`KppDecomp`, `KppSolve`) with
essentially no intra-cell parallelism at a granularity the hardware can
use.  One thread per cell with per-thread local workspace already
extracts the parallelism that exists — across cells — at the granularity
the machine wants.  ~29.5k c/s saturated is what that structure costs;
it is a latency-and-dependency limit, not a bandwidth or occupancy one.

### Ruled out

- Explicit coalescing / cell-first layouts (Exp-1).
- Any mapping that spends more than one hardware thread per cell while
  the generated kernels stay scalar straight-line code (Exp-2, and the
  `1/(f*L + 1-f) <= 1` argument, which is general).
- By extension: converting `Fun_SPLIT`/`Jac_SP` to data-driven
  index/coefficient-table loops **for the purpose of enabling warp
  cooperation** is not worth doing — it would remove the redundant-lane
  waste, but the best it can then reach is parity with gpul.

### What could still move it

1. **Per-thread ILP — the natural Experiment 3.**  Keep one thread per
   cell, but give each thread 2–4 cells in flight and interleave their
   independent instruction streams.  This attacks the actual limiter
   diagnosed in Phase 0 (dependent-chain latency against DRAM-backed
   local memory) by *increasing* work per thread rather than splitting
   it, so it does not pay the `f*L` penalty at all — it is the mirror
   image of Exp-2 and the only mapping change not yet ruled out.  Cost:
   per-thread local frame multiplies, so it trades directly against
   occupancy and will need the `NV_ACC_CUDA_STACKSIZE` sizing care that
   bit us at 384 KB.
2. **Algorithmic — fewer/cheaper serial chains rather than more
   parallelism.**  Reuse the Jacobian and its LU factorization across
   accepted steps (Rodas3 refactorizes every step; the LU is the longest
   serial chain); enable KPP auto-reduction (the AR masks are currently
   constant-folded to `.TRUE.`, i.e. every species active every step);
   mixed precision for the Jacobian/LU with a double-precision residual.
   These cut the work instead of redistributing it.
3. **A solver with more exposed parallelism per cell** — blocked or
   dense batched LU (turns row-sequential sparse work into
   lane-friendly tiles), or an integrator whose stages are independent.
   This is a much larger change and should only follow a measurement of
   how much of the step time the LU actually is.
4. **Load balance (secondary, complementary).**  Sorting/binning cells
   by expected step count reduces inter-warp imbalance; worth doing
   alongside whichever of the above is chosen, not instead of it.

The immediate recommendation is (1), with (2) as the higher-ceiling but
higher-effort follow-on.

## Reproduction

```
# login node (needs python3)
bash batch_src/build_batch.sh prep
# CPU bit-exact gate
bsub -G compute-rvmartin -g /y.zhuge/bench -q general -n 24 \
     -R "span[hosts=1] rusage[mem=96GB]" \
     -a "docker(billzhuge/geos-chem-deps:14.7-lsf)" \
     -o logs/cpuw_gate.%J.log bash batch_src/job_cpuw.sh
# GPU benchmark + lane sweep (gpul run in the same job)
bash batch_src/sub_gpuw.sh
```

Logs: `logs/cpuw_gate.382514.log` (bit-exact gate),
`logs/gpuw_20k.383040.log` (all-`loop seq` control),
`logs/gpuw_20k.383335.log` (final, correctly mapped),
`logs/build_gpuw.txt` (-Minfo=accel listing of the final build).
