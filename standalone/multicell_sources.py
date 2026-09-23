#!/usr/bin/env python3
"""Generate gckpp_MultiCell<NC>.F90 -- MULTI-CELL (per-thread ILP) variants of
the four hot per-cell kernels, for Phase-1 Experiment 3.

Run inside KPP-Standalone-batch/ AFTER patch_sources.py (needs python3, so
login node only).

Idea under test
---------------
Experiment 2 showed that SPLITTING one cell across L lanes loses (the cell has
no usable intra-cell parallelism, and cells-in-flight drop by L).  The mirror
image is to give ONE thread NC independent cells and INTERLEAVE their
instruction streams, so the machine can hide one cell's dependency stalls with
another cell's arithmetic.  That only works if the interleaving happens INSIDE
the generated straight-line kernels (Fun_SPLIT, Jac_SP, KppDecomp, KppSolve) --
calling a scalar kernel NC times back-to-back gives no overlap at all, since a
GPU thread issues in order.

Transformation (mechanical, per routine)
----------------------------------------
  - every per-cell array NAME(expr) -> NAME(mc, expr)  (cell index FIRST)
  - every executable statement is emitted NC times, ONCE PER CELL, with the
    NC copies ADJACENT in program order.  This is an explicit unroll: it does
    not depend on the compiler's unroll heuristics, and it puts NC mutually
    independent instances of every instruction next to each other for the
    scheduler.
  - per-cell operation ORDER is untouched (for a fixed cell, the sequence of
    statements is exactly the original), so results are BIT-EXACT.

Bodies are extracted verbatim from the patched generated sources.  Multi-line
continued statements are first joined into one logical statement, rewritten,
then re-emitted with exact-resume ('&' ... '&') continuations, so a statement
may be split at any character without changing its meaning.
"""
import re, sys, os

NC_LIST = [2, 4]

# ------------------------------------------------------------------ helpers

def read(f):
    return open(f).read()


def extract(src, start_pat, end_pat):
    m0 = re.search(start_pat, src)
    if not m0:
        sys.exit('extract: start not found: %s' % start_pat)
    m1 = re.search(end_pat, src[m0.end():])
    if not m1:
        sys.exit('extract: end not found: %s' % end_pat)
    return src[m0.end():m0.end() + m1.start()]


def body_after(text, anchor):
    i = text.find(anchor)
    if i < 0:
        sys.exit('body_after: anchor not found: %s' % anchor)
    return text[i + len(anchor):]


def logical_statements(text):
    """Join Fortran free-form continuation lines into logical statements.

    Comments and blank lines are dropped (these generated files contain no
    character literals, so a bare '!' always starts a comment).
    """
    stmts = []
    cur = ''
    for raw in text.split('\n'):
        s = raw
        if '!' in s:
            s = s[:s.index('!')]
        s = s.strip()
        if not s:
            continue
        cont_in = s.startswith('&')
        if cont_in:
            s = s[1:]
        cont_out = s.rstrip().endswith('&')
        if cont_out:
            s = s.rstrip()[:-1]
        if cur == '':
            cur = s.strip()
        elif cont_in:
            cur = cur + s.rstrip()          # exact resume: no space inserted
        else:
            cur = cur + ' ' + s.strip()
        if not cont_out:
            if cur.strip():
                stmts.append(cur.strip())
            cur = ''
    if cur.strip():
        stmts.append(cur.strip())
    return stmts


def cellify(stmt, names, marker='@'):
    """NAME( -> NAME(@,   for every name in 'names' (dict name -> suffix)."""
    for n, suf in names.items():
        stmt = re.sub(r'\b%s\(' % re.escape(n), '%s(%s%s' % (n, marker, suf), stmt)
    return stmt


def wrap(stmt, indent='  ', width=100):
    """Emit one logical statement, split with exact-resume continuations."""
    if len(stmt) + len(indent) <= width:
        return [indent + stmt]
    out = []
    s = stmt
    first = True
    while s:
        take = (width - len(indent)) if first else (width - 6)
        chunk, s = s[:take], s[take:]
        tail = '&' if s else ''
        out.append((indent + chunk + tail) if first else ('     &' + chunk + tail))
        first = False
    return out


def emit_unrolled(stmts, names, nc, indent='  '):
    """Emit each statement NC times (cell copies adjacent)."""
    out = []
    for st in stmts:
        for c in range(1, nc + 1):
            out.extend(wrap(cellify(st, names).replace('@', str(c)), indent))
    return out


# ---------------------------------------------------------------- Fun_SPLIT
src = read('gckpp_Function.F90')
t = extract(src, r'(?m)^SUBROUTINE Fun_SPLIT \(', r'(?m)^END SUBROUTINE Fun_SPLIT\b')
t = body_after(t, '! Computation of equation rates')
fun_stmts = logical_statements(t)
fun_stmts = [s for s in fun_stmts if 'PRESENT( Aout )' not in s]
VDOT_WHOLE = 'Vdot = P_VAR - D_VAR*V'
if VDOT_WHOLE not in fun_stmts:
    sys.exit('Fun_SPLIT: whole-array Vdot statement not found')
FUN_NAMES = {n: ',' for n in ['A', 'V', 'F', 'RCT', 'Vdot', 'P_VAR', 'D_VAR']}

# ------------------------------------------------------------------ Jac_SP
src = read('gckpp_Jacobian.F90')
t = extract(src, r'(?m)^SUBROUTINE Jac_SP \(', r'(?m)^END SUBROUTINE Jac_SP\s*$')
mB = re.search(r'REAL\(kind=dp\) :: B\((\d+)\)', t)
if not mB:
    sys.exit('Jac_SP: local B(...) not found')
NB_JAC = int(mB.group(1))
jac_stmts = logical_statements(body_after(t, mB.group(0)))
JAC_NAMES = {n: ',' for n in ['V', 'F', 'RCT', 'JVS', 'B']}

# ---------------------------------------------------------------- KppSolve
src = read('gckpp_LinearAlgebra.F90')
t = extract(src, r'(?m)^SUBROUTINE KppSolve \(', r'(?m)^END SUBROUTINE KppSolve\s*$')
sol_stmts = logical_statements(body_after(t, 'REAL(kind=dp) :: X(NVAR)'))
SOL_NAMES = {'JVS': ',', 'X': ',noff+'}

print('statements: Fun_SPLIT=%d  Jac_SP=%d  KppSolve=%d  (NB_JAC=%d)'
      % (len(fun_stmts), len(jac_stmts), len(sol_stmts), NB_JAC))

# ------------------------------------------------------------------ KppDecomp
# Hand-written from the patched KppDecomp: identical loop structure, the NC
# cells unrolled at the innermost statement level.  The only deviation is that
# a singular pivot records IER(mc) and CONTINUES instead of returning early --
# equivalent for the caller, which rebuilds Ghimj from Jac0 with H/2 on any
# IER/=0, so the partially/fully overwritten JVS is discarded either way.
def kppdecomp(nc):
    L = []
    a = L.append
    a('SUBROUTINE KppDecomp_M ( JVS, IER )')
    a('  !$acc routine seq')
    a('  REAL(kind=dp) :: JVS(NC,LU_NONZERO)')
    a('  INTEGER  :: IER(NC)')
    a('  REAL(kind=dp) :: W(NC,NVAR)')
    a('  REAL(kind=dp) :: acc(NC), atiny')
    a('  INTEGER  :: k, kk, j, jj')
    a('  atiny = 0.0_dp')
    for c in range(1, nc + 1):
        a('  IER(%d) = 0' % c)
    a('  DO k = 1, NVAR')
    for c in range(1, nc + 1):
        a('    IF ( ABS(JVS(%d,LU_DIAG(k))) < TINY(atiny) .AND. IER(%d) == 0 ) IER(%d) = k'
          % (c, c, c))
    a('    DO kk = LU_CROW(k), LU_CROW(k+1)-1')
    for c in range(1, nc + 1):
        a('      W(%d,LU_ICOL(kk)) = JVS(%d,kk)' % (c, c))
    a('    END DO')
    a('    DO kk = LU_CROW(k), LU_DIAG(k)-1')
    a('      j = LU_ICOL(kk)')
    for c in range(1, nc + 1):
        a('      acc(%d) = -W(%d,j) / JVS(%d,LU_DIAG(j))' % (c, c, c))
    for c in range(1, nc + 1):
        a('      W(%d,j) = -acc(%d)' % (c, c))
    a('      DO jj = LU_DIAG(j)+1, LU_CROW(j+1)-1')
    for c in range(1, nc + 1):
        a('        W(%d,LU_ICOL(jj)) = W(%d,LU_ICOL(jj)) + acc(%d)*JVS(%d,jj)' % (c, c, c, c))
    a('      END DO')
    a('    END DO')
    a('    DO kk = LU_CROW(k), LU_CROW(k+1)-1')
    for c in range(1, nc + 1):
        a('      JVS(%d,kk) = W(%d,LU_ICOL(kk))' % (c, c))
    a('    END DO')
    a('  END DO')
    a('END SUBROUTINE KppDecomp_M')
    return L


# ------------------------------------------------------------------ output
HEAD = '''!------------------------------------------------------------------------------
! gckpp_MultiCell{NC}.F90 -- GENERATED by multicell_sources.py.  DO NOT EDIT.
!
! MULTI-CELL (per-thread ILP) kernels: one thread carries NC = {NC} independent
! grid cells.  Every per-cell array carries the cell index FIRST, and every
! executable statement is emitted {NC} times (one per cell, copies adjacent), so
! the {NC} independent instruction streams are interleaved for the scheduler.
! Per-cell operation ORDER is identical to the one-cell kernels => bit-exact.
!------------------------------------------------------------------------------
MODULE gckpp_MultiCell{NC}

  USE gckpp_Precision,  ONLY : dp
  USE gckpp_Parameters
  USE gckpp_JacobianSP
  IMPLICIT NONE
  PUBLIC

  INTEGER, PARAMETER :: NC = {NC}          ! cells in flight per thread

  ! Auto-reduce masks, constant-folded (AR is off in all harvested cells)
  LOGICAL, PARAMETER :: DO_FUN(353)  = .TRUE.
  LOGICAL, PARAMETER :: DO_JVS(5683) = .TRUE.
  LOGICAL, PARAMETER :: DO_SLV(354)  = .TRUE.

  INTEGER, PARAMETER :: NB_JAC = {NB}      ! size of the Jac_SP temporary B

  ! Rodas3 has 4 stages
  INTEGER,  PARAMETER :: ROS_S   = 4
  INTEGER,  PARAMETER :: NWORK_K = NVAR*ROS_S
  INTEGER,  PARAMETER :: NWORK_J = LU_NONZERO
  REAL(dp), PARAMETER :: ROUNDOFF = 2.0_dp**(-52)
  REAL(dp), PARAMETER :: ZERO = 0.0_dp, ONE = 1.0_dp, HALF = 0.5_dp
  REAL(dp), PARAMETER :: DELTAMIN = 1.0E-5_dp

CONTAINS

SUBROUTINE Fun_SPLIT_M ( V, F, RCT, Vdot, P_VAR, D_VAR )
  !$acc routine seq
  REAL(kind=dp) :: V(NC,NVAR)
  REAL(kind=dp) :: F(NC,NFIX)
  REAL(kind=dp) :: RCT(NC,NREACT)
  REAL(kind=dp) :: Vdot(NC,NVAR)
  REAL(kind=dp) :: P_VAR(NC,NVAR)
  REAL(kind=dp) :: D_VAR(NC,NVAR)
  REAL(kind=dp) :: A(NC,NREACT)
  INTEGER :: kv_i
'''

TAIL = '''
INCLUDE 'multicell_integrator.inc'

END MODULE gckpp_MultiCell{NC}
'''

for nc in NC_LIST:
    lines = [HEAD.format(NC=nc, NB=NB_JAC)]

    body = []
    for st in fun_stmts:
        if st == VDOT_WHOLE:
            body.append('  DO kv_i = 1, NVAR')
            for c in range(1, nc + 1):
                body.append('    Vdot(%d,kv_i) = P_VAR(%d,kv_i) - D_VAR(%d,kv_i)*V(%d,kv_i)'
                            % (c, c, c, c))
            body.append('  ENDDO')
        else:
            for c in range(1, nc + 1):
                body.extend(wrap(cellify(st, FUN_NAMES).replace('@', str(c))))
    lines.append('\n'.join(body))
    lines.append('\nEND SUBROUTINE Fun_SPLIT_M\n')

    lines.append('SUBROUTINE Jac_SP_M ( V, F, RCT, JVS )\n'
                 '  !$acc routine seq\n'
                 '  REAL(kind=dp) :: V(NC,NVAR)\n'
                 '  REAL(kind=dp) :: F(NC,NFIX)\n'
                 '  REAL(kind=dp) :: RCT(NC,NREACT)\n'
                 '  REAL(kind=dp) :: JVS(NC,LU_NONZERO)\n'
                 '  REAL(kind=dp) :: B(NC,NB_JAC)\n')
    lines.append('\n'.join(emit_unrolled(jac_stmts, JAC_NAMES, nc)))
    lines.append('\nEND SUBROUTINE Jac_SP_M\n')

    lines.append('\n'.join(kppdecomp(nc)))
    lines.append('\n')

    lines.append('SUBROUTINE KppSolve_M ( JVS, X, noff )\n'
                 '  !$acc routine seq\n'
                 '  REAL(kind=dp) :: JVS(NC,LU_NONZERO)\n'
                 '  REAL(kind=dp) :: X(NC,*)\n'
                 '  INTEGER, INTENT(IN) :: noff\n')
    lines.append('\n'.join(emit_unrolled(sol_stmts, SOL_NAMES, nc)))
    lines.append('\nEND SUBROUTINE KppSolve_M\n')

    lines.append(TAIL.format(NC=nc))

    fn = 'gckpp_MultiCell%d.F90' % nc
    open(fn, 'w').write('\n'.join(lines))
    print('generated %s  (%d lines)' % (fn, open(fn).read().count('\n')))

print('ALL_MULTICELL')
