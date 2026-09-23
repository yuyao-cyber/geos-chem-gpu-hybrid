#!/usr/bin/env python3
"""Prepare the in-tree KPP/fullchem sources for the batched CPU/GPU path.

Adapted from $WS/batch_src/patch_sources.py (written for the
KPP-Standalone-batch file set).  Differences for the in-model tree:

  - All introduced module-level entities are PRIVATE.  This is REQUIRED
    in the model tree: gckpp_Integrator.F90 does a wholesale
    "USE gckpp_Global" and (in Rosenbrock) "USE gckpp_LinearAlgebra",
    and it assigns DO_SLV/DO_FUN/DO_JVS from gckpp_Global.  A public
    DO_* entity in the patched modules would make those names ambiguous
    and break compilation of the stock integrator.
  - The gckpp_Global DO_* arrays are left in place and are still
    assigned by ros_Integrator (harmlessly); the patched Function /
    Jacobian / LinearAlgebra modules simply no longer read them.

Behavioral note: with auto-reduction disabled (the only supported mode of
the batch tree), the stock code sets DO_FUN/DO_JVS/DO_SLV = .TRUE. at
every integrator entry, so constant-folding them to .TRUE. is
behavior-preserving for the stock path as well.  Auto-reduce runs are NOT
supported in this tree (they would ignore the reduced-mechanism masks).

Idempotent: each transformation checks whether it has been applied.
Run inside GCClassic-14.7.1-batch/src/GEOS-Chem/KPP/fullchem/.
"""
import re, sys

def read(f):
    return open(f).read()

def write(f, s):
    open(f, 'w').write(s)

def annotate_subroutine(src, name):
    """Insert '!$acc routine seq' right after 'SUBROUTINE <name> (' line."""
    if re.search(r'(?m)^SUBROUTINE %s\b[^\n]*\n  !\$acc routine seq' % re.escape(name), src):
        return src  # already annotated
    pat = re.compile(r'(?m)^(SUBROUTINE %s\b[^\n]*)$' % re.escape(name))
    def rep(m):
        return m.group(1) + '\n  !$acc routine seq'
    out, n = pat.subn(rep, src, count=1)
    if n != 1:
        sys.exit('FAILED to annotate %s' % name)
    return out

# ---------------- gckpp_Function.F90 ----------------
f = 'gckpp_Function.F90'
s = read(f)
if 'PRIVATE, PARAMETER :: DO_FUN' not in s:
    # constant-fold DO_FUN
    s = re.sub(r'(?m)^\s*USE gckpp_Global, only : DO_FUN.*\n', '', s)
    s = s.replace('IMPLICIT NONE',
                  'IMPLICIT NONE\n'
                  '  LOGICAL, PRIVATE, PARAMETER :: DO_FUN(353) = .TRUE.', 1)
    # remove module-level scratch A + its THREADPRIVATE
    s = re.sub(r'(?m)^\s*REAL\(kind=dp\) :: A\(NREACT\)\s*\n', '', s, count=1)
    s = re.sub(r'(?m)^\s*!\$OMP THREADPRIVATE\(\s*A\s*\)\s*\n', '', s, count=1)
    # for each subroutine, annotate + add local A where the body assigns A(
    parts = re.split(r'(?m)^(SUBROUTINE \w+[^\n]*)$', s)
    out = [parts[0]]
    for i in range(1, len(parts), 2):
        header, body = parts[i], parts[i+1]
        add = '\n  !$acc routine seq'
        if re.search(r'(?m)^\s*A\(\d+\) =', body) or re.search(r'\bA\(1:NREACT\)|Aout\s*=\s*A|\bA\b\s*\(', body):
            add += '\n  REAL(kind=dp) :: A(NREACT)'
        out.append(header + add)
        out.append(body)
    s = ''.join(out)
    write(f, s)
    print('patched', f)
else:
    print('already patched', f)

# ---------------- gckpp_Jacobian.F90 ----------------
f = 'gckpp_Jacobian.F90'
s = read(f)
if 'PRIVATE, PARAMETER :: DO_JVS' not in s:
    s = re.sub(r'(?m)^\s*USE gckpp_Global, ONLY: DO_JVS.*\n', '', s)
    s = s.replace('IMPLICIT NONE',
                  'IMPLICIT NONE\n'
                  '  LOGICAL, PRIVATE, PARAMETER :: DO_JVS(5683) = .TRUE.', 1)
    s = annotate_subroutine(s, 'Jac_SP')
    write(f, s)
    print('patched', f)
else:
    print('already patched', f)

# ---------------- gckpp_LinearAlgebra.F90 ----------------
f = 'gckpp_LinearAlgebra.F90'
s = read(f)
if 'PRIVATE, SAVE :: DO_SLV' not in s:
    s = re.sub(r'(?m)^\s*USE gckpp_Global, ONLY: DO_SLV.*\n', '', s)
    s = s.replace('IMPLICIT NONE',
                  'IMPLICIT NONE\n'
                  '  LOGICAL, PRIVATE, SAVE :: DO_SLV(354) = .TRUE.\n'
                  '  !$acc declare copyin(DO_SLV)', 1)
    s = annotate_subroutine(s, 'KppDecomp')
    s = annotate_subroutine(s, 'KppSolve')
    write(f, s)
    print('patched', f)
else:
    print('already patched', f)
print('ALL_PATCHED')
