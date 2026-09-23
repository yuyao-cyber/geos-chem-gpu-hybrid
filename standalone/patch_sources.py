#!/usr/bin/env python3
"""Prepare KPP-Standalone sources for batched CPU/GPU builds.

Idempotent transformations (run inside KPP-Standalone-batch/):
  gckpp_Function.F90
    - DO_FUN mask: module USE -> local PARAMETER (.TRUE.)  [constant-fold]
    - module scratch array A(NREACT) -> local per routine  [GPU race fix]
    - '!$acc routine seq' on every subroutine               [device codegen]
  gckpp_Jacobian.F90
    - DO_JVS mask -> PARAMETER; '!$acc routine seq' on Jac_SP
  gckpp_LinearAlgebra.F90
    - DO_SLV mask -> module SAVE var (assigned by AR helpers) with
      '!$acc declare copyin' for device visibility
    - '!$acc routine seq' on KppDecomp and KppSolve
"""
import re, sys

def read(f):
    return open(f).read()

def write(f, s):
    open(f, 'w').write(s)

def annotate_subroutine(src, name):
    """Insert '!$acc routine seq' right after 'SUBROUTINE <name> (' line."""
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
# constant-fold DO_FUN
s = re.sub(r'(?m)^\s*USE gckpp_Global, only : DO_FUN.*\n', '', s)
s = s.replace('IMPLICIT NONE',
              'IMPLICIT NONE\n  LOGICAL, PARAMETER :: DO_FUN(353) = .TRUE.', 1)
# remove module-level scratch A + its THREADPRIVATE
s = re.sub(r'(?m)^\s*REAL\(kind=dp\) :: A\(NREACT\)\s*\n', '', s, count=1)
s = re.sub(r'(?m)^\s*!\$OMP THREADPRIVATE\(\s*A\s*\)\s*\n', '', s, count=1)
# for each subroutine, annotate + add local A where the body assigns A(
subs = re.findall(r'(?m)^SUBROUTINE (\w+)', s)
parts = re.split(r'(?m)^(SUBROUTINE \w+[^\n]*)$', s)
# parts: [pre, header1, body1, header2, body2, ...]
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

# ---------------- gckpp_Jacobian.F90 ----------------
f = 'gckpp_Jacobian.F90'
s = read(f)
s = re.sub(r'(?m)^\s*USE gckpp_Global, ONLY: DO_JVS.*\n', '', s)
s = s.replace('IMPLICIT NONE',
              'IMPLICIT NONE\n  LOGICAL, PARAMETER :: DO_JVS(5683) = .TRUE.', 1)
s = annotate_subroutine(s, 'Jac_SP')
write(f, s)
print('patched', f)

# ---------------- gckpp_LinearAlgebra.F90 ----------------
f = 'gckpp_LinearAlgebra.F90'
s = read(f)
s = re.sub(r'(?m)^\s*USE gckpp_Global, ONLY: DO_SLV.*\n', '', s)
s = s.replace('IMPLICIT NONE',
              'IMPLICIT NONE\n  LOGICAL, SAVE :: DO_SLV(354) = .TRUE.\n'
              '  !$acc declare copyin(DO_SLV)', 1)
s = annotate_subroutine(s, 'KppDecomp')
s = annotate_subroutine(s, 'KppSolve')
write(f, s)
print('patched', f)
print('ALL_PATCHED')
