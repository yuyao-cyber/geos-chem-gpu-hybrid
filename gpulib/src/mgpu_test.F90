!------------------------------------------------------------------------------
! mgpu_test.F90 -- gfortran+OpenMP driver: nDev host threads each launch the
! batched solver on a DIFFERENT GPU at the same time through
! gc_gpu_integrate_batch_dev; checks all cells converge and times it against
! a single-device launch of the same total batch.
!------------------------------------------------------------------------------
PROGRAM mgpu_test
  USE ISO_C_BINDING
  USE OMP_LIB
  IMPLICIT NONE
  INTEGER, PARAMETER :: NVAR=353, NFIX=3, NSPEC=356, NREACT=1058
  INTERFACE
     SUBROUTINE gc_gpu_info( nDev, devName, nameLen ) BIND(C, name='gc_gpu_info')
       IMPORT :: c_int, c_char
       INTEGER(c_int),         INTENT(OUT) :: nDev
       INTEGER(c_int),         VALUE       :: nameLen
       CHARACTER(KIND=c_char), INTENT(OUT) :: devName(*)
     END SUBROUTINE gc_gpu_info
     SUBROUTINE gc_gpu_integrate_batch_dev( dev, nCell, Tstart, Tend, C, RCONST, &
          ATOL, RTOL, ICNTRL, RCNTRL, Perm, ISTAT, RSTAT, IERR, tTotal, tKernel ) &
          BIND(C, name='gc_gpu_integrate_batch_dev')
       IMPORT :: c_int, c_double
       INTEGER(c_int), VALUE         :: dev, nCell
       REAL(c_double), VALUE         :: Tstart, Tend
       REAL(c_double), INTENT(INOUT) :: C(*)
       REAL(c_double), INTENT(IN)    :: RCONST(*), ATOL(*), RTOL(*), RCNTRL(*)
       INTEGER(c_int), INTENT(IN)    :: ICNTRL(*), Perm(*)
       INTEGER(c_int), INTENT(OUT)   :: ISTAT(*), IERR(*)
       REAL(c_double), INTENT(OUT)   :: RSTAT(*), tTotal, tKernel
     END SUBROUTINE gc_gpu_integrate_batch_dev
  END INTERFACE
  INTEGER(c_int) :: nDev, i, d, n, nTot, lo, hi, nUse
  CHARACTER(KIND=c_char) :: devName(256)
  REAL(c_double), ALLOCATABLE :: C(:,:), RCONST(:,:), RCNTRL(:,:), RSTAT(:,:)
  INTEGER(c_int), ALLOCATABLE :: ICNTRL(:,:), ISTAT(:,:), IERR(:), Perm(:)
  REAL(c_double) :: ATOL(NVAR), RTOL(NVAR), tT(0:15), tK(0:15), t0, t1, tSingle, tMulti
  CHARACTER(LEN=32) :: arg
  nTot = 100000
  CALL GET_COMMAND_ARGUMENT( 1, arg ); IF ( LEN_TRIM(arg) > 0 ) READ(arg,*) nTot
  CALL gc_gpu_info( nDev, devName, 256_c_int )
  WRITE(*,'(a,i0)') 'devices visible: ', nDev
  nUse = MIN( nDev, 8 )
  ALLOCATE( C(NSPEC,nTot), RCONST(NREACT,nTot), RCNTRL(20,nTot), RSTAT(3,nTot), &
            ICNTRL(20,nTot), ISTAT(8,nTot), IERR(nTot), Perm(nTot) )
  ATOL = 1.0d-2; RTOL = 0.5d-2
  ICNTRL = 0; ICNTRL(1,:) = 1; ICNTRL(3,:) = 4; ICNTRL(7,:) = 1; ICNTRL(15,:) = -1
  RCNTRL = 0.0d0
  ! dummy chemistry: a couple of first-order loss reactions so the solver does work
  RCONST = 0.0d0
  DO i = 1, nTot
     C(:,i) = 1.0d6 + REAL(MOD(i,97),8)
     RCONST(1:50,i) = 1.0d-4 * (1.0d0 + REAL(MOD(i,13),8)/13.0d0)
  ENDDO
  ! ---- single device, whole batch ----
  DO i = 1, nTot; Perm(i) = i; ENDDO
  t0 = omp_get_wtime()
  CALL gc_gpu_integrate_batch_dev( 0_c_int, nTot, 0.0d0, 1200.0d0, C, RCONST, ATOL, RTOL, &
                                   ICNTRL, RCNTRL, Perm, ISTAT, RSTAT, IERR, tT(0), tK(0) )
  t1 = omp_get_wtime(); tSingle = t1 - t0
  WRITE(*,'(a,i0,a,f8.3,a,i0)') 'single dev0: n=', nTot, ' t=', tSingle, ' nFail=', COUNT(IERR /= 1)
  ! ---- nUse devices concurrently, contiguous slices, one host thread each ----
  DO i = 1, nTot
     C(:,i) = 1.0d6 + REAL(MOD(i,97),8)
  ENDDO
  IERR = 0
  t0 = omp_get_wtime()
  !$OMP PARALLEL NUM_THREADS(nUse) PRIVATE(d, lo, hi, n)
  d  = omp_get_thread_num()
  lo = 1 + (nTot * d) / nUse
  hi = (nTot * (d+1)) / nUse
  n  = hi - lo + 1
  CALL gc_gpu_integrate_batch_dev( INT(d,c_int), n, 0.0d0, 1200.0d0, C(1,lo), RCONST(1,lo), &
                                   ATOL, RTOL, ICNTRL(1,lo), RCNTRL(1,lo), Perm, &
                                   ISTAT(1,lo), RSTAT(1,lo), IERR(lo), tT(d), tK(d) )
  !$OMP END PARALLEL
  t1 = omp_get_wtime(); tMulti = t1 - t0
  DO d = 0, nUse-1
     WRITE(*,'(a,i0,a,f8.3,a,f8.3)') '  dev', d, ': api_s=', tT(d), ' kernel_s=', tK(d)
  ENDDO
  WRITE(*,'(a,i0,a,f8.3,a,i0,a,f6.2)') 'multi ', nUse, ' devs: t=', tMulti, ' nFail=', COUNT(IERR /= 1), &
        ' speedup_vs_single=', tSingle / MAX(tMulti,1.0d-9)
  IF ( COUNT(IERR /= 1) == 0 .and. nUse >= 2 ) WRITE(*,'(a)') 'MGPU TEST OK'
  IF ( nUse < 2 ) WRITE(*,'(a)') 'MGPU TEST INCONCLUSIVE (only one device visible)'
END PROGRAM mgpu_test
SUBROUTINE acc_register_library( reg, unreg, lookup ) BIND(C, name='acc_register_library')
  USE ISO_C_BINDING, ONLY : C_PTR
  TYPE(C_PTR), VALUE :: reg, unreg, lookup
END SUBROUTINE acc_register_library
