!------------------------------------------------------------------------------
! gc_gpu_batch_api.F90 -- C-interoperable GPU entry points for the GEOS-Chem
! batched chemistry path (workstream C).
!
! Compiled with nvfortran (OpenACC) into libgcchemgpu.so together with the
! MODEL TREE's own patched KPP sources (gckpp_Precision/Parameters/JacobianSP/
! Function/Jacobian/LinearAlgebra) and gckpp_BatchIntegrator.F90.  The
! gfortran-built model calls the two BIND(C) routines below through a plain
! Fortran INTERFACE; no .mod files cross the compiler boundary.
!
! Design rules for the mixed-compiler link:
!   * no Fortran I/O anywhere in this library (the gfortran and nvfortran
!     Fortran runtimes must not both own stdout)
!   * built WITHOUT -mp (the model brings libgomp; we must not add libnvomp)
!   * only plain arrays / scalars by reference, C_INT / C_DOUBLE kinds
!
! gc_gpu_integrate_batch:
!   one launch per chemistry timestep.  Cell ic of the launch solves batch
!   column Perm(ic) (so the caller's cost-sort permutation is honoured
!   without repacking the host arrays).  Per cell it replicates
!   Integrate_Model (option merging: zeroed controls, ICNTRL(15)=5 default,
!   WHERE(ICNTRL_U/=0) and WHERE(RCNTRL_U>0) merges) and then calls the
!   statistics-carrying Integrate_Cell_S with ROUTINE-LOCAL workspace and
!   local copies of the per-cell state (the Integrate_Cell_L pattern that
!   the standalone GPU benchmark showed is what makes the kernel fast).
!   Returns ISTAT(1:8) (Nfun,Njac,Nstp,Nacc,Nrej,Ndec,Nsol,Nsng), RSTAT(1:3)
!   (Texit,Hexit,Hnew) and IERR per cell -- exactly what the model's scatter
!   consumes (ISTATUS(1:8) for KppDiags, RSTATE(3)=Hnew for the warm start).
!
! gc_gpu_selftest:
!   reports the number of NVIDIA devices + the device name and solves one
!   dummy cell on the device, so the link + runtime can be verified before
!   the real chemistry runs.
!------------------------------------------------------------------------------
MODULE gc_gpu_batch_api

  USE ISO_C_BINDING
  USE gckpp_Precision,       ONLY : dp
  USE gckpp_Parameters,      ONLY : NVAR, NFIX, NSPEC, NREACT
  USE gckpp_BatchIntegrator, ONLY : Integrate_Cell_S, NWORK_K, NWORK_J

  IMPLICIT NONE
  PRIVATE
  PUBLIC :: gc_gpu_integrate_batch
  PUBLIC :: gc_gpu_integrate_batch_dev
  PUBLIC :: gc_gpu_selftest

CONTAINS

  !--------------------------------------------------------------------------
  ! Device-side per-cell solve: Integrate_Model semantics with local
  ! workspace (mirrors Integrate_Cell_L / Integrate_Model of the model tree).
  !--------------------------------------------------------------------------
  SUBROUTINE gpu_solve_cell( Y, FIXc, RCONSTc, Tstart, Tend,                 &
                             ATOL_U, RTOL_U, ICNTRL_U, RCNTRL_U,             &
                             ISTAT, RSTAT, IERR )
    !$acc routine seq
    REAL(dp), INTENT(INOUT) :: Y(NVAR)
    REAL(dp), INTENT(IN)    :: FIXc(NFIX)
    REAL(dp), INTENT(IN)    :: RCONSTc(NREACT)
    REAL(dp), INTENT(IN)    :: Tstart, Tend
    REAL(dp), INTENT(IN)    :: ATOL_U(NVAR), RTOL_U(NVAR)
    INTEGER,  INTENT(IN)    :: ICNTRL_U(20)
    REAL(dp), INTENT(IN)    :: RCNTRL_U(20)
    INTEGER,  INTENT(OUT)   :: ISTAT(8)
    REAL(dp), INTENT(OUT)   :: RSTAT(3)
    INTEGER,  INTENT(OUT)   :: IERR

    INTEGER  :: ICNTRL(20), kk
    REAL(dp) :: RCNTRL(20)
    ! Per-thread local workspace (interleaved local memory on the GPU)
    REAL(dp) :: Kw(NWORK_K), Ghimj(NWORK_J), Jac0(NWORK_J)
    REAL(dp) :: Ynew(NVAR), Fcn0(NVAR), Fcn(NVAR)
    REAL(dp) :: dFdT(NVAR), Yerr(NVAR), wP(NVAR), wD(NVAR)
    ! Local copies of the per-cell state
    REAL(dp) :: Yl(NVAR), FIXl(NFIX), RCONSTl(NREACT)
    REAL(dp) :: ATOLl(NVAR), RTOLl(NVAR)

    ! --- option merging, identical to Integrate_Model / stock INTEGRATE ---
    DO kk = 1, 20
       ICNTRL(kk) = 0
       RCNTRL(kk) = 0.0_dp
    ENDDO
    ICNTRL(15) = 5
    DO kk = 1, 20
       IF ( ICNTRL_U(kk) /= 0      ) ICNTRL(kk) = ICNTRL_U(kk)
       IF ( RCNTRL_U(kk) >  0.0_dp ) RCNTRL(kk) = RCNTRL_U(kk)
    ENDDO

    DO kk = 1, NVAR
       Yl(kk)    = Y(kk)
       ATOLl(kk) = ATOL_U(kk)
       RTOLl(kk) = RTOL_U(kk)
    ENDDO
    DO kk = 1, NFIX
       FIXl(kk) = FIXc(kk)
    ENDDO
    DO kk = 1, NREACT
       RCONSTl(kk) = RCONSTc(kk)
    ENDDO

    CALL Integrate_Cell_S( Yl, FIXl, RCONSTl, Tstart, Tend,                  &
                           ATOLl, RTOLl, ICNTRL, RCNTRL,                     &
                           Kw, Ghimj, Jac0, Ynew, Fcn0, Fcn, dFdT, Yerr,     &
                           wP, wD, ISTAT, RSTAT, IERR )

    DO kk = 1, NVAR
       Y(kk) = Yl(kk)
    ENDDO
  END SUBROUTINE gpu_solve_cell

  !--------------------------------------------------------------------------
  ! Batch launch (one call per chemistry timestep)
  !--------------------------------------------------------------------------
  SUBROUTINE gc_gpu_integrate_batch( nCell, Tstart, Tend, C, RCONST,         &
                                     ATOL, RTOL, ICNTRL, RCNTRL, Perm,       &
                                     ISTAT, RSTAT, IERR, tTotal, tKernel )   &
             BIND(C, name='gc_gpu_integrate_batch')
    ! Single-device entry point (unchanged behaviour): runs on the calling
    ! host thread's current device.
    INTEGER(c_int),    VALUE       :: nCell
    REAL(c_double),    VALUE       :: Tstart, Tend
    REAL(c_double),    INTENT(INOUT) :: C(NSPEC, nCell)
    REAL(c_double),    INTENT(IN)    :: RCONST(NREACT, nCell)
    REAL(c_double),    INTENT(IN)    :: ATOL(NVAR), RTOL(NVAR)
    INTEGER(c_int),    INTENT(IN)    :: ICNTRL(20, nCell)
    REAL(c_double),    INTENT(IN)    :: RCNTRL(20, nCell)
    INTEGER(c_int),    INTENT(IN)    :: Perm(nCell)
    INTEGER(c_int),    INTENT(OUT)   :: ISTAT(8, nCell)
    REAL(c_double),    INTENT(OUT)   :: RSTAT(3, nCell)
    INTEGER(c_int),    INTENT(OUT)   :: IERR(nCell)
    REAL(c_double),    INTENT(OUT)   :: tTotal, tKernel
    CALL gc_gpu_integrate_batch_dev( -1_c_int, nCell, Tstart, Tend, C, RCONST, &
                                     ATOL, RTOL, ICNTRL, RCNTRL, Perm,         &
                                     ISTAT, RSTAT, IERR, tTotal, tKernel )
  END SUBROUTINE gc_gpu_integrate_batch

  !--------------------------------------------------------------------------
  ! Multi-device entry point: dev >= 0 selects the NVIDIA device the CALLING
  ! HOST THREAD uses for this and later launches (OpenACC keeps a current
  ! device per host thread); dev < 0 leaves it unchanged.  The model calls
  ! this from nDev different OpenMP threads at the same time, each with a
  ! disjoint contiguous range of cells, so nDev GPUs work concurrently.
  ! The `set device_num` DIRECTIVE is used (not the acc_set_device_num API
  ! routine, which the gfortran model would bind to libgomp at run time).
  !--------------------------------------------------------------------------
  SUBROUTINE gc_gpu_integrate_batch_dev( dev, nCell, Tstart, Tend, C, RCONST, &
                                     ATOL, RTOL, ICNTRL, RCNTRL, Perm,       &
                                     ISTAT, RSTAT, IERR, tTotal, tKernel )   &
             BIND(C, name='gc_gpu_integrate_batch_dev')
    INTEGER(c_int),    VALUE       :: dev
    INTEGER(c_int),    VALUE       :: nCell
    REAL(c_double),    VALUE       :: Tstart, Tend
    REAL(c_double),    INTENT(INOUT) :: C(NSPEC, nCell)     ! VAR=1:NVAR, FIX=NVAR+1:NSPEC
    REAL(c_double),    INTENT(IN)    :: RCONST(NREACT, nCell)
    REAL(c_double),    INTENT(IN)    :: ATOL(NVAR), RTOL(NVAR)
    INTEGER(c_int),    INTENT(IN)    :: ICNTRL(20, nCell)
    REAL(c_double),    INTENT(IN)    :: RCNTRL(20, nCell)
    INTEGER(c_int),    INTENT(IN)    :: Perm(nCell)          ! launch order
    INTEGER(c_int),    INTENT(OUT)   :: ISTAT(8, nCell)
    REAL(c_double),    INTENT(OUT)   :: RSTAT(3, nCell)
    INTEGER(c_int),    INTENT(OUT)   :: IERR(nCell)
    REAL(c_double),    INTENT(OUT)   :: tTotal, tKernel      ! seconds

    INTEGER    :: ic, idx
    INTEGER(c_int) :: devSel
    INTEGER(8) :: c0, c1, c2, c3, crate
    INTERFACE
       FUNCTION gc_gpu_set_device( d ) BIND(C, name='gc_gpu_set_device')
         IMPORT :: c_int
         INTEGER(c_int), VALUE :: d
         INTEGER(c_int)        :: gc_gpu_set_device
       END FUNCTION gc_gpu_set_device
    END INTERFACE

    tTotal  = 0.0_dp
    tKernel = 0.0_dp
    IF ( dev >= 0 ) THEN
       ! Select the device for this host thread through NVIDIA's own runtime
       ! (dlsym on libacchost, see gc_gpu_devinfo.c).  Neither the
       ! acc_set_device_num API nor the `!$acc set` directive can be used
       ! here: in the gfortran-built model both bind to libgomp.
       devSel = gc_gpu_set_device( dev )
       IF ( devSel /= dev ) THEN
          IERR(1:nCell) = -99          ! device selection failed
          RETURN
       ENDIF
    ENDIF
    IF ( nCell <= 0 ) RETURN

    CALL SYSTEM_CLOCK( c0, crate )
    !$acc data copy(C) copyin(RCONST, ATOL, RTOL, ICNTRL, RCNTRL, Perm)      &
    !$acc      copyout(ISTAT, RSTAT, IERR)
    CALL SYSTEM_CLOCK( c1 )
    !$acc parallel loop gang vector private(idx)
    DO ic = 1, nCell
       idx = Perm(ic)
       CALL gpu_solve_cell( C(1:NVAR,idx), C(NVAR+1:NSPEC,idx),              &
                            RCONST(:,idx), Tstart, Tend, ATOL, RTOL,         &
                            ICNTRL(:,idx), RCNTRL(:,idx),                    &
                            ISTAT(:,idx), RSTAT(:,idx), IERR(idx) )
    ENDDO
    CALL SYSTEM_CLOCK( c2 )
    !$acc end data
    CALL SYSTEM_CLOCK( c3 )

    tKernel = REAL( c2 - c1, dp ) / REAL( crate, dp )
    tTotal  = REAL( c3 - c0, dp ) / REAL( crate, dp )
  END SUBROUTINE gc_gpu_integrate_batch_dev

  !--------------------------------------------------------------------------
  ! Self test: solve one dummy cell (all RCONST = 0) on the device.
  ! rc = IERR of that solve (1 on success); nsteps = internal steps taken.
  !--------------------------------------------------------------------------
  SUBROUTINE gc_gpu_selftest( rc, nsteps, tTotal ) BIND(C, name='gc_gpu_selftest')
    ! NOTE: gc_gpu_info (device count / name) is implemented in C
    ! (gc_gpu_devinfo.c) via the CUDA driver API -- see the comment there:
    ! the public acc_* API would bind to the model's libgomp at runtime.
    INTEGER(c_int), INTENT(OUT) :: rc, nsteps
    REAL(c_double), INTENT(OUT) :: tTotal
    REAL(dp) :: C(NSPEC,1), RCONST(NREACT,1), ATOL(NVAR), RTOL(NVAR)
    REAL(dp) :: RCNTRL(20,1), RSTAT(3,1), tK
    INTEGER  :: ICNTRL(20,1), ISTAT(8,1), IERR(1), Perm(1)

    C      = 1.0e6_dp
    RCONST = 0.0_dp
    ATOL   = 1.0e-2_dp
    RTOL   = 0.5e-2_dp
    ICNTRL = 0
    ICNTRL(1,1)  = 1      ! autonomous
    ICNTRL(2,1)  = 0      ! vector tolerances
    ICNTRL(3,1)  = 4      ! Rodas3
    ICNTRL(7,1)  = 1
    ICNTRL(15,1) = -1
    RCNTRL = 0.0_dp
    Perm(1) = 1
    CALL gc_gpu_integrate_batch( 1, 0.0_dp, 1200.0_dp, C, RCONST, ATOL, RTOL, &
                                 ICNTRL, RCNTRL, Perm, ISTAT, RSTAT, IERR,    &
                                 tTotal, tK )
    rc     = IERR(1)
    nsteps = ISTAT(3,1)
  END SUBROUTINE gc_gpu_selftest

END MODULE gc_gpu_batch_api
