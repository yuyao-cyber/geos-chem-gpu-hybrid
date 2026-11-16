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
    IF ( ws_global_mode() ) THEN
       ! low-memory mode: workspace in device global arrays, chunked launches
       CALL integrate_batch_global_ws( nCell, Tstart, Tend, C, RCONST,       &
                                       ATOL, RTOL, ICNTRL, RCNTRL, Perm,    &
                                       ISTAT, RSTAT, IERR, tTotal, tKernel )
       RETURN
    ENDIF

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
  ! LOW-MEMORY MODE (env GC_GPU_WORKSPACE=global).
  ! The per-cell solver workspace (stage vectors, Jacobian, LU factors,
  ! ~122 KB per cell) lives in device GLOBAL arrays sized for a CHUNK of
  ! cells (env GC_GPU_CHUNK, default 32768) instead of in per-thread local
  ! memory.  The per-thread stack then only holds the generated kernels' own
  ! scratch (~30 KB), so NV_ACC_CUDA_STACKSIZE can drop from 150-160 KB to
  ! ~40 KB and the device footprint from 25-35 GB to roughly
  !   stack 40 KB x resident threads  +  chunk x 122 KB  +  batch arrays,
  ! i.e. ~12 GB on a V100 with the default chunk.  Measured in the standalone
  ! benchmark as ~25% slower per kernel than the local-workspace mode; it
  ! exists so the solver can share a GPU with other jobs.
  !--------------------------------------------------------------------------
  LOGICAL FUNCTION ws_global_mode()
    CHARACTER(LEN=16) :: v
    INTEGER :: st
    v = ''
    CALL GET_ENVIRONMENT_VARIABLE( 'GC_GPU_WORKSPACE', v, STATUS=st )
    ws_global_mode = ( st == 0 .and. ( v(1:6) == 'global' .or. v(1:6) == 'GLOBAL' ) )
  END FUNCTION ws_global_mode

  INTEGER FUNCTION env_int( name, default )
    ! integer environment variable without Fortran I/O (manual parse)
    CHARACTER(LEN=*), INTENT(IN) :: name
    INTEGER,          INTENT(IN) :: default
    CHARACTER(LEN=24) :: v
    INTEGER :: st, i, d, val
    v = ''
    CALL GET_ENVIRONMENT_VARIABLE( name, v, STATUS=st )
    env_int = default
    IF ( st /= 0 ) RETURN
    val = 0
    DO i = 1, LEN_TRIM(v)
       d = ICHAR(v(i:i)) - ICHAR('0')
       IF ( d < 0 .or. d > 9 ) RETURN
       val = val*10 + d
    ENDDO
    IF ( val > 0 ) env_int = val
  END FUNCTION env_int

  SUBROUTINE gpu_solve_cell_ws( Y, FIXc, RCONSTc, Tstart, Tend,              &
                                ATOL_U, RTOL_U, ICNTRL_U, RCNTRL_U,          &
                                Kw, Ghimj, Jac0, Ynew, Fcn0, Fcn, dFdT, Yerr,&
                                wP, wD, ISTAT, RSTAT, IERR )
    !$acc routine seq
    ! Same option merging as gpu_solve_cell, but the workspace is passed in
    ! (slices of the chunk arrays) and the inputs are used in place.
    REAL(dp), INTENT(INOUT) :: Y(NVAR)
    REAL(dp), INTENT(IN)    :: FIXc(NFIX), RCONSTc(NREACT)
    REAL(dp), INTENT(IN)    :: Tstart, Tend, ATOL_U(NVAR), RTOL_U(NVAR)
    INTEGER,  INTENT(IN)    :: ICNTRL_U(20)
    REAL(dp), INTENT(IN)    :: RCNTRL_U(20)
    REAL(dp), INTENT(INOUT) :: Kw(NWORK_K), Ghimj(NWORK_J), Jac0(NWORK_J)
    REAL(dp), INTENT(INOUT) :: Ynew(NVAR), Fcn0(NVAR), Fcn(NVAR), dFdT(NVAR)
    REAL(dp), INTENT(INOUT) :: Yerr(NVAR), wP(NVAR), wD(NVAR)
    INTEGER,  INTENT(OUT)   :: ISTAT(8), IERR
    REAL(dp), INTENT(OUT)   :: RSTAT(3)
    INTEGER  :: ICNTRL(20), kk
    REAL(dp) :: RCNTRL(20)
    DO kk = 1, 20
       ICNTRL(kk) = 0
       RCNTRL(kk) = 0.0_dp
    ENDDO
    ICNTRL(15) = 5
    DO kk = 1, 20
       IF ( ICNTRL_U(kk) /= 0      ) ICNTRL(kk) = ICNTRL_U(kk)
       IF ( RCNTRL_U(kk) >  0.0_dp ) RCNTRL(kk) = RCNTRL_U(kk)
    ENDDO
    CALL Integrate_Cell_S( Y, FIXc, RCONSTc, Tstart, Tend,                   &
                           ATOL_U, RTOL_U, ICNTRL, RCNTRL,                   &
                           Kw, Ghimj, Jac0, Ynew, Fcn0, Fcn, dFdT, Yerr,     &
                           wP, wD, ISTAT, RSTAT, IERR )
  END SUBROUTINE gpu_solve_cell_ws

  SUBROUTINE integrate_batch_global_ws( nCell, Tstart, Tend, C, RCONST,      &
                                        ATOL, RTOL, ICNTRL, RCNTRL, Perm,   &
                                        ISTAT, RSTAT, IERR, tTotal, tKernel )
    INTEGER(c_int), VALUE         :: nCell
    REAL(c_double), VALUE         :: Tstart, Tend
    REAL(c_double), INTENT(INOUT) :: C(NSPEC, nCell)
    REAL(c_double), INTENT(IN)    :: RCONST(NREACT, nCell)
    REAL(c_double), INTENT(IN)    :: ATOL(NVAR), RTOL(NVAR)
    INTEGER(c_int), INTENT(IN)    :: ICNTRL(20, nCell)
    REAL(c_double), INTENT(IN)    :: RCNTRL(20, nCell)
    INTEGER(c_int), INTENT(IN)    :: Perm(nCell)
    INTEGER(c_int), INTENT(OUT)   :: ISTAT(8, nCell)
    REAL(c_double), INTENT(OUT)   :: RSTAT(3, nCell)
    INTEGER(c_int), INTENT(OUT)   :: IERR(nCell)
    REAL(c_double), INTENT(OUT)   :: tTotal, tKernel
    REAL(dp), ALLOCATABLE :: Kw(:,:), Ghimj(:,:), Jac0(:,:)
    REAL(dp), ALLOCATABLE :: Ynew(:,:), Fcn0(:,:), Fcn(:,:), dFdT(:,:)
    REAL(dp), ALLOCATABLE :: Yerr(:,:), wP(:,:), wD(:,:)
    INTEGER    :: nchunk, nch, cbase, ic, idx
    INTEGER(8) :: t0, t1, t2, t3, crate
    nchunk = env_int( 'GC_GPU_CHUNK', 32768 )
    nchunk = MAX( 1024, MIN( nchunk, nCell ) )
    ALLOCATE( Kw(NWORK_K,nchunk), Ghimj(NWORK_J,nchunk), Jac0(NWORK_J,nchunk), &
              Ynew(NVAR,nchunk), Fcn0(NVAR,nchunk), Fcn(NVAR,nchunk),         &
              dFdT(NVAR,nchunk), Yerr(NVAR,nchunk), wP(NVAR,nchunk),           &
              wD(NVAR,nchunk) )
    tKernel = 0.0_dp
    CALL SYSTEM_CLOCK( t0, crate )
    !$acc data copy(C) copyin(RCONST, ATOL, RTOL, ICNTRL, RCNTRL, Perm)      &
    !$acc      copyout(ISTAT, RSTAT, IERR)                                   &
    !$acc      create(Kw, Ghimj, Jac0, Ynew, Fcn0, Fcn, dFdT, Yerr, wP, wD)
    DO cbase = 0, nCell-1, nchunk
       nch = MIN( nchunk, nCell - cbase )
       CALL SYSTEM_CLOCK( t1 )
       !$acc parallel loop gang vector private(idx)
       DO ic = 1, nch
          idx = Perm(cbase+ic)
          CALL gpu_solve_cell_ws( C(1:NVAR,idx), C(NVAR+1:NSPEC,idx),         &
                   RCONST(:,idx), Tstart, Tend, ATOL, RTOL,                  &
                   ICNTRL(:,idx), RCNTRL(:,idx),                             &
                   Kw(:,ic), Ghimj(:,ic), Jac0(:,ic), Ynew(:,ic), Fcn0(:,ic),&
                   Fcn(:,ic), dFdT(:,ic), Yerr(:,ic), wP(:,ic), wD(:,ic),    &
                   ISTAT(:,idx), RSTAT(:,idx), IERR(idx) )
       ENDDO
       CALL SYSTEM_CLOCK( t2 )
       tKernel = tKernel + REAL( t2 - t1, dp ) / REAL( crate, dp )
    ENDDO
    !$acc end data
    CALL SYSTEM_CLOCK( t3 )
    tTotal = REAL( t3 - t0, dp ) / REAL( crate, dp )
    DEALLOCATE( Kw, Ghimj, Jac0, Ynew, Fcn0, Fcn, dFdT, Yerr, wP, wD )
  END SUBROUTINE integrate_batch_global_ws

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
