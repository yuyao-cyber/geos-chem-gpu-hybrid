!------------------------------------------------------------------------------
! gckpp_BatchIntegrator.F90  (GC-Classic in-model version)
!
! A batched, GPU-offloadable version of the KPP Rosenbrock integrator
! (Rodas3 path), refactored from gckpp_Integrator.F90 so that ALL per-cell
! state is passed explicitly as arguments instead of living in module
! globals (C, VAR, FIX, RCONST, TIME).  This is the enabling transformation
! for running many grid cells concurrently on a GPU:
!
!   - gckpp_Integrator.F90 (original): one cell at a time, state in
!     THREADPRIVATE module globals; OpenMP CPU threads only.
!   - this file: Integrate_Cell() is self-contained ("!$acc routine seq"),
!     called from a parallel loop over cells on CPU (OpenMP) or GPU
!     (OpenACC).  Large work arrays are passed in as workspace slices so
!     no device-side allocation is needed.
!
! Numerical behavior replicates the original ros_Integrator EXACTLY
! (same expressions, same operation order) for the configuration used by
! GEOS-Chem full-chemistry cells with auto-reduction disabled:
!   ICNTRL(1)=1 (autonomous), ICNTRL(2)=0 (vector tolerances),
!   ICNTRL(3)=4 (Rodas3), ICNTRL(15)=-1 (no Update_* calls in integrator).
! Differences from the original, all deliberate:
!   - No PRINT statements (device code); failures return IERR codes.
!   - Roundoff is the compile-time constant 2**-52 (the exact value that
!     WLAMCH('E') computes at runtime).
!   - Auto-reduce (ICNTRL(12)) paths are not implemented (IERR=-90 if
!     requested).
!
! This in-model copy differs from the Phase-0 validated file
! ($WS/batch_src/gckpp_BatchIntegrator.F90) as follows:
!   - The interleaved (Integrate_Cell_I) and warp-cooperative
!     (Integrate_Cell_W) experimental variants are removed (they depend on
!     modules that only exist in the KPP-Standalone-batch experiment tree).
!   - Integrate_Cell_S is added: an exact copy of Integrate_Cell that
!     additionally maintains the ISTATUS(1:8) counters and RSTATUS(1:3)
!     outputs with the SAME semantics as the stock ros_Integrator
!     (Nfun, Njac, Nstp, Nacc, Nrej, Ndec, Nsol, Nsng / Texit, Hexit,
!     Hnew).  The numerics are untouched; only counter bookkeeping and
!     the extra outputs are added.
!   - Integrate_Model is added: the model-facing wrapper that replicates
!     the option-merging done by SUBROUTINE INTEGRATE in
!     gckpp_Integrator.F90 (defaults, WHERE(ICNTRL_U/=0),
!     WHERE(RCNTRL_U>0)) and then calls Integrate_Cell_S with thread-local
!     workspace.  It is what GeosCore/fullchem_mod.F90 calls per cell on
!     the batched path (GC_BATCH_CHEM=1).
!------------------------------------------------------------------------------
MODULE gckpp_BatchIntegrator

  USE gckpp_Precision,  ONLY : dp
  USE gckpp_Parameters, ONLY : NVAR, NFIX, NSPEC, NREACT, LU_NONZERO
  IMPLICIT NONE
  PRIVATE
  PUBLIC :: Integrate_Cell
  PUBLIC :: Integrate_Cell_L
  PUBLIC :: Integrate_Cell_S
  PUBLIC :: Integrate_Model
  PUBLIC :: NWORK_K, NWORK_J

  ! Rodas3 has 4 stages
  INTEGER, PARAMETER :: ROS_S = 4
  INTEGER, PARAMETER :: NWORK_K = NVAR*ROS_S     ! size of stage workspace
  INTEGER, PARAMETER :: NWORK_J = LU_NONZERO     ! size of Jacobian workspace

  ! Exact value computed by WLAMCH('E') in the original code
  ! (measured: 2**-52, i.e. EPSILON(1.0_dp))
  REAL(dp), PARAMETER :: ROUNDOFF = 2.0_dp**(-52)

  REAL(dp), PARAMETER :: ZERO = 0.0_dp, ONE = 1.0_dp, HALF = 0.5_dp
  REAL(dp), PARAMETER :: DELTAMIN = 1.0E-5_dp

CONTAINS

  SUBROUTINE Integrate_Cell( Y, FIXc, RCONSTc, Tstart, Tend,                 &
                             AbsTol, RelTol, ICNTRL, RCNTRL,                 &
                             Kw, Ghimj, Jac0, Ynew, Fcn0, Fcn, dFdT, Yerr,   &
                             wP, wD, Nsteps, Hexit, IERR )
    !$acc routine seq
    ! One grid cell: integrate Y over [Tstart,Tend] with rate constants
    ! RCONSTc and fixed species FIXc.  Workspace arrays (Kw..Yerr) are
    ! caller-provided per-cell slices; nothing here allocates.
    USE gckpp_Function,      ONLY : Fun_SPLIT
    USE gckpp_Jacobian,      ONLY : Jac_SP
    USE gckpp_LinearAlgebra, ONLY : KppDecomp, KppSolve

    REAL(dp), INTENT(INOUT) :: Y(NVAR)          ! in: initial, out: final
    REAL(dp), INTENT(IN)    :: FIXc(NFIX)
    REAL(dp), INTENT(IN)    :: RCONSTc(NREACT)
    REAL(dp), INTENT(IN)    :: Tstart, Tend
    REAL(dp), INTENT(IN)    :: AbsTol(NVAR), RelTol(NVAR)
    INTEGER,  INTENT(IN)    :: ICNTRL(20)
    REAL(dp), INTENT(IN)    :: RCNTRL(20)
    REAL(dp), INTENT(INOUT) :: Kw(NWORK_K)      ! stage vectors K
    REAL(dp), INTENT(INOUT) :: Ghimj(NWORK_J)   ! LU-factored matrix
    REAL(dp), INTENT(INOUT) :: Jac0(NWORK_J)    ! Jacobian
    REAL(dp), INTENT(INOUT) :: Ynew(NVAR), Fcn0(NVAR), Fcn(NVAR)
    REAL(dp), INTENT(INOUT) :: dFdT(NVAR), Yerr(NVAR)
    REAL(dp), INTENT(INOUT) :: wP(NVAR), wD(NVAR)
    INTEGER,  INTENT(OUT)   :: Nsteps
    REAL(dp), INTENT(OUT)   :: Hexit
    INTEGER,  INTENT(OUT)   :: IERR

    ! Rodas3 coefficients (from SUBROUTINE Rodas3 of the original)
    REAL(dp), PARAMETER :: ros_A(6) = (/ 0.0_dp, 2.0_dp, 0.0_dp,             &
                                         2.0_dp, 0.0_dp, 1.0_dp /)
    REAL(dp), PARAMETER :: ros_C(6) = (/ 4.0_dp, 1.0_dp, -1.0_dp,            &
                                         1.0_dp, -1.0_dp, -(8.0_dp/3.0_dp) /)
    LOGICAL,  PARAMETER :: ros_NewF(4)  = (/ .TRUE., .FALSE., .TRUE., .TRUE. /)
    REAL(dp), PARAMETER :: ros_M(4)     = (/ 2.0_dp, 0.0_dp, 1.0_dp, 1.0_dp /)
    REAL(dp), PARAMETER :: ros_E(4)     = (/ 0.0_dp, 0.0_dp, 0.0_dp, 1.0_dp /)
    REAL(dp), PARAMETER :: ros_ELO      = 3.0_dp
    REAL(dp), PARAMETER :: ros_Alpha(4) = (/ 0.0_dp, 0.0_dp, 1.0_dp, 1.0_dp /)
    REAL(dp), PARAMETER :: ros_Gamma(4) = (/ 0.5_dp, 1.5_dp, 0.0_dp, 0.0_dp /)

    ! Locals (small)
    INTEGER  :: Max_no_steps, Direction, ioffset, i, j, k, istage
    INTEGER  :: ising, Nconsecutive, Nacc
    REAL(dp) :: Hmin, Hmax, Hstart, FacMin, FacMax, FacRej, FacSafe
    REAL(dp) :: T, H, Hnew, HC, HG, Fac, Tau, Err, ghinv, Delta
    REAL(dp) :: Ymax, Scal, aa
    LOGICAL  :: Autonomous, VectorTol, RejectLastH, RejectMoreH, Singular

    IERR   = 0
    Nsteps = 0
    Nacc   = 0
    Hexit  = ZERO

    !--- Option decoding (replicates Rosenbrock() defaults) ---
    IF ( ICNTRL(12) == 1 ) THEN
       IERR = -90                     ! auto-reduce not supported in batch
       RETURN
    ENDIF
    Autonomous = .NOT. ( ICNTRL(1) == 0 )
    VectorTol  = ( ICNTRL(2) == 0 )
    IF ( .NOT. ( ICNTRL(3) == 0 .OR. ICNTRL(3) == 4 ) ) THEN
       IERR = -2                      ! only Rodas3 implemented
       RETURN
    ENDIF
    IF ( ICNTRL(4) == 0 ) THEN
       Max_no_steps = 200000
    ELSE
       Max_no_steps = ICNTRL(4)
    ENDIF

    IF ( RCNTRL(1) > ZERO ) THEN
       Hmin = RCNTRL(1)
    ELSE
       Hmin = ZERO
    ENDIF
    IF ( RCNTRL(2) > ZERO ) THEN
       Hmax = MIN( ABS(RCNTRL(2)), ABS(Tend-Tstart) )
    ELSE
       Hmax = ABS(Tend-Tstart)
    ENDIF
    IF ( RCNTRL(3) > ZERO ) THEN
       Hstart = MIN( ABS(RCNTRL(3)), ABS(Tend-Tstart) )
    ELSE
       Hstart = MAX( Hmin, DELTAMIN )
    ENDIF
    IF ( RCNTRL(4) > ZERO ) THEN
       FacMin = RCNTRL(4)
    ELSE
       FacMin = 0.2_dp
    ENDIF
    IF ( RCNTRL(5) > ZERO ) THEN
       FacMax = RCNTRL(5)
    ELSE
       FacMax = 6.0_dp
    ENDIF
    IF ( RCNTRL(6) > ZERO ) THEN
       FacRej = RCNTRL(6)
    ELSE
       FacRej = 0.1_dp
    ENDIF
    IF ( RCNTRL(7) > ZERO ) THEN
       FacSafe = RCNTRL(7)
    ELSE
       FacSafe = 0.9_dp
    ENDIF

    !--- ros_Integrator body (exact replica, PRINTs removed) ---
    T = Tstart
    H = MIN( MAX( ABS(Hmin), ABS(Hstart) ), ABS(Hmax) )
    IF ( ABS(H) <= 10.0_dp*ROUNDOFF ) H = DELTAMIN

    IF ( Tend >= Tstart ) THEN
       Direction = +1
    ELSE
       Direction = -1
    ENDIF
    H = Direction*H

    RejectLastH = .FALSE.
    RejectMoreH = .FALSE.

    TimeLoop: DO WHILE ( (Direction > 0) .AND. ((T-Tend)+ROUNDOFF <= ZERO)   &
                    .OR. (Direction < 0) .AND. ((Tend-T)+ROUNDOFF <= ZERO) )

       IF ( Nsteps > Max_no_steps ) THEN
          IERR = -6
          RETURN
       ENDIF
       IF ( ((T+0.1_dp*H) == T) .OR. (H <= ROUNDOFF) ) THEN
          IERR = -7
          RETURN
       ENDIF

       H = MIN( H, ABS(Tend-T) )

       ! Fcn0 <- F(T, Y)  (Fun_SPLIT: identical path to the original
       ! standalone FunTemplate, for bit-faithful comparison)
       CALL Fun_SPLIT( Y, FIXc, RCONSTc, Fcn0, wP, wD )

       IF ( .NOT. Autonomous ) THEN
          ! dFdT by finite difference (ros_FunTimeDerivative replica)
          Delta = SQRT(ROUNDOFF)*MAX( 1.0E-6_dp, ABS(T) )
          CALL Fun_SPLIT( Y, FIXc, RCONSTc, dFdT, wP, wD )
          DO k = 1, NVAR
             dFdT(k) = ( dFdT(k) - Fcn0(k) ) / Delta
          ENDDO
       ENDIF

       ! Jac0 <- J(T, Y)
       CALL Jac_SP( Y, FIXc, RCONSTc, Jac0 )

  UntilAccepted: DO

       ! --- ros_PrepareMatrix replica ---
       Nconsecutive = 0
       Singular     = .TRUE.
       DO WHILE ( Singular )
          DO k = 1, LU_NONZERO
             Ghimj(k) = -Jac0(k)
          ENDDO
          ghinv = ONE/(Direction*H*ros_Gamma(1))
          CALL AddGhinvDiag( Ghimj, ghinv )
          CALL KppDecomp( Ghimj, ising )
          IF ( ising == 0 ) THEN
             Singular = .FALSE.
          ELSE
             Nconsecutive = Nconsecutive + 1
             IF ( Nconsecutive <= 5 ) THEN
                H = H*HALF
             ELSE
                IERR = -8
                RETURN
             ENDIF
          ENDIF
       ENDDO

       ! --- Stages ---
  Stage: DO istage = 1, ROS_S
          ioffset = NVAR*(istage-1)
          IF ( istage == 1 ) THEN
             DO k = 1, NVAR
                Fcn(k) = Fcn0(k)
             ENDDO
          ELSEIF ( ros_NewF(istage) ) THEN
             DO k = 1, NVAR
                Ynew(k) = Y(k)
             ENDDO
             DO j = 1, istage-1
                aa = ros_A( (istage-1)*(istage-2)/2 + j )
                DO k = 1, NVAR
                   Ynew(k) = Ynew(k) + aa*Kw(NVAR*(j-1)+k)
                ENDDO
             ENDDO
             Tau = T + ros_Alpha(istage)*Direction*H
             CALL Fun_SPLIT( Ynew, FIXc, RCONSTc, Fcn, wP, wD )
          ENDIF
          DO k = 1, NVAR
             Kw(ioffset+k) = Fcn(k)
          ENDDO
          DO j = 1, istage-1
             HC = ros_C( (istage-1)*(istage-2)/2 + j )/(Direction*H)
             DO k = 1, NVAR
                Kw(ioffset+k) = Kw(ioffset+k) + HC*Kw(NVAR*(j-1)+k)
             ENDDO
          ENDDO
          IF ( (.NOT. Autonomous) .AND. (ros_Gamma(istage) /= ZERO) ) THEN
             HG = Direction*H*ros_Gamma(istage)
             DO k = 1, NVAR
                Kw(ioffset+k) = Kw(ioffset+k) + HG*dFdT(k)
             ENDDO
          ENDIF
          CALL KppSolve( Ghimj, Kw(ioffset+1:ioffset+NVAR) )
       ENDDO Stage

       ! --- New solution ---
       DO k = 1, NVAR
          Ynew(k) = Y(k)
       ENDDO
       DO j = 1, ROS_S
          DO k = 1, NVAR
             Ynew(k) = Ynew(k) + ros_M(j)*Kw(NVAR*(j-1)+k)
          ENDDO
       ENDDO

       ! --- Error estimate ---
       DO k = 1, NVAR
          Yerr(k) = ZERO
       ENDDO
       DO j = 1, ROS_S
          DO k = 1, NVAR
             Yerr(k) = Yerr(k) + ros_E(j)*Kw(NVAR*(j-1)+k)
          ENDDO
       ENDDO
       ! ros_ErrorNorm replica
       Err = ZERO
       DO i = 1, NVAR
          Ymax = MAX( ABS(Y(i)), ABS(Ynew(i)) )
          IF ( VectorTol ) THEN
             Scal = AbsTol(i) + RelTol(i)*Ymax
          ELSE
             Scal = AbsTol(1) + RelTol(1)*Ymax
          ENDIF
          Err = Err + ( Yerr(i)/Scal )**2
       ENDDO
       Err = SQRT( Err/NVAR )
       Err = MAX( Err, 1.0d-10 )

       Fac  = MIN( FacMax, MAX( FacMin, FacSafe/Err**(ONE/ros_ELO) ) )
       Hnew = H*Fac

       Nsteps = Nsteps + 1
       IF ( (Err <= ONE) .OR. (H <= Hmin) ) THEN     ! accept
          Nacc = Nacc + 1
          IF ( ICNTRL(16) == 1 ) THEN
             DO k = 1, NVAR
                Y(k) = MAX( Ynew(k), ZERO )
             ENDDO
          ELSE
             DO k = 1, NVAR
                Y(k) = Ynew(k)
             ENDDO
          ENDIF
          T = T + Direction*H
          Hnew = MAX( Hmin, MIN( Hnew, Hmax ) )
          IF ( RejectLastH ) Hnew = MIN( Hnew, H )
          Hexit = H
          RejectLastH = .FALSE.
          RejectMoreH = .FALSE.
          H = Hnew
          EXIT UntilAccepted
       ELSE                                          ! reject
          IF ( RejectMoreH ) Hnew = H*FacRej
          RejectMoreH = RejectLastH
          RejectLastH = .TRUE.
          H = Hnew
       ENDIF

       ENDDO UntilAccepted

    ENDDO TimeLoop

    IERR = 1     ! success

  END SUBROUTINE Integrate_Cell

  SUBROUTINE Integrate_Cell_L( Y, FIXc, RCONSTc, Tstart, Tend,               &
                               AbsTol, RelTol, ICNTRL, RCNTRL,               &
                               Nsteps, Hexit, IERR )
    !$acc routine seq
    ! Same as Integrate_Cell, but the workspace lives in ROUTINE-LOCAL
    ! automatic arrays.  On the GPU these are placed in per-thread local
    ! memory, which the hardware interleaves across threads -- giving
    ! coalesced access without transforming the generated kernels.
    REAL(dp), INTENT(INOUT) :: Y(NVAR)
    REAL(dp), INTENT(IN)    :: FIXc(NFIX)
    REAL(dp), INTENT(IN)    :: RCONSTc(NREACT)
    REAL(dp), INTENT(IN)    :: Tstart, Tend
    REAL(dp), INTENT(IN)    :: AbsTol(NVAR), RelTol(NVAR)
    INTEGER,  INTENT(IN)    :: ICNTRL(20)
    REAL(dp), INTENT(IN)    :: RCNTRL(20)
    INTEGER,  INTENT(OUT)   :: Nsteps
    REAL(dp), INTENT(OUT)   :: Hexit
    INTEGER,  INTENT(OUT)   :: IERR
    ! Per-thread local workspace (interleaved local memory on GPU)
    REAL(dp) :: Kw(NWORK_K), Ghimj(NWORK_J), Jac0(NWORK_J)
    REAL(dp) :: Ynew(NVAR), Fcn0(NVAR), Fcn(NVAR)
    REAL(dp) :: dFdT(NVAR), Yerr(NVAR), wP(NVAR), wD(NVAR)
    ! Local copies of the INPUT state as well: RCONST (1058 values) is
    ! re-read by Fun/Jac on every internal stage -- paying one strided
    ! copy-in here converts all those re-reads to interleaved local
    ! accesses.  Same for Y (read/write), FIX, and the tolerances.
    REAL(dp) :: Yl(NVAR), FIXl(NFIX), RCONSTl(NREACT)
    REAL(dp) :: ATOLl(NVAR), RTOLl(NVAR)
    INTEGER  :: kk
    DO kk = 1, NVAR
       Yl(kk)    = Y(kk)
       ATOLl(kk) = AbsTol(kk)
       RTOLl(kk) = RelTol(kk)
    ENDDO
    DO kk = 1, NFIX
       FIXl(kk) = FIXc(kk)
    ENDDO
    DO kk = 1, NREACT
       RCONSTl(kk) = RCONSTc(kk)
    ENDDO
    CALL Integrate_Cell( Yl, FIXl, RCONSTl, Tstart, Tend,                    &
                         ATOLl, RTOLl, ICNTRL, RCNTRL,                       &
                         Kw, Ghimj, Jac0, Ynew, Fcn0, Fcn, dFdT, Yerr,       &
                         wP, wD, Nsteps, Hexit, IERR )
    DO kk = 1, NVAR
       Y(kk) = Yl(kk)
    ENDDO
  END SUBROUTINE Integrate_Cell_L

  SUBROUTINE Integrate_Cell_S( Y, FIXc, RCONSTc, Tstart, Tend,               &
                               AbsTol, RelTol, ICNTRL, RCNTRL,               &
                               Kw, Ghimj, Jac0, Ynew, Fcn0, Fcn, dFdT, Yerr, &
                               wP, wD, ISTAT, RSTAT, IERR )
    !$acc routine seq
    ! EXACT copy of Integrate_Cell, plus the statistics bookkeeping of the
    ! stock ros_Integrator so that the model's KppDiags diagnostics and the
    ! warm-start cache (Hnew) are preserved bit-for-bit:
    !   ISTAT(1)=Nfun  ISTAT(2)=Njac  ISTAT(3)=Nstp  ISTAT(4)=Nacc
    !   ISTAT(5)=Nrej  ISTAT(6)=Ndec  ISTAT(7)=Nsol  ISTAT(8)=Nsng
    !   RSTAT(1)=Texit RSTAT(2)=Hexit RSTAT(3)=Hnew
    ! The numerics are IDENTICAL to Integrate_Cell (counter increments and
    ! output stores only; no floating-point expression changed).
    USE gckpp_Function,      ONLY : Fun_SPLIT
    USE gckpp_Jacobian,      ONLY : Jac_SP
    USE gckpp_LinearAlgebra, ONLY : KppDecomp, KppSolve

    REAL(dp), INTENT(INOUT) :: Y(NVAR)          ! in: initial, out: final
    REAL(dp), INTENT(IN)    :: FIXc(NFIX)
    REAL(dp), INTENT(IN)    :: RCONSTc(NREACT)
    REAL(dp), INTENT(IN)    :: Tstart, Tend
    REAL(dp), INTENT(IN)    :: AbsTol(NVAR), RelTol(NVAR)
    INTEGER,  INTENT(IN)    :: ICNTRL(20)
    REAL(dp), INTENT(IN)    :: RCNTRL(20)
    REAL(dp), INTENT(INOUT) :: Kw(NWORK_K)      ! stage vectors K
    REAL(dp), INTENT(INOUT) :: Ghimj(NWORK_J)   ! LU-factored matrix
    REAL(dp), INTENT(INOUT) :: Jac0(NWORK_J)    ! Jacobian
    REAL(dp), INTENT(INOUT) :: Ynew(NVAR), Fcn0(NVAR), Fcn(NVAR)
    REAL(dp), INTENT(INOUT) :: dFdT(NVAR), Yerr(NVAR)
    REAL(dp), INTENT(INOUT) :: wP(NVAR), wD(NVAR)
    INTEGER,  INTENT(OUT)   :: ISTAT(8)
    REAL(dp), INTENT(OUT)   :: RSTAT(3)
    INTEGER,  INTENT(OUT)   :: IERR

    ! Rodas3 coefficients (from SUBROUTINE Rodas3 of the original)
    REAL(dp), PARAMETER :: ros_A(6) = (/ 0.0_dp, 2.0_dp, 0.0_dp,             &
                                         2.0_dp, 0.0_dp, 1.0_dp /)
    REAL(dp), PARAMETER :: ros_C(6) = (/ 4.0_dp, 1.0_dp, -1.0_dp,            &
                                         1.0_dp, -1.0_dp, -(8.0_dp/3.0_dp) /)
    LOGICAL,  PARAMETER :: ros_NewF(4)  = (/ .TRUE., .FALSE., .TRUE., .TRUE. /)
    REAL(dp), PARAMETER :: ros_M(4)     = (/ 2.0_dp, 0.0_dp, 1.0_dp, 1.0_dp /)
    REAL(dp), PARAMETER :: ros_E(4)     = (/ 0.0_dp, 0.0_dp, 0.0_dp, 1.0_dp /)
    REAL(dp), PARAMETER :: ros_ELO      = 3.0_dp
    REAL(dp), PARAMETER :: ros_Alpha(4) = (/ 0.0_dp, 0.0_dp, 1.0_dp, 1.0_dp /)
    REAL(dp), PARAMETER :: ros_Gamma(4) = (/ 0.5_dp, 1.5_dp, 0.0_dp, 0.0_dp /)

    ! Locals (small)
    INTEGER  :: Max_no_steps, Direction, ioffset, i, j, k, istage
    INTEGER  :: ising, Nconsecutive, Nsteps
    REAL(dp) :: Hmin, Hmax, Hstart, FacMin, FacMax, FacRej, FacSafe
    REAL(dp) :: T, H, Hnew, HC, HG, Fac, Tau, Err, ghinv, Delta
    REAL(dp) :: Ymax, Scal, aa
    LOGICAL  :: Autonomous, VectorTol, RejectLastH, RejectMoreH, Singular

    IERR   = 0
    Nsteps = 0
    ISTAT  = 0
    RSTAT  = ZERO

    !--- Option decoding (replicates Rosenbrock() defaults) ---
    IF ( ICNTRL(12) == 1 ) THEN
       IERR = -90                     ! auto-reduce not supported in batch
       RETURN
    ENDIF
    Autonomous = .NOT. ( ICNTRL(1) == 0 )
    VectorTol  = ( ICNTRL(2) == 0 )
    IF ( .NOT. ( ICNTRL(3) == 0 .OR. ICNTRL(3) == 4 ) ) THEN
       IERR = -2                      ! only Rodas3 implemented
       RETURN
    ENDIF
    IF ( ICNTRL(4) == 0 ) THEN
       Max_no_steps = 200000
    ELSE
       Max_no_steps = ICNTRL(4)
    ENDIF

    IF ( RCNTRL(1) > ZERO ) THEN
       Hmin = RCNTRL(1)
    ELSE
       Hmin = ZERO
    ENDIF
    IF ( RCNTRL(2) > ZERO ) THEN
       Hmax = MIN( ABS(RCNTRL(2)), ABS(Tend-Tstart) )
    ELSE
       Hmax = ABS(Tend-Tstart)
    ENDIF
    IF ( RCNTRL(3) > ZERO ) THEN
       Hstart = MIN( ABS(RCNTRL(3)), ABS(Tend-Tstart) )
    ELSE
       Hstart = MAX( Hmin, DELTAMIN )
    ENDIF
    IF ( RCNTRL(4) > ZERO ) THEN
       FacMin = RCNTRL(4)
    ELSE
       FacMin = 0.2_dp
    ENDIF
    IF ( RCNTRL(5) > ZERO ) THEN
       FacMax = RCNTRL(5)
    ELSE
       FacMax = 6.0_dp
    ENDIF
    IF ( RCNTRL(6) > ZERO ) THEN
       FacRej = RCNTRL(6)
    ELSE
       FacRej = 0.1_dp
    ENDIF
    IF ( RCNTRL(7) > ZERO ) THEN
       FacSafe = RCNTRL(7)
    ELSE
       FacSafe = 0.9_dp
    ENDIF

    !--- ros_Integrator body (exact replica, PRINTs removed) ---
    T = Tstart
    H = MIN( MAX( ABS(Hmin), ABS(Hstart) ), ABS(Hmax) )
    IF ( ABS(H) <= 10.0_dp*ROUNDOFF ) H = DELTAMIN

    IF ( Tend >= Tstart ) THEN
       Direction = +1
    ELSE
       Direction = -1
    ENDIF
    H = Direction*H

    RejectLastH = .FALSE.
    RejectMoreH = .FALSE.

    TimeLoop: DO WHILE ( (Direction > 0) .AND. ((T-Tend)+ROUNDOFF <= ZERO)   &
                    .OR. (Direction < 0) .AND. ((Tend-T)+ROUNDOFF <= ZERO) )

       IF ( Nsteps > Max_no_steps ) THEN
          IERR = -6
          RETURN
       ENDIF
       IF ( ((T+0.1_dp*H) == T) .OR. (H <= ROUNDOFF) ) THEN
          IERR = -7
          RETURN
       ENDIF

       H = MIN( H, ABS(Tend-T) )

       ! Fcn0 <- F(T, Y)
       CALL Fun_SPLIT( Y, FIXc, RCONSTc, Fcn0, wP, wD )
       ISTAT(1) = ISTAT(1) + 1

       IF ( .NOT. Autonomous ) THEN
          ! dFdT by finite difference (ros_FunTimeDerivative replica)
          Delta = SQRT(ROUNDOFF)*MAX( 1.0E-6_dp, ABS(T) )
          CALL Fun_SPLIT( Y, FIXc, RCONSTc, dFdT, wP, wD )
          ISTAT(1) = ISTAT(1) + 1
          DO k = 1, NVAR
             dFdT(k) = ( dFdT(k) - Fcn0(k) ) / Delta
          ENDDO
       ENDIF

       ! Jac0 <- J(T, Y)
       CALL Jac_SP( Y, FIXc, RCONSTc, Jac0 )
       ISTAT(2) = ISTAT(2) + 1

  UntilAccepted: DO

       ! --- ros_PrepareMatrix replica ---
       Nconsecutive = 0
       Singular     = .TRUE.
       DO WHILE ( Singular )
          DO k = 1, LU_NONZERO
             Ghimj(k) = -Jac0(k)
          ENDDO
          ghinv = ONE/(Direction*H*ros_Gamma(1))
          CALL AddGhinvDiag( Ghimj, ghinv )
          CALL KppDecomp( Ghimj, ising )
          ISTAT(6) = ISTAT(6) + 1
          IF ( ising == 0 ) THEN
             Singular = .FALSE.
          ELSE
             ISTAT(8) = ISTAT(8) + 1
             Nconsecutive = Nconsecutive + 1
             IF ( Nconsecutive <= 5 ) THEN
                H = H*HALF
             ELSE
                IERR = -8
                RETURN
             ENDIF
          ENDIF
       ENDDO

       ! --- Stages ---
  Stage: DO istage = 1, ROS_S
          ioffset = NVAR*(istage-1)
          IF ( istage == 1 ) THEN
             DO k = 1, NVAR
                Fcn(k) = Fcn0(k)
             ENDDO
          ELSEIF ( ros_NewF(istage) ) THEN
             DO k = 1, NVAR
                Ynew(k) = Y(k)
             ENDDO
             DO j = 1, istage-1
                aa = ros_A( (istage-1)*(istage-2)/2 + j )
                DO k = 1, NVAR
                   Ynew(k) = Ynew(k) + aa*Kw(NVAR*(j-1)+k)
                ENDDO
             ENDDO
             Tau = T + ros_Alpha(istage)*Direction*H
             CALL Fun_SPLIT( Ynew, FIXc, RCONSTc, Fcn, wP, wD )
             ISTAT(1) = ISTAT(1) + 1
          ENDIF
          DO k = 1, NVAR
             Kw(ioffset+k) = Fcn(k)
          ENDDO
          DO j = 1, istage-1
             HC = ros_C( (istage-1)*(istage-2)/2 + j )/(Direction*H)
             DO k = 1, NVAR
                Kw(ioffset+k) = Kw(ioffset+k) + HC*Kw(NVAR*(j-1)+k)
             ENDDO
          ENDDO
          IF ( (.NOT. Autonomous) .AND. (ros_Gamma(istage) /= ZERO) ) THEN
             HG = Direction*H*ros_Gamma(istage)
             DO k = 1, NVAR
                Kw(ioffset+k) = Kw(ioffset+k) + HG*dFdT(k)
             ENDDO
          ENDIF
          CALL KppSolve( Ghimj, Kw(ioffset+1:ioffset+NVAR) )
          ISTAT(7) = ISTAT(7) + 1
       ENDDO Stage

       ! --- New solution ---
       DO k = 1, NVAR
          Ynew(k) = Y(k)
       ENDDO
       DO j = 1, ROS_S
          DO k = 1, NVAR
             Ynew(k) = Ynew(k) + ros_M(j)*Kw(NVAR*(j-1)+k)
          ENDDO
       ENDDO

       ! --- Error estimate ---
       DO k = 1, NVAR
          Yerr(k) = ZERO
       ENDDO
       DO j = 1, ROS_S
          DO k = 1, NVAR
             Yerr(k) = Yerr(k) + ros_E(j)*Kw(NVAR*(j-1)+k)
          ENDDO
       ENDDO
       ! ros_ErrorNorm replica
       Err = ZERO
       DO i = 1, NVAR
          Ymax = MAX( ABS(Y(i)), ABS(Ynew(i)) )
          IF ( VectorTol ) THEN
             Scal = AbsTol(i) + RelTol(i)*Ymax
          ELSE
             Scal = AbsTol(1) + RelTol(1)*Ymax
          ENDIF
          Err = Err + ( Yerr(i)/Scal )**2
       ENDDO
       Err = SQRT( Err/NVAR )
       Err = MAX( Err, 1.0d-10 )

       Fac  = MIN( FacMax, MAX( FacMin, FacSafe/Err**(ONE/ros_ELO) ) )
       Hnew = H*Fac

       Nsteps   = Nsteps + 1
       ISTAT(3) = ISTAT(3) + 1
       IF ( (Err <= ONE) .OR. (H <= Hmin) ) THEN     ! accept
          ISTAT(4) = ISTAT(4) + 1
          IF ( ICNTRL(16) == 1 ) THEN
             DO k = 1, NVAR
                Y(k) = MAX( Ynew(k), ZERO )
             ENDDO
          ELSE
             DO k = 1, NVAR
                Y(k) = Ynew(k)
             ENDDO
          ENDIF
          T = T + Direction*H
          Hnew = MAX( Hmin, MIN( Hnew, Hmax ) )
          IF ( RejectLastH ) Hnew = MIN( Hnew, H )
          RSTAT(2) = H                    ! Hexit (RSTATUS(Nhexit))
          RSTAT(3) = Hnew                 ! Hnew  (RSTATUS(Nhnew))
          RSTAT(1) = T                    ! Texit (RSTATUS(Ntexit))
          RejectLastH = .FALSE.
          RejectMoreH = .FALSE.
          H = Hnew
          EXIT UntilAccepted
       ELSE                                          ! reject
          IF ( RejectMoreH ) Hnew = H*FacRej
          RejectMoreH = RejectLastH
          RejectLastH = .TRUE.
          H = Hnew
          IF ( ISTAT(4) >= 1 ) ISTAT(5) = ISTAT(5) + 1
       ENDIF

       ENDDO UntilAccepted

    ENDDO TimeLoop

    IERR = 1     ! success

  END SUBROUTINE Integrate_Cell_S

  SUBROUTINE Integrate_Model( TIN, TOUT, ICNTRL_U, RCNTRL_U,                 &
                              ATOL_U, RTOL_U, Y, FIXc, RCONSTc,              &
                              ISTATUS_U, RSTATUS_U, IERR_U )
    ! Model-facing per-cell entry point for the batched chemistry path.
    ! Replicates the option handling of SUBROUTINE INTEGRATE in
    ! gckpp_Integrator.F90 (zeroed control vectors, default ICNTRL(15)=5,
    ! WHERE(ICNTRL_U /= 0) and WHERE(RCNTRL_U > 0) merges), then calls the
    ! statistics-carrying batched integrator with routine-local workspace
    ! and local copies of the per-cell state (mirrors Integrate_Cell_L).
    !
    ! NOTE: GEOS-Chem's fullchem driver always passes ICNTRL(15) = -1, and
    ! the batched integrator never calls the Update_* routines, so the
    ! Integrator_Update_Options logic of the stock wrapper is not needed.
    REAL(dp), INTENT(IN)    :: TIN, TOUT
    INTEGER,  INTENT(IN)    :: ICNTRL_U(20)
    REAL(dp), INTENT(IN)    :: RCNTRL_U(20)
    REAL(dp), INTENT(IN)    :: ATOL_U(NVAR), RTOL_U(NVAR)
    REAL(dp), INTENT(INOUT) :: Y(NVAR)
    REAL(dp), INTENT(IN)    :: FIXc(NFIX)
    REAL(dp), INTENT(IN)    :: RCONSTc(NREACT)
    INTEGER,  INTENT(OUT)   :: ISTATUS_U(20)
    REAL(dp), INTENT(OUT)   :: RSTATUS_U(20)
    INTEGER,  INTENT(OUT)   :: IERR_U

    INTEGER  :: ICNTRL(20), ISTAT(8), kk
    REAL(dp) :: RCNTRL(20), RSTAT(3)
    ! Per-thread local workspace
    REAL(dp) :: Kw(NWORK_K), Ghimj(NWORK_J), Jac0(NWORK_J)
    REAL(dp) :: Ynew(NVAR), Fcn0(NVAR), Fcn(NVAR)
    REAL(dp) :: dFdT(NVAR), Yerr(NVAR), wP(NVAR), wD(NVAR)
    ! Local copies of per-cell state (mirrors Integrate_Cell_L)
    REAL(dp) :: Yl(NVAR), FIXl(NFIX), RCONSTl(NREACT)
    REAL(dp) :: ATOLl(NVAR), RTOLl(NVAR)

    !~~~> Zero input and output arrays for safety's sake
    ICNTRL     = 0
    RCNTRL     = 0.0_dp

    !~~~> fine-tune the integrator (same default as the stock wrapper;
    !~~~> always overridden to -1 by the fullchem driver's ICNTRL)
    ICNTRL(15) = 5

    !~~~> if optional parameters are given, and if they are /= 0,
    !     then use them to overwrite default settings
    WHERE( ICNTRL_U /= 0 ) ICNTRL = ICNTRL_U
    WHERE( RCNTRL_U > 0 ) RCNTRL = RCNTRL_U

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

    CALL Integrate_Cell_S( Yl, FIXl, RCONSTl, TIN, TOUT,                     &
                           ATOLl, RTOLl, ICNTRL, RCNTRL,                     &
                           Kw, Ghimj, Jac0, Ynew, Fcn0, Fcn, dFdT, Yerr,     &
                           wP, wD, ISTAT, RSTAT, IERR_U )

    DO kk = 1, NVAR
       Y(kk) = Yl(kk)
    ENDDO

    ISTATUS_U      = 0
    RSTATUS_U      = 0.0_dp
    ISTATUS_U(1:8) = ISTAT
    RSTATUS_U(1:3) = RSTAT

  END SUBROUTINE Integrate_Model

  SUBROUTINE AddGhinvDiag( Ghimj, ghinv )
    !$acc routine seq
    ! Adds ghinv on the LU diagonal (separated out so LU_DIAG stays a
    ! host-associated module constant of gckpp_JacobianSP)
    USE gckpp_JacobianSP, ONLY : LU_DIAG
    REAL(dp), INTENT(INOUT) :: Ghimj(LU_NONZERO)
    REAL(dp), INTENT(IN)    :: ghinv
    INTEGER :: i
    DO i = 1, NVAR
       Ghimj(LU_DIAG(i)) = Ghimj(LU_DIAG(i)) + ghinv
    ENDDO
  END SUBROUTINE AddGhinvDiag

END MODULE gckpp_BatchIntegrator
