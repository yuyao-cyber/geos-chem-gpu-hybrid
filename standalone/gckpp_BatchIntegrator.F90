!------------------------------------------------------------------------------
! gckpp_BatchIntegrator.F90
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
! GEOS-Chem full-chemistry cells sampled with auto-reduction disabled:
!   ICNTRL(1)=1 (autonomous), ICNTRL(2)=0 (vector tolerances),
!   ICNTRL(3)=4 (Rodas3), ICNTRL(15)=-1 (no Update_* calls in integrator).
! Differences from the original, all deliberate:
!   - No PRINT statements (device code); failures return IERR codes.
!   - Roundoff is the compile-time constant 2**-53 (the exact value that
!     WLAMCH('E') computes at runtime).
!   - Auto-reduce (ICNTRL(12)) paths are not implemented (IERR=-90 if
!     requested); harvested cells were produced with AR off.
!
! Validation: kpp_batch.F90 compares results against the original
! integrator cell-by-cell; agreement is expected to machine precision.
!------------------------------------------------------------------------------
MODULE gckpp_BatchIntegrator

  USE gckpp_Precision,  ONLY : dp
  USE gckpp_Parameters, ONLY : NVAR, NFIX, NSPEC, NREACT, LU_NONZERO
  IMPLICIT NONE
  PRIVATE
  PUBLIC :: Integrate_Cell
  PUBLIC :: Integrate_Cell_L
  PUBLIC :: Integrate_Cell_I
  PUBLIC :: Integrate_Cell_W
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
    ! (The coalescing experiment: compare mode 'gpu' vs mode 'gpul'.)
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
    ! re-read by Fun/Jac on every internal stage — paying one strided
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

  SUBROUTINE Integrate_Cell_I( ic, nP, Y, FIXc, RCONSTc, Tstart, Tend,       &
                               AbsTol, RelTol, ICNTRL, RCNTRL,               &
                               Kw, Ghimj, Jac0, Ynew, Fcn0, Fcn, dFdT, Yerr, &
                               wP, wD, wA, wB, wW, Nsteps, Hexit, IERR )
    !$acc routine seq
    ! INTERLEAVED-layout variant of Integrate_Cell: identical Rodas3 control
    ! flow and per-cell operation order, but every per-cell array (state and
    ! workspace) is a caller-provided GLOBAL array with the cell index FIRST,
    ! dimensioned (nP, :).  Thread ic accessing element k then touches an
    ! address adjacent to thread ic+1 => coalesced global-memory traffic.
    ! Scalars (T, H, Err, ...) stay in registers as before.
    USE gckpp_Interleaved, ONLY : Fun_SPLIT_I, Jac_SP_I, KppDecomp_I,        &
                                  KppSolve_I, NB_JAC

    INTEGER,  INTENT(IN)    :: ic, nP
    REAL(dp), INTENT(INOUT) :: Y(nP,NVAR)       ! in: initial, out: final
    REAL(dp), INTENT(IN)    :: FIXc(nP,NFIX)
    REAL(dp), INTENT(IN)    :: RCONSTc(nP,NREACT)
    REAL(dp), INTENT(IN)    :: Tstart, Tend
    REAL(dp), INTENT(IN)    :: AbsTol(nP,NVAR)
    REAL(dp), INTENT(IN)    :: RelTol(NVAR)     ! uniform across cells
    INTEGER,  INTENT(IN)    :: ICNTRL(nP,20)
    REAL(dp), INTENT(IN)    :: RCNTRL(nP,20)
    REAL(dp), INTENT(INOUT) :: Kw(nP,NWORK_K)   ! stage vectors K
    REAL(dp), INTENT(INOUT) :: Ghimj(nP,NWORK_J)
    REAL(dp), INTENT(INOUT) :: Jac0(nP,NWORK_J)
    REAL(dp), INTENT(INOUT) :: Ynew(nP,NVAR), Fcn0(nP,NVAR), Fcn(nP,NVAR)
    REAL(dp), INTENT(INOUT) :: dFdT(nP,NVAR), Yerr(nP,NVAR)
    REAL(dp), INTENT(INOUT) :: wP(nP,NVAR), wD(nP,NVAR)
    REAL(dp), INTENT(INOUT) :: wA(nP,NREACT)    ! Fun_SPLIT_I rate scratch
    REAL(dp), INTENT(INOUT) :: wB(nP,NB_JAC)    ! Jac_SP_I temporary
    REAL(dp), INTENT(INOUT) :: wW(nP,NVAR)      ! KppDecomp_I row scratch
    INTEGER,  INTENT(OUT)   :: Nsteps
    REAL(dp), INTENT(OUT)   :: Hexit
    INTEGER,  INTENT(OUT)   :: IERR

    ! Rodas3 coefficients (identical to Integrate_Cell)
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
    IF ( ICNTRL(ic,12) == 1 ) THEN
       IERR = -90                     ! auto-reduce not supported in batch
       RETURN
    ENDIF
    Autonomous = .NOT. ( ICNTRL(ic,1) == 0 )
    VectorTol  = ( ICNTRL(ic,2) == 0 )
    IF ( .NOT. ( ICNTRL(ic,3) == 0 .OR. ICNTRL(ic,3) == 4 ) ) THEN
       IERR = -2                      ! only Rodas3 implemented
       RETURN
    ENDIF
    IF ( ICNTRL(ic,4) == 0 ) THEN
       Max_no_steps = 200000
    ELSE
       Max_no_steps = ICNTRL(ic,4)
    ENDIF

    IF ( RCNTRL(ic,1) > ZERO ) THEN
       Hmin = RCNTRL(ic,1)
    ELSE
       Hmin = ZERO
    ENDIF
    IF ( RCNTRL(ic,2) > ZERO ) THEN
       Hmax = MIN( ABS(RCNTRL(ic,2)), ABS(Tend-Tstart) )
    ELSE
       Hmax = ABS(Tend-Tstart)
    ENDIF
    IF ( RCNTRL(ic,3) > ZERO ) THEN
       Hstart = MIN( ABS(RCNTRL(ic,3)), ABS(Tend-Tstart) )
    ELSE
       Hstart = MAX( Hmin, DELTAMIN )
    ENDIF
    IF ( RCNTRL(ic,4) > ZERO ) THEN
       FacMin = RCNTRL(ic,4)
    ELSE
       FacMin = 0.2_dp
    ENDIF
    IF ( RCNTRL(ic,5) > ZERO ) THEN
       FacMax = RCNTRL(ic,5)
    ELSE
       FacMax = 6.0_dp
    ENDIF
    IF ( RCNTRL(ic,6) > ZERO ) THEN
       FacRej = RCNTRL(ic,6)
    ELSE
       FacRej = 0.1_dp
    ENDIF
    IF ( RCNTRL(ic,7) > ZERO ) THEN
       FacSafe = RCNTRL(ic,7)
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
       CALL Fun_SPLIT_I( ic, nP, Y, FIXc, RCONSTc, Fcn0, wP, wD, wA )

       IF ( .NOT. Autonomous ) THEN
          Delta = SQRT(ROUNDOFF)*MAX( 1.0E-6_dp, ABS(T) )
          CALL Fun_SPLIT_I( ic, nP, Y, FIXc, RCONSTc, dFdT, wP, wD, wA )
          DO k = 1, NVAR
             dFdT(ic,k) = ( dFdT(ic,k) - Fcn0(ic,k) ) / Delta
          ENDDO
       ENDIF

       ! Jac0 <- J(T, Y)
       CALL Jac_SP_I( ic, nP, Y, FIXc, RCONSTc, Jac0, wB )

  UntilAccepted: DO

       ! --- ros_PrepareMatrix replica ---
       Nconsecutive = 0
       Singular     = .TRUE.
       DO WHILE ( Singular )
          DO k = 1, LU_NONZERO
             Ghimj(ic,k) = -Jac0(ic,k)
          ENDDO
          ghinv = ONE/(Direction*H*ros_Gamma(1))
          CALL AddGhinvDiag_I( ic, nP, Ghimj, ghinv )
          CALL KppDecomp_I( ic, nP, Ghimj, wW, ising )
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
                Fcn(ic,k) = Fcn0(ic,k)
             ENDDO
          ELSEIF ( ros_NewF(istage) ) THEN
             DO k = 1, NVAR
                Ynew(ic,k) = Y(ic,k)
             ENDDO
             DO j = 1, istage-1
                aa = ros_A( (istage-1)*(istage-2)/2 + j )
                DO k = 1, NVAR
                   Ynew(ic,k) = Ynew(ic,k) + aa*Kw(ic,NVAR*(j-1)+k)
                ENDDO
             ENDDO
             Tau = T + ros_Alpha(istage)*Direction*H
             CALL Fun_SPLIT_I( ic, nP, Ynew, FIXc, RCONSTc, Fcn, wP, wD, wA )
          ENDIF
          DO k = 1, NVAR
             Kw(ic,ioffset+k) = Fcn(ic,k)
          ENDDO
          DO j = 1, istage-1
             HC = ros_C( (istage-1)*(istage-2)/2 + j )/(Direction*H)
             DO k = 1, NVAR
                Kw(ic,ioffset+k) = Kw(ic,ioffset+k) + HC*Kw(ic,NVAR*(j-1)+k)
             ENDDO
          ENDDO
          IF ( (.NOT. Autonomous) .AND. (ros_Gamma(istage) /= ZERO) ) THEN
             HG = Direction*H*ros_Gamma(istage)
             DO k = 1, NVAR
                Kw(ic,ioffset+k) = Kw(ic,ioffset+k) + HG*dFdT(ic,k)
             ENDDO
          ENDIF
          CALL KppSolve_I( ic, nP, ioffset, Ghimj, Kw )
       ENDDO Stage

       ! --- New solution ---
       DO k = 1, NVAR
          Ynew(ic,k) = Y(ic,k)
       ENDDO
       DO j = 1, ROS_S
          DO k = 1, NVAR
             Ynew(ic,k) = Ynew(ic,k) + ros_M(j)*Kw(ic,NVAR*(j-1)+k)
          ENDDO
       ENDDO

       ! --- Error estimate ---
       DO k = 1, NVAR
          Yerr(ic,k) = ZERO
       ENDDO
       DO j = 1, ROS_S
          DO k = 1, NVAR
             Yerr(ic,k) = Yerr(ic,k) + ros_E(j)*Kw(ic,NVAR*(j-1)+k)
          ENDDO
       ENDDO
       ! ros_ErrorNorm replica
       Err = ZERO
       DO i = 1, NVAR
          Ymax = MAX( ABS(Y(ic,i)), ABS(Ynew(ic,i)) )
          IF ( VectorTol ) THEN
             Scal = AbsTol(ic,i) + RelTol(i)*Ymax
          ELSE
             Scal = AbsTol(ic,1) + RelTol(1)*Ymax
          ENDIF
          Err = Err + ( Yerr(ic,i)/Scal )**2
       ENDDO
       Err = SQRT( Err/NVAR )
       Err = MAX( Err, 1.0d-10 )

       Fac  = MIN( FacMax, MAX( FacMin, FacSafe/Err**(ONE/ros_ELO) ) )
       Hnew = H*Fac

       Nsteps = Nsteps + 1
       IF ( (Err <= ONE) .OR. (H <= Hmin) ) THEN     ! accept
          Nacc = Nacc + 1
          IF ( ICNTRL(ic,16) == 1 ) THEN
             DO k = 1, NVAR
                Y(ic,k) = MAX( Ynew(ic,k), ZERO )
             ENDDO
          ELSE
             DO k = 1, NVAR
                Y(ic,k) = Ynew(ic,k)
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

  END SUBROUTINE Integrate_Cell_I

  SUBROUTINE AddGhinvDiag_I( ic, nP, Ghimj, ghinv )
    !$acc routine seq
    ! Interleaved-layout version of AddGhinvDiag
    USE gckpp_JacobianSP, ONLY : LU_DIAG
    INTEGER,  INTENT(IN)    :: ic, nP
    REAL(dp), INTENT(INOUT) :: Ghimj(nP,LU_NONZERO)
    REAL(dp), INTENT(IN)    :: ghinv
    INTEGER :: i
    DO i = 1, NVAR
       Ghimj(ic,LU_DIAG(i)) = Ghimj(ic,LU_DIAG(i)) + ghinv
    ENDDO
  END SUBROUTINE AddGhinvDiag_I

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



  !--------------------------------------------------------------------
  ! Vector-level helper kernels.
  !
  ! nvfortran 24.7 only maps a '!$acc loop vector' to threadIdx%x when the
  ! loop is the ENTIRE body of a '!$acc routine vector' procedure.  A
  ! vector loop written inline inside sequential control flow (a DO WHILE,
  ! a stage loop, an IF block) of a routine vector is silently demoted to
  ! '!$acc loop seq' -- confirmed from -Minfo=accel.  Every element-wise
  ! loop of the warp-cooperative integrator is therefore hoisted into one
  ! of these one-loop routines, which DO get mapped to the lanes.
  !
  ! All of them are element-wise (each output element written by exactly
  ! one iteration), so no summation is reordered and results are
  ! bit-identical to the scalar integrator.
  !--------------------------------------------------------------------

  SUBROUTINE VZERO( n, y, yoff )
    !$acc routine vector
    INTEGER,  INTENT(IN)    :: n, yoff
    REAL(dp), INTENT(INOUT) :: y(*)
    INTEGER :: i
    !$acc loop vector
    DO i = 1, n
       y(yoff+i) = ZERO
    ENDDO
  END SUBROUTINE VZERO

  SUBROUTINE VCOPY( n, x, xoff, y, yoff )
    !$acc routine vector
    INTEGER,  INTENT(IN)    :: n, xoff, yoff
    REAL(dp), INTENT(IN)    :: x(*)
    REAL(dp), INTENT(INOUT) :: y(*)
    INTEGER :: i
    !$acc loop vector
    DO i = 1, n
       y(yoff+i) = x(xoff+i)
    ENDDO
  END SUBROUTINE VCOPY

  SUBROUTINE VCOPYPOS( n, x, y )
    !$acc routine vector
    INTEGER,  INTENT(IN)    :: n
    REAL(dp), INTENT(IN)    :: x(n)
    REAL(dp), INTENT(INOUT) :: y(n)
    INTEGER :: i
    !$acc loop vector
    DO i = 1, n
       y(i) = MAX( x(i), ZERO )
    ENDDO
  END SUBROUTINE VCOPYPOS

  SUBROUTINE VNEG( n, x, y )
    !$acc routine vector
    INTEGER,  INTENT(IN)    :: n
    REAL(dp), INTENT(IN)    :: x(n)
    REAL(dp), INTENT(INOUT) :: y(n)
    INTEGER :: i
    !$acc loop vector
    DO i = 1, n
       y(i) = -x(i)
    ENDDO
  END SUBROUTINE VNEG

  SUBROUTINE VAXPY( n, a, x, xoff, y, yoff )
    !$acc routine vector
    INTEGER,  INTENT(IN)    :: n, xoff, yoff
    REAL(dp), INTENT(IN)    :: a, x(*)
    REAL(dp), INTENT(INOUT) :: y(*)
    INTEGER :: i
    !$acc loop vector
    DO i = 1, n
       y(yoff+i) = y(yoff+i) + a*x(xoff+i)
    ENDDO
  END SUBROUTINE VAXPY

  SUBROUTINE VDFDT( n, Fcn0, dFdT, Delta )
    !$acc routine vector
    INTEGER,  INTENT(IN)    :: n
    REAL(dp), INTENT(IN)    :: Fcn0(n), Delta
    REAL(dp), INTENT(INOUT) :: dFdT(n)
    INTEGER :: i
    !$acc loop vector
    DO i = 1, n
       dFdT(i) = ( dFdT(i) - Fcn0(i) ) / Delta
    ENDDO
  END SUBROUTINE VDFDT

  SUBROUTINE VERRTERM( n, Y, Ynew, Yerr, AbsTol, RelTol, VectorTol, wS )
    !$acc routine vector
    ! Per-species error-norm TERMS only.  The sum itself is done by the
    ! caller in the original i = 1..NVAR order, so the reduction is NOT
    ! reordered and the norm is bit-identical to the scalar integrator.
    INTEGER,  INTENT(IN)    :: n
    REAL(dp), INTENT(IN)    :: Y(n), Ynew(n), Yerr(n), AbsTol(n), RelTol(n)
    LOGICAL,  INTENT(IN)    :: VectorTol
    REAL(dp), INTENT(INOUT) :: wS(n)
    INTEGER  :: i
    REAL(dp) :: Ymax, Scal
    !$acc loop vector private(Ymax,Scal)
    DO i = 1, n
       Ymax = MAX( ABS(Y(i)), ABS(Ynew(i)) )
       IF ( VectorTol ) THEN
          Scal = AbsTol(i) + RelTol(i)*Ymax
       ELSE
          Scal = AbsTol(1) + RelTol(1)*Ymax
       ENDIF
       wS(i) = ( Yerr(i)/Scal )**2
    ENDDO
  END SUBROUTINE VERRTERM

  SUBROUTINE VLU_GATHER( i0, i1, JVS, W )
    !$acc routine vector
    USE gckpp_JacobianSP, ONLY : LU_ICOL
    INTEGER,  INTENT(IN)    :: i0, i1
    REAL(dp), INTENT(IN)    :: JVS(LU_NONZERO)
    REAL(dp), INTENT(INOUT) :: W(NVAR)
    INTEGER :: kk
    !$acc loop vector
    DO kk = i0, i1
       W( LU_ICOL(kk) ) = JVS(kk)
    ENDDO
  END SUBROUTINE VLU_GATHER

  SUBROUTINE VLU_SCATTER( i0, i1, JVS, W )
    !$acc routine vector
    USE gckpp_JacobianSP, ONLY : LU_ICOL
    INTEGER,  INTENT(IN)    :: i0, i1
    REAL(dp), INTENT(INOUT) :: JVS(LU_NONZERO)
    REAL(dp), INTENT(IN)    :: W(NVAR)
    INTEGER :: kk
    !$acc loop vector
    DO kk = i0, i1
       JVS(kk) = W( LU_ICOL(kk) )
    ENDDO
  END SUBROUTINE VLU_SCATTER

  SUBROUTINE VLU_UPDATE( j0, j1, a, JVS, W )
    !$acc routine vector
    USE gckpp_JacobianSP, ONLY : LU_ICOL
    INTEGER,  INTENT(IN)    :: j0, j1
    REAL(dp), INTENT(IN)    :: a, JVS(LU_NONZERO)
    REAL(dp), INTENT(INOUT) :: W(NVAR)
    INTEGER :: jj
    !$acc loop vector
    DO jj = j0, j1
       W( LU_ICOL(jj) ) = W( LU_ICOL(jj) ) + a*JVS(jj)
    ENDDO
  END SUBROUTINE VLU_UPDATE

  SUBROUTINE KppDecomp_W( JVS, W, IER )
    !$acc routine vector
    ! WARP-COOPERATIVE sparse LU factorization.
    ! Same algorithm and same per-element arithmetic as KppDecomp; the two
    ! row-gather/scatter loops and the row-update loop are executed by the
    ! vector lanes of ONE gang (one cell).  Every W element is written by
    ! exactly one iteration of each parallel loop, so the result is
    ! BIT-IDENTICAL to the serial version (no reordered reductions).
    ! The outer k loop and the sub-diagonal kk loop carry true dependencies
    ! and stay serial (executed redundantly by all lanes, which keeps the
    ! lanes converged and every scalar identical across the warp).
    ! W is caller-provided (must be SHARED by the lanes of the gang).
    USE gckpp_JacobianSP, ONLY : LU_CROW, LU_DIAG, LU_ICOL
    REAL(dp), INTENT(INOUT) :: JVS(LU_NONZERO)
    REAL(dp), INTENT(INOUT) :: W(NVAR)
    INTEGER,  INTENT(OUT)   :: IER
    INTEGER  :: k, kk, j
    REAL(dp) :: a
    a   = 0.0_dp
    IER = 0
    DO k = 1, NVAR
       IF ( ABS(JVS(LU_DIAG(k))) < TINY(a) ) THEN
          IER = k
          RETURN
       ENDIF
       CALL VLU_GATHER( LU_CROW(k), LU_CROW(k+1)-1, JVS, W )
       DO kk = LU_CROW(k), LU_DIAG(k)-1
          j = LU_ICOL(kk)
          a = -W(j) / JVS( LU_DIAG(j) )
          W(j) = -a
          CALL VLU_UPDATE( LU_DIAG(j)+1, LU_CROW(j+1)-1, a, JVS, W )
       ENDDO
       CALL VLU_SCATTER( LU_CROW(k), LU_CROW(k+1)-1, JVS, W )
    ENDDO
  END SUBROUTINE KppDecomp_W

  SUBROUTINE AddGhinvDiag_W( Ghimj, ghinv )
    !$acc routine vector
    USE gckpp_JacobianSP, ONLY : LU_DIAG
    REAL(dp), INTENT(INOUT) :: Ghimj(LU_NONZERO)
    REAL(dp), INTENT(IN)    :: ghinv
    INTEGER :: i
    !$acc loop vector independent
    DO i = 1, NVAR
       Ghimj(LU_DIAG(i)) = Ghimj(LU_DIAG(i)) + ghinv
    ENDDO
  END SUBROUTINE AddGhinvDiag_W

  SUBROUTINE Integrate_Cell_W( Y, FIXc, RCONSTc, Tstart, Tend,               &
                               AbsTol, RelTol, ICNTRL, RCNTRL,               &
                               Kw, Ghimj, Jac0, Ynew, Fcn0, Fcn, dFdT, Yerr, &
                               wP, wD, wW, wS, Nsteps, Hexit, IERR )
    !$acc routine vector
    ! WARP-COOPERATIVE variant of Integrate_Cell: ONE gang (a warp, or a
    ! sub-warp of vector_length lanes) solves ONE cell.  The Rodas3 scalar
    ! control flow is executed REDUNDANTLY by every lane -- all lanes
    ! therefore follow exactly the same accept/reject path and the warp
    ! never diverges.  The wide element-wise loops (length NVAR=353 or
    ! LU_NONZERO=5683) are split across the lanes with '!$acc loop vector',
    ! as are the row loops inside KppDecomp_W.
    !
    ! Bit-exactness: every vectorised loop is element-wise (each output
    ! element written by exactly one iteration), so no summation is
    ! reordered.  The one reduction in the algorithm -- the error norm --
    ! is computed in TWO phases: the per-species terms are formed in a
    ! vector loop into wS, then summed SERIALLY in the original i=1..NVAR
    ! order (redundantly on every lane).  Result: identical arithmetic to
    ! Integrate_Cell on CPU *and* GPU.
    !
    ! Workspace arrays are caller-provided per-CELL slices (cell-last
    ! layout).  With one warp per cell the lanes of a vector loop touch
    ! consecutive elements k, so these accesses are fully coalesced.
    ! wW is the KppDecomp row scratch; wS the error-norm term scratch.
    USE gckpp_Function,      ONLY : Fun_SPLIT
    USE gckpp_Jacobian,      ONLY : Jac_SP
    USE gckpp_LinearAlgebra, ONLY : KppSolve

    REAL(dp), INTENT(INOUT) :: Y(NVAR)
    REAL(dp), INTENT(IN)    :: FIXc(NFIX)
    REAL(dp), INTENT(IN)    :: RCONSTc(NREACT)
    REAL(dp), INTENT(IN)    :: Tstart, Tend
    REAL(dp), INTENT(IN)    :: AbsTol(NVAR), RelTol(NVAR)
    INTEGER,  INTENT(IN)    :: ICNTRL(20)
    REAL(dp), INTENT(IN)    :: RCNTRL(20)
    REAL(dp), INTENT(INOUT) :: Kw(NWORK_K)
    REAL(dp), INTENT(INOUT) :: Ghimj(NWORK_J)
    REAL(dp), INTENT(INOUT) :: Jac0(NWORK_J)
    REAL(dp), INTENT(INOUT) :: Ynew(NVAR), Fcn0(NVAR), Fcn(NVAR)
    REAL(dp), INTENT(INOUT) :: dFdT(NVAR), Yerr(NVAR)
    REAL(dp), INTENT(INOUT) :: wP(NVAR), wD(NVAR)
    REAL(dp), INTENT(INOUT) :: wW(NVAR), wS(NVAR)
    INTEGER,  INTENT(OUT)   :: Nsteps
    REAL(dp), INTENT(OUT)   :: Hexit
    INTEGER,  INTENT(OUT)   :: IERR

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

    IF ( ICNTRL(12) == 1 ) THEN
       IERR = -90
       RETURN
    ENDIF
    Autonomous = .NOT. ( ICNTRL(1) == 0 )
    VectorTol  = ( ICNTRL(2) == 0 )
    IF ( .NOT. ( ICNTRL(3) == 0 .OR. ICNTRL(3) == 4 ) ) THEN
       IERR = -2
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

       ! Straight-line generated kernels: executed redundantly by every
       ! lane (all lanes compute and store identical values).  These are
       ! the SERIAL FRACTION of the cell solve for this mapping.
       CALL Fun_SPLIT( Y, FIXc, RCONSTc, Fcn0, wP, wD )

       IF ( .NOT. Autonomous ) THEN
          Delta = SQRT(ROUNDOFF)*MAX( 1.0E-6_dp, ABS(T) )
          CALL Fun_SPLIT( Y, FIXc, RCONSTc, dFdT, wP, wD )
          CALL VDFDT( NVAR, Fcn0, dFdT, Delta )
       ENDIF

       CALL Jac_SP( Y, FIXc, RCONSTc, Jac0 )

  UntilAccepted: DO

       Nconsecutive = 0
       Singular     = .TRUE.
       DO WHILE ( Singular )
          CALL VNEG( LU_NONZERO, Jac0, Ghimj )
          ghinv = ONE/(Direction*H*ros_Gamma(1))
          CALL AddGhinvDiag_W( Ghimj, ghinv )
          CALL KppDecomp_W( Ghimj, wW, ising )
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

  Stage: DO istage = 1, ROS_S
          ioffset = NVAR*(istage-1)
          IF ( istage == 1 ) THEN
             CALL VCOPY( NVAR, Fcn0, 0, Fcn, 0 )
          ELSEIF ( ros_NewF(istage) ) THEN
             CALL VCOPY( NVAR, Y, 0, Ynew, 0 )
             DO j = 1, istage-1
                aa = ros_A( (istage-1)*(istage-2)/2 + j )
                CALL VAXPY( NVAR, aa, Kw, NVAR*(j-1), Ynew, 0 )
             ENDDO
             Tau = T + ros_Alpha(istage)*Direction*H
             CALL Fun_SPLIT( Ynew, FIXc, RCONSTc, Fcn, wP, wD )
          ENDIF
          CALL VCOPY( NVAR, Fcn, 0, Kw, ioffset )
          DO j = 1, istage-1
             HC = ros_C( (istage-1)*(istage-2)/2 + j )/(Direction*H)
             CALL VAXPY( NVAR, HC, Kw, NVAR*(j-1), Kw, ioffset )
          ENDDO
          IF ( (.NOT. Autonomous) .AND. (ros_Gamma(istage) /= ZERO) ) THEN
             HG = Direction*H*ros_Gamma(istage)
             CALL VAXPY( NVAR, HG, dFdT, 0, Kw, ioffset )
          ENDIF
          ! Triangular solves: true serial dependency chains over rows of
          ! ~8 entries; parallelising them would reorder sums.  Executed
          ! redundantly by all lanes.
          CALL KppSolve( Ghimj, Kw(ioffset+1:ioffset+NVAR) )
       ENDDO Stage

       CALL VCOPY( NVAR, Y, 0, Ynew, 0 )
       DO j = 1, ROS_S
          CALL VAXPY( NVAR, ros_M(j), Kw, NVAR*(j-1), Ynew, 0 )
       ENDDO

       CALL VZERO( NVAR, Yerr, 0 )
       DO j = 1, ROS_S
          CALL VAXPY( NVAR, ros_E(j), Kw, NVAR*(j-1), Yerr, 0 )
       ENDDO
       ! ros_ErrorNorm, two-phase: terms in parallel, sum in ORIGINAL order
       CALL VERRTERM( NVAR, Y, Ynew, Yerr, AbsTol, RelTol, VectorTol, wS )
       Err = ZERO
       DO i = 1, NVAR
          Err = Err + wS(i)
       ENDDO
       Err = SQRT( Err/NVAR )
       Err = MAX( Err, 1.0d-10 )

       Fac  = MIN( FacMax, MAX( FacMin, FacSafe/Err**(ONE/ros_ELO) ) )
       Hnew = H*Fac

       Nsteps = Nsteps + 1
       IF ( (Err <= ONE) .OR. (H <= Hmin) ) THEN
          Nacc = Nacc + 1
          IF ( ICNTRL(16) == 1 ) THEN
             CALL VCOPYPOS( NVAR, Ynew, Y )
          ELSE
             CALL VCOPY( NVAR, Ynew, 0, Y, 0 )
          ENDIF
          T = T + Direction*H
          Hnew = MAX( Hmin, MIN( Hnew, Hmax ) )
          IF ( RejectLastH ) Hnew = MIN( Hnew, H )
          Hexit = H
          RejectLastH = .FALSE.
          RejectMoreH = .FALSE.
          H = Hnew
          EXIT UntilAccepted
       ELSE
          IF ( RejectMoreH ) Hnew = H*FacRej
          RejectMoreH = RejectLastH
          RejectLastH = .TRUE.
          H = Hnew
       ENDIF

       ENDDO UntilAccepted

    ENDDO TimeLoop

    IERR = 1

  END SUBROUTINE Integrate_Cell_W

END MODULE gckpp_BatchIntegrator
