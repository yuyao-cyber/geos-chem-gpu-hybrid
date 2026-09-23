!------------------------------------------------------------------------------
! multicell_integrator.inc  --  INCLUDEd by the generated gckpp_MultiCell<NC>
! modules (Phase-1 Experiment 3: per-thread ILP).  Uses the module PARAMETER
! NC = number of cells carried by one thread.
!
! Integrate_Cell_M drives NC INDEPENDENT grid cells through Rodas3 in
! LOCKSTEP: all NC cells take the Rosenbrock stages together, but each keeps
! its OWN T, H, error estimate and accept/reject decision.  The thread loops
! until every cell is finished; a cell that finishes (or fails) early is
! masked out of the BOOKKEEPING while its arithmetic lanes keep being
! computed and discarded.  That masked-off waste is the cost of the design.
!
! Bit-exactness: for any single cell, the sequence of floating-point
! operations is exactly the sequence Integrate_Cell performs.  Nothing is
! reordered or reassociated; the NC dimension is only ever an extra index.
! Two places recompute rather than skip, both provably value-identical:
!   (1) Fun_SPLIT_M/Jac_SP_M are re-evaluated for every lane whenever ANY
!       lane starts a new time step.  A lane that is mid-rejection has an
!       UNCHANGED Y (a rejected step never writes Y), so re-evaluating F(Y)
!       and J(Y) reproduces exactly the values it already held.
!   (2) the singular-matrix retry loop rebuilds Ghimj from Jac0 for every
!       active lane.  A lane that is not singular has unchanged Jac0 and H,
!       so it recomputes the identical factorization.
!------------------------------------------------------------------------------

  SUBROUTINE AddGhinvDiag_M( Ghimj, ghinv )
    !$acc routine seq
    ! Adds ghinv on the LU diagonal, NC cells at a time
    REAL(dp), INTENT(INOUT) :: Ghimj(NC,LU_NONZERO)
    REAL(dp), INTENT(IN)    :: ghinv(NC)
    INTEGER :: i, mc
    DO i = 1, NVAR
       DO mc = 1, NC
          Ghimj(mc,LU_DIAG(i)) = Ghimj(mc,LU_DIAG(i)) + ghinv(mc)
       ENDDO
    ENDDO
  END SUBROUTINE AddGhinvDiag_M

  SUBROUTINE Integrate_Cell_M( g, nCell, perm, Yb, FIXb, RCONSTb, Tendb,      &
                               ATOLb, RTOLv, ICNTRLb, RCNTRLb,                &
                               Nstepsb, Hexitb, IERRb )
    !$acc routine seq
    ! Solve the NC cells  perm( (g-1)*NC + 1 : g*NC )  in one thread.
    INTEGER,  INTENT(IN)    :: g, nCell
    INTEGER,  INTENT(IN)    :: perm(nCell)
    REAL(dp), INTENT(INOUT) :: Yb(NVAR,nCell)
    REAL(dp), INTENT(IN)    :: FIXb(NFIX,nCell), RCONSTb(NREACT,nCell)
    REAL(dp), INTENT(IN)    :: Tendb(nCell), ATOLb(NVAR,nCell), RTOLv(NVAR)
    INTEGER,  INTENT(IN)    :: ICNTRLb(20,nCell)
    REAL(dp), INTENT(IN)    :: RCNTRLb(20,nCell)
    INTEGER,  INTENT(INOUT) :: Nstepsb(nCell), IERRb(nCell)
    REAL(dp), INTENT(INOUT) :: Hexitb(nCell)

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

    ! Per-thread local workspace, NC cells wide (cell index FIRST)
    REAL(dp) :: Kw(NC,NWORK_K), Ghimj(NC,NWORK_J), Jac0(NC,NWORK_J)
    REAL(dp) :: Ynew(NC,NVAR), Fcn0(NC,NVAR), Fcn(NC,NVAR)
    REAL(dp) :: dFdT(NC,NVAR), Yerr(NC,NVAR), wPl(NC,NVAR), wDl(NC,NVAR)
    REAL(dp) :: Yl(NC,NVAR), FIXl(NC,NFIX), RCONSTl(NC,NREACT)
    REAL(dp) :: ATOLl(NC,NVAR), RTOLl(NC,NVAR)

    ! Per-cell control state
    INTEGER  :: cid(NC), Max_no_steps(NC), Direction(NC), Nst(NC)
    INTEGER  :: Nconsecutive(NC), ising(NC), ierr_c(NC)
    REAL(dp) :: Tc(NC), Hc(NC), Hnew(NC), Err(NC), Fac(NC), Hex(NC)
    REAL(dp) :: Hmin(NC), Hmax(NC), Hstart(NC), Tnd(NC)
    REAL(dp) :: FacMin(NC), FacMax(NC), FacRej(NC), FacSafe(NC)
    REAL(dp) :: ghinv(NC), HCc(NC), HGc(NC)
    LOGICAL  :: Autonomous(NC), VectorTol(NC), RejectLastH(NC), RejectMoreH(NC)
    LOGICAL  :: Singular(NC), active(NC), newstep(NC), valid(NC)

    INTEGER  :: mc, i, j, k, istage, ioffset, ii, mIter
    REAL(dp) :: Ymax, Scal, aa, Delta
    LOGICAL  :: anyAct, anySing, needFJ, anyNonAuto

    !--- gather the NC cells this thread owns -------------------------------
    DO mc = 1, NC
       ii = (g-1)*NC + mc
       valid(mc) = ( ii <= nCell )
       IF ( valid(mc) ) THEN
          cid(mc) = perm(ii)
       ELSE
          cid(mc) = perm(nCell)      ! pad lane: computed, never written back
       ENDIF
    ENDDO
    DO k = 1, NVAR
       DO mc = 1, NC
          Yl(mc,k)    = Yb(k,cid(mc))
          ATOLl(mc,k) = ATOLb(k,cid(mc))
          RTOLl(mc,k) = RTOLv(k)
       ENDDO
    ENDDO
    DO k = 1, NFIX
       DO mc = 1, NC
          FIXl(mc,k) = FIXb(k,cid(mc))
       ENDDO
    ENDDO
    DO k = 1, NREACT
       DO mc = 1, NC
          RCONSTl(mc,k) = RCONSTb(k,cid(mc))
       ENDDO
    ENDDO

    !--- option decoding, per cell (replicates Rosenbrock() defaults) -------
    anyNonAuto = .FALSE.
    DO mc = 1, NC
       ierr_c(mc) = 0
       Nst(mc)    = 0
       Hex(mc)    = ZERO
       active(mc) = .TRUE.
       Tnd(mc)    = Tendb(cid(mc))

       IF ( ICNTRLb(12,cid(mc)) == 1 ) THEN
          ierr_c(mc) = -90
          active(mc) = .FALSE.
       ENDIF
       Autonomous(mc) = .NOT. ( ICNTRLb(1,cid(mc)) == 0 )
       VectorTol(mc)  = ( ICNTRLb(2,cid(mc)) == 0 )
       IF ( .NOT. Autonomous(mc) ) anyNonAuto = .TRUE.
       IF ( .NOT. ( ICNTRLb(3,cid(mc)) == 0 .OR. ICNTRLb(3,cid(mc)) == 4 ) ) THEN
          ierr_c(mc) = -2
          active(mc) = .FALSE.
       ENDIF
       IF ( ICNTRLb(4,cid(mc)) == 0 ) THEN
          Max_no_steps(mc) = 200000
       ELSE
          Max_no_steps(mc) = ICNTRLb(4,cid(mc))
       ENDIF

       IF ( RCNTRLb(1,cid(mc)) > ZERO ) THEN
          Hmin(mc) = RCNTRLb(1,cid(mc))
       ELSE
          Hmin(mc) = ZERO
       ENDIF
       IF ( RCNTRLb(2,cid(mc)) > ZERO ) THEN
          Hmax(mc) = MIN( ABS(RCNTRLb(2,cid(mc))), ABS(Tnd(mc)-ZERO) )
       ELSE
          Hmax(mc) = ABS(Tnd(mc)-ZERO)
       ENDIF
       IF ( RCNTRLb(3,cid(mc)) > ZERO ) THEN
          Hstart(mc) = MIN( ABS(RCNTRLb(3,cid(mc))), ABS(Tnd(mc)-ZERO) )
       ELSE
          Hstart(mc) = MAX( Hmin(mc), DELTAMIN )
       ENDIF
       IF ( RCNTRLb(4,cid(mc)) > ZERO ) THEN
          FacMin(mc) = RCNTRLb(4,cid(mc))
       ELSE
          FacMin(mc) = 0.2_dp
       ENDIF
       IF ( RCNTRLb(5,cid(mc)) > ZERO ) THEN
          FacMax(mc) = RCNTRLb(5,cid(mc))
       ELSE
          FacMax(mc) = 6.0_dp
       ENDIF
       IF ( RCNTRLb(6,cid(mc)) > ZERO ) THEN
          FacRej(mc) = RCNTRLb(6,cid(mc))
       ELSE
          FacRej(mc) = 0.1_dp
       ENDIF
       IF ( RCNTRLb(7,cid(mc)) > ZERO ) THEN
          FacSafe(mc) = RCNTRLb(7,cid(mc))
       ELSE
          FacSafe(mc) = 0.9_dp
       ENDIF

       Tc(mc) = ZERO
       Hc(mc) = MIN( MAX( ABS(Hmin(mc)), ABS(Hstart(mc)) ), ABS(Hmax(mc)) )
       IF ( ABS(Hc(mc)) <= 10.0_dp*ROUNDOFF ) Hc(mc) = DELTAMIN
       IF ( Tnd(mc) >= ZERO ) THEN
          Direction(mc) = +1
       ELSE
          Direction(mc) = -1
       ENDIF
       Hc(mc) = Direction(mc)*Hc(mc)
       RejectLastH(mc) = .FALSE.
       RejectMoreH(mc) = .FALSE.
       newstep(mc)     = .TRUE.
    ENDDO

    anyAct = .FALSE.
    DO mc = 1, NC
       IF ( active(mc) ) anyAct = .TRUE.
    ENDDO

    mIter = 0
    MasterLoop: DO WHILE ( anyAct )
       mIter = mIter + 1
       IF ( mIter > 4000000 ) EXIT MasterLoop

       !--- Phase A: TimeLoop entry test for cells starting a new step ------
       needFJ = .FALSE.
       DO mc = 1, NC
          IF ( active(mc) .AND. newstep(mc) ) THEN
             IF ( .NOT. ( ( (Direction(mc) > 0) .AND.                         &
                            ((Tc(mc)-Tnd(mc))+ROUNDOFF <= ZERO) )             &
                     .OR. ( (Direction(mc) < 0) .AND.                         &
                            ((Tnd(mc)-Tc(mc))+ROUNDOFF <= ZERO) ) ) ) THEN
                ierr_c(mc) = 1                      ! normal completion
                active(mc) = .FALSE.
             ELSEIF ( Nst(mc) > Max_no_steps(mc) ) THEN
                ierr_c(mc) = -6
                active(mc) = .FALSE.
             ELSEIF ( ((Tc(mc)+0.1_dp*Hc(mc)) == Tc(mc)) .OR.                 &
                      (Hc(mc) <= ROUNDOFF) ) THEN
                ierr_c(mc) = -7
                active(mc) = .FALSE.
             ELSE
                Hc(mc) = MIN( Hc(mc), ABS(Tnd(mc)-Tc(mc)) )
                needFJ = .TRUE.
             ENDIF
          ENDIF
       ENDDO

       anyAct = .FALSE.
       DO mc = 1, NC
          IF ( active(mc) ) anyAct = .TRUE.
       ENDDO
       IF ( .NOT. anyAct ) EXIT MasterLoop

       !--- Phase B: F(Y) and J(Y) ------------------------------------------
       IF ( needFJ ) THEN
          CALL Fun_SPLIT_M( Yl, FIXl, RCONSTl, Fcn0, wPl, wDl )
          IF ( anyNonAuto ) THEN
             CALL Fun_SPLIT_M( Yl, FIXl, RCONSTl, dFdT, wPl, wDl )
             DO mc = 1, NC
                IF ( .NOT. Autonomous(mc) ) THEN
                   Delta = SQRT(ROUNDOFF)*MAX( 1.0E-6_dp, ABS(Tc(mc)) )
                   DO k = 1, NVAR
                      dFdT(mc,k) = ( dFdT(mc,k) - Fcn0(mc,k) ) / Delta
                   ENDDO
                ENDIF
             ENDDO
          ENDIF
          CALL Jac_SP_M( Yl, FIXl, RCONSTl, Jac0 )
       ENDIF

       !--- Phase C: ros_PrepareMatrix --------------------------------------
       DO mc = 1, NC
          Nconsecutive(mc) = 0
          Singular(mc)     = active(mc)
       ENDDO
       anySing = .TRUE.
       DO WHILE ( anySing )
          DO k = 1, NWORK_J
             DO mc = 1, NC
                Ghimj(mc,k) = -Jac0(mc,k)
             ENDDO
          ENDDO
          DO mc = 1, NC
             ghinv(mc) = ONE/(Direction(mc)*Hc(mc)*ros_Gamma(1))
          ENDDO
          CALL AddGhinvDiag_M( Ghimj, ghinv )
          CALL KppDecomp_M( Ghimj, ising )
          anySing = .FALSE.
          DO mc = 1, NC
             IF ( Singular(mc) ) THEN
                IF ( ising(mc) == 0 ) THEN
                   Singular(mc) = .FALSE.
                ELSE
                   Nconsecutive(mc) = Nconsecutive(mc) + 1
                   IF ( Nconsecutive(mc) <= 5 ) THEN
                      Hc(mc)  = Hc(mc)*HALF
                      anySing = .TRUE.
                   ELSE
                      ierr_c(mc)   = -8
                      active(mc)   = .FALSE.
                      Singular(mc) = .FALSE.
                   ENDIF
                ENDIF
             ENDIF
          ENDDO
       ENDDO

       !--- Phase D: Rosenbrock stages --------------------------------------
       Stage: DO istage = 1, ROS_S
          ioffset = NVAR*(istage-1)
          IF ( istage == 1 ) THEN
             DO k = 1, NVAR
                DO mc = 1, NC
                   Fcn(mc,k) = Fcn0(mc,k)
                ENDDO
             ENDDO
          ELSEIF ( ros_NewF(istage) ) THEN
             DO k = 1, NVAR
                DO mc = 1, NC
                   Ynew(mc,k) = Yl(mc,k)
                ENDDO
             ENDDO
             DO j = 1, istage-1
                aa = ros_A( (istage-1)*(istage-2)/2 + j )
                DO k = 1, NVAR
                   DO mc = 1, NC
                      Ynew(mc,k) = Ynew(mc,k) + aa*Kw(mc,NVAR*(j-1)+k)
                   ENDDO
                ENDDO
             ENDDO
             CALL Fun_SPLIT_M( Ynew, FIXl, RCONSTl, Fcn, wPl, wDl )
          ENDIF
          DO k = 1, NVAR
             DO mc = 1, NC
                Kw(mc,ioffset+k) = Fcn(mc,k)
             ENDDO
          ENDDO
          DO j = 1, istage-1
             DO mc = 1, NC
                HCc(mc) = ros_C( (istage-1)*(istage-2)/2 + j )                &
                          /(Direction(mc)*Hc(mc))
             ENDDO
             DO k = 1, NVAR
                DO mc = 1, NC
                   Kw(mc,ioffset+k) = Kw(mc,ioffset+k)                        &
                                      + HCc(mc)*Kw(mc,NVAR*(j-1)+k)
                ENDDO
             ENDDO
          ENDDO
          IF ( anyNonAuto ) THEN
             DO mc = 1, NC
                IF ( (.NOT. Autonomous(mc)) .AND. (ros_Gamma(istage) /= ZERO) ) THEN
                   HGc(mc) = Direction(mc)*Hc(mc)*ros_Gamma(istage)
                   DO k = 1, NVAR
                      Kw(mc,ioffset+k) = Kw(mc,ioffset+k) + HGc(mc)*dFdT(mc,k)
                   ENDDO
                ENDIF
             ENDDO
          ENDIF
          CALL KppSolve_M( Ghimj, Kw, ioffset )
       ENDDO Stage

       !--- Phase E: new solution, error estimate, accept/reject ------------
       DO k = 1, NVAR
          DO mc = 1, NC
             Ynew(mc,k) = Yl(mc,k)
          ENDDO
       ENDDO
       DO j = 1, ROS_S
          DO k = 1, NVAR
             DO mc = 1, NC
                Ynew(mc,k) = Ynew(mc,k) + ros_M(j)*Kw(mc,NVAR*(j-1)+k)
             ENDDO
          ENDDO
       ENDDO
       DO k = 1, NVAR
          DO mc = 1, NC
             Yerr(mc,k) = ZERO
          ENDDO
       ENDDO
       DO j = 1, ROS_S
          DO k = 1, NVAR
             DO mc = 1, NC
                Yerr(mc,k) = Yerr(mc,k) + ros_E(j)*Kw(mc,NVAR*(j-1)+k)
             ENDDO
          ENDDO
       ENDDO
       DO mc = 1, NC
          Err(mc) = ZERO
       ENDDO
       DO i = 1, NVAR
          DO mc = 1, NC
             Ymax = MAX( ABS(Yl(mc,i)), ABS(Ynew(mc,i)) )
             IF ( VectorTol(mc) ) THEN
                Scal = ATOLl(mc,i) + RTOLl(mc,i)*Ymax
             ELSE
                Scal = ATOLl(mc,1) + RTOLl(mc,1)*Ymax
             ENDIF
             Err(mc) = Err(mc) + ( Yerr(mc,i)/Scal )**2
          ENDDO
       ENDDO
       DO mc = 1, NC
          Err(mc)  = SQRT( Err(mc)/NVAR )
          Err(mc)  = MAX( Err(mc), 1.0d-10 )
          Fac(mc)  = MIN( FacMax(mc), MAX( FacMin(mc),                        &
                          FacSafe(mc)/Err(mc)**(ONE/ros_ELO) ) )
          Hnew(mc) = Hc(mc)*Fac(mc)
       ENDDO

       DO mc = 1, NC
          IF ( active(mc) ) THEN
             Nst(mc) = Nst(mc) + 1
             IF ( (Err(mc) <= ONE) .OR. (Hc(mc) <= Hmin(mc)) ) THEN
                IF ( ICNTRLb(16,cid(mc)) == 1 ) THEN
                   DO k = 1, NVAR
                      Yl(mc,k) = MAX( Ynew(mc,k), ZERO )
                   ENDDO
                ELSE
                   DO k = 1, NVAR
                      Yl(mc,k) = Ynew(mc,k)
                   ENDDO
                ENDIF
                Tc(mc)   = Tc(mc) + Direction(mc)*Hc(mc)
                Hnew(mc) = MAX( Hmin(mc), MIN( Hnew(mc), Hmax(mc) ) )
                IF ( RejectLastH(mc) ) Hnew(mc) = MIN( Hnew(mc), Hc(mc) )
                Hex(mc)  = Hc(mc)
                RejectLastH(mc) = .FALSE.
                RejectMoreH(mc) = .FALSE.
                Hc(mc)      = Hnew(mc)
                newstep(mc) = .TRUE.
             ELSE
                IF ( RejectMoreH(mc) ) Hnew(mc) = Hc(mc)*FacRej(mc)
                RejectMoreH(mc) = RejectLastH(mc)
                RejectLastH(mc) = .TRUE.
                Hc(mc)      = Hnew(mc)
                newstep(mc) = .FALSE.
             ENDIF
          ENDIF
       ENDDO

    ENDDO MasterLoop

    !--- scatter results back ----------------------------------------------
    DO mc = 1, NC
       IF ( valid(mc) ) THEN
          DO k = 1, NVAR
             Yb(k,cid(mc)) = Yl(mc,k)
          ENDDO
          Nstepsb(cid(mc)) = Nst(mc)
          Hexitb(cid(mc))  = Hex(mc)
          IERRb(cid(mc))   = ierr_c(mc)
       ENDIF
    ENDDO

  END SUBROUTINE Integrate_Cell_M
