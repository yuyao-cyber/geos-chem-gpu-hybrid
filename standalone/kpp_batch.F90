!------------------------------------------------------------------------------
! kpp_batch.F90 — batched driver for the GEOS-Chem chemistry solver
!
! Loads many harvested grid-cell states (KPP-Standalone sample files) and
! solves them all with the batched integrator (gckpp_BatchIntegrator),
! on CPU (serial or OpenMP) or GPU (OpenACC).  Validates against the
! ORIGINAL global-state integrator cell by cell in "check" mode.
!
! Usage:  ./kpp_batch.exe <listfile> <mode> [ncap]
!   listfile : text file, one sample-file path per line
!   mode     : check | cpu1 | omp | gpu
!   ncap     : optional max number of cells (default: all lines)
!
! Outputs: throughput (cells/s), per-mode wall time, accuracy vs serial.
!------------------------------------------------------------------------------
PROGRAM kpp_batch

  USE gckpp_Precision,       ONLY : dp
  USE gckpp_Parameters,      ONLY : NVAR, NFIX, NSPEC, NREACT
  USE gckpp_Monitor,         ONLY : SPC_NAMES
  USE gckpp_BatchIntegrator, ONLY : Integrate_Cell, Integrate_Cell_L,        &
                                    Integrate_Cell_I, Integrate_Cell_W,      &
                                    NWORK_K, NWORK_J
  USE gckpp_Interleaved,     ONLY : NB_JAC
  USE gckpp_MultiCell2,      ONLY : Integrate_Cell_M2 => Integrate_Cell_M,    &
                                    NC2 => NC
#ifdef WITH_M4
  USE gckpp_MultiCell4,      ONLY : Integrate_Cell_M4 => Integrate_Cell_M,    &
                                    NC4 => NC
#endif
  USE kpp_standalone_init,   ONLY : read_input

  IMPLICIT NONE

  CHARACTER(LEN=256) :: listFile, mode, arg3, line
  INTEGER            :: nMax, nCell, i, u, ios, nErr, nDiff
  INTEGER(8)         :: c0, c1, crate
  REAL(dp)           :: tWall, maxRel, rel, denom

  ! Per-cell packed state (cell index last => each cell contiguous)
  REAL(dp), ALLOCATABLE :: Yb(:,:), Y0b(:,:), FIXb(:,:), RCONSTb(:,:)
  REAL(dp), ALLOCATABLE :: ATOLb(:,:), RTOLv(:), RCNTRLb(:,:)
  REAL(dp), ALLOCATABLE :: Tendb(:), Hexitb(:)
  INTEGER,  ALLOCATABLE :: ICNTRLb(:,:), Nstepsb(:), IERRb(:)
  REAL(dp), ALLOCATABLE :: Yref(:,:)

  ! Workspace (cell index last)
  REAL(dp), ALLOCATABLE :: wK(:,:), wG(:,:), wJ(:,:)
  REAL(dp), ALLOCATABLE :: wYn(:,:), wF0(:,:), wF(:,:), wDF(:,:), wYe(:,:)
  REAL(dp), ALLOCATABLE :: wP(:,:), wD(:,:)
  REAL(dp), ALLOCATABLE :: wWs(:,:), wSs(:,:)   ! warp-coop scratch
  INTEGER(8) :: nGT3, nGT6

  ! Interleaved layout (cell index FIRST; leading dim padded to 128)
  INTEGER :: nPad, k
  REAL(dp), ALLOCATABLE :: Y_i(:,:), FIX_i(:,:), RCONST_i(:,:), ATOL_i(:,:)
  REAL(dp), ALLOCATABLE :: RCNTRL_i(:,:)
  INTEGER,  ALLOCATABLE :: ICNTRL_i(:,:)
  REAL(dp), ALLOCATABLE :: wK_i(:,:), wG_i(:,:), wJ_i(:,:)
  REAL(dp), ALLOCATABLE :: wYn_i(:,:), wF0_i(:,:), wF_i(:,:)
  REAL(dp), ALLOCATABLE :: wDF_i(:,:), wYe_i(:,:), wP_i(:,:), wD_i(:,:)
  REAL(dp), ALLOCATABLE :: wA_i(:,:), wB_i(:,:), wW_i(:,:)

  ! Multi-cell (per-thread ILP) support: cell->thread assignment maps
  INTEGER,  ALLOCATABLE :: permId(:), permSort(:)
  INTEGER               :: gg, nGroup

  ! Scratch for reading one file
  REAL(dp) :: C1cell(NSPEC), R1(NREACT), ATOL1(NSPEC), RCNTRL1(20)
  REAL(dp) :: Hstart, Hexit, cosSZA, OperatorTimestep
  INTEGER  :: ICNTRL1(20), level, fileTotSteps

  CALL Get_Command_Argument( 1, listFile )
  CALL Get_Command_Argument( 2, mode )
  nMax = HUGE(1)
  IF ( Command_Argument_Count() >= 3 ) THEN
     CALL Get_Command_Argument( 3, arg3 )
     READ( arg3, * ) nMax
  ENDIF

  !--------------------------------------------------------------------
  ! Pass 1: count lines
  !--------------------------------------------------------------------
  OPEN( NEWUNIT=u, FILE=TRIM(listFile), STATUS='OLD', ACTION='READ' )
  nCell = 0
  DO
     READ( u, '(A)', IOSTAT=ios ) line
     IF ( ios /= 0 ) EXIT
     IF ( LEN_TRIM(line) > 0 ) nCell = nCell + 1
     IF ( nCell >= nMax ) EXIT
  ENDDO
  CLOSE( u )
  PRINT '(A,I8)', 'Cells to load: ', nCell

  ALLOCATE( Yb(NVAR,nCell), Y0b(NVAR,nCell), FIXb(NFIX,nCell) )
  ALLOCATE( RCONSTb(NREACT,nCell), ATOLb(NVAR,nCell), RTOLv(NVAR) )
  ALLOCATE( RCNTRLb(20,nCell), ICNTRLb(20,nCell), Tendb(nCell) )
  ALLOCATE( Hexitb(nCell), Nstepsb(nCell), IERRb(nCell) )
  RTOLv = 0.5e-2_dp

  !--------------------------------------------------------------------
  ! Pass 2: load all cells
  !--------------------------------------------------------------------
  CALL SYSTEM_CLOCK( c0, crate )
  OPEN( NEWUNIT=u, FILE=TRIM(listFile), STATUS='OLD', ACTION='READ' )
  i = 0
  DO
     READ( u, '(A)', IOSTAT=ios ) line
     IF ( ios /= 0 ) EXIT
     IF ( LEN_TRIM(line) == 0 ) CYCLE
     i = i + 1
     IF ( i > nCell ) EXIT
     CALL read_input( TRIM(line), R1, C1cell, SPC_NAMES, Hstart, Hexit,      &
                      cosSZA, level, fileTotSteps, OperatorTimestep,         &
                      ICNTRL1, RCNTRL1, ATOL1 )
     Yb(:,i)      = C1cell(1:NVAR)
     FIXb(:,i)    = C1cell(NVAR+1:NSPEC)
     RCONSTb(:,i) = R1
     WHERE( ATOL1(1:NVAR) < 0.0_dp ) ATOL1(1:NVAR) = 1.0e-2_dp
     ATOLb(:,i)   = ATOL1(1:NVAR)
     ICNTRLb(:,i) = ICNTRL1
     RCNTRL1(3)   = Hstart
     RCNTRLb(:,i) = RCNTRL1
     Tendb(i)     = OperatorTimestep
  ENDDO
  CLOSE( u )
  ! i may overshoot by 1 when capped; never grow past allocation
  nCell = MIN( nCell, i )
  Y0b = Yb
  CALL SYSTEM_CLOCK( c1 )
  PRINT '(A,I8,A,F8.1,A)', 'Loaded cells:  ', nCell, '   in ',               &
        REAL(c1-c0,dp)/REAL(crate,dp), ' s'

  !--------------------------------------------------------------------
  ! Workspace
  !--------------------------------------------------------------------
  ALLOCATE( wK(NWORK_K,nCell), wG(NWORK_J,nCell), wJ(NWORK_J,nCell) )
  ALLOCATE( wYn(NVAR,nCell), wF0(NVAR,nCell), wF(NVAR,nCell) )
  ALLOCATE( wDF(NVAR,nCell), wYe(NVAR,nCell) )
  ALLOCATE( wP(NVAR,nCell), wD(NVAR,nCell) )
  ALLOCATE( wWs(NVAR,nCell), wSs(NVAR,nCell) )

  SELECT CASE ( TRIM(mode) )

  !====================================================================
  CASE ( 'check' )
     ! Batched solver vs ORIGINAL integrator, cell by cell
     CALL SYSTEM_CLOCK( c0, crate )
     CALL Run_Serial()
     CALL SYSTEM_CLOCK( c1 )
     tWall = REAL(c1-c0,dp)/REAL(crate,dp)
     PRINT '(A,F10.2,A,F10.1,A)', 'Batched serial: ', tWall, ' s  (',        &
           nCell/tWall, ' cells/s)'
     ALLOCATE( Yref(NVAR,nCell) )
     CALL SYSTEM_CLOCK( c0 )
     CALL Run_Original( Yref )
     CALL SYSTEM_CLOCK( c1 )
     tWall = REAL(c1-c0,dp)/REAL(crate,dp)
     PRINT '(A,F10.2,A,F10.1,A)', 'Original serial:', tWall, ' s  (',        &
           nCell/tWall, ' cells/s)'
     nDiff  = 0
     nGT6   = 0
     nGT3   = 0
     maxRel = 0.0_dp
     DO i = 1, nCell
        DO u = 1, NVAR
           IF ( Yb(u,i) /= Yref(u,i) ) THEN
              nDiff = nDiff + 1
              denom = MAX( ABS(Yref(u,i)), 1.0e-30_dp )
              rel   = ABS( Yb(u,i) - Yref(u,i) ) / denom
              IF ( rel > 1.0e-6_dp ) nGT6 = nGT6 + 1
              IF ( rel > 1.0e-3_dp ) nGT3 = nGT3 + 1
              IF ( rel > maxRel ) maxRel = rel
           ENDIF
        ENDDO
     ENDDO
     PRINT '(A,I10,A,I10)', 'CHECK: rel>1e-6: ', nGT6, '   rel>1e-3: ', nGT3
     ! WLAMCH probe: what unit roundoff does the ORIGINAL actually use?
     BLOCK
       USE gckpp_LinearAlgebra, ONLY : WLAMCH
       REAL(dp) :: wl
       wl = WLAMCH('E')
       PRINT '(A,ES18.10,A,ES18.10)', 'WLAMCH(E) = ', wl, '   2**-53 = ', 2.0_dp**(-53)
     END BLOCK
     ! identify the differing cells
     DO i = 1, nCell
        rel = 0.0_dp
        DO u = 1, NVAR
           denom = MAX( ABS(Yref(u,i)), 1.0e-30_dp )
           rel   = MAX( rel, ABS( Yb(u,i) - Yref(u,i) ) / denom )
        ENDDO
        IF ( rel > 1.0e-6_dp ) THEN
           PRINT '(A,I7,A,ES11.3,A,I5,A,ES11.3)', 'DIFFCELL i=', i,          &
                 '  maxrel=', rel, '  batchNsteps=', Nstepsb(i),             &
                 '  Hexit=', Hexitb(i)
        ENDIF
     ENDDO
     PRINT '(A,I10,A,I12,A)', 'CHECK: differing values: ', nDiff, '  of ',   &
           INT(nCell,8)*NVAR, ' (batched vs original)'
     PRINT '(A,ES12.4)',      'CHECK: max relative diff: ', maxRel

  !====================================================================
  CASE ( 'cpu1' )
     CALL SYSTEM_CLOCK( c0, crate )
     CALL Run_Serial()
     CALL SYSTEM_CLOCK( c1 )
     tWall = REAL(c1-c0,dp)/REAL(crate,dp)
     PRINT '(A,F10.2,A,F10.1,A)', 'CPU serial:  ', tWall, ' s  (',           &
           nCell/tWall, ' cells/s)'

  !====================================================================
  CASE ( 'omp' )
     CALL SYSTEM_CLOCK( c0, crate )
     !$omp parallel do schedule(dynamic,16)
     DO i = 1, nCell
        CALL Integrate_Cell( Yb(:,i), FIXb(:,i), RCONSTb(:,i), 0.0_dp,       &
                             Tendb(i), ATOLb(:,i), RTOLv, ICNTRLb(:,i),      &
                             RCNTRLb(:,i), wK(:,i), wG(:,i), wJ(:,i),        &
                             wYn(:,i), wF0(:,i), wF(:,i), wDF(:,i),          &
                             wYe(:,i), wP(:,i), wD(:,i),                         &
                             Nstepsb(i), Hexitb(i), IERRb(i) )
     ENDDO
     !$omp end parallel do
     CALL SYSTEM_CLOCK( c1 )
     tWall = REAL(c1-c0,dp)/REAL(crate,dp)
     PRINT '(A,F10.2,A,F10.1,A)', 'CPU OpenMP:  ', tWall, ' s  (',           &
           nCell/tWall, ' cells/s)'

  !====================================================================
  CASE ( 'gpu' )
     ! Reference on CPU first (serial), into Yref
     ALLOCATE( Yref(NVAR,nCell) )
     CALL SYSTEM_CLOCK( c0, crate )
     CALL Run_Serial()
     CALL SYSTEM_CLOCK( c1 )
     tWall = REAL(c1-c0,dp)/REAL(crate,dp)
     PRINT '(A,F10.2,A,F10.1,A)', 'CPU serial:  ', tWall, ' s  (',           &
           nCell/tWall, ' cells/s)'
     Yref = Yb
     Yb   = Y0b    ! reset inputs for the GPU pass

     !$acc data copy(Yb) copyin(FIXb, RCONSTb, ATOLb, RTOLv, ICNTRLb,        &
     !$acc              RCNTRLb, Tendb)                                      &
     !$acc      create(wK, wG, wJ, wYn, wF0, wF, wDF, wYe, wP, wD)                   &
     !$acc      copyout(Nstepsb, Hexitb, IERRb)
     CALL SYSTEM_CLOCK( c0 )
     !$acc parallel loop gang vector
     DO i = 1, nCell
        CALL Integrate_Cell( Yb(:,i), FIXb(:,i), RCONSTb(:,i), 0.0_dp,       &
                             Tendb(i), ATOLb(:,i), RTOLv, ICNTRLb(:,i),      &
                             RCNTRLb(:,i), wK(:,i), wG(:,i), wJ(:,i),        &
                             wYn(:,i), wF0(:,i), wF(:,i), wDF(:,i),          &
                             wYe(:,i), wP(:,i), wD(:,i),                         &
                             Nstepsb(i), Hexitb(i), IERRb(i) )
     ENDDO
     CALL SYSTEM_CLOCK( c1 )
     !$acc end data
     tWall = REAL(c1-c0,dp)/REAL(crate,dp)
     PRINT '(A,F10.2,A,F10.1,A)', 'GPU kernel:  ', tWall, ' s  (',           &
           nCell/tWall, ' cells/s)'

     nErr   = COUNT( IERRb(1:nCell) /= 1 )
     maxRel = 0.0_dp
     DO i = 1, nCell
        DO u = 1, NVAR
           denom = MAX( ABS(Yref(u,i)), 1.0e-20_dp )
           rel   = ABS( Yb(u,i) - Yref(u,i) ) / denom
           IF ( rel > maxRel ) maxRel = rel
        ENDDO
     ENDDO
     PRINT '(A,I8)',     'GPU failed cells (IERR/=1): ', nErr
     PRINT '(A,ES12.4)', 'GPU vs CPU max relative diff: ', maxRel

  !====================================================================
  CASE ( 'gpul' )
     ! GPU with routine-LOCAL workspace (hardware-interleaved local
     ! memory => coalesced) -- compare against mode 'gpu'
     ALLOCATE( Yref(NVAR,nCell) )
     CALL SYSTEM_CLOCK( c0, crate )
     CALL Run_Serial()
     CALL SYSTEM_CLOCK( c1 )
     tWall = REAL(c1-c0,dp)/REAL(crate,dp)
     PRINT '(A,F10.2,A,F10.1,A)', 'CPU serial:  ', tWall, ' s  (',           &
           nCell/tWall, ' cells/s)'
     Yref = Yb
     Yb   = Y0b
     !$acc data copy(Yb) copyin(FIXb, RCONSTb, ATOLb, RTOLv, ICNTRLb,        &
     !$acc              RCNTRLb, Tendb)                                      &
     !$acc      copyout(Nstepsb, Hexitb, IERRb)
     CALL SYSTEM_CLOCK( c0 )
     !$acc parallel loop gang vector
     DO i = 1, nCell
        CALL Integrate_Cell_L( Yb(:,i), FIXb(:,i), RCONSTb(:,i), 0.0_dp,     &
                               Tendb(i), ATOLb(:,i), RTOLv, ICNTRLb(:,i),    &
                               RCNTRLb(:,i), Nstepsb(i), Hexitb(i), IERRb(i) )
     ENDDO
     CALL SYSTEM_CLOCK( c1 )
     !$acc end data
     tWall = REAL(c1-c0,dp)/REAL(crate,dp)
     PRINT '(A,F10.2,A,F10.1,A)', 'GPU-local kernel:  ', tWall, ' s  (',     &
           nCell/tWall, ' cells/s)'
     nErr   = COUNT( IERRb(1:nCell) /= 1 )
     maxRel = 0.0_dp
     DO i = 1, nCell
        DO u = 1, NVAR
           denom = MAX( ABS(Yref(u,i)), 1.0e-20_dp )
           rel   = ABS( Yb(u,i) - Yref(u,i) ) / denom
           IF ( rel > maxRel ) maxRel = rel
        ENDDO
     ENDDO
     PRINT '(A,I8)',     'GPU-local failed cells (IERR/=1): ', nErr
     PRINT '(A,ES12.4)', 'GPU-local vs CPU max relative diff: ', maxRel

  !====================================================================
  CASE ( 'cpui' )
     ! INTERLEAVED layout on CPU (OpenMP): correctness gate for gpui.
     ! Must be BIT-EXACT vs the original integrator (same op order/cell).
     CALL Alloc_Interleaved()
     CALL SYSTEM_CLOCK( c0, crate )
     CALL Transpose_In()
     CALL SYSTEM_CLOCK( c1 )
     PRINT '(A,F10.2,A)', 'Transpose-in:  ',                                 &
           REAL(c1-c0,dp)/REAL(crate,dp), ' s  (excluded from solve time)'
     CALL SYSTEM_CLOCK( c0 )
     !$omp parallel do schedule(dynamic,16)
     DO i = 1, nCell
        CALL Integrate_Cell_I( i, nPad, Y_i, FIX_i, RCONST_i, 0.0_dp,        &
                               Tendb(i), ATOL_i, RTOLv, ICNTRL_i, RCNTRL_i,  &
                               wK_i, wG_i, wJ_i, wYn_i, wF0_i, wF_i, wDF_i,  &
                               wYe_i, wP_i, wD_i, wA_i, wB_i, wW_i,          &
                               Nstepsb(i), Hexitb(i), IERRb(i) )
     ENDDO
     !$omp end parallel do
     CALL SYSTEM_CLOCK( c1 )
     tWall = REAL(c1-c0,dp)/REAL(crate,dp)
     PRINT '(A,F10.2,A,F10.1,A)', 'CPU-OMP interleaved: ', tWall, ' s  (',   &
           nCell/tWall, ' cells/s)'
     CALL Transpose_Back()
     ALLOCATE( Yref(NVAR,nCell) )
     CALL SYSTEM_CLOCK( c0 )
     CALL Run_Original( Yref )
     CALL SYSTEM_CLOCK( c1 )
     tWall = REAL(c1-c0,dp)/REAL(crate,dp)
     PRINT '(A,F10.2,A,F10.1,A)', 'Original serial:', tWall, ' s  (',        &
           nCell/tWall, ' cells/s)'
     nDiff  = 0
     maxRel = 0.0_dp
     DO i = 1, nCell
        DO u = 1, NVAR
           IF ( Yb(u,i) /= Yref(u,i) ) THEN
              nDiff = nDiff + 1
              denom = MAX( ABS(Yref(u,i)), 1.0e-30_dp )
              rel   = ABS( Yb(u,i) - Yref(u,i) ) / denom
              IF ( rel > maxRel ) maxRel = rel
           ENDIF
        ENDDO
     ENDDO
     PRINT '(A,I10,A,I12,A)', 'CPUI CHECK: differing values: ', nDiff,       &
           '  of ', INT(nCell,8)*NVAR, ' (interleaved vs original)'
     PRINT '(A,ES12.4)',      'CPUI CHECK: max relative diff: ', maxRel

  !====================================================================
  CASE ( 'gpui' )
     ! GPU with INTERLEAVED global layout: thread ic and ic+1 touch
     ! adjacent addresses for the same element k => coalesced.
     ALLOCATE( Yref(NVAR,nCell) )
     CALL SYSTEM_CLOCK( c0, crate )
     CALL Run_Serial()
     CALL SYSTEM_CLOCK( c1 )
     tWall = REAL(c1-c0,dp)/REAL(crate,dp)
     PRINT '(A,F10.2,A,F10.1,A)', 'CPU serial:  ', tWall, ' s  (',           &
           nCell/tWall, ' cells/s)'
     Yref = Yb
     Yb   = Y0b
     CALL Alloc_Interleaved()
     CALL SYSTEM_CLOCK( c0 )
     CALL Transpose_In()
     CALL SYSTEM_CLOCK( c1 )
     PRINT '(A,F10.2,A)', 'Transpose-in:  ',                                 &
           REAL(c1-c0,dp)/REAL(crate,dp), ' s  (excluded from solve time)'

     !$acc data copy(Y_i)                                                    &
     !$acc      copyin(FIX_i, RCONST_i, ATOL_i, RTOLv, ICNTRL_i,             &
     !$acc             RCNTRL_i, Tendb)                                      &
     !$acc      create(wK_i, wG_i, wJ_i, wYn_i, wF0_i, wF_i, wDF_i,          &
     !$acc             wYe_i, wP_i, wD_i, wA_i, wB_i, wW_i)                  &
     !$acc      copyout(Nstepsb, Hexitb, IERRb)
     CALL SYSTEM_CLOCK( c0 )
     !$acc parallel loop gang vector
     DO i = 1, nCell
        CALL Integrate_Cell_I( i, nPad, Y_i, FIX_i, RCONST_i, 0.0_dp,        &
                               Tendb(i), ATOL_i, RTOLv, ICNTRL_i, RCNTRL_i,  &
                               wK_i, wG_i, wJ_i, wYn_i, wF0_i, wF_i, wDF_i,  &
                               wYe_i, wP_i, wD_i, wA_i, wB_i, wW_i,          &
                               Nstepsb(i), Hexitb(i), IERRb(i) )
     ENDDO
     CALL SYSTEM_CLOCK( c1 )
     !$acc end data
     tWall = REAL(c1-c0,dp)/REAL(crate,dp)
     PRINT '(A,F10.2,A,F10.1,A)', 'GPU-interleaved kernel:  ', tWall,        &
           ' s  (', nCell/tWall, ' cells/s)'
     CALL Transpose_Back()
     nErr   = COUNT( IERRb(1:nCell) /= 1 )
     maxRel = 0.0_dp
     DO i = 1, nCell
        DO u = 1, NVAR
           denom = MAX( ABS(Yref(u,i)), 1.0e-20_dp )
           rel   = ABS( Yb(u,i) - Yref(u,i) ) / denom
           IF ( rel > maxRel ) maxRel = rel
        ENDDO
     ENDDO
     PRINT '(A,I8)',     'GPU-interleaved failed cells (IERR/=1): ', nErr
     PRINT '(A,ES12.4)', 'GPU-interleaved vs CPU max relative diff: ', maxRel


  !====================================================================
  CASE ( 'cpuw' )
     ! WARP-COOPERATIVE code path compiled for CPU (gfortran; the
     ! '!$acc loop vector' directives are inert, so the loops run
     ! serially).  BIT-EXACT gate vs the ORIGINAL integrator.
     CALL SYSTEM_CLOCK( c0, crate )
     !$omp parallel do schedule(dynamic,16)
     DO i = 1, nCell
        CALL Integrate_Cell_W( Yb(:,i), FIXb(:,i), RCONSTb(:,i), 0.0_dp,     &
                               Tendb(i), ATOLb(:,i), RTOLv, ICNTRLb(:,i),    &
                               RCNTRLb(:,i), wK(:,i), wG(:,i), wJ(:,i),      &
                               wYn(:,i), wF0(:,i), wF(:,i), wDF(:,i),        &
                               wYe(:,i), wP(:,i), wD(:,i), wWs(:,i),         &
                               wSs(:,i), Nstepsb(i), Hexitb(i), IERRb(i) )
     ENDDO
     !$omp end parallel do
     CALL SYSTEM_CLOCK( c1 )
     tWall = REAL(c1-c0,dp)/REAL(crate,dp)
     PRINT '(A,F10.2,A,F10.1,A)', 'CPU-OMP warp-path: ', tWall, ' s  (',     &
           nCell/tWall, ' cells/s)'
     ALLOCATE( Yref(NVAR,nCell) )
     CALL SYSTEM_CLOCK( c0 )
     CALL Run_Original( Yref )
     CALL SYSTEM_CLOCK( c1 )
     tWall = REAL(c1-c0,dp)/REAL(crate,dp)
     PRINT '(A,F10.2,A,F10.1,A)', 'Original serial:', tWall, ' s  (',        &
           nCell/tWall, ' cells/s)'
     nDiff  = 0
     maxRel = 0.0_dp
     DO i = 1, nCell
        DO u = 1, NVAR
           IF ( Yb(u,i) /= Yref(u,i) ) THEN
              nDiff = nDiff + 1
              denom = MAX( ABS(Yref(u,i)), 1.0e-30_dp )
              rel   = ABS( Yb(u,i) - Yref(u,i) ) / denom
              IF ( rel > maxRel ) maxRel = rel
           ENDIF
        ENDDO
     ENDDO
     PRINT '(A,I10,A,I12,A)', 'CPUW CHECK: differing values: ', nDiff,       &
           '  of ', INT(nCell,8)*NVAR, ' (warp-path vs original)'
     PRINT '(A,ES12.4)',      'CPUW CHECK: max relative diff: ', maxRel

  !====================================================================
  CASE ( 'gpuw' )
     ! WARP-COOPERATIVE on GPU: one gang (vector_length lanes) per cell.
     ! Sweeps vector lengths 32/16/8/4 in one run; gpul is run first on
     ! the same GPU for a clean same-job comparison.
     ALLOCATE( Yref(NVAR,nCell) )
     CALL SYSTEM_CLOCK( c0, crate )
     CALL Run_Serial()
     CALL SYSTEM_CLOCK( c1 )
     tWall = REAL(c1-c0,dp)/REAL(crate,dp)
     PRINT '(A,F10.2,A,F10.1,A)', 'CPU serial:  ', tWall, ' s  (',           &
           nCell/tWall, ' cells/s)'
     Yref = Yb

     !$acc data copy(Yb) copyin(FIXb, RCONSTb, ATOLb, RTOLv, ICNTRLb,        &
     !$acc              RCNTRLb, Tendb)                                      &
     !$acc      create(wK, wG, wJ, wYn, wF0, wF, wDF, wYe, wP, wD,           &
     !$acc             wWs, wSs)                                             &
     !$acc      copyout(Nstepsb, Hexitb, IERRb)

     ! --- reference: one-thread-per-cell, routine-local workspace ---
     Yb = Y0b
     !$acc update device(Yb)
     CALL SYSTEM_CLOCK( c0 )
     !$acc parallel loop gang vector
     DO i = 1, nCell
        CALL Integrate_Cell_L( Yb(:,i), FIXb(:,i), RCONSTb(:,i), 0.0_dp,     &
                               Tendb(i), ATOLb(:,i), RTOLv, ICNTRLb(:,i),    &
                               RCNTRLb(:,i), Nstepsb(i), Hexitb(i), IERRb(i) )
     ENDDO
     CALL SYSTEM_CLOCK( c1 )
     !$acc update self(Yb, IERRb)
     tWall = REAL(c1-c0,dp)/REAL(crate,dp)
     nErr = COUNT( IERRb(1:nCell) /= 1 )
     maxRel = 0.0_dp
     DO i = 1, nCell
        DO u = 1, NVAR
           denom = MAX( ABS(Yref(u,i)), 1.0e-20_dp )
           rel   = ABS( Yb(u,i) - Yref(u,i) ) / denom
           IF ( rel > maxRel ) maxRel = rel
        ENDDO
     ENDDO
     PRINT '(A,F10.2,A,F10.1,A,I8,A,ES11.3)', 'GPUL  (1 thr/cell) t= ',      &
           tWall, ' s  ', nCell/tWall, ' cells/s  nFail=', nErr,             &
           '  maxrel=', maxRel

     Yb = Y0b
     !$acc update device(Yb)
     CALL SYSTEM_CLOCK( c0 )
     !$acc parallel loop gang vector_length(32)
     DO i = 1, nCell
        CALL Integrate_Cell_W( Yb(:,i), FIXb(:,i), RCONSTb(:,i), 0.0_dp,     &
                               Tendb(i), ATOLb(:,i), RTOLv, ICNTRLb(:,i),    &
                               RCNTRLb(:,i), wK(:,i), wG(:,i), wJ(:,i),      &
                               wYn(:,i), wF0(:,i), wF(:,i), wDF(:,i),        &
                               wYe(:,i), wP(:,i), wD(:,i), wWs(:,i),         &
                               wSs(:,i), Nstepsb(i), Hexitb(i), IERRb(i) )
     ENDDO
     CALL SYSTEM_CLOCK( c1 )
     !$acc update self(Yb, IERRb)
     tWall = REAL(c1-c0,dp)/REAL(crate,dp)
     nErr   = COUNT( IERRb(1:nCell) /= 1 )
     maxRel = 0.0_dp
     DO i = 1, nCell
        DO u = 1, NVAR
           denom = MAX( ABS(Yref(u,i)), 1.0e-20_dp )
           rel   = ABS( Yb(u,i) - Yref(u,i) ) / denom
           IF ( rel > maxRel ) maxRel = rel
        ENDDO
     ENDDO
     PRINT '(A,I3,A,F10.2,A,F10.1,A,I8,A,ES11.3)', 'GPUW vlen=', 32,         &
           '  t= ', tWall, ' s  ', nCell/tWall, ' cells/s  nFail=', nErr,     &
           '  maxrel=', maxRel

     Yb = Y0b
     !$acc update device(Yb)
     CALL SYSTEM_CLOCK( c0 )
     !$acc parallel loop gang vector_length(16)
     DO i = 1, nCell
        CALL Integrate_Cell_W( Yb(:,i), FIXb(:,i), RCONSTb(:,i), 0.0_dp,     &
                               Tendb(i), ATOLb(:,i), RTOLv, ICNTRLb(:,i),    &
                               RCNTRLb(:,i), wK(:,i), wG(:,i), wJ(:,i),      &
                               wYn(:,i), wF0(:,i), wF(:,i), wDF(:,i),        &
                               wYe(:,i), wP(:,i), wD(:,i), wWs(:,i),         &
                               wSs(:,i), Nstepsb(i), Hexitb(i), IERRb(i) )
     ENDDO
     CALL SYSTEM_CLOCK( c1 )
     !$acc update self(Yb, IERRb)
     tWall = REAL(c1-c0,dp)/REAL(crate,dp)
     nErr   = COUNT( IERRb(1:nCell) /= 1 )
     maxRel = 0.0_dp
     DO i = 1, nCell
        DO u = 1, NVAR
           denom = MAX( ABS(Yref(u,i)), 1.0e-20_dp )
           rel   = ABS( Yb(u,i) - Yref(u,i) ) / denom
           IF ( rel > maxRel ) maxRel = rel
        ENDDO
     ENDDO
     PRINT '(A,I3,A,F10.2,A,F10.1,A,I8,A,ES11.3)', 'GPUW vlen=', 16,         &
           '  t= ', tWall, ' s  ', nCell/tWall, ' cells/s  nFail=', nErr,     &
           '  maxrel=', maxRel

     Yb = Y0b
     !$acc update device(Yb)
     CALL SYSTEM_CLOCK( c0 )
     !$acc parallel loop gang vector_length(8)
     DO i = 1, nCell
        CALL Integrate_Cell_W( Yb(:,i), FIXb(:,i), RCONSTb(:,i), 0.0_dp,     &
                               Tendb(i), ATOLb(:,i), RTOLv, ICNTRLb(:,i),    &
                               RCNTRLb(:,i), wK(:,i), wG(:,i), wJ(:,i),      &
                               wYn(:,i), wF0(:,i), wF(:,i), wDF(:,i),        &
                               wYe(:,i), wP(:,i), wD(:,i), wWs(:,i),         &
                               wSs(:,i), Nstepsb(i), Hexitb(i), IERRb(i) )
     ENDDO
     CALL SYSTEM_CLOCK( c1 )
     !$acc update self(Yb, IERRb)
     tWall = REAL(c1-c0,dp)/REAL(crate,dp)
     nErr   = COUNT( IERRb(1:nCell) /= 1 )
     maxRel = 0.0_dp
     DO i = 1, nCell
        DO u = 1, NVAR
           denom = MAX( ABS(Yref(u,i)), 1.0e-20_dp )
           rel   = ABS( Yb(u,i) - Yref(u,i) ) / denom
           IF ( rel > maxRel ) maxRel = rel
        ENDDO
     ENDDO
     PRINT '(A,I3,A,F10.2,A,F10.1,A,I8,A,ES11.3)', 'GPUW vlen=', 8,         &
           '  t= ', tWall, ' s  ', nCell/tWall, ' cells/s  nFail=', nErr,     &
           '  maxrel=', maxRel

     Yb = Y0b
     !$acc update device(Yb)
     CALL SYSTEM_CLOCK( c0 )
     !$acc parallel loop gang vector_length(4)
     DO i = 1, nCell
        CALL Integrate_Cell_W( Yb(:,i), FIXb(:,i), RCONSTb(:,i), 0.0_dp,     &
                               Tendb(i), ATOLb(:,i), RTOLv, ICNTRLb(:,i),    &
                               RCNTRLb(:,i), wK(:,i), wG(:,i), wJ(:,i),      &
                               wYn(:,i), wF0(:,i), wF(:,i), wDF(:,i),        &
                               wYe(:,i), wP(:,i), wD(:,i), wWs(:,i),         &
                               wSs(:,i), Nstepsb(i), Hexitb(i), IERRb(i) )
     ENDDO
     CALL SYSTEM_CLOCK( c1 )
     !$acc update self(Yb, IERRb)
     tWall = REAL(c1-c0,dp)/REAL(crate,dp)
     nErr   = COUNT( IERRb(1:nCell) /= 1 )
     maxRel = 0.0_dp
     DO i = 1, nCell
        DO u = 1, NVAR
           denom = MAX( ABS(Yref(u,i)), 1.0e-20_dp )
           rel   = ABS( Yb(u,i) - Yref(u,i) ) / denom
           IF ( rel > maxRel ) maxRel = rel
        ENDDO
     ENDDO
     PRINT '(A,I3,A,F10.2,A,F10.1,A,I8,A,ES11.3)', 'GPUW vlen=', 4,         &
           '  t= ', tWall, ' s  ', nCell/tWall, ' cells/s  nFail=', nErr,     &
           '  maxrel=', maxRel

     !$acc end data

  !====================================================================
  CASE ( 'cpum' )
     ! MULTI-CELL (per-thread ILP) code path compiled for CPU (gfortran).
     ! BIT-EXACT gate vs the ORIGINAL integrator, for BOTH cell->thread
     ! assignments (identity = the list order, and cost-sorted pairing).
     ALLOCATE( Yref(NVAR,nCell) )
     CALL SYSTEM_CLOCK( c0, crate )
     CALL Run_Original( Yref )
     CALL SYSTEM_CLOCK( c1 )
     PRINT '(A,F10.2,A,F10.1,A)', 'Original serial:', REAL(c1-c0,dp)/crate,   &
           ' s  (', nCell*crate/REAL(c1-c0,dp), ' cells/s)'
     ! per-cell cost (step count) for the cost-sorted pairing
     Yb = Y0b
     CALL Run_Serial()
     CALL Build_Perms()
     CALL Gate_M2( permId,   'cpum M2 identity ' )
     CALL Gate_M2( permSort, 'cpum M2 sorted   ' )
#ifdef WITH_M4
     CALL Gate_M4( permId,   'cpum M4 identity ' )
     CALL Gate_M4( permSort, 'cpum M4 sorted   ' )
#endif

  !====================================================================
  CASE ( 'gpum', 'gpum4' )
     ! MULTI-CELL (per-thread ILP) on GPU: one thread carries NC cells.
     ! gpul is run FIRST on the SAME GPU inside the SAME data region, so
     ! the comparison is clean.  Both pairings are measured.
     ALLOCATE( Yref(NVAR,nCell) )
     CALL SYSTEM_CLOCK( c0, crate )
     CALL Run_Serial()
     CALL SYSTEM_CLOCK( c1 )
     tWall = REAL(c1-c0,dp)/REAL(crate,dp)
     PRINT '(A,F10.2,A,F10.1,A)', 'CPU serial:  ', tWall, ' s  (',            &
           nCell/tWall, ' cells/s)'
     Yref = Yb
     CALL Build_Perms()

     !$acc data copy(Yb) copyin(FIXb, RCONSTb, ATOLb, RTOLv, ICNTRLb,         &
     !$acc              RCNTRLb, Tendb, permId, permSort)                     &
     !$acc      copyout(Nstepsb, Hexitb, IERRb)

     ! --- baseline: one thread per cell, routine-local workspace ---
     Yb = Y0b
     !$acc update device(Yb)
     CALL SYSTEM_CLOCK( c0 )
     !$acc parallel loop gang vector
     DO i = 1, nCell
        CALL Integrate_Cell_L( Yb(:,i), FIXb(:,i), RCONSTb(:,i), 0.0_dp,      &
                               Tendb(i), ATOLb(:,i), RTOLv, ICNTRLb(:,i),     &
                               RCNTRLb(:,i), Nstepsb(i), Hexitb(i), IERRb(i) )
     ENDDO
     CALL SYSTEM_CLOCK( c1 )
     !$acc update self(Yb, IERRb)
     CALL Report( 'GPUL  NC=1 identity', REAL(c1-c0,dp)/REAL(crate,dp) )

     ! --- control: same NC=1 kernel, cost-sorted cell order (warp balance) --
     Yb = Y0b
     !$acc update device(Yb)
     CALL SYSTEM_CLOCK( c0 )
     !$acc parallel loop gang vector private(k)
     DO i = 1, nCell
        k = permSort(i)
        CALL Integrate_Cell_L( Yb(:,k), FIXb(:,k), RCONSTb(:,k), 0.0_dp,      &
                               Tendb(k), ATOLb(:,k), RTOLv, ICNTRLb(:,k),     &
                               RCNTRLb(:,k), Nstepsb(k), Hexitb(k), IERRb(k) )
     ENDDO
     CALL SYSTEM_CLOCK( c1 )
     !$acc update self(Yb, IERRb)
     CALL Report( 'GPUL  NC=1 sorted  ', REAL(c1-c0,dp)/REAL(crate,dp) )

     IF ( TRIM(mode) == 'gpum' ) THEN
     nGroup = ( nCell + NC2 - 1 ) / NC2
     Yb = Y0b
     !$acc update device(Yb)
     CALL SYSTEM_CLOCK( c0 )
     !$acc parallel loop gang vector
     DO gg = 1, nGroup
        CALL Integrate_Cell_M2( gg, nCell, permId, Yb, FIXb, RCONSTb, Tendb,  &
                                ATOLb, RTOLv, ICNTRLb, RCNTRLb,               &
                                Nstepsb, Hexitb, IERRb )
     ENDDO
     CALL SYSTEM_CLOCK( c1 )
     !$acc update self(Yb, IERRb)
     CALL Report( 'GPUM  NC=2 identity', REAL(c1-c0,dp)/REAL(crate,dp) )

     Yb = Y0b
     !$acc update device(Yb)
     CALL SYSTEM_CLOCK( c0 )
     !$acc parallel loop gang vector
     DO gg = 1, nGroup
        CALL Integrate_Cell_M2( gg, nCell, permSort, Yb, FIXb, RCONSTb, Tendb,&
                                ATOLb, RTOLv, ICNTRLb, RCNTRLb,               &
                                Nstepsb, Hexitb, IERRb )
     ENDDO
     CALL SYSTEM_CLOCK( c1 )
     !$acc update self(Yb, IERRb)
     CALL Report( 'GPUM  NC=2 sorted  ', REAL(c1-c0,dp)/REAL(crate,dp) )
     ENDIF

#ifdef WITH_M4
     IF ( TRIM(mode) == 'gpum4' ) THEN
     nGroup = ( nCell + NC4 - 1 ) / NC4
     Yb = Y0b
     !$acc update device(Yb)
     CALL SYSTEM_CLOCK( c0 )
     !$acc parallel loop gang vector
     DO gg = 1, nGroup
        CALL Integrate_Cell_M4( gg, nCell, permId, Yb, FIXb, RCONSTb, Tendb,  &
                                ATOLb, RTOLv, ICNTRLb, RCNTRLb,               &
                                Nstepsb, Hexitb, IERRb )
     ENDDO
     CALL SYSTEM_CLOCK( c1 )
     !$acc update self(Yb, IERRb)
     CALL Report( 'GPUM  NC=4 identity', REAL(c1-c0,dp)/REAL(crate,dp) )

     Yb = Y0b
     !$acc update device(Yb)
     CALL SYSTEM_CLOCK( c0 )
     !$acc parallel loop gang vector
     DO gg = 1, nGroup
        CALL Integrate_Cell_M4( gg, nCell, permSort, Yb, FIXb, RCONSTb, Tendb,&
                                ATOLb, RTOLv, ICNTRLb, RCNTRLb,               &
                                Nstepsb, Hexitb, IERRb )
     ENDDO
     CALL SYSTEM_CLOCK( c1 )
     !$acc update self(Yb, IERRb)
     CALL Report( 'GPUM  NC=4 sorted  ', REAL(c1-c0,dp)/REAL(crate,dp) )
     ENDIF
#endif

     !$acc end data

  CASE DEFAULT
     PRINT *, 'Unknown mode: ', TRIM(mode)
  END SELECT

  PRINT '(A,I8)', 'Cells with IERR/=1: ', COUNT( IERRb(1:nCell) /= 1 )

CONTAINS

  SUBROUTINE Run_Serial()
    INTEGER :: ii
    DO ii = 1, nCell
       CALL Integrate_Cell( Yb(:,ii), FIXb(:,ii), RCONSTb(:,ii), 0.0_dp,     &
                            Tendb(ii), ATOLb(:,ii), RTOLv, ICNTRLb(:,ii),    &
                            RCNTRLb(:,ii), wK(:,ii), wG(:,ii), wJ(:,ii),     &
                            wYn(:,ii), wF0(:,ii), wF(:,ii), wDF(:,ii),       &
                            wYe(:,ii), wP(:,ii), wD(:,ii),                       &
                            Nstepsb(ii), Hexitb(ii), IERRb(ii) )
    ENDDO
  END SUBROUTINE Run_Serial

  SUBROUTINE Alloc_Interleaved()
    ! Leading (cell) dimension padded to a multiple of 128 for alignment
    nPad = ( (nCell + 127) / 128 ) * 128
    PRINT '(A,I8,A,I8)', 'Interleaved: nCell = ', nCell, '   nPad = ', nPad
    ALLOCATE( Y_i(nPad,NVAR), FIX_i(nPad,NFIX), RCONST_i(nPad,NREACT) )
    ALLOCATE( ATOL_i(nPad,NVAR), ICNTRL_i(nPad,20), RCNTRL_i(nPad,20) )
    ALLOCATE( wK_i(nPad,NWORK_K), wG_i(nPad,NWORK_J), wJ_i(nPad,NWORK_J) )
    ALLOCATE( wYn_i(nPad,NVAR), wF0_i(nPad,NVAR), wF_i(nPad,NVAR) )
    ALLOCATE( wDF_i(nPad,NVAR), wYe_i(nPad,NVAR), wP_i(nPad,NVAR) )
    ALLOCATE( wD_i(nPad,NVAR), wA_i(nPad,NREACT), wB_i(nPad,NB_JAC) )
    ALLOCATE( wW_i(nPad,NVAR) )
  END SUBROUTINE Alloc_Interleaved

  SUBROUTINE Transpose_In()
    ! Pack cell-last inputs into interleaved (cell-first) arrays.
    ! Padded rows (nCell+1..nPad) are zero-filled and never touched by the
    ! solver loop (which runs i = 1..nCell only).
    INTEGER :: ii, kk
    Y_i = 0.0_dp;  FIX_i = 0.0_dp;  RCONST_i = 0.0_dp;  ATOL_i = 0.0_dp
    ICNTRL_i = 0;  RCNTRL_i = 0.0_dp
    DO kk = 1, NVAR
       DO ii = 1, nCell
          Y_i(ii,kk)    = Yb(kk,ii)
          ATOL_i(ii,kk) = ATOLb(kk,ii)
       ENDDO
    ENDDO
    DO kk = 1, NFIX
       DO ii = 1, nCell
          FIX_i(ii,kk) = FIXb(kk,ii)
       ENDDO
    ENDDO
    DO kk = 1, NREACT
       DO ii = 1, nCell
          RCONST_i(ii,kk) = RCONSTb(kk,ii)
       ENDDO
    ENDDO
    DO kk = 1, 20
       DO ii = 1, nCell
          ICNTRL_i(ii,kk) = ICNTRLb(kk,ii)
          RCNTRL_i(ii,kk) = RCNTRLb(kk,ii)
       ENDDO
    ENDDO
  END SUBROUTINE Transpose_In

  SUBROUTINE Transpose_Back()
    ! Unpack the solved state back into Yb for comparison/reporting
    INTEGER :: ii, kk
    DO kk = 1, NVAR
       DO ii = 1, nCell
          Yb(kk,ii) = Y_i(ii,kk)
       ENDDO
    ENDDO
  END SUBROUTINE Transpose_Back

  SUBROUTINE Run_Original( Yout )
    ! Reference path through the ORIGINAL integrator (module-global state)
    USE gckpp_Global,     ONLY : C, RCONST, ATOL, RTOL
    USE gckpp_Integrator, ONLY : Integrate
    REAL(dp), INTENT(OUT) :: Yout(NVAR,nCell)
    INTEGER  :: ii, ISTATUS_o(20), IERR_o
    REAL(dp) :: RSTATE_o(20)
    DO ii = 1, nCell
       C(1:NVAR)       = Y0b(:,ii)
       C(NVAR+1:NSPEC) = FIXb(:,ii)
       RCONST          = RCONSTb(:,ii)
       ATOL(1:NVAR)    = ATOLb(:,ii)
       RTOL            = RTOLv
       CALL Integrate( 0.0_dp, Tendb(ii), ICNTRLb(:,ii), RCNTRLb(:,ii),      &
                       ISTATUS_o, RSTATE_o, IERR_o )
       Yout(:,ii) = C(1:NVAR)
    ENDDO
  END SUBROUTINE Run_Original

  !--------------------------------------------------------------------
  ! Multi-cell (per-thread ILP) helpers
  !--------------------------------------------------------------------
  SUBROUTINE Build_Perms()
    ! permId   : cells assigned to threads in list order (RANDOM pairing,
    !            because lists/random.txt is a random order)
    ! permSort : cells assigned in ascending order of the solver step count
    !            measured by the serial reference run, so the NC cells that
    !            share a thread have near-equal cost (COST-SORTED pairing)
    INTEGER :: ii, s, maxs, pos, tmp
    INTEGER, ALLOCATABLE :: cnt(:)
    ALLOCATE( permId(nCell), permSort(nCell) )
    DO ii = 1, nCell
       permId(ii) = ii
    ENDDO
    maxs = MAXVAL( Nstepsb(1:nCell) )
    ALLOCATE( cnt(0:maxs) )
    cnt = 0
    DO ii = 1, nCell
       cnt(Nstepsb(ii)) = cnt(Nstepsb(ii)) + 1
    ENDDO
    pos = 1
    DO s = 0, maxs
       tmp    = cnt(s)
       cnt(s) = pos
       pos    = pos + tmp
    ENDDO
    DO ii = 1, nCell
       s = Nstepsb(ii)
       permSort(cnt(s)) = ii
       cnt(s) = cnt(s) + 1
    ENDDO
    DEALLOCATE( cnt )
    PRINT '(A,I6,A,I6,A,F8.2)', 'Cost sort: steps min= ',                     &
          MINVAL(Nstepsb(1:nCell)), '  max= ', maxs, '  mean= ',              &
          REAL(SUM(Nstepsb(1:nCell)),dp)/nCell
  END SUBROUTINE Build_Perms

  SUBROUTINE Report( tag, t )
    ! Accuracy + throughput of the run that just finished (Yb/IERRb must
    ! already have been copied back from the device).
    CHARACTER(LEN=*), INTENT(IN) :: tag
    REAL(dp),         INTENT(IN) :: t
    INTEGER  :: ii, kk, ne
    REAL(dp) :: mr, dn, rl
    ne = COUNT( IERRb(1:nCell) /= 1 )
    mr = 0.0_dp
    DO ii = 1, nCell
       DO kk = 1, NVAR
          dn = MAX( ABS(Yref(kk,ii)), 1.0e-20_dp )
          rl = ABS( Yb(kk,ii) - Yref(kk,ii) ) / dn
          IF ( rl > mr ) mr = rl
       ENDDO
    ENDDO
    PRINT '(A,A,F10.3,A,F10.1,A,I8,A,ES11.3)', TRIM(tag), '  t= ', t,         &
          ' s  ', nCell/t, ' cells/s  nFail=', ne, '  maxrel=', mr
  END SUBROUTINE Report

  SUBROUTINE Gate_Report( tag, tw )
    ! BIT-EXACT gate: batched multi-cell path vs the ORIGINAL integrator
    CHARACTER(LEN=*), INTENT(IN) :: tag
    REAL(dp),         INTENT(IN) :: tw
    INTEGER  :: ii, kk, nd
    REAL(dp) :: mr, dn, rl
    nd = 0
    mr = 0.0_dp
    DO ii = 1, nCell
       DO kk = 1, NVAR
          IF ( Yb(kk,ii) /= Yref(kk,ii) ) THEN
             nd = nd + 1
             dn = MAX( ABS(Yref(kk,ii)), 1.0e-30_dp )
             rl = ABS( Yb(kk,ii) - Yref(kk,ii) ) / dn
             IF ( rl > mr ) mr = rl
          ENDIF
       ENDDO
    ENDDO
    PRINT '(A,A,F9.2,A,F10.1,A)', TRIM(tag), ' t=', tw, ' s (',               &
          nCell/tw, ' cells/s)'
    PRINT '(A,A,I10,A,I12)', TRIM(tag), ' differing values: ', nd, '  of ',   &
          INT(nCell,8)*NVAR
    PRINT '(A,A,ES12.4,A,I8)', TRIM(tag), ' max relative diff: ', mr,         &
          '   cells with IERR/=1: ', COUNT( IERRb(1:nCell) /= 1 )
  END SUBROUTINE Gate_Report

  SUBROUTINE Gate_M2( perm, tag )
    INTEGER,          INTENT(IN) :: perm(:)
    CHARACTER(LEN=*), INTENT(IN) :: tag
    INTEGER    :: ng, jj
    INTEGER(8) :: a0, a1, ar
    Yb = Y0b
    ng = ( nCell + NC2 - 1 ) / NC2
    CALL SYSTEM_CLOCK( a0, ar )
    !$omp parallel do schedule(dynamic,16)
    DO jj = 1, ng
       CALL Integrate_Cell_M2( jj, nCell, perm, Yb, FIXb, RCONSTb, Tendb,     &
                               ATOLb, RTOLv, ICNTRLb, RCNTRLb,                &
                               Nstepsb, Hexitb, IERRb )
    ENDDO
    !$omp end parallel do
    CALL SYSTEM_CLOCK( a1 )
    CALL Gate_Report( tag, REAL(a1-a0,dp)/REAL(ar,dp) )
  END SUBROUTINE Gate_M2

#ifdef WITH_M4
  SUBROUTINE Gate_M4( perm, tag )
    INTEGER,          INTENT(IN) :: perm(:)
    CHARACTER(LEN=*), INTENT(IN) :: tag
    INTEGER    :: ng, jj
    INTEGER(8) :: a0, a1, ar
    Yb = Y0b
    ng = ( nCell + NC4 - 1 ) / NC4
    CALL SYSTEM_CLOCK( a0, ar )
    !$omp parallel do schedule(dynamic,16)
    DO jj = 1, ng
       CALL Integrate_Cell_M4( jj, nCell, perm, Yb, FIXb, RCONSTb, Tendb,     &
                               ATOLb, RTOLv, ICNTRLb, RCNTRLb,                &
                               Nstepsb, Hexitb, IERRb )
    ENDDO
    !$omp end parallel do
    CALL SYSTEM_CLOCK( a1 )
    CALL Gate_Report( tag, REAL(a1-a0,dp)/REAL(ar,dp) )
  END SUBROUTINE Gate_M4
#endif

END PROGRAM kpp_batch
