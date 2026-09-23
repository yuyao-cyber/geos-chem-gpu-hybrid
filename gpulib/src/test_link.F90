!------------------------------------------------------------------------------
! test_link.F90 -- gfortran-built driver that links libgcchemgpu.so
! (nvfortran/OpenACC) through the same BIND(C) INTERFACE the model uses.
! Proves the mixed-compiler link + GPU runtime in isolation from the model.
!------------------------------------------------------------------------------
PROGRAM test_link
  USE ISO_C_BINDING
  IMPLICIT NONE

  INTERFACE
     SUBROUTINE gc_gpu_info( nDev, devName, nameLen ) BIND(C, name='gc_gpu_info')
       IMPORT :: c_int, c_char
       INTEGER(c_int),         INTENT(OUT) :: nDev
       INTEGER(c_int),         VALUE       :: nameLen
       CHARACTER(KIND=c_char), INTENT(OUT) :: devName(*)
     END SUBROUTINE gc_gpu_info
     SUBROUTINE gc_gpu_selftest( rc, nsteps, tTotal ) BIND(C, name='gc_gpu_selftest')
       IMPORT :: c_int, c_double
       INTEGER(c_int), INTENT(OUT) :: rc, nsteps
       REAL(c_double), INTENT(OUT) :: tTotal
     END SUBROUTINE gc_gpu_selftest
  END INTERFACE

  INTEGER(c_int)         :: nDev, rc, nsteps, i
  REAL(c_double)         :: t
  CHARACTER(KIND=c_char) :: devName(256)
  CHARACTER(LEN=256)     :: name

  CALL gc_gpu_info( nDev, devName, 256_c_int )
  name = ''
  DO i = 1, 255
     IF ( devName(i) == C_NULL_CHAR ) EXIT
     name(i:i) = devName(i)
  ENDDO
  WRITE(*,'(a,i0,a,a)') 'gc_gpu_info: nDev=', nDev, ' name=', TRIM(name)

  CALL gc_gpu_selftest( rc, nsteps, t )
  WRITE(*,'(a,i0,a,i0,a,f8.4)') 'gc_gpu_selftest: rc=', rc, ' nsteps=', nsteps, ' t=', t
  IF ( rc == 1 .and. nDev > 0 ) THEN
     WRITE(*,'(a)') 'GPU selftest OK'
  ELSE
     WRITE(*,'(a)') 'GPU selftest FAILED'
  ENDIF
END PROGRAM test_link

!------------------------------------------------------------------------------
! No-op OpenACC profiling-registration hook.  The NVIDIA OpenACC runtime
! (libacchost) references acc_register_library WEAKLY and calls it at init if
! any loaded object defines it; gfortran's libgomp exports a stub of that name
! that aborts with "libgomp: TODO".  Defining it in the executable (first in
! the dynamic-linker global scope) makes the weak reference bind here.
!------------------------------------------------------------------------------
SUBROUTINE acc_register_library( reg, unreg, lookup )                         &
           BIND(C, name='acc_register_library')
  USE ISO_C_BINDING, ONLY : C_PTR
  TYPE(C_PTR), VALUE :: reg, unreg, lookup
END SUBROUTINE acc_register_library
