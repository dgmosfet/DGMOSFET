#include "mosfetproblem.h"
#include "debug_flags.h"


/**
   PURPOSE:        

   FILE:           cuda_bte.cu

   NAME:           MOSFETProblem::bte

   PARAMETERS:

   RETURN VALUE:   none

   CALLS TO:       MOSFETProblem::set_fluxes_a3         (cuda_bte_rk.cu)
                   MOSFETProblem::set_fluxes_a2         (cuda_bte_rk.cu)
                   MOSFETProblem::set_rhs_to_zero       (cuda_bte_rk.cu)
		   MOSFETProblem::scatterings           (cuda_bte_scatterings.cu)
		   MOSFETProblem::weno_X                (cuda_bte_weno.cu)
		   MOSFETProblem::weno_W                (cuda_bte_weno.cu)
		   MOSFETProblem::weno_PHI              (cuda_bte_weno.cu)
                   MOSFETProblem::perform_RK            (cuda_bte_rk.cu)

   CALLED FROM:    MOSFETProblem::perform_step_2        (cuda_solve.cu)

   DATA MODIFIED:  

   METHOD
   ======
   The function performs the following steps
   in order to perform the advection (BTE) phase:
   (1) set_fluxes_a3();
   (2) set_fluxes_a2();
   (3) set_rhs_to_zero();
   (4) scatterings( s );
   (5) WENO_X(s);
   (6) WENO_W(s);
   (7) WENO_PHI(s);
   (8) perform_RK ( _dt, s );
   
   LAST MODIFIED: 2023/06/05


   DATA FLOW
   =========
   From a macroscopic point of view, this method works like this
   
      s [parameter], pdf(,,,,,s), eps(,,), deps_dx(,,), chi(,,,) ---> pdf(,,,,,(s+1) mod RK_ORDER)
**/

void MOSFETProblemCuda::bte( const int s )
{
  set_fluxes_a3();
  set_fluxes_a2();
  set_rhs_to_zero();
  scatterings( s );
  WENO_X(s);
  WENO_W(s);
  WENO_PHI(s);
  perform_RK ( _dt, s );

  return;
}
