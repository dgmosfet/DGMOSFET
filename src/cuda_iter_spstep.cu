#include "mosfetproblem.h"
#include "debug_flags.h"

/**
   PURPOSE:        

   FILE:           cuda_iter_spstep.cu

   NAME:           MOSFETProblem::spstep_CPU_LIS

   PARAMETERS:

   RETURN VALUE:   none

   CALLS TO:       CPU_frechet                               (cuda_iter_spstep_frechet.cu)
                   CPU_constr_linsys_LIS                     (cuda_iter_spstep_constrlinsys.cu)
                   CPU_solve_linsys_LIS                      (cuda_iter_spstep_solvelinsys.cu)

   CALLED FROM:    MOSFETProblem::thermequi_on_cuda          (cuda_iter.cu)
                   MOSFETProblem::initcond_on_cuda           (cuda_iter.cu)

   DATA MODIFIED:  

   METHOD:         

   LAST UPDATE:    2022 February 16
*/
void MOSFETProblemCuda::spstep_CPU_LIS( const FixedPointType fpt )
{
  /**********************************************
   *     COMPUTE THE NEWTON-RAPHSON FRECHET     *
   **********************************************/
  if( fpt == NEWTON_RAPHSON )
    {
      CPU_compute_frechet();
    }

  /************************************************************************************
   *                           construct the linear system                            *
   ************************************************************************************/
  CPU_constr_linsys_LIS( fpt );

  // // delete from here
  // lis_output_matrix( _matrix_2d_LIS, LIS_FMT_MM, "linsys.dat" );
  // lis_output_vector( _rhs_LIS, LIS_FMT_MM, "rhs.dat" );
  // // to here

  /************************************************************************************
   *                       call the solver and copy the result                        *
   ************************************************************************************/
  CPU_solve_linsys_LIS();

  return;
}

