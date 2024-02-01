#include "mosfetproblem.h"
#include "debug_flags.h"

/**
   PURPOSE:        This is the top level function in class MOSFETProblem in order
                   to perform the simulation.

   FILE:           cuda_solve.cu

   NAME:           MOSFETProblem::solve_2

   PARAMETERS:

   RETURN VALUE:   none

   CALLS TO:       MOSFETProblem::compute_eta         (cuda_testing.h - declared inline)
                   MOSFETProblem::perform_step_2      (cuda_time_integration.cu)
		   MOSFETProblem::stopcondition       (cuda_solve.cu)
		   MOSFETProblem::store_data          (cuda_filestorage.cu)
		   MOSFETProblem::show_eta            (cuda_testing.cu)
		   MOSFETProblem::analysis_comptime   (cuda_comptime.cu)

   CALLED FROM:    main

   DATA MODIFIED:  all data except constants

   METHOD:         

   LAST UPDATE:    2022 February 15
*/
void MOSFETProblemCuda::solve_2()
{
  const int FREQ  = host_solvpar->get_FREQ();

  if( FREQ > 0 )
    store_data();

  ++_step;

  _time_init = gettime();
  _ep = POTENTIAL;

  for( ; ; ++_step )
    {
      compute_eta();
      perform_step_3();
      // perform_step_4();  // FOR DEBUGGING PURPOSES
      if( stopcondition() ) break;

      if( FREQ > 0 )
	store_data();

      ostringstream message;
      message << "(after) STEP phase --- _step = " << _step << endl;
      show_eta(message.str());
    }
  
  if( FREQ > 0 )
    store_data(true);

#ifdef _COMPTIMES
  analysis_comptime();
#endif
  show_eta("Finishing...");
  cerr << endl << endl
       << " ************************************************************************************************************************************************************************ " << endl
       << " ./generate_films_and_graphs.sh        --- to generate graphically all the results concerning the simulation, to be stored in films_and_graphs/ "                           << endl
       << " ./generate_comptimes_analysis.sh      --- to generate piecharts and histograms concerning the performances of the GPU/CPU execution, to be stored in comptimes_analysis/ " << endl
       << " ************************************************************************************************************************************************************************ " << endl << endl;
}


bool MOSFETProblemCuda::stopcondition()
{
  const int STEPMAX = host_solvpar->get_STEPMAX();
  const double TMAX = host_solvpar->get_TMAX();

  return( _time+1.e-16 >= TMAX/host_rescpar->get_tstar() || _step >= STEPMAX );
}
