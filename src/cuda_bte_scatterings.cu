#include "mosfetproblem.h"
#include "debug_flags.h"

/**
   PURPOSE:        

   FILE:           cuda_bte_scatterings.cu

   NAME:           MOSFETProblem::scatterings

   PARAMETERS:

   RETURN VALUE:   none

   CALLS TO:       MOSFETProblem::start_time                    (cuda_comptime.h - declared inline)
		   MOSFETProblem::stop_time                     (cuda_comptime.h - declared inline)
		   MOSFETProblem::compute_Wm1                   (cuda_bte_scatterings_phonons.cu)
		   MOSFETProblem::phonons_gain                  (cuda_bte_scatterings_phonons.cu)
		   MOSFETProblem::phonons_loss                  (cuda_bte_scatterings_phonons.cu)
		   MOSFETProblem::compute_I_SR                  (cuda_bte_scatterings_roughness.cu)
		   MOSFETProblem::roughness_gain                (cuda_bte_scatterings_roughness.cu)
		   MOSFETProblem::roughness_loss                (cuda_bte_scatterings_roughness.cu)

   CALLED FROM:    MOSFETProblem::perform_step_2                (cuda_time_integration.cu)

   DATA MODIFIED:  

   METHOD:         

   LAST UPDATE:    2022 February 16
*/
void MOSFETProblemCuda::scatterings( const int stage )
{
  start_time( _PHASE_BTE_SCATTERINGS );

  /**************************************
   *      COPY CONSTANT DATA TO GPU     *
   **************************************/
#ifdef __CUDACC__
  checkCudaErrors( cudaMemcpyToSymbol(_GPU_sqrtmassXY,  _sqrtmassXY,  _NVALLEYS*sizeof(double),             0, cudaMemcpyHostToDevice) );
  checkCudaErrors( cudaMemcpyToSymbol(_GPU_mass,        _mass,        9*sizeof(double),                     0, cudaMemcpyHostToDevice) );
  checkCudaErrors( cudaMemcpyToSymbol(_GPU_kane,        _kane,        _NVALLEYS*sizeof(double),             0, cudaMemcpyHostToDevice) );
  checkCudaErrors( cudaMemcpyToSymbol(_GPU_occupations, _occupations, 8*_NVALLEYS*_NVALLEYS*sizeof(double), 0, cudaMemcpyHostToDevice) );
  checkCudaErrors( cudaMemcpyToSymbol(_GPU_cscatt,      _cscatt,      8*sizeof(double),                     0, cudaMemcpyHostToDevice) );
  checkCudaErrors( cudaMemcpyToSymbol(_GPU_omega,       _omega,       8*sizeof(double),                     0, cudaMemcpyHostToDevice) );
#endif

  /***********************************
   *              PHONONS            *
   ***********************************/
  if( host_solvpar->get_PHONONS() == true )
    {
      start_time( _PHASE_BTE_SCATTERINGS_PHONONS );

      compute_Wm1(); // preliminary computations
      phonons_gain ( stage );
      phonons_loss ( stage );

      stop_time( _PHASE_BTE_SCATTERINGS_PHONONS );
    }

  /***********************************
   *             ROUGHNESS           *
   ***********************************/
  if( host_solvpar->get_ROUGHNESS() == true )
    {
      start_time( _PHASE_BTE_SCATTERINGS_ROUGHNESS );

      compute_I_SR( stage ); // preliminary computations
      roughness_gain ( stage );
      roughness_loss ( stage );

// #define _CHECK_ROUGHNESS_GAIN_LOSS_BALANCE
#ifdef _CHECK_ROUGHNESS_GAIN_LOSS_BALANCE
#warning   _CHECK_ROUGHNESS_GAIN_LOSS_BALANCE is activated
      const int NSBN     = host_dm -> get_NSBN();
      const int NX       = host_dm -> get_X()   -> get_N();
      const int NZ       = host_dm -> get_Z()   -> get_N();
      const int NW       = host_dm -> get_W()   -> get_N();
      const int NPHI     = host_dm -> get_PHI() -> get_N();

#ifdef __CUDACC__
      checkCudaErrors( cudaMemcpy(_test_gain, _GPU_test_gain, _NVALLEYS*NSBN*NX*NW*NPHI*sizeof(double), cudaMemcpyDeviceToHost) );
      checkCudaErrors( cudaMemcpy(_test_loss, _GPU_test_loss, _NVALLEYS*NSBN*NX*NW*NPHI*sizeof(double), cudaMemcpyDeviceToHost) );
#endif
      double sum_gain = 0, sum_loss = 0;
      for(int nu=0; nu<_NVALLEYS; ++nu)
	for(int p=0; p<NSBN; ++p)
	  {
	    for(int i=0; i<NX; ++i)
	      for(int l=0; l<NW; ++l)
		for(int m=0; m<NSBN; ++m)
		  {
		    sum_gain += test_gain(nu,p,i,l,m);
		    sum_loss += test_loss(nu,p,i,l,m);
		  }
	  }
      cerr << " --- sum_gain = " << sum_gain
	   << " --- sum_loss = " << sum_loss
	   << " --- difference = " << fabs(sum_gain - sum_loss)
	   << endl;
      // for(int nu=0; nu<_NVALLEYS; ++nu)
      // 	for(int p=0; p<NSBN; ++p)
      // 	  {
      // 	    double sum_gain = 0, sum_loss = 0;
      // 	    for(int i=0; i<NX; ++i)
      // 	      for(int l=0; l<NW; ++l)
      // 		for(int m=0; m<NSBN; ++m)
      // 		  {
      // 		    sum_gain += test_gain(nu,p,i,l,m);
      // 		    sum_loss += test_loss(nu,p,i,l,m);
      // 		  }
      // 	    cerr << " (nu,p) = (" << nu << "," << p << ")"
      // 		 << " --- sum_gain = " << sum_gain
      // 		 << " --- sum_loss = " << sum_loss
      // 		 << " --- difference = " << fabs(sum_gain - sum_loss)
      // 		 << endl;
      // 	  }
#endif

      stop_time( _PHASE_BTE_SCATTERINGS_ROUGHNESS );
}

  stop_time( _PHASE_BTE_SCATTERINGS );

  return;
}





