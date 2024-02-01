#include "mosfetproblem.h"
#include "debug_flags.h"

void MOSFETProblemCuda::perform_step_3()
{
  const int NSBN  = host_dm -> get_NSBN();
  const int NX    = host_dm -> get_X()   -> get_N();
  const int NZ    = host_dm -> get_Z()   -> get_N();
  const int NW    = host_dm -> get_W()   -> get_N();
  const int NPHI  = host_dm -> get_PHI() -> get_N();

  start_time( _PHASE_STEP );

  for( int s = 0; s < RK_ORDER; ++s )
    {
      start_time( _PHASE_BTE );
      bte( s );
      dens((s+1)%RK_ORDER);
      voldens_totvoldens(NEW);
      stop_time( _PHASE_BTE );
      
      start_time( _PHASE_ITER );
      _stage_ep = (s+1)%RK_ORDER;

      // if( _step == 1 )
      // 	{
      // 	  int iter2ret = iter_2( NEWTON_RAPHSON, ABSOLUTE, host_solvpar->get_TOL_NR_POTENTIAL(), host_solvpar->get_TOL_EIGVALS(), host_solvpar->get_TOL_EIGVECS_IPIM(), host_solvpar->get_TOL_LINSYS_SRJ() );

      // 	  if( iter2ret != 0 )
      // 	    {
      // 	      cerr << " From file " << __FILE__ << ", from function " << __func__ << ", from line " << __LINE__ << " : s = " << s
      // 		   << ", iter2ret = " << iter2ret << endl;
      // 	      cerr << " Newton-Raphson did not succeed in converging. Exiting." << endl;
      // 	      exit( _ERROR_ITER_NR_NOTCONV );
      // 	    }
      // 	}
      // else
      // 	{
      // 	  // int iter2ret = iter_2( NEWTON_RAPHSON, RELATIVE, host_solvpar->get_TOL_NR_POTENTIAL(), host_solvpar->get_TOL_EIGVALS(), host_solvpar->get_TOL_EIGVECS_IPIM(), host_solvpar->get_TOL_LINSYS_SRJ() );
      // 	  int iter2ret = iter_2( NEWTON_RAPHSON, ABSOLUTE, host_solvpar->get_TOL_NR_POTENTIAL(), host_solvpar->get_TOL_EIGVALS(), host_solvpar->get_TOL_EIGVECS_IPIM(), host_solvpar->get_TOL_LINSYS_SRJ() );
      // 	  if( iter2ret != 0 )
      // 	    {
      // 	      int iter2ret_2 = iter_2( GUMMEL, RELATIVE, 1.E-8, host_solvpar->get_TOL_EIGVALS(), host_solvpar->get_TOL_EIGVECS_IPIM(), 1.E-8 );
	      
      // 	      if( iter2ret_2 != 0 )
      // 		{
      // 		  cerr << " From file " << __FILE__ << ", from function " << __func__ << ", from line " << __LINE__ << " : s = " << s
      // 		       << ", iter2ret_2 = " << iter2ret << " --- iter2ret_2 = " << iter2ret_2 << endl;
      // 		  cerr << " Neither Newton-Raphson nor Gummel succeeded in converging. Exiting." << endl;
      // 		  exit( _ERROR_ITER_NR_NOTCONV );
      // 		}
      // 	    }
      // 	}

      FixedPointTest test_type = RELATIVE;
      
      int iter2ret = iter_2( NEWTON_RAPHSON, test_type, host_solvpar->get_TOL_NR_POTENTIAL(), host_solvpar->get_TOL_EIGVALS(), host_solvpar->get_TOL_EIGVECS_IPIM(), host_solvpar->get_TOL_LINSYS_SRJ() );
      if( iter2ret != 0 )
	{
	  int iter2ret_2 = iter_2( GUMMEL, test_type, host_solvpar->get_TOL_NR_POTENTIAL(), host_solvpar->get_TOL_EIGVALS(), host_solvpar->get_TOL_EIGVECS_IPIM(), host_solvpar->get_TOL_LINSYS_SRJ() );
	  
	  if( iter2ret_2 != 0 )
	    {
	      cerr << " From file " << __FILE__ << ", from function " << __func__ << ", from line " << __LINE__ << " : s = " << s
		   << ", iter2ret_2 = " << iter2ret << " --- iter2ret_2 = " << iter2ret_2 << endl;
	      cerr << " Neither Newton-Raphson nor Gummel succeeded in converging. Exiting." << endl;
		  exit( _ERROR_ITER_NR_NOTCONV );
	    }
	}

      
      stop_time( _PHASE_ITER );
    }

  /**************************
   * update time step       *
   **************************/
  _time += _dt;
  update_cfl();

  /**********************************
   * compute macroscopic magnitudes *
   **********************************/
  macro(0);

  stop_time( _PHASE_STEP );
}




/**
   PURPOSE:        

   FILE:           cuda_time_integration.cu

   NAME:           MOSFETProblem::update_cfl

   PARAMETERS:

   RETURN VALUE:   none

   CALLS TO:       none

   CALLED FROM:    MOSFETProblem::perform_step_3        (cuda_time_integration.cu)

   DATA MODIFIED:  

   METHOD:         

   LAST UPDATE:    2023/01/26
*/
void MOSFETProblemCuda::update_cfl()
{
  // recompute maximum step size
  const double DXM1 = host_dm->get_X()->get_delta_m1();
  const double DWM1 = host_dm->get_W()->get_delta_m1();
  const double DPHIM1 = host_dm->get_PHI()->get_delta_m1();
  
  _dtmax = 1./(_max_a1*DXM1+_max_a2*DWM1+_max_a3*DPHIM1);
  
  _dt = min( host_solvpar->get_CFL()*_dtmax, host_solvpar->get_TMAX()/host_rescpar->get_tstar()-_time );

  return;
}




void MOSFETProblemCuda::perform_step_4()
{
  const int NSBN  = host_dm -> get_NSBN();
  const int NX    = host_dm -> get_X()   -> get_N();
  const int NZ    = host_dm -> get_Z()   -> get_N();
  const int NW    = host_dm -> get_W()   -> get_N();
  const int NPHI  = host_dm -> get_PHI() -> get_N();

  start_time( _PHASE_STEP );

  for( int s = 0; s < RK_ORDER; ++s )
    {
      start_time( _PHASE_BTE );
      bte( s );
      dens((s+1)%RK_ORDER);
      voldens_totvoldens(NEW);
      stop_time( _PHASE_BTE );
      
      start_time( _PHASE_ITER );
      _stage_ep = (s+1)%RK_ORDER;


/*
  begin section : timint_02
  description   : See header file.
 */
      // INITIALIZATION OF THE ITERATIVE METHOD
      double *_init_pot     = new double[ NX * NZ ];
      double *_init_voldens = new double[ _NVALLEYS*NSBN*NX*NZ ];
#ifdef __CUDACC__
      checkCudaErrors( cudaMemcpy(_init_pot,      _GPU_pot,        NX*NZ*sizeof(double),                                      cudaMemcpyDeviceToHost) );
      checkCudaErrors( cudaMemcpy(_init_voldens,  _GPU_voldens,    _NVALLEYS*NSBN*NX*NZ*sizeof(double),                      cudaMemcpyDeviceToHost) );
#else
      exit(-2);
#endif

      // NEWTON RESULTS
      double *_Newton_pot     = new double[ NX * NZ ];
      double *_Newton_eps     = new double[ _NVALLEYS*NSBN*NX ];
      double *_Newton_chi     = new double[ _NVALLEYS*NSBN*NX*_SCHROED_MATRIX_SIZE_PAD ];
      double *_Newton_voldens = new double[ _NVALLEYS*NSBN*NX*NZ ];
      int iter2ret_NEWTON = iter_2( NEWTON_RAPHSON, RELATIVE, host_solvpar->get_TOL_NR_POTENTIAL(), host_solvpar->get_TOL_EIGVALS(), host_solvpar->get_TOL_EIGVECS_IPIM(), host_solvpar->get_TOL_LINSYS_SRJ() );
      if( iter2ret_NEWTON != 0 )
	{
	  cerr << " problem with NEWTON." << endl;
	  exit(-1);
	}
      checkCudaErrors( cudaMemcpy(_Newton_pot,  _GPU_pot,   NX*NZ*sizeof(double),                                       cudaMemcpyDeviceToHost) );
      checkCudaErrors( cudaMemcpy(_Newton_eps,  _GPU_eps,   _NVALLEYS*NSBN*NX*sizeof(double),                           cudaMemcpyDeviceToHost) );
      checkCudaErrors( cudaMemcpy(_Newton_chi,  _GPU_chi,   _NVALLEYS*NSBN*NX*_SCHROED_MATRIX_SIZE_PAD*sizeof(double),  cudaMemcpyDeviceToHost) );
      checkCudaErrors( cudaMemcpy(_Newton_voldens,  _GPU_voldens,   _NVALLEYS*NSBN*NX*NZ*sizeof(double),  cudaMemcpyDeviceToHost) );

      // RESTORE INITIAL CONDITIONS
      checkCudaErrors( cudaMemcpy(_GPU_pot,      _init_pot,        NX*NZ*sizeof(double),                                     cudaMemcpyHostToDevice) );
      checkCudaErrors( cudaMemcpy(_GPU_voldens,  _init_voldens,    _NVALLEYS*NSBN*NX*NZ*sizeof(double),                      cudaMemcpyHostToDevice) );
      
      // GUMMEL RESULTS
      double *_Gummel_pot     = new double[ NX * NZ ];
      double *_Gummel_eps     = new double[ _NVALLEYS*NSBN*NX ];
      double *_Gummel_chi     = new double[ _NVALLEYS*NSBN*NX*_SCHROED_MATRIX_SIZE_PAD ];
      double *_Gummel_voldens = new double[ _NVALLEYS*NSBN*NX*NZ ];
      int iter2ret_GUMMEL = iter_2( GUMMEL, RELATIVE, 1.E-12, host_solvpar->get_TOL_EIGVALS(), host_solvpar->get_TOL_EIGVECS_IPIM(), 1.E-12 );
      if( iter2ret_GUMMEL != 0 )
	{
	  cerr << " problem with GUMMEL." << endl;
	  exit(-1);
	}
      checkCudaErrors( cudaMemcpy(_Gummel_pot,  _GPU_pot,   NX*NZ*sizeof(double),                                       cudaMemcpyDeviceToHost) );
      checkCudaErrors( cudaMemcpy(_Gummel_eps,  _GPU_eps,   _NVALLEYS*NSBN*NX*sizeof(double),                           cudaMemcpyDeviceToHost) );
      checkCudaErrors( cudaMemcpy(_Gummel_chi,  _GPU_chi,   _NVALLEYS*NSBN*NX*_SCHROED_MATRIX_SIZE_PAD*sizeof(double),  cudaMemcpyDeviceToHost) );
      checkCudaErrors( cudaMemcpy(_Gummel_voldens,  _GPU_voldens,   _NVALLEYS*NSBN*NX*NZ*sizeof(double),  cudaMemcpyDeviceToHost) );

      // COMPARE RESULTS
      cerr << " comparing Gummel and Newton-Raphson...";
      double linf_pot = 0;
      for(int i=0; i<NX; ++i)
	{
	  for(int j=0; j<NZ; ++j)
	    {
	      linf_pot = max( linf_pot,  fabs( _Gummel_pot[ indx_i_j(i,j)] - _Newton_pot[ indx_i_j(i,j)] ) );
	    }
	}
      
      double linf_eps = 0;
      for(int nu=0; nu<_NVALLEYS; ++nu)
	{
	  for(int p=0; p<NSBN; ++p)
	    {
	      for(int i=0; i<NX; ++i)
		{
		  linf_eps = max( linf_eps,  fabs( _Gummel_eps[ indx_i_nu_p(nu,p,i)] - _Newton_eps[ indx_i_nu_p(nu,p,i)] ) );
		}
	    }
	}

      double linf_chi = 0;
      for(int nu=0; nu<_NVALLEYS; ++nu)
	{
	  for(int p=0; p<NSBN; ++p)
	    {
	      for(int i=0; i<NX; ++i)
		{
		  for(int j=1; j<NZ-1; ++j)
		    {
		      linf_chi = max( linf_chi,  fabs( _Gummel_chi[ indx_i_nu_p_jschr  (nu,p,i,j) ]*_Gummel_chi[ indx_i_nu_p_jschr  (nu,p,i,j) ] - _Newton_chi[ indx_i_nu_p_jschr  (nu,p,i,j) ]*_Newton_chi[ indx_i_nu_p_jschr  (nu,p,i,j) ] ) );
		    }
		}
	    }
	}

      double linf_voldens = 0;
      for(int nu=0; nu<_NVALLEYS; ++nu)
	{
	  for(int p=0; p<NSBN; ++p)
	    {
	      for(int i=0; i<NX; ++i)
		{
		  for(int j=1; j<NZ-1; ++j)
		    {
		      linf_voldens = max( linf_voldens,  fabs( _Gummel_voldens[ indx_i_j_nu_p(nu,p,i,j) ] - _Newton_voldens[ indx_i_j_nu_p(nu,p,i,j) ] ) );
		    }
		}
	    }
	}

      cerr << "linf_pot = " << linf_pot << endl;
      cerr << "linf_eps = " << linf_eps << endl;
      cerr << "linf_chi = " << linf_chi << endl;
      cerr << "linf_voldens = " << linf_voldens << endl;
      
      // FREE MEMORY
      delete [] _Newton_voldens;
      delete [] _Newton_chi;
      delete [] _Newton_eps;
      delete [] _Newton_pot;
      delete [] _Gummel_voldens;
      delete [] _Gummel_chi;
      delete [] _Gummel_eps;
      delete [] _Gummel_pot;
      delete [] _init_voldens;
      delete [] _init_pot;
/*
  end section : timint_02
 */



      stop_time( _PHASE_ITER );
    }

  /**************************
   * update time step       *
   **************************/
  _time += _dt;
  update_cfl();

  /**********************************
   * compute macroscopic magnitudes *
   **********************************/
  macro(0);

  stop_time( _PHASE_STEP );
}



