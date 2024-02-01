#include "mosfetproblem.h"
#include "debug_flags.h"

#ifdef __CUDACC__
#define CUDA_SAFE_CALL( call ) {                                         \
 cudaError_t err = call;                                                 \
 if( cudaSuccess != err ) {                                              \
   fprintf(stderr,"CUDA: error occurred in cuda routine. Exiting...\n"); \
   exit(err);                                                            \
 } }
#endif

/**
   PURPOSE:        This function implements the Newton-Raphson iterative method for
                   the computation of the eigenstates. 
		   Macroscopically, it receives the surface densities
		   (plus the former potential and the total volume density,
		   but just as a good initialization)
		   and returns the computed eigenvalues (energy levels), 
		   eigenvectors (the wave functions) and potential,
		   plus some derived quantities (total volume densities and deps_dx).
		   Input: surfdens (pot, totvoldens)
		   Output: pot, eps, chi (voldens, totvoldens, deps_dx)

		   Brief description of the method. 
		   The method works by iteratively updating the potential, 
		   then keeping consistency with the eigenvalues/eigenvectors and the volume densities,
		   until some convergence parameters are fulfilled.
		   As convergence parameters, the L2 and Linf differences
		   between the potential at the two last stages
		   and the total volume density at the two last stages.

   FILE:           cuda_iter.cu

   NAME:           MOSFETProblem::iter_2

   PARAMETERS:

   RETURN VALUE:   integer, with the following meaning
                   0    : exit with no errors
		   1    : non converence of the method, but without any 'nan' in the test magnitudes
		   +2   : if l2V is nan, in case of non convergence
		   +4   : if l2N is nan, in case of non convergence
		   +8   : if linfV is nan, in case of non convergence
		   +16  : if linfN is nan, in case of non convergence
		   +32  : if l2V is inf, in case of non convergence
		   +64  : if l2N is inf, in case of non convergence
		   +128 : if linfV is inf, in case of non convergence
		   +256 : if linfN is inf, in case of non convergence

   CALLS TO:       eigenstates                   (cuda_iter_eigenstates.cu)
                   voldens_totvoldens            (cuda_reductions.cu)
		               update_potential              (cuda_iter.cu)
		               keep_eigenstates_consistent   (cuda_iter.cu)
		               test_conv_POTENTIAL           (cuda_iter.cu)
		               reverse_into_OLD              (cuda_iter.cu)
		               compute_deps_dx               (cuda_iter.cu)
                   start_time                    (cuda_comptime.h - declared inline)
		               stop_time                     (cuda_comptime.h - declared inline)
                   show_eta                      (cuda_testing.h - declared inline)
                   
   CALLED FROM:    perform_step_2                (cuda_time_integration.cu)

   DATA MODIFIED:  

   METHOD:         

   LAST UPDATE:    2022 February 16
*/


// #define   _CHECK_ITER_2_INPUT_OUTPUT
int MOSFETProblemCuda::iter_2( const FixedPointType fpt, const FixedPointTest test_type, const double fp_tolpar, const double eigvals_tolpar, const double eigvecs_tolpar, const double linsys_tolpar )
{
#ifdef   _CHECK_ITER_2_INPUT_OUTPUT
#warning   _CHECK_ITER_2_INPUT_OUTPUT is activated
  check_iter_2_input();
#endif

  const int NSBN  = host_dm -> get_NSBN();
  const int NX    = host_dm -> get_X()   -> get_N();
  const int NZ    = host_dm -> get_Z()   -> get_N();
  const int NW    = host_dm -> get_W()   -> get_N();
  const int NPHI  = host_dm -> get_PHI() -> get_N();

  _ep = POTENTIAL;  /* the kind of eigenproblem being solved (not the profile INITCONDITION, *
		     * not the thermal equilibrium THERMEQUIL).                              */

  ostringstream message;            // for debugging purposes, show
  message << "ITER phase " << _ep;  // a message if the
#ifdef _SHOW_ETA                    // _SHOW_ETA flag is active
  show_eta(message.str());
#endif

#ifdef   _CHECK_ITER_2_INPUT_OUTPUT
  cerr << " ---------- iter_2 initialization ---------- " << endl;
#endif
  reverse_into_OLD();               
  eigenstates(OLD, eigvals_tolpar, eigvecs_tolpar);
  voldens_totvoldens(OLD);
  bool stopcondition = false;
  iter_counter = 1;
  // const int itermax = 50;
  const int itermax = host_solvpar -> get_ITERMAX_NR_POTENTIAL();
  double vect_l2V[itermax];
  double vect_l2N[itermax];
  double vect_linfV[itermax];
  double vect_linfN[itermax];

  start_time(_PHASE_ITER_SPSTEP);
#ifdef   _CHECK_ITER_2_INPUT_OUTPUT
  cerr << " ---------- iter_2 enter iterative phase ---------- " << endl;
#endif
  while( stopcondition == false && iter_counter <= itermax )
    {
      // delete from here
      if( fpt == NEWTON_RAPHSON )
	{
	  cerr << "-";
	}
      if( fpt == GUMMEL )
	{
	  cerr << "+";
	}
      // to here
      
#ifdef   _CHECK_ITER_2_INPUT_OUTPUT
      cerr << " ---------- iter_2 iter_counter = " << iter_counter << " ---------- " << endl;
#endif
      int success_update_potential = update_potential( fpt, linsys_tolpar );

      if( success_update_potential != 0 )
	return success_update_potential;
      
      eigenstates(NEW, eigvals_tolpar, eigvecs_tolpar);
      voldens_totvoldens(NEW);

      if( test_type == RELATIVE )
	{
	  stopcondition = test_conv_POTENTIAL(fp_tolpar);
	}
      else if( test_type == ABSOLUTE )
	{
	  stopcondition = test_conv_POTENTIAL_ABSOLUTE(fp_tolpar);
	}
      else
	{
	  cerr << " ERROR." << endl;
	  exit(-9);
	}

      vect_l2V[iter_counter-1] = _l2V;
      vect_l2N[iter_counter-1] = _l2N;
      vect_linfV[iter_counter-1] = _linfV;
      vect_linfN[iter_counter-1] = _linfN;
      /* 
	 begin section 'cpn26' (random name, haha)
	 modified : 2023/01/26
	 who      : Francesco Vecil

	 Here, the presence of nan in the L2 and Linf norms of the potential
	 and the volume density is checked. In case, function 'iter_2' 
	 exits by returning an error code to signal that.
	 This section is meant to be PERMANENT, it is not particularly
	 costly and introduces a useful safety measure.
      */
      int errcode = 0;
      if( isnan(_l2V) == true )
	{
	  errcode += 2;
	}
      if( isnan(_l2N) == true )
	{
	  errcode += 4;
	}
      if( isnan(_linfV) == true )
	{
	  errcode += 8;
	}
      if( isnan(_linfN) == true )
	{
	  errcode += 16;
	}
      if( isinf(_l2V) == true )
	{
	  errcode += 32;
	}
      if( isinf(_l2N) == true )
	{
	  errcode += 64;
	}
      if( isinf(_linfV) == true )
	{
	  errcode += 128;
	}
      if( isinf(_linfN) == true )
	{
	  errcode += 256;
	}
      if( errcode > 0 )
	{
	  cerr << "function " << __func__ << ", line " << __LINE__ << ", file '" << __FILE__ << "'" << endl;
	  for(int i=0; i<iter_counter; ++i)
	    {
	      cerr << " vect_L2V["   << setw(3) << i << "] = " << setw(16) << vect_l2V[i]   << setw(16) << " " << " vect_L2N["   << setw(3) << i << "] = " << setw(16) << vect_l2N[i]
		   << setw(16)
		   << " vect_LinfV[" << setw(3) << i << "] = " << setw(16) << vect_linfV[i] << setw(16) << " " << " vect_LinfN[" << setw(3) << i << "] = " << setw(16) << vect_linfN[i]
		   << endl;
	    }
	  return (errcode + 1);
	}
      /* 
	 end section 'cpn26'
      */
      reverse_into_OLD();
      ++iter_counter;
    }

  if( iter_counter > itermax )
    {
      cerr << "function " << __func__ << ", line " << __LINE__ << ", file '" << __FILE__ << "'" << endl;
      for(int i=0; i<itermax; ++i)
	{
	  cerr << " vect_L2V["   << setw(3) << i << "] = " << setw(16) << vect_l2V[i]   << setw(16) << " " << " vect_L2N["   << setw(3) << i << "] = " << setw(16) << vect_l2N[i]
	       << setw(16)
	       << " vect_LinfV[" << setw(3) << i << "] = " << setw(16) << vect_linfV[i] << setw(16) << " " << " vect_LinfN[" << setw(3) << i << "] = " << setw(16) << vect_linfN[i]
	       << endl;
	}
      return 1;
    }

  stop_time(_PHASE_ITER_SPSTEP);

  compute_deps_dx();
  compute_max_deps_dx();
  _max_a2 = _max_deps_dx*_max_a1;
  _max_a3 = _max_deps_dx*_max_a3const;

#ifdef _SHOW_ETA
  cerr << "[ok] (function " << __func__ << ", line " << __LINE__ << " (" << iter_counter-1 << " iterations)" << endl;
#endif

#ifdef   _CHECK_ITER_2_INPUT_OUTPUT
  check_iter_2_output();
#endif

  return 0;
}


/**
   PURPOSE:        

   FILE:           cuda_iter.cu

   NAME:           MOSFETProblem::update_potential

   PARAMETERS:

   RETURN VALUE:   none

   CALLS TO:       compute_frechet        (cuda_iter_spstep_frechet.cu)
                   constr_linsys          (cuda_iter_spstep_constrlinsys.cu)
                   solve_linsys           (cuda_iter_spstep_solvelinsys.cu)

   CALLED FROM:    MOSFETProblem::iter_2  (cuda_iter.cu)

   DATA MODIFIED:  

   METHOD:         

   LAST UPDATE:    2022 February 16
*/

// #define   _CHECK_UPDATE_POTENTIAL_INPUT_OUTPUT
int MOSFETProblemCuda::update_potential( const FixedPointType fpt, const double linsys_tolpar )
{
#ifdef   _CHECK_UPDATE_POTENTIAL_INPUT_OUTPUT
#warning   _CHECK_UPDATE_POTENTIAL_INPUT_OUTPUT is activated
  check_update_potential_input();
#endif

  if( fpt == NEWTON_RAPHSON )
    {
      compute_frechet();
    }
  constr_linsys( fpt );
  int success = solve_linsys(linsys_tolpar);

#ifdef   _CHECK_UPDATE_POTENTIAL_INPUT_OUTPUT
  check_update_potential_output();
#endif

  return success;
}


/**
   PURPOSE:        

   FILE:           cuda_iter.cu

   NAME:           MOSFETProblem::thermequil_on_cuda

   PARAMETERS:

   RETURN VALUE:   none

   CALLS TO:       show_eta                            (cuda_testing.cu)
                   spstep_CPU_LIS                      (cuda_iter_spstep.cu)
                   CPU_eigenstates                     (cuda_iter_eigenstates.cu)
                   CPU_compute_surfdens                (cuda_reductions.cu)
                   CPU_voldens_totvoldens              (cuda_reductions.cu)
		   CPU_test_conv_INITCOND_THERMEQUIL   (cuda_iter.cu)

   CALLED FROM:    MOSFETProblem::cuda_config          (cuda_config.cu)

   DATA MODIFIED:  

   METHOD:         

   LAST UPDATE:    2022 February 16
*/
void MOSFETProblemCuda::thermequil_on_cuda()
{
  enterfunc;

  const int NSBN  = host_dm -> get_NSBN();
  const int NX    = host_dm -> get_X()   -> get_N();
  const int NZ    = host_dm -> get_Z()   -> get_N();

  _ep = THERMEQUIL;
#ifdef _SHOW_ETA
  ostringstream message;
  message << "on Cuda code --- ITER phase " << _ep;
  show_eta(message.str());
#endif
  for(int i=0; i<NX; ++i)
    for(int j=0; j<NZ; ++j)
      pot_OLD(i,j) = _pot_b[j];

  CPU_eigenstates(OLD, 1.E-12, 1.E-12);
  CPU_compute_surfdens();
  CPU_voldens_totvoldens(OLD);

  bool stopcondition = false;
  int iter_counter = 1;
  const int itermax = 1000;
  for(; stopcondition == false; ++iter_counter)
    {
      spstep_CPU_LIS( NEWTON_RAPHSON );
#ifdef _SHOW_ETA
      cerr << ".";
#endif

      CPU_eigenstates(NEW, 1.E-12, 1.E-12);
      CPU_compute_surfdens();
      CPU_voldens_totvoldens(NEW);

      stopcondition = CPU_test_conv_INITCOND_THERMEQUIL();
      if( iter_counter > itermax )
	{
	  throw error_ITER_NR_NOTCONV_THERMEQUIL();
	}
      memcpy( _pot_OLD, _pot, NX*NZ*sizeof(double) );
      memcpy( _totvoldens_OLD, _totvoldens, NX*NZ*sizeof(double) );
    }
  for( int i=0; i<NX; ++i )
    {
      _totsurfdens_eq[i] = 0;		
      for(int nu=0; nu<_NVALLEYS; ++nu)
	for(int p=0; p<NSBN; ++p)
	  {
	    surfdens_eq(nu,p,i) = surfdens(nu,p,i);
	    _totsurfdens_eq[i] += 2.*surfdens_eq(nu,p,i);
	  }
    }
#pragma omp parallel for
  for(int index = 0; index < _NVALLEYS*NSBN; ++index)
    {
      int nu = index/NSBN;
      int p  = index - nu*NSBN;
      const double DXM1 = host_dm->get_X()->get_delta_m1();

      deps_dx(nu,p,0) = (eps(nu,p,1) - eps(nu,p,0))*DXM1;
      for(int i=1; i<=NX-2; ++i)
	deps_dx(nu,p,i) = (eps(nu,p,i+1) - eps(nu,p,i-1))*0.5*DXM1;
      deps_dx(nu,p,NX-1) = (eps(nu,p,NX-1) - eps(nu,p,NX-2))*DXM1;
    }
  _max_deps_dx = 0;
  for(int i=0; i<_NVALLEYS*NSBN*NX; ++i)
    if(fabs(_deps_dx[i]) > _max_deps_dx)
      _max_deps_dx = fabs(_deps_dx[i]);
  _max_a2 = _max_deps_dx*_max_a1;
  _max_a3 = _max_deps_dx*_max_a3const;
#ifdef __CUDACC__
  checkCudaErrors( cudaMemcpy(_GPU_surfdens,       _surfdens,       _NVALLEYS*NSBN*NX*sizeof(double),                          cudaMemcpyHostToDevice) );
  checkCudaErrors( cudaMemcpy(_GPU_surfdens_eq,    _surfdens_eq,    _NVALLEYS*NSBN*NX*sizeof(double),                          cudaMemcpyHostToDevice) );
  checkCudaErrors( cudaMemcpy(_GPU_pot,            _pot,            NX*NZ*sizeof(double),                                      cudaMemcpyHostToDevice) );
  checkCudaErrors( cudaMemcpy(_GPU_eps,            _eps,            _NVALLEYS*NSBN*NX*sizeof(double),                          cudaMemcpyHostToDevice) );
  checkCudaErrors( cudaMemcpy(_GPU_deps_dx,        _deps_dx,        _NVALLEYS*NSBN*NX*sizeof(double),                          cudaMemcpyHostToDevice) );
  checkCudaErrors( cudaMemcpy(_GPU_chi,            _chi,            _NVALLEYS*NSBN*NX*_SCHROED_MATRIX_SIZE_PAD*sizeof(double), cudaMemcpyHostToDevice) );
  checkCudaErrors( cudaMemcpy(_GPU_pot_OLD,        _pot_OLD,        NX*NZ*sizeof(double),                                      cudaMemcpyHostToDevice) );
  checkCudaErrors( cudaMemcpy(_GPU_totvoldens_OLD, _totvoldens_OLD, NX*NZ*sizeof(double),                                      cudaMemcpyHostToDevice) );
#endif

  exitfunc;
}



/**
   PURPOSE:        

   FILE:           cuda_iter.cu

   NAME:           MOSFETProblem::thermequil_on_cuda

   PARAMETERS:

   RETURN VALUE:   none

   CALLS TO:       MOSFETPrroblem::show_eta                            (cuda_testing.h --- declared inline)
                   MOSFETPrroblem::spstep_CPU_LIS                      (cuda_iter_spstep.cu)
                   MOSFETPrroblem::CPU_eigenstates                     (cuda_iter_eigenstates.cu)
                   MOSFETPrroblem::CPU_compute_surfdens                (cuda_reductions.cu)
                   MOSFETPrroblem::CPU_voldens_totvoldens              (cuda_reductions.cu)
		   MOSFETPrroblem::CPU_test_conv_INITCOND_THERMEQUIL   (cuda_iter.cu)
		   MOSFETPrroblem::config_rhs                          (cuda_config.cu)

   CALLED FROM:    MOSFETProblem::cuda_config                          (cuda_config.cu)

   DATA MODIFIED:  

   METHOD:         

   LAST UPDATE:    2022 February 16
*/
void MOSFETProblemCuda::initcond_on_cuda()
{
  enterfunc;
  
  const int NSBN  = host_dm -> get_NSBN();
  const int NX    = host_dm -> get_X()   -> get_N();
  const int NZ    = host_dm -> get_Z()   -> get_N();
  const double VBIAS = host_physdesc -> get_VBIAS();

  // INITCOND on Cuda
  _ep = INITCOND;
#ifdef _SHOW_ETA
  ostringstream message;
  message << "on Cuda code --- ITER phase " << _ep;
  show_eta(message.str());
#endif
  // INITIAL GUESS FOR THE POTENTIAL
  for(int i=0; i<NX; ++i)
    for(int j=0; j<NZ; ++j)
      pot_OLD(i,j) = 1.0;
  CPU_eigenstates(OLD, 1.E-12, 1.E-12);
  CPU_compute_surfdens();
  CPU_voldens_totvoldens(OLD);
  bool stopcondition = false;
  int iter_counter = 1;
  const int itermax = 1000;
  for(; stopcondition == false; ++iter_counter)
    {
      spstep_CPU_LIS( NEWTON_RAPHSON );
#ifdef _SHOW_ETA
      cerr << ".";
#endif
      CPU_eigenstates(NEW, 1.E-12, 1.E-12);
      CPU_compute_surfdens();
      CPU_voldens_totvoldens(NEW);
      stopcondition = CPU_test_conv_INITCOND_THERMEQUIL();
      memcpy( _pot_OLD, _pot, NX*NZ*sizeof(double) );
      memcpy( _totvoldens_OLD, _totvoldens, NX*NZ*sizeof(double) );
      if( iter_counter > itermax )
	{
	  throw error_ITER_NR_NOTCONV_INITCOND();
	}
    }
  const double DXM1 = host_dm->get_X()->get_delta_m1();
  for(int j=0; j<NZ; ++j)
    _pot_b[j] = pot(0,j);
#ifdef __CUDACC__
  checkCudaErrors( cudaMemcpy(_GPU_pot_b, _pot_b, NZ*sizeof(double), cudaMemcpyHostToDevice) );
#endif
  for(int i=0; i<NX; ++i)
    for(int j=0; j<NZ; ++j)
      {
	if( 0<=j && j<NZ && i==0 )
	  {
	    bc_confined(THERMEQUIL,i,j).point_value = _pot_b[j];
	    bc_confined(POTENTIAL,i,j).point_value = _pot_b[j];
	  }
	if( 0<=j && j<NZ && i==NX-1 )
	  {
	    bc_confined(THERMEQUIL,i,j).point_value = _pot_b[j];
	    bc_confined(POTENTIAL,i,j).point_value = _pot_b[j] + VBIAS/host_rescpar->get_potstar();
	  }
      }
  for( int i=0; i<NX; ++i )
    {
      _totsurfdens[i] = 0;
      for(int nu=0; nu<_NVALLEYS; ++nu)
	for(int p=0; p<NSBN; ++p)
	  _totsurfdens[i] += 2.*surfdens(nu,p,i);
      _totsurfdens_eq[i] = 0;		
      for(int nu=0; nu<_NVALLEYS; ++nu)
	for(int p=0; p<NSBN; ++p)
	  {
	    surfdens_eq(nu,p,i) = surfdens(nu,p,i);
	    _totsurfdens_eq[i] += 2.*surfdens_eq(nu,p,i);
	  }
    }	
  for(int nu=0; nu<_NVALLEYS; ++nu)
    for(int p=0; p<NSBN; ++p)
      deps_dx(nu,p,0) = (eps(nu,p,1) - eps(nu,p,0))*DXM1;
  for(int i=1; i<=NX-2; ++i)
    for(int nu=0; nu<_NVALLEYS; ++nu)
      for(int p=0; p<NSBN; ++p)
	deps_dx(nu,p,i) = (eps(nu,p,i+1) - eps(nu,p,i-1))*0.5*DXM1;
  for(int nu=0; nu<_NVALLEYS; ++nu)
    for(int p=0; p<NSBN; ++p)
      deps_dx(nu,p,NX-1) = (eps(nu,p,NX-1) - eps(nu,p,NX-2))*DXM1;
  _repartition_b = 0;
  for(int nu=0; nu<_NVALLEYS; ++nu)
    for(int p=0; p<NSBN; ++p)
      _repartition_b += 2.*sqrt(mass(Xdim,nu)*mass(Ydim,nu)) * exp(-eps(nu,p,0));
#ifdef __CUDACC__	
  checkCudaErrors( cudaMemcpy(_GPU_surfdens,    _surfdens,    _NVALLEYS*NSBN*NX*sizeof(double),                          cudaMemcpyHostToDevice) );
  checkCudaErrors( cudaMemcpy(_GPU_surfdens_eq, _surfdens_eq, _NVALLEYS*NSBN*NX*sizeof(double),                          cudaMemcpyHostToDevice) );
  checkCudaErrors( cudaMemcpy(_GPU_voldens,     _voldens,     _NVALLEYS*NSBN*NX*NZ*sizeof(double),                      cudaMemcpyHostToDevice) );
  checkCudaErrors( cudaMemcpy(_GPU_totvoldens,  _totvoldens,  NX*NZ*sizeof(double),                                      cudaMemcpyHostToDevice) );
  checkCudaErrors( cudaMemcpy(_GPU_chi,         _chi,         _NVALLEYS*NSBN*NX*_SCHROED_MATRIX_SIZE_PAD*sizeof(double), cudaMemcpyHostToDevice) );
  checkCudaErrors( cudaMemcpy(_GPU_eps,         _eps,         _NVALLEYS*NSBN*NX*sizeof(double),                          cudaMemcpyHostToDevice) );
  checkCudaErrors( cudaMemcpy(_GPU_deps_dx,     _deps_dx,     _NVALLEYS*NSBN*NX*sizeof(double),                          cudaMemcpyHostToDevice) );
#endif

  exitfunc;

  config_rhs();
}












void MOSFETProblemCuda::CPU_compute_max_deps_dx()
{
  const int NSBN  = host_dm -> get_NSBN();
  const int NX    = host_dm -> get_X()   -> get_N();

  _max_deps_dx = 0;
  for(int index = 0; index < _NVALLEYS*NSBN*NX; ++index)
    _max_deps_dx = max( _max_deps_dx, fabs(_deps_dx[index]) );

  return;
}

#ifdef __CUDACC__
template <class T> __device__ __host__ T reduce_maxabs(volatile T *sdata, int tid)
{
  T res = fabs(sdata[tid]);
  for(int offset=16; offset>0; offset/=2)
    res = max( res, __shfl_xor_sync( 0xFFFFFFFF, res, offset, 32 ) );
  return res;
}
void MOSFETProblemCuda::GPU_compute_max_deps_dx() // DE VERAS QUE SE PUEDE MEJORAR...
{
  const int NSBN  = host_dm -> get_NSBN();
  const int NX    = host_dm -> get_X()   -> get_N();

  checkCudaErrors( cudaMemcpy(_deps_dx, _GPU_deps_dx, _NVALLEYS*NSBN*NX*sizeof(double), cudaMemcpyDeviceToHost) );
  CPU_compute_max_deps_dx();

  /**
     WORKINPROGRESS. Voy a intervenir aquí para hacer esta reducción a nivel de warp
     en lugar de hacer una copia de memoria GPU-RAM a CPU-RAM.

     LAST MODIFIED: 20220503

     DESCRIPTION: I have added function 
     template <class T> __device__ T reduce_maxabs(volatile T *sdata, int tid)
     to perform a reduction at warp level.
  */
  
  return;
}
#endif

void MOSFETProblemCuda::compute_max_deps_dx()
/********************************************************************
 * This function computes the derivative of the energy levels.
 * 
 * Input: eps
 * Output: deps_dx
 ********************************************************************/
{
#ifdef __CUDACC__
  GPU_compute_max_deps_dx();
#else
  CPU_compute_max_deps_dx();
#endif
  
  return;
}


void MOSFETProblemCuda::CPU_compute_deps_dx()
{
  const int NSBN  = host_dm -> get_NSBN();
  const int NX    = host_dm -> get_X()   -> get_N();
  const double DXM1 = host_dm->get_X()->get_delta_m1();

  // this is not suitable for GPU code
#pragma omp parallel for
  for(int index = 0; index < _NVALLEYS*NSBN; ++index)
    {
      int nu = index/NSBN;
      int p  = index - nu*NSBN;

      deps_dx(nu,p,0) = (eps(nu,p,1) - eps(nu,p,0))*DXM1;
      for(int i=1; i<=NX-2; ++i)
	deps_dx(nu,p,i) = (eps(nu,p,i+1) - eps(nu,p,i-1))*0.5*DXM1;
      deps_dx(nu,p,NX-1) = (eps(nu,p,NX-1) - eps(nu,p,NX-2))*DXM1;
    }

  return;
}


#ifdef __CUDACC__
__global__ void cuda_compute_deps_dx( const discrMeshes *dm, double *_GPU_deps_dx, const double *_GPU_eps )
{
  const int index = blockIdx.x*blockDim.x + threadIdx.x;

  const int NSBN     = dm->get_NSBN();
  const int NX       = dm->get_X()->get_N();
  const double DXM1  = dm->get_X()->get_delta_m1();

  if(index < _NVALLEYS*NSBN)
    {
      int nu, p;
      GPU_map_1D_to_2D( index, &nu, _NVALLEYS, &p, NSBN );
      
      GPU_deps_dx(nu,p,0) = (GPU_eps(nu,p,1) - GPU_eps(nu,p,0))*DXM1;
      for(int i=1; i<=NX-2; ++i)
	GPU_deps_dx(nu,p,i) = (GPU_eps(nu,p,i+1) - GPU_eps(nu,p,i-1))*0.5*DXM1;
      GPU_deps_dx(nu,p,NX-1) = (GPU_eps(nu,p,NX-1) - GPU_eps(nu,p,NX-2))*DXM1;
    }           
}
#endif


#ifdef __CUDACC__
void MOSFETProblemCuda::GPU_compute_deps_dx()
{
  const int NSBN  = host_dm -> get_NSBN();

  // const int TPB = 128;
  // cudaDeviceSetCacheConfig(cudaFuncCachePreferNone);
  const int gridSize      = host_gridconfig -> cuda_compute_deps_dx_config -> get_gridSize();
  const int blockSize     = host_gridconfig -> cuda_compute_deps_dx_config -> get_blockSize();
  const int shmemSize     = host_gridconfig -> cuda_compute_deps_dx_config -> get_shmemSize();
  const cudaFuncCache cfc = host_gridconfig -> cuda_compute_deps_dx_config -> get_cfc();
  cudaDeviceSetCacheConfig( cfc );
  cuda_compute_deps_dx <<< gridSize, blockSize, shmemSize >>> ( device_dm, _GPU_deps_dx, _GPU_eps );
  
  return;
}
#endif


void MOSFETProblemCuda::compute_deps_dx()
  /********************************************************************
   * This function computes the derivative of the energy levels.
   * 
   * Input: eps
   * Output: deps_dx
   ********************************************************************/
{
#ifdef __CUDACC__
  GPU_compute_deps_dx();
#else
  CPU_compute_deps_dx();
#endif
  
  return;
}


void MOSFETProblemCuda::reverse_into_OLD()
  /********************************************************************
   * This function just copies the potential and the total 
   * volume density into an auxiliary OLD vector.
   * 
   * Input: pot, totvoldens
   * Output: pot_OLD, totvoldens_OLD
   ********************************************************************/
{
#ifdef __CUDACC__
  GPU_reverse_into_OLD();
#else
  CPU_reverse_into_OLD();
#endif
      
  return;
}


void MOSFETProblemCuda::CPU_reverse_into_OLD()
{
  const int NX    = host_dm -> get_X()   -> get_N();
  const int NZ    = host_dm -> get_Z()   -> get_N();

  memcpy( _pot_OLD,        _pot,        NX*NZ*sizeof(double) );
  memcpy( _totvoldens_OLD, _totvoldens, NX*NZ*sizeof(double) );
      
  return;
}


#ifdef __CUDACC__
void MOSFETProblemCuda::GPU_reverse_into_OLD()
{
  const int NX    = host_dm -> get_X()   -> get_N();
  const int NZ    = host_dm -> get_Z()   -> get_N();

  checkCudaErrors( cudaMemcpy(_GPU_pot_OLD,        _GPU_pot,        NX*NZ*sizeof(double), cudaMemcpyDeviceToDevice) );
  checkCudaErrors( cudaMemcpy(_GPU_totvoldens_OLD, _GPU_totvoldens, NX*NZ*sizeof(double), cudaMemcpyDeviceToDevice) );

  return;
}
#endif


bool MOSFETProblemCuda::CPU_test_conv_POTENTIAL(const double tolpar)
{
  const int NX    = host_dm -> get_X()   -> get_N();
  const int NZ    = host_dm -> get_Z()   -> get_N();

  double *_aux_pot, *_aux_pot_OLD, *_aux_totvoldens, *_aux_totvoldens_OLD;
  _aux_pot            = _pot;
  _aux_pot_OLD        = _pot_OLD;
  _aux_totvoldens     = _totvoldens;
  _aux_totvoldens_OLD = _totvoldens_OLD;
  
  if( _ep != POTENTIAL )
    {
      cerr << " ERORR, from file '" << __FILE__ << "', from line " << __LINE__ << ", from function '" << __func__ << "' <--- I should not reach this point!" << endl;
      exit(-1);
    }

  bool res = false;
  // double _linfV, _l2V, _linfN, _l2N;
  _linfV = 0; _l2V = 0; _linfN = 0; _l2N = 0; /* 
						 global variables, defined as lass members
						 they are useful outside the scope of this function
						 for error testing in case of non convergence 
						 of the Newton-Raphson method for potential
					      */
  
  // #pragma parallel for reduction (max:_linfV, max:_linfN, +:_l2V, +:_l2N)
  for(int i=0; i<NX; ++i)
    for(int j=0; j<NZ; ++j)
      {
	if(fabs(_aux_pot_OLD[i*NZ+j] - _aux_pot[i*NZ+j]) > _linfV)
	  _linfV = fabs(_aux_pot_OLD[i*NZ+j] - _aux_pot[i*NZ+j]);
      
	if(fabs(_aux_totvoldens_OLD[i*NZ+j] - _aux_totvoldens[i*NZ+j]) > _linfN)
	  _linfN = fabs(_aux_totvoldens_OLD[i*NZ+j] - _aux_totvoldens[i*NZ+j]);

	// delete from here
	const double valV = _aux_pot_OLD[i*NZ+j] - _aux_pot[i*NZ+j];
	if( isnan(valV * valV) || isinf(valV * valV) )
	  {
	    cerr << " From file " << __FILE__ << ", from function " << __func__ << ", from line " << __LINE__ << " : (i,j)=(" << i << "," << j << "), valV=" << valV << endl;
	    cerr << " From file " << __FILE__ << ", from function " << __func__ << ", from line " << __LINE__ << " : (i,j)=(" << i << "," << j << "), valV*valV=" << valV*valV << endl;
	  }
	const double valN = _aux_totvoldens_OLD[i*NZ+j] - _aux_totvoldens[i*NZ+j];
	if( isnan(valN * valN) || isinf(valN * valN) )
	  {
	    cerr << " From file " << __FILE__ << ", from function " << __func__ << ", from line " << __LINE__ << " : (i,j)=(" << i << "," << j << "), valN=" << valN << endl;
	    cerr << " From file " << __FILE__ << ", from function " << __func__ << ", from line " << __LINE__ << " : (i,j)=(" << i << "," << j << "), valN*valN=" << valN*valN << endl;
	  }
	// to here
	_l2V += (_aux_pot_OLD[i*NZ+j] - _aux_pot[i*NZ+j]) * (_aux_pot_OLD[i*NZ+j] - _aux_pot[i*NZ+j]);
	_l2N += (_aux_totvoldens_OLD[i*NZ+j] - _aux_totvoldens[i*NZ+j]) * (_aux_totvoldens_OLD[i*NZ+j] - _aux_totvoldens[i*NZ+j]);
      }

  // delete from here
  if( isnan(_l2V) || isinf(_l2V) )
    {
      cerr << " From file " << __FILE__ << ", from function " << __func__ << ", from line " << __LINE__ << " : _l2V=" << _l2V << endl;
    }
  if( isnan(_l2N) || isinf(_l2N) )
    {
      cerr << " From file " << __FILE__ << ", from function " << __func__ << ", from line " << __LINE__ << " : _l2N=" << _l2N << endl;
    }
  // to here

  const double DX = host_dm->get_X()->get_delta();
  const double DZ = host_dm->get_Z()->get_delta();
  _l2V *= DX*DZ;    _l2N *= DX*DZ;
  _l2V = sqrt(_l2V); _l2N = sqrt(_l2N);

  // delete from here
  if( isnan(_l2V) || isinf(_l2V) )
    {
      cerr << " From file " << __FILE__ << ", from function " << __func__ << ", from line " << __LINE__ << " : _l2V=" << _l2V << endl;
    }
  if( isnan(_l2N) || isinf(_l2N) )
    {
      cerr << " From file " << __FILE__ << ", from function " << __func__ << ", from line " << __LINE__ << " : _l2N=" << _l2N << endl;
    }
  // to here

  if(_linfV < tolpar && _linfN < tolpar && _l2V < tolpar && _l2N < tolpar )  
    res = true;

  // delete from here
  if( isnan(_l2V) || isinf(_l2V) )
    {
      cerr << " From file " << __FILE__ << ", from function " << __func__ << ", from line " << __LINE__ << " : _l2V=" << _l2V << endl;
    }
  if( isnan(_l2N) || isinf(_l2N) )
    {
      cerr << " From file " << __FILE__ << ", from function " << __func__ << ", from line " << __LINE__ << " : _l2N=" << _l2N << endl;
    }
  // to here

  return res;
}



#ifdef __CUDACC__
bool MOSFETProblemCuda::GPU_test_conv_POTENTIAL(const double tolpar)
{
  const int NX    = host_dm -> get_X()   -> get_N();
  const int NZ    = host_dm -> get_Z()   -> get_N();

  double *_aux_pot, *_aux_pot_OLD, *_aux_totvoldens, *_aux_totvoldens_OLD;
  _aux_pot            = new double [ NX*NZ ];
  _aux_pot_OLD        = new double [ NX*NZ ];
  _aux_totvoldens     = new double [ NX*NZ ];
  _aux_totvoldens_OLD = new double [ NX*NZ ];

  checkCudaErrors( cudaMemcpy(_aux_pot,            _GPU_pot,            NX*NZ*sizeof(double), cudaMemcpyDeviceToHost) );
  checkCudaErrors( cudaMemcpy(_aux_pot_OLD,        _GPU_pot_OLD,        NX*NZ*sizeof(double), cudaMemcpyDeviceToHost) );
  checkCudaErrors( cudaMemcpy(_aux_totvoldens,     _GPU_totvoldens,     NX*NZ*sizeof(double), cudaMemcpyDeviceToHost) );
  checkCudaErrors( cudaMemcpy(_aux_totvoldens_OLD, _GPU_totvoldens_OLD, NX*NZ*sizeof(double), cudaMemcpyDeviceToHost) );
  
  if( _ep != POTENTIAL )
    {
      cerr << " ERORR, from file '" << __FILE__ << "', from line " << __LINE__ << ", from function '" << __func__ << "' <--- I should not reach this point!" << endl;
      exit(-1);
    }

  bool res = false;
  // double _linfV, _l2V, _linfN, _l2N;
  _linfV = 0; _l2V = 0; _linfN = 0; _l2N = 0;

  // #pragma parallel for reduction (max:_linfV, max:_linfN, +:_l2V, +:_l2N)
  for(int i=0; i<NX; ++i)
    for(int j=0; j<NZ; ++j)
      {
	if(fabs(_aux_pot_OLD[i*NZ+j] - _aux_pot[i*NZ+j]) > _linfV)
	  _linfV = fabs(_aux_pot_OLD[i*NZ+j] - _aux_pot[i*NZ+j]);
      
	if(fabs(_aux_totvoldens_OLD[i*NZ+j] - _aux_totvoldens[i*NZ+j]) > _linfN)
	  _linfN = fabs(_aux_totvoldens_OLD[i*NZ+j] - _aux_totvoldens[i*NZ+j]);

	_l2V += (_aux_pot_OLD[i*NZ+j] - _aux_pot[i*NZ+j]) * (_aux_pot_OLD[i*NZ+j] - _aux_pot[i*NZ+j]);
	_l2N += (_aux_totvoldens_OLD[i*NZ+j] - _aux_totvoldens[i*NZ+j]) * (_aux_totvoldens_OLD[i*NZ+j] - _aux_totvoldens[i*NZ+j]);
      }

  const double DX = host_dm->get_X()->get_delta();
  const double DZ = host_dm->get_Z()->get_delta();
  _l2V *= DX*DZ;    _l2N *= DX*DZ;
  _l2V = sqrt(_l2V); _l2N = sqrt(_l2N);

  if(_linfV < tolpar && _linfN < tolpar && _l2V < tolpar && _l2N < tolpar )  
    res = true;

  delete [] _aux_pot;
  delete [] _aux_pot_OLD;
  delete [] _aux_totvoldens;
  delete [] _aux_totvoldens_OLD;

  return res;
}
#endif


bool MOSFETProblemCuda::test_conv_POTENTIAL(const double tolpar)
  /********************************************************************
   * This function checks the convergence tolerance parameters
   * inside the Newton-Raphson iterative method.
   * 
   * Input: pot, pot_OLD, totvoldens, totvoldens_OLD
   * Output: (boolean)
   ********************************************************************/
{
#ifdef __CUDACC__
  return GPU_test_conv_POTENTIAL(tolpar);
#else
  return CPU_test_conv_POTENTIAL(tolpar);
#endif
}


bool MOSFETProblemCuda::test_conv_POTENTIAL_ABSOLUTE(const double tolpar)
  /********************************************************************
   * This function checks the convergence tolerance parameters
   * inside the Newton-Raphson iterative method.
   * 
   * Input: pot, pot_OLD, totvoldens, totvoldens_OLD
   * Output: (boolean)
   ********************************************************************/
{
#ifdef __CUDACC__
  return GPU_test_conv_POTENTIAL_ABSOLUTE(tolpar);
#else
  return CPU_test_conv_POTENTIAL_ABSOLUTE(tolpar);
#endif
}




bool MOSFETProblemCuda::CPU_test_conv_INITCOND_THERMEQUIL()
{
  const int NX    = host_dm -> get_X()   -> get_N();
  const int NZ    = host_dm -> get_Z()   -> get_N();

  double tol;

  if( _ep == POTENTIAL )
    tol = host_solvpar -> get_TOL_NR_POTENTIAL();
  else if( _ep == INITCOND )
    tol = host_solvpar -> get_TOL_NR_INITCOND();
  else if( _ep == THERMEQUIL )
    tol = host_solvpar -> get_TOL_NR_THERMEQUIL();
  else
    {
      cerr << " ERORR, from file '" << __FILE__ << "', from line " << __LINE__ << ", from function '" << __func__ << "' <--- I should not reach this point!" << endl;
      exit(-1);
    }

  bool res = false;
  double _linfV, _l2V, _linfN, _l2N;
  _linfV = 0; _l2V = 0; _linfN = 0; _l2N = 0;

  // #pragma parallel for reduction (max:_linfV, max:_linfN, +:_l2V, +:_l2N)
  for(int i=0; i<NX; ++i)
    for(int j=0; j<NZ; ++j)
      {
	if(fabs(pot_OLD(i,j) - pot(i,j)) > _linfV)
	  _linfV = fabs(pot_OLD(i,j) - pot(i,j));
      
	if(fabs(totvoldens_OLD(i,j) - totvoldens(i,j)) > _linfN)
	  _linfN = fabs(totvoldens_OLD(i,j) - totvoldens(i,j));

	_l2V += (pot_OLD(i,j) - pot(i,j)) * (pot_OLD(i,j) - pot(i,j));
	_l2N += (totvoldens_OLD(i,j) - totvoldens(i,j)) * (totvoldens_OLD(i,j) - totvoldens(i,j));
      }
  const double DX = host_dm->get_X()->get_delta();
  const double DZ = host_dm->get_Z()->get_delta();
  _l2V *= DX*DZ;    _l2N *= DX*DZ;
  _l2V = sqrt(_l2V); _l2N = sqrt(_l2N);

  if(_linfV < tol && _linfN < tol && _l2V < tol && _l2N < tol )  
    res = true;

  return res;

}




void MOSFETProblemCuda::store_corrupt_data(const double* _init_pot, const double* _init_voldens, const double* _init_eps, const double* _init_chi, const double* _init_surfdens)
{
  const int NSBN  = host_dm -> get_NSBN();
  const int NX    = host_dm -> get_X()   -> get_N();
  const int NZ    = host_dm -> get_Z()   -> get_N();
  const int NW    = host_dm -> get_W()   -> get_N();
  const int NPHI  = host_dm -> get_PHI() -> get_N();

  ofstream ostr;

  ostr.open("init_pot.dat", ios_base::out);
  for(int i=0; i<NX; ++i)
    {
      for(int j=0; j<NZ; ++j)
	{
	  const double val = _init_pot[ indx_i_j(i,j) ];

	  if( isinf(val) || isnan(val) )
	    {
	      cerr << " _init_pot[ indx_i_j(" << i << "," << j << ") ] = " << val;
	    }
	  
	  ostr << setw(16) << host_dm -> get_X() -> mesh(i)
	       << setw(16) << host_dm -> get_Z() -> mesh(j)
	       << setw(16) << val
	       << endl;
	}
      ostr << endl;
    }
  ostr.close();
  
  ostr.open("init_voldens.dat", ios_base::out);
  for(int i=0; i<NX; ++i)
    {
      for(int j=0; j<NZ; ++j)
	{
	  ostr << setw(16) << host_dm -> get_X() -> mesh(i)
	       << setw(16) << host_dm -> get_Z() -> mesh(j);
	  double aux = 0;
	  for(int nu=0; nu<_NVALLEYS; ++nu)
	    {
	      for(int p=0; p<NSBN; ++p)
		{
		  const double val = _init_voldens[ indx_i_j_nu_p(nu,p,i,j) ];

		  if( isinf(val) || isnan(val) )
		    {
		      cerr << " _init_voldens[ indx_i_j_nu_p(" << nu << "," << p << "," << i << "," << j << ") ] = " << val;
		    }
		  
		  ostr << setw(16) << val;
		  aux += val;
		}
	    }
	  ostr << setw(16) << aux << endl;
	}
      ostr << endl;
    }
  ostr.close();
  
  ostr.open("init_eps.dat", ios_base::out);
  for(int i=0; i<NX; ++i)
    {
      ostr << setw(16) << host_dm -> get_X() -> mesh(i);
      for(int nu=0; nu<_NVALLEYS; ++nu)
	{
	  for(int p=0; p<NSBN; ++p)
	    {
	      const double val = _init_eps[ indx_i_nu_p(nu,p,i) ];

	      if( isinf(val) || isnan(val) )
		{
		  cerr << " _init_eps[ indx_i_nu_p(" << nu << "," << p << "," << i << ") ] = " << val;
		}

	      ostr << setw(16) << val;
	    }
	}
      ostr << endl;
    }
  ostr.close();
  
  ostr.open("init_chi.dat", ios_base::out);
  for(int i=0; i<NX; ++i)
    {
      for(int j=1; j<NZ-1; ++j)
	{
	  ostr << setw(16) << host_dm -> get_X() -> mesh(i)
	       << setw(16) << host_dm -> get_Z() -> mesh(j);
	  for(int nu=0; nu<_NVALLEYS; ++nu)
	    {
	      for(int p=0; p<NSBN; ++p)
		{
		  const double val = _init_chi[ indx_i_nu_p_jschr(nu,p,i,j) ];

		  if( isinf(val) || isnan(val) )
		    {
		      cerr << " _init_chi[ indx_i_nu_p_jschr(" << nu << "," << p << "," << i << "," << j << ") ] = " << val;
		    }

		  ostr << setw(16) << val;
		}
	    }
	  ostr << endl;
	}
      ostr << endl;
    }
  ostr.close();
  
  ostr.open("init_surfdens.dat", ios_base::out);
  for(int i=0; i<NX; ++i)
    {
      ostr << setw(16) << host_dm -> get_X() -> mesh(i);
      double aux = 0;
      for(int nu=0; nu<_NVALLEYS; ++nu)
	{
	  for(int p=0; p<NSBN; ++p)
	    {
	      const double val = _init_surfdens[ indx_i_nu_p(nu,p,i) ];
	      ostr << setw(16) << val;
	      aux += val;
	    }
	}
      ostr << setw(16) << aux << endl;
    }
  ostr.close();

  return;
}





cuda_iter_kernels_config::cuda_iter_kernels_config(const discrMeshes *dm)
{
  const int NSBN  = dm -> get_NSBN();

  const int TPB = 128;
  cuda_compute_deps_dx_config = new kernelConfig( nblocks( _NVALLEYS*NSBN, TPB ), TPB, NOSHMEM, cudaFuncCachePreferNone, "cuda_compute_deps_dx" );
}





/*
  name          : 'check_iter_2_input', 'check_iter_2_output'
  last modified : 2023/06/06
  author        : Francesco VECIL

  description   : This functions check the input and the output of the 'iter_2' method.
                  As input  : (i) surfdens, (ii) pot [as initialization]
*/
void MOSFETProblemCuda::check_iter_2_input()
{
  cerr << "called '" << __func__ << "'...";

  const int NSBN  = host_dm -> get_NSBN();
  const int NX    = host_dm -> get_X()   -> get_N();
  const int NZ    = host_dm -> get_Z()   -> get_N();

  bool stopcheck = false;
  
  /*
    begin section : Check surfdens
  */
  // cerr << "check surfdens...";
  double *_aux_surfdens = new double[ _NVALLEYS*NSBN*NX ];
#ifdef __CUDACC__
  CUDA_SAFE_CALL( cudaMemcpy( _aux_surfdens, _GPU_surfdens, _NVALLEYS*NSBN*NX*sizeof(double), cudaMemcpyDeviceToHost ) );
#else
  memcpy( _aux_surfdens, _surfdens, _NVALLEYS*NSBN*NX*sizeof(double) );
#endif
  
  for(int nu=0; nu<_NVALLEYS; ++nu)
    {
      for(int p=0; p<NSBN; ++p)
	{
	  for(int i=0; i<NX; ++i)
	    {
	      const double val = _aux_surfdens[ indx_i_nu_p(nu,p,i) ];
	      if( isnan( val ) || isinf( val ) )
		{
		  cerr << " ERROR from function " << __func__ << ", from line " << __LINE__ << ", from file '" << __FILE__
		       << "' --- surfdens(" << nu << "," << p << "," << i << ") = " << val << endl;
		  stopcheck = true;
		}
	    }
	}
    }
  delete [] _aux_surfdens;
  // cerr << "[ok]";
  /*
    end section : Check surfdens
   */



  
  /*
    begin section : Check pot[spphase]
   */
  // cerr << "check pot...";
  double *_aux_pot = new double[ NX*NZ ];
#ifdef __CUDACC__
  CUDA_SAFE_CALL( cudaMemcpy( _aux_pot, _GPU_pot, NX*NZ*sizeof(double), cudaMemcpyDeviceToHost ) );
#else
  memcpy( _aux_pot, _pot, NX*NZ*sizeof(double) );
#endif

  for(int i=0; i<NX; ++i)
    {
      for(int j=0; j<NZ; ++j)
	{
	  const double val = _aux_pot[ indx_i_j(i,j) ];
	  if( isnan( val ) || isinf( val ) )
	    {
	      cerr << " ERROR from function " << __func__ << ", from line " << __LINE__ << ", from file '" << __FILE__
		   << "' --- pot(" << i << "," << j << ") = " << val << endl;
	      stopcheck = true;
	    }
	}
    }
  delete [] _aux_pot;
  // cerr << "[ok]";
  /*
    end section : Check pot[spphase]
  */

  if( stopcheck == true )
    {
      cerr << " ERROR from function " << __func__ << ", from line " << __LINE__ << ", from file '" << __FILE__
	   << "' --- exiting..." << endl;
      exit(_ERR_CHECK_ITER_2_INPUT_OUTPUT);
    }
  
  cerr << "[ok]" << endl;

  return;
}



/*
  name          : 'check_iter_2_input', 'check_iter_2_output'
  last modified : 2023/06/06
  author        : Francesco VECIL

  description   : This functions check the input and the output of the 'iter_2' method.
		  As output : (iii) eps, (iv) deps_dx, (v) chi, (vi) pot
 */
void MOSFETProblemCuda::check_iter_2_output()
{
  cerr << "called '" << __func__ << "'...";

  const int NSBN  = host_dm -> get_NSBN();
  const int NX    = host_dm -> get_X()   -> get_N();
  const int NZ    = host_dm -> get_Z()   -> get_N();

  bool stopcheck = false;
  
  /*
    begin section : Check pot
   */
  // cerr << "check pot...";
  double *_aux_pot = new double[ NX*NZ ];
#ifdef __CUDACC__
  CUDA_SAFE_CALL( cudaMemcpy( _aux_pot, _GPU_pot, NX*NZ*sizeof(double), cudaMemcpyDeviceToHost ) );
#else
  memcpy( _aux_pot, _pot, NX*NZ*sizeof(double) );
#endif

  for(int i=0; i<NX; ++i)
    {
      for(int j=0; j<NZ; ++j)
	{
	  const double val = _aux_pot[ indx_i_j(i,j) ];
	  if( isnan( val ) || isinf( val ) )
	    {
	      cerr << " ERROR from function " << __func__ << ", from line " << __LINE__ << ", from file '" << __FILE__
		   << "' --- pot(" << i << "," << j << ") = " << val << endl;
	      stopcheck = true;
	    }
	}
    }
  delete [] _aux_pot;
  // cerr << "[ok]";
  /*
    end section : Check pot[spphase]
  */

  if( stopcheck == true )
    {
      cerr << " ERROR from function " << __func__ << ", from line " << __LINE__ << ", from file '" << __FILE__
	   << "' --- exiting..." << endl;
      exit(_ERR_CHECK_ITER_2_INPUT_OUTPUT);
    }
  
  cerr << "[ok]" << endl;

  return;
}






/*
  name          : 'check_update_potential_input', 'check_update_potential_output'
  last modified : 2023/06/06
  author        : Francesco VECIL

  description   : This functions check the input and the output of the 'update_potential' method.
                  As input  : (i) eps, (ii) chi, (iii) pot[OLD]
 */
void MOSFETProblemCuda::check_update_potential_input()
{
  cerr << "called '" << __func__ << "'...";

  const int NSBN  = host_dm -> get_NSBN();
  const int NX    = host_dm -> get_X()   -> get_N();
  const int NZ    = host_dm -> get_Z()   -> get_N();

  bool stopcheck = false;
  
  /*
    begin section : Check eps
   */
  cerr << "check eps...";
  double *_aux_eps = new double[ _NVALLEYS*NSBN*NX ];

#ifdef __CUDACC__
  CUDA_SAFE_CALL( cudaMemcpy( _aux_eps, _GPU_eps, _NVALLEYS*NSBN*NX*sizeof(double), cudaMemcpyDeviceToHost ) );
#else
  memcpy( _aux_eps, _eps, _NVALLEYS*NSBN*NX*sizeof(double) );
#endif
  
  for(int nu=0; nu<_NVALLEYS; ++nu)
    {
      for(int p=0; p<NSBN; ++p)
	{
	  for(int i=0; i<NX; ++i)
	    {
	      const double val = _aux_eps[ indx_i_nu_p(nu,p,i) ];
	      if( isnan( val ) || isinf( val ) )
		{
		  cerr << " ERROR from function " << __func__ << ", from line " << __LINE__ << ", from file '" << __FILE__
		       << "' --- eps(" << nu << "," << p << "," << i << ") = " << val << endl;
		  stopcheck = true;
		}
	    }
	}
    }
  delete [] _aux_eps;
  cerr << "[ok]";
  /*
    end section : Check eps
   */
  

  
  /*
    begin section : Check chi
   */
  cerr << "check chi...";
  double *_aux_chi = new double[ _NVALLEYS*NSBN*NX*_SCHROED_MATRIX_SIZE_PAD ];
#ifdef __CUDACC__
  CUDA_SAFE_CALL( cudaMemcpy( _aux_chi, _GPU_chi, _NVALLEYS*NSBN*NX*_SCHROED_MATRIX_SIZE_PAD*sizeof(double), cudaMemcpyDeviceToHost ) );
#else
  memcpy( _aux_chi, _chi, _NVALLEYS*NSBN*NX*_SCHROED_MATRIX_SIZE_PAD*sizeof(double) );
#endif
  
  for(int nu=0; nu<_NVALLEYS; ++nu)
    {
      for(int p=0; p<NSBN; ++p)
	{
	  for(int i=0; i<NX; ++i)
	    {
	      for(int j=1; j<NZ-1; ++j)
		{
		  const double val = _aux_chi[ indx_i_nu_p_jschr(nu,p,i,j) ];
		  if( isnan( val ) || isinf( val ) )
		    {
		      cerr << " ERROR from function " << __func__ << ", from line " << __LINE__ << ", from file '" << __FILE__
			   << "' --- chi(" << nu << "," << p << "," << i << "," << j << ") = " << val << endl;
		      stopcheck = true;
		    }
		}
	    }
	}
    }
  
  delete [] _aux_chi;
  cerr << "[ok]";
  /*
    end section : Check chi
   */

  /*
    begin section : Check pot
   */
  cerr << "check pot OLD...";
  double *_aux_pot = new double[ NX*NZ ];
#ifdef __CUDACC__
  CUDA_SAFE_CALL( cudaMemcpy( _aux_pot, _GPU_pot_OLD, NX*NZ*sizeof(double), cudaMemcpyDeviceToHost ) );
#else
  memcpy( _aux_pot, _pot_OLD, NX*NZ*sizeof(double) );
#endif

  for(int i=0; i<NX; ++i)
    {
      for(int j=0; j<NZ; ++j)
	{
	  const double val = _aux_pot[ indx_i_j(i,j) ];
	  if( isnan( val ) || isinf( val ) )
	    {
	      cerr << " ERROR from function " << __func__ << ", from line " << __LINE__ << ", from file '" << __FILE__
		   << "' --- pot(" << i << "," << j << ") = " << val << endl;
	      stopcheck = true;
	    }
	}
    }
  delete [] _aux_pot;
  cerr << "[ok]";
  /*
    end section : Check pot[spphase]
  */

  if( stopcheck == true )
    {
      cerr << " ERROR from function " << __func__ << ", from line " << __LINE__ << ", from file '" << __FILE__
	   << "' --- exiting..." << endl;
      exit(_ERR_CHECK_UPDATE_POTENTIAL_INPUT_OUTPUT);
    }
  
  return;
}



/*
  name          : 'check_update_potential_input', 'check_update_potential_output'
  last modified : 2023/06/06
  author        : Francesco VECIL

  description   : This functions check the input and the output of the 'update_potential' method.
		  As output : (iv) pot[NEW]
 */
void MOSFETProblemCuda::check_update_potential_output()
{
  cerr << "called '" << __func__ << "'...";

  const int NSBN  = host_dm -> get_NSBN();
  const int NX    = host_dm -> get_X()   -> get_N();
  const int NZ    = host_dm -> get_Z()   -> get_N();

  bool stopcheck = false;
  
  /*
    begin section : Check pot
   */
  cerr << "check pot NEW...";
  double *_aux_pot = new double[ NX*NZ ];
#ifdef __CUDACC__
  CUDA_SAFE_CALL( cudaMemcpy( _aux_pot, _GPU_pot, NX*NZ*sizeof(double), cudaMemcpyDeviceToHost ) );
#else
  memcpy( _aux_pot, _pot, NX*NZ*sizeof(double) );
#endif

  for(int i=0; i<NX; ++i)
    {
      for(int j=0; j<NZ; ++j)
	{
	  const double val = _aux_pot[ indx_i_j(i,j) ];
	  if( isnan( val ) || isinf( val ) )
	    {
	      cerr << " ERROR from function " << __func__ << ", from line " << __LINE__ << ", from file '" << __FILE__
		   << "' --- pot(" << i << "," << j << ") = " << val << endl;
	      stopcheck = true;
	    }
	}
    }
  delete [] _aux_pot;
  cerr << "[ok]";
  /*
    end section : Check pot[spphase]
  */

  if( stopcheck == true )
    {
      cerr << " ERROR from function " << __func__ << ", from line " << __LINE__ << ", from file '" << __FILE__
	   << "' --- exiting..." << endl;
      exit(_ERR_CHECK_UPDATE_POTENTIAL_INPUT_OUTPUT);
    }
  
  return;
}




/*
  name          : new_test_pot
  last modified : 2023/06/12
  author        : Francesco VECIL

  description   : See header file.
 */
#ifdef __CUDACC__
bool MOSFETProblemCuda::GPU_test_conv_POTENTIAL_ABSOLUTE(const double tolpar)
{
  /*
    As a first attempt, to check whether it works or not, the test will be
    performed on CPU. If it is successful, it will be ported onto GPU.
   */

  const int NX    = host_dm -> get_X()   -> get_N();
  const int NZ    = host_dm -> get_Z()   -> get_N();
  const double CP = host_adimpar -> get_cp();
  
  double *_aux_pot        = new double [ NX*NZ ];
  double *_aux_totvoldens = new double [ NX*NZ ];
  double *_aux_residual   = new double [ NX*NZ ];

  checkCudaErrors( cudaMemcpy(_aux_pot,            _GPU_pot,            NX*NZ*sizeof(double), cudaMemcpyDeviceToHost) );
  checkCudaErrors( cudaMemcpy(_aux_totvoldens,     _GPU_totvoldens,     NX*NZ*sizeof(double), cudaMemcpyDeviceToHost) );

  // cerr << "1)..." << endl;
  // fill residual vector
  double *mat = &_matrix_2dconst[ 2*NX*NZ*(2*NZ+1) ];
  for(int i=0; i<NX; ++i)
    {
      for(int j=0; j<NZ; ++j)
	{
	  int line = indx_i_j(i,j);
	  
	  double aux = 0;
	  
	  for(int k=max(0, line-NZ); k<=min(NX*NZ-1, line+NZ); ++k)
	    {
	      // aux += matrix_2dconst(POTENTIAL, line, k) * _aux_pot[k];
	      aux += mat[ line*(2*NZ+1)+(k+NZ-line) ] * _aux_pot[k];
	    }
	  _aux_residual[ indx_i_j(i,j) ] = aux;
	}
    }

  for(int i=1; i<NX-1; ++i)
    {
      for(int j=1; j<NZ-1; ++j)
	{
	  _aux_residual[ indx_i_j(i,j) ] += CP * (_aux_totvoldens[ indx_i_j(i,j) ] - nd(i,j) );
	}
    }

  // cerr << "2)..." << endl;
  // compute scalar residual
  double linfnorm = 0;
  for(int i=1; i<NX-1; ++i)
    {
      for(int j=1; j<NZ-1; ++j)
	{
	  linfnorm = max( linfnorm, fabs(_aux_residual[ indx_i_j(i,j) ]) );
	}
    }
  // cerr << "linfnorm = " << linfnorm << " / " << tolpar << "\r";

  delete [] _aux_residual;
  delete [] _aux_totvoldens;
  delete [] _aux_pot;

  if(linfnorm < tolpar)
    {
      return true;
    }
  else
    {
      return false;
    }
}
#endif


bool MOSFETProblemCuda::CPU_test_conv_POTENTIAL_ABSOLUTE(const double tolpar)
{
  /*
    As a first attempt, to check whether it works or not, the test will be
    performed on CPU. If it is successful, it will be ported onto GPU.
   */

  const int NX    = host_dm -> get_X()   -> get_N();
  const int NZ    = host_dm -> get_Z()   -> get_N();
  const double CP = host_adimpar -> get_cp();
  
  double *_aux_pot        = new double [ NX*NZ ];
  double *_aux_totvoldens = new double [ NX*NZ ];
  double *_aux_residual   = new double [ NX*NZ ];

  memcpy(_aux_pot,            _pot,            NX*NZ*sizeof(double));
  memcpy(_aux_totvoldens,     _totvoldens,     NX*NZ*sizeof(double));

  // cerr << "1)..." << endl;
  // fill residual vector
  double *mat = &_matrix_2dconst[ 2*NX*NZ*(2*NZ+1) ];
  for(int i=0; i<NX; ++i)
    {
      for(int j=0; j<NZ; ++j)
	{
	  int line = indx_i_j(i,j);
	  
	  double aux = 0;
	  
	  for(int k=max(0, line-NZ); k<=min(NX*NZ-1, line+NZ); ++k)
	    {
	      // aux += matrix_2dconst(POTENTIAL, line, k) * _aux_pot[k];
	      aux += mat[ line*(2*NZ+1)+(k+NZ-line) ] * _aux_pot[k];
	    }
	  _aux_residual[ indx_i_j(i,j) ] = aux;
	}
    }

  for(int i=1; i<NX-1; ++i)
    {
      for(int j=1; j<NZ-1; ++j)
	{
	  _aux_residual[ indx_i_j(i,j) ] += CP * (_aux_totvoldens[ indx_i_j(i,j) ] - nd(i,j) );
	}
    }

  // cerr << "2)..." << endl;
  // compute scalar residual
  double linfnorm = 0;
  for(int i=1; i<NX-1; ++i)
    {
      for(int j=1; j<NZ-1; ++j)
	{
	  linfnorm = max( linfnorm, fabs(_aux_residual[ indx_i_j(i,j) ]) );
	}
    }
  // cerr << "linfnorm = " << linfnorm << endl;

  delete [] _aux_residual;
  delete [] _aux_totvoldens;
  delete [] _aux_pot;

  if(linfnorm < tolpar)
    {
      return true;
    }
  else
    {
      return false;
    }
}
/*
  name          : new_test_pot
 */
