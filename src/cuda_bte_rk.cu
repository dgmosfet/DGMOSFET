#include "mosfetproblem.h"
#include "debug_flags.h"



#ifdef __CUDACC__
__global__ void cuda_set_rhs_to_zero( const discrMeshes *dm, double *_GPU_rhs_pdf )
{
  const int NSBN     = dm -> get_NSBN();
  const int NX       = dm -> get_X()   -> get_N();
  const int NW       = dm -> get_W()   -> get_N();
  const int NPHI     = dm -> get_PHI() -> get_N();

  int global_index = blockIdx.x*blockDim.x + threadIdx.x;

  if(global_index < _NVALLEYS*NSBN*NX*NW*NPHI)
    {
      _GPU_rhs_pdf[ global_index ] = 0;
    }
}
#else
void MOSFETProblemCuda::CPU_set_rhs_to_zero()
{
  const int NSBN  = host_dm -> get_NSBN();
  const int NX    = host_dm -> get_X()   -> get_N();
  const int NW    = host_dm -> get_W()   -> get_N();
  const int NPHI  = host_dm -> get_PHI() -> get_N();

#pragma omp parallel for
  for( int global_index=0; global_index < _NVALLEYS*NSBN*NX*NW*NPHI; global_index++ )
    {
      int nu,p,i,l,m;
      GPU_map_1D_to_5D( global_index, &i, NX, &nu, _NVALLEYS, &p, NSBN, &l, NW, &m, NPHI );
      rhs_pdf(nu,p,i,l,m) = 0;
    }
  return;
}
#endif

/**
   PURPOSE:        

   FILE:           cuda_bte_rk.cu

   NAME:           MOSFETProblem::set_rhs_to_zero

   PARAMETERS:

   RETURN VALUE:   none

   CALLS TO:       MOSFETProblem::show_eta             (cuda_testing.h - declared inline)

   CALLED FROM:    MOSFETProblem::perform_step_2       (cuda_time_integration.cu)

   DATA MODIFIED:  

   METHOD:         

   LAST UPDATE:    2022 February 16
*/
void MOSFETProblemCuda::set_rhs_to_zero()
{
#ifdef _SHOW_ETA
  ostringstream message;
  message << "called " << __func__;
  show_eta(message.str());
#endif
#ifdef __CUDACC__
  const int NSBN  = host_dm -> get_NSBN();
  const int NX    = host_dm -> get_X()   -> get_N();
  const int NW    = host_dm -> get_W()   -> get_N();
  const int NPHI  = host_dm -> get_PHI() -> get_N();

  const int gridSize      = host_gridconfig -> cuda_set_rhs_to_zero_config -> get_gridSize();
  const int blockSize     = host_gridconfig -> cuda_set_rhs_to_zero_config -> get_blockSize();
  const int shmemSize     = host_gridconfig -> cuda_set_rhs_to_zero_config -> get_shmemSize();
  const cudaFuncCache cfc = host_gridconfig -> cuda_set_rhs_to_zero_config -> get_cfc();
  cudaDeviceSetCacheConfig( cfc );
  cuda_set_rhs_to_zero <<< gridSize, blockSize, shmemSize >>> ( device_dm, _GPU_rhs_pdf );
#else
  CPU_set_rhs_to_zero ();
#endif

#ifdef _SHOW_ETA
  cerr << "[ok] (function " << __func__ << ", line " << __LINE__ << ")" << endl;
#endif
  return;
}






#ifdef __CUDACC__
__global__ void cuda_perform_RK_1_3( const discrMeshes *dm, double *_GPU_pdf, const double *_GPU_rhs_pdf, const double DT )
{
  const int NSBN     = dm -> get_NSBN();
  const int NX       = dm -> get_X()   -> get_N();
  const int NW       = dm -> get_W()   -> get_N();
  const int NPHI     = dm -> get_PHI() -> get_N();

  int global_index = blockIdx.x*blockDim.x + threadIdx.x;

  if(global_index < _NVALLEYS*NSBN*NX*NW*NPHI)
    {
      int nu,p,i,l,m;
      GPU_map_1D_to_5D( global_index, &i, NX, &nu, _NVALLEYS, &p, NSBN, &l, NW, &m, NPHI );

      GPU_pdf( nu,p,i,l,m,1 ) = GPU_pdf( nu,p,i,l,m,0 ) + DT*GPU_rhs_pdf( nu,p,i,l,m );
    }
}
#else
void MOSFETProblemCuda::CPU_perform_RK_1_3( const double DT )
{
  const int NSBN  = host_dm -> get_NSBN();
  const int NX    = host_dm -> get_X()   -> get_N();
  const int NW    = host_dm -> get_W()   -> get_N();
  const int NPHI  = host_dm -> get_PHI() -> get_N();

#pragma omp parallel for
  for( int global_index=0; global_index < _NVALLEYS*NSBN*NX*NW*NPHI; global_index++ )
    {
      int nu,p,i,l,m;
      GPU_map_1D_to_5D( global_index, &i, NX, &nu, _NVALLEYS, &p, NSBN, &l, NW, &m, NPHI );
      
      pdf( nu,p,i,l,m,1 ) = pdf( nu,p,i,l,m,0 ) + DT*rhs_pdf( nu,p,i,l,m );
      // TASK : uncomment
      // rhs_pdf( nu,p,i,l,m ) = 0;
    }
  return;
}
#endif
/**
   PURPOSE:        

   FILE:           cuda_bte_rk.cu

   NAME:           MOSFETProblem::perform_RK_1_3

   PARAMETERS:

   RETURN VALUE:   none

   CALLS TO:       MOSFETProblem::start_time           (cuda_comptime.h - declared inline)
		   MOSFETProblem::stop_time            (cuda_comptime.h - declared inline)
                   MOSFETProblem::show_eta             (cuda_testing.h - declared inline)

   CALLED FROM:    MOSFETProblem::perform_step_2       (cuda_time_integration.cu)

   DATA MODIFIED:  

   METHOD:         

   LAST UPDATE:    2022 February 16
*/
void MOSFETProblemCuda::perform_RK_1_3( const double DT )
{
#ifdef _SHOW_ETA
  ostringstream message;
  message << "called " << __func__;
  show_eta(message.str());
#endif

  start_time( _PHASE_BTE_RK );

#ifdef __CUDACC__
  const int NSBN  = host_dm -> get_NSBN();
  const int NX    = host_dm -> get_X()   -> get_N();
  const int NW    = host_dm -> get_W()   -> get_N();
  const int NPHI  = host_dm -> get_PHI() -> get_N();

  const int gridSize      = host_gridconfig -> cuda_perform_RK_1_3_config -> get_gridSize();
  const int blockSize     = host_gridconfig -> cuda_perform_RK_1_3_config -> get_blockSize();
  const int shmemSize     = host_gridconfig -> cuda_perform_RK_1_3_config -> get_shmemSize();
  const cudaFuncCache cfc = host_gridconfig -> cuda_perform_RK_1_3_config -> get_cfc();
  cudaDeviceSetCacheConfig( cfc );
  cuda_perform_RK_1_3 <<< gridSize, blockSize, shmemSize >>> ( device_dm, _GPU_pdf, _GPU_rhs_pdf, DT );
#else
  CPU_perform_RK_1_3 ( DT );
#endif

  stop_time( _PHASE_BTE_RK );

#ifdef _SHOW_ETA
  cerr << "[ok] (function " << __func__ << ", line " << __LINE__ << ")" << endl;
#endif
  return;
}


#ifdef __CUDACC__
__global__ void cuda_perform_RK_2_3( const discrMeshes *dm, double *_GPU_pdf, const double *_GPU_rhs_pdf, const double DT )
{
  const int NSBN     = dm -> get_NSBN();
  const int NX       = dm -> get_X()   -> get_N();
  const int NW       = dm -> get_W()   -> get_N();
  const int NPHI     = dm -> get_PHI() -> get_N();

  int global_index = blockIdx.x*blockDim.x + threadIdx.x;

  if(global_index < _NVALLEYS*NSBN*NX*NW*NPHI)
    {
      int nu,p,i,l,m;
      GPU_map_1D_to_5D( global_index, &i, NX, &nu, _NVALLEYS, &p, NSBN, &l, NW, &m, NPHI );

      GPU_pdf( nu,p,i,l,m,2 ) = .75*GPU_pdf( nu,p,i,l,m,0 ) + .25*GPU_pdf( nu,p,i,l,m,1 ) + .25*DT*GPU_rhs_pdf( nu,p,i,l,m );
      // GPU_rhs_pdf( nu,p,i,l,m ) = 0;
    }
}
#else
void MOSFETProblemCuda::CPU_perform_RK_2_3( const double DT )
{
  const int NSBN  = host_dm -> get_NSBN();
  const int NX    = host_dm -> get_X()   -> get_N();
  const int NW    = host_dm -> get_W()   -> get_N();
  const int NPHI  = host_dm -> get_PHI() -> get_N();

#pragma omp parallel for
  for( int global_index=0; global_index < _NVALLEYS*NSBN*NX*NW*NPHI; global_index++ )
    {
      int nu,p,i,l,m;
      GPU_map_1D_to_5D( global_index, &i, NX, &nu, _NVALLEYS, &p, NSBN, &l, NW, &m, NPHI );

      pdf( nu,p,i,l,m,2 ) = .75*pdf( nu,p,i,l,m,0 ) + .25*pdf( nu,p,i,l,m,1 ) + .25*DT*rhs_pdf( nu,p,i,l,m );
      // rhs_pdf( nu,p,i,l,m ) = 0;
    }
  return;
}
#endif
/**
   PURPOSE:        

   FILE:           cuda_bte_rk.cu

   NAME:           MOSFETProblem::perform_RK_2_3

   PARAMETERS:

   RETURN VALUE:   none

   CALLS TO:       MOSFETProblem::start_time           (cuda_comptime.h - declared inline)
		   MOSFETProblem::stop_time            (cuda_comptime.h - declared inline)
                   MOSFETProblem::show_eta             (cuda_testing.h - declared inline)

   CALLED FROM:    MOSFETProblem::perform_step_2       (cuda_time_integration.cu)

   DATA MODIFIED:  

   METHOD:         

   LAST UPDATE:    2022 February 16
*/
void MOSFETProblemCuda::perform_RK_2_3( const double DT )
{
#ifdef _SHOW_ETA
  ostringstream message;
  message << "called " << __func__;
  show_eta(message.str());
#endif
  start_time( _PHASE_BTE_RK );

#ifdef __CUDACC__
  const int NSBN  = host_dm -> get_NSBN();
  const int NX    = host_dm -> get_X()   -> get_N();
  const int NW    = host_dm -> get_W()   -> get_N();
  const int NPHI  = host_dm -> get_PHI() -> get_N();

  const int gridSize      = host_gridconfig -> cuda_perform_RK_2_3_config -> get_gridSize();
  const int blockSize     = host_gridconfig -> cuda_perform_RK_2_3_config -> get_blockSize();
  const int shmemSize     = host_gridconfig -> cuda_perform_RK_2_3_config -> get_shmemSize();
  const cudaFuncCache cfc = host_gridconfig -> cuda_perform_RK_2_3_config -> get_cfc();
  cudaDeviceSetCacheConfig( cfc );
  cuda_perform_RK_2_3 <<< gridSize, blockSize, shmemSize >>> ( device_dm, _GPU_pdf, _GPU_rhs_pdf, DT );
#else
  CPU_perform_RK_2_3 ( DT );
#endif

  stop_time( _PHASE_BTE_RK );

#ifdef _SHOW_ETA
  cerr << "[ok] (function " << __func__ << ", line " << __LINE__ << ")" << endl;
#endif
  return;
}


#ifdef __CUDACC__
__global__ void cuda_perform_RK_3_3( const discrMeshes *dm, double *_GPU_pdf, const double *_GPU_rhs_pdf, const double DT )
{
  const int NSBN     = dm -> get_NSBN();
  const int NX       = dm -> get_X()   -> get_N();
  const int NW       = dm -> get_W()   -> get_N();
  const int NPHI     = dm -> get_PHI() -> get_N();

  int global_index = blockIdx.x*blockDim.x + threadIdx.x;

  if(global_index < _NVALLEYS*NSBN*NX*NW*NPHI)
    {
      int nu,p,i,l,m;
      GPU_map_1D_to_5D( global_index, &i, NX, &nu, _NVALLEYS, &p, NSBN, &l, NW, &m, NPHI );

      GPU_pdf( nu,p,i,l,m,0 ) = 1./3.*GPU_pdf( nu,p,i,l,m,0 ) + 2./3.*GPU_pdf( nu,p,i,l,m,2 ) + 2./3.*DT*GPU_rhs_pdf( nu,p,i,l,m );
      // GPU_rhs_pdf( nu,p,i,l,m ) = 0;
    }
}
#else
void MOSFETProblemCuda::CPU_perform_RK_3_3( const double DT )
{
  const int NSBN  = host_dm -> get_NSBN();
  const int NX    = host_dm -> get_X()   -> get_N();
  const int NW    = host_dm -> get_W()   -> get_N();
  const int NPHI  = host_dm -> get_PHI() -> get_N();

#pragma omp parallel for
  for( int global_index=0; global_index < _NVALLEYS*NSBN*NX*NW*NPHI; global_index++ )
    {
      int nu,p,i,l,m;
      GPU_map_1D_to_5D( global_index, &i, NX, &nu, _NVALLEYS, &p, NSBN, &l, NW, &m, NPHI );

      pdf( nu,p,i,l,m,0 ) = 1./3.*pdf( nu,p,i,l,m,0 ) + 2./3.*pdf( nu,p,i,l,m,2 ) + 2./3.*DT*rhs_pdf( nu,p,i,l,m );
      // rhs_pdf( nu,p,i,l,m ) = 0;
    }
  return;
}
#endif
/**
   PURPOSE:        

   FILE:           cuda_bte_rk.cu

   NAME:           MOSFETProblem::perform_RK_3_3

   PARAMETERS:

   RETURN VALUE:   none

   CALLS TO:       MOSFETProblem::start_time           (cuda_comptime.h - declared inline)
		   MOSFETProblem::stop_time            (cuda_comptime.h - declared inline)
                   MOSFETProblem::show_eta             (cuda_testing.h - declared inline)

   CALLED FROM:    MOSFETProblem::perform_step_2       (cuda_time_integration.cu)

   DATA MODIFIED:  

   METHOD:         

   LAST UPDATE:    2022 February 16
*/
void MOSFETProblemCuda::perform_RK_3_3( const double DT )
{
#ifdef _SHOW_ETA
  ostringstream message;
  message << "called " << __func__;
  show_eta(message.str());
#endif
  start_time( _PHASE_BTE_RK );

#ifdef __CUDACC__
  const int NSBN  = host_dm -> get_NSBN();
  const int NX    = host_dm -> get_X()   -> get_N();
  const int NW    = host_dm -> get_W()   -> get_N();
  const int NPHI  = host_dm -> get_PHI() -> get_N();

  const int gridSize      = host_gridconfig -> cuda_perform_RK_3_3_config -> get_gridSize();
  const int blockSize     = host_gridconfig -> cuda_perform_RK_3_3_config -> get_blockSize();
  const int shmemSize     = host_gridconfig -> cuda_perform_RK_3_3_config -> get_shmemSize();
  const cudaFuncCache cfc = host_gridconfig -> cuda_perform_RK_3_3_config -> get_cfc();
  cudaDeviceSetCacheConfig( cfc );
  cuda_perform_RK_3_3 <<< gridSize, blockSize, shmemSize >>> ( device_dm, _GPU_pdf, _GPU_rhs_pdf, DT );
#else
  CPU_perform_RK_3_3 ( DT );
#endif

  stop_time( _PHASE_BTE_RK );

#ifdef _SHOW_ETA
  cerr << "[ok] (function " << __func__ << ", line " << __LINE__ << ")" << endl;
#endif

  return;
}



/*
  Function 'perform_RK' performs time integration using 3rd order Total-Variation Diminishing Runge-Kutta scheme.
  
  Input:
  ------
  DT : double, representing the time step
  s  : integer (0, 1, 2, any other value will produce an exception)

  Output:
  -------
  In debugging mode, the function returns:
  0    : no error
  +1   : a  NAN/INF value on entry for rhs
  +4   : a  NAN/INF value on entry for pdf
  +16  : a  NAN/INF value on exit  for rhs
  +64  : a  NAN/INF value on exit  for pdf

  Functioning:
  ------------
  The function takes rhs_pdf and pdf( 0 <= stage <= s ) and overwrites pdf( (s+1)%3 ).
 */
// #define _PERFORM_RK_DEBUG
// #define _DEBUG_TOB1659
// #define _DEBUG_TOB1660
int MOSFETProblemCuda::perform_RK( const double DT, const int s )
{
  switch( s )
    {
    case 0:
      {
	perform_RK_1_3( DT );
	break;
      }
    case 1:
      {
	perform_RK_2_3( DT );
	break;
      }
    case 2:
      {
	perform_RK_3_3( DT );
	break;
      }
    default:
      {
	cerr << " ERROR : por aqui no he de pasar." << endl;
	throw error_RK();
	break;
      }
    }

  return 0;
}




/********************************
 *        FLUXES                *
 ********************************/
#ifdef __CUDACC__
__global__ void cuda_set_fluxes_a3( const discrMeshes *dm, double *_GPU_a3, const double *_GPU_deps_dx, const double *_GPU_a3_const )
{
  const int NSBN     = dm -> get_NSBN();
  const int NX       = dm -> get_X()   -> get_N();
  const int NW       = dm -> get_W()   -> get_N();
  const int NPHI     = dm -> get_PHI() -> get_N();

  int global_index = blockIdx.x*blockDim.x + threadIdx.x;

  if(global_index < _NVALLEYS*NSBN*NX*NW*NPHI)
    {
      int nu,p,i,l,m;
      GPU_map_1D_to_5D( global_index, &i, NX, &nu, _NVALLEYS, &p, NSBN, &l, NW, &m, NPHI );
      GPU_a3( nu,p,i,l,m ) = GPU_deps_dx( nu,p,i )*GPU_a3_const(nu,l,m);
    }
}

__global__ void cuda_set_fluxes_a2_testversion( const discrMeshes *dm, double *_GPU_a2, const double *_GPU_deps_dx, const double *_GPU_vel )
{
  const int NSBN     = dm -> get_NSBN();
  const int NX       = dm -> get_X()   -> get_N();
  const int NW       = dm -> get_W()   -> get_N();
  const int NPHI     = dm -> get_PHI() -> get_N();

  int global_index = blockIdx.x*blockDim.x + threadIdx.x;

  if(global_index < _NVALLEYS*NSBN*NX*NW*NPHI)
    {
      int nu,p,i,l,m;
      GPU_map_1D_to_5D( global_index, &i, NX, &nu, _NVALLEYS, &p, NSBN, &l, NW, &m, NPHI );

      GPU_a2( nu, p, i, l, m ) = -GPU_deps_dx( nu,p,i )*GPU_vel(Xdim, nu, l, m);
    }
}

#else
void MOSFETProblemCuda::CPU_set_fluxes_a3()
{
  const double DW = host_dm->get_W()->get_delta();
  
  const int NSBN  = host_dm -> get_NSBN();
  const int NX    = host_dm -> get_X()   -> get_N();
  const int NW    = host_dm -> get_W()   -> get_N();
  const int NPHI  = host_dm -> get_PHI() -> get_N();

#pragma omp parallel for
  for(int global_index=0; global_index < _NVALLEYS*NSBN*NX*NW*NPHI; ++global_index)
    {
      int nu,p,i,l,m;
      GPU_map_1D_to_5D( global_index, &i, NX, &nu, _NVALLEYS, &p, NSBN, &l, NW, &m, NPHI );

      a3( nu,p,i,l,m ) = deps_dx( nu,p,i )*_sinphi[m]/( sqrtmass(Xdim,nu)*sqrt(2.*(l+0.5)*DW*(1.+host_rescpar->get_epsstar()*_kane[nu]*(l+0.5)*DW)) );
    }

  return;
}

#endif

/**
   PURPOSE:        

   FILE:           cuda_bte_rk.cu

   NAME:           MOSFETProblem::set_fluxes_a3

   PARAMETERS:

   RETURN VALUE:   none

   CALLS TO:       MOSFETProblem::start_time           (cuda_comptime.h - declared inline)
		   MOSFETProblem::stop_time            (cuda_comptime.h - declared inline)
                   MOSFETProblem::show_eta             (cuda_testing.h - declared inline)

   CALLED FROM:    MOSFETProblem::perform_step_2       (cuda_time_integration.cu)

   DATA MODIFIED:  

   METHOD:         

   LAST UPDATE:    2022 February 16
*/
void MOSFETProblemCuda::set_fluxes_a3()
{
#ifdef _SHOW_ETA
  ostringstream message;
  message << "called " << __func__;
  show_eta(message.str());
#endif

  start_time( _PHASE_BTE_SETFLUXES );

#ifdef __CUDACC__
  const int NSBN  = host_dm -> get_NSBN();
  const int NX    = host_dm -> get_X()   -> get_N();
  const int NW    = host_dm -> get_W()   -> get_N();
  const int NPHI  = host_dm -> get_PHI() -> get_N();
  
  checkCudaErrors( cudaMemcpyToSymbol(_GPU_sqrtmass,  _sqrtmass,     9*sizeof(double), 0, cudaMemcpyHostToDevice) );
  checkCudaErrors( cudaMemcpyToSymbol(_GPU_kane,      _kane,         3*sizeof(double), 0, cudaMemcpyHostToDevice) );
  // delete from here
  checkCudaErrors( cudaMemcpyToSymbol(_GPU_mass,  _mass,     9*sizeof(double), 0, cudaMemcpyHostToDevice) );
  // to here
  const int gridSize      = host_gridconfig -> cuda_set_fluxes_a3_config -> get_gridSize();
  const int blockSize     = host_gridconfig -> cuda_set_fluxes_a3_config -> get_blockSize();
  const int shmemSize     = host_gridconfig -> cuda_set_fluxes_a3_config -> get_shmemSize();
  const cudaFuncCache cfc = host_gridconfig -> cuda_set_fluxes_a3_config -> get_cfc();
  cudaDeviceSetCacheConfig( cfc );
  cuda_set_fluxes_a3 <<< gridSize, blockSize, shmemSize >>> ( device_dm, _GPU_a3, _GPU_deps_dx, _GPU_a3_const );
#else
  CPU_set_fluxes_a3();
#endif
  
  _max_a3 = _max_a3const * _max_deps_dx;
  
  stop_time( _PHASE_BTE_SETFLUXES );

#ifdef _SHOW_ETA
  cerr << "[ok] (function " << __func__ << ", line " << __LINE__ << ")" << endl;
#endif
  return;
}




/**
   PURPOSE:        

   FILE:           cuda_bte_rk.cu

   NAME:           MOSFETProblem::set_fluxes_a2

   PARAMETERS:

   RETURN VALUE:   none

   CALLS TO:       MOSFETProblem::start_time           (cuda_comptime.h - declared inline)
		   MOSFETProblem::stop_time            (cuda_comptime.h - declared inline)
                   MOSFETProblem::show_eta             (cuda_testing.h - declared inline)

   CALLED FROM:    MOSFETProblem::perform_step_2       (cuda_time_integration.cu)

   DATA MODIFIED:  

   METHOD:         

   LAST UPDATE:    2022 February 16
*/
void MOSFETProblemCuda::set_fluxes_a2()
{
#ifdef _SHOW_ETA
  ostringstream message;
  message << "called " << __func__;
  show_eta(message.str());
#endif
  start_time( _PHASE_BTE_SETFLUXES );

#ifdef __CUDACC__
  const int NSBN  = host_dm -> get_NSBN();
  const int NX    = host_dm -> get_X()   -> get_N();
  const int NW    = host_dm -> get_W()   -> get_N();
  const int NPHI  = host_dm -> get_PHI() -> get_N();

  checkCudaErrors( cudaMemcpyToSymbol(_GPU_sqrtmass,  _sqrtmass,     9*sizeof(double), 0, cudaMemcpyHostToDevice) );
  checkCudaErrors( cudaMemcpyToSymbol(_GPU_kane,      _kane,         3*sizeof(double), 0, cudaMemcpyHostToDevice) );
  // delete from here
  checkCudaErrors( cudaMemcpyToSymbol(_GPU_mass,  _mass,     9*sizeof(double), 0, cudaMemcpyHostToDevice) );
  // to here
  const int gridSize      = host_gridconfig -> cuda_set_fluxes_a2_testversion_config -> get_gridSize();
  const int blockSize     = host_gridconfig -> cuda_set_fluxes_a2_testversion_config -> get_blockSize();
  const int shmemSize     = host_gridconfig -> cuda_set_fluxes_a2_testversion_config -> get_shmemSize();
  const cudaFuncCache cfc = host_gridconfig -> cuda_set_fluxes_a2_testversion_config -> get_cfc();
  cudaDeviceSetCacheConfig( cfc );
  cuda_set_fluxes_a2_testversion <<< gridSize, blockSize, shmemSize >>> ( device_dm, _GPU_a2, _GPU_deps_dx, _GPU_vel );
#else
  // CPU_set_fluxes_a2();
  /**
     NOTA: el código en CPU no necesita el cálculo del flujo a2. De hecho, ni siquiera
     existe una estructura para guardar estos valores. El WENO_W hace los cálculos
     del flujo por su cuenta y ya está.
  */
#endif

  stop_time( _PHASE_BTE_SETFLUXES );

#ifdef _SHOW_ETA
  cerr << "[ok] (function " << __func__ << ", line " << __LINE__ << ")" << endl;
#endif
  return;
}




cuda_bte_rk_kernels_config::cuda_bte_rk_kernels_config( const discrMeshes *dm )
{
  const int NSBN  = dm -> get_NSBN();
  const int NX    = dm -> get_X()   -> get_N();
  const int NW    = dm -> get_W()   -> get_N();
  const int NPHI  = dm -> get_PHI() -> get_N();
  
  const int RK_TPB = 128;
  cuda_perform_RK_1_3_config            = new kernelConfig( nblocks( _NVALLEYS*NSBN*NX*NW*NPHI, RK_TPB ),     RK_TPB,     NOSHMEM, cudaFuncCachePreferNone, "cuda_perform_RK_1_3"            );
  cuda_perform_RK_2_3_config            = new kernelConfig( nblocks( _NVALLEYS*NSBN*NX*NW*NPHI, RK_TPB ),     RK_TPB,     NOSHMEM, cudaFuncCachePreferL1  , "cuda_perform_RK_2_3"            );
  cuda_perform_RK_3_3_config            = new kernelConfig( nblocks( _NVALLEYS*NSBN*NX*NW*NPHI, RK_TPB ),     RK_TPB,     NOSHMEM, cudaFuncCachePreferL1  , "cuda_perform_RK_3_3"            );
  cuda_set_rhs_to_zero_config           = new kernelConfig( nblocks( _NVALLEYS*NSBN*NX*NW*NPHI, RK_TPB ),     RK_TPB,     NOSHMEM, cudaFuncCachePreferL1  , "cuda_set_rhs_to_zero"           );

  const int fluxes_TPB = 128;
  cuda_set_fluxes_a3_config             = new kernelConfig( nblocks( _NVALLEYS*NSBN*NX*NW*NPHI, fluxes_TPB ), fluxes_TPB, NOSHMEM, cudaFuncCachePreferL1  , "cuda_set_fluxes_a3_config"      );
  cuda_set_fluxes_a2_testversion_config = new kernelConfig( nblocks( _NVALLEYS*NSBN*NX*NW*NPHI, fluxes_TPB ), fluxes_TPB, NOSHMEM, cudaFuncCachePreferL1  , "cuda_set_fluxes_a2_testversion" );
}
