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






#ifdef __CUDACC__
__global__ void cuda_compute_eps_diff_m1( const discrMeshes *dm, double *_GPU_eps_diff_m1, const double *_GPU_eps )
{
  int global_index = blockIdx.x*blockDim.x + threadIdx.x;

  const int NSBN     = dm -> get_NSBN();
  const int NX       = dm -> get_X() -> get_N();

  if( global_index < _NVALLEYS*NSBN*NSBN*NX )
    {
      int nu, p, pp, i;
      GPU_map_1D_to_4D( global_index, &i, NX, &nu, _NVALLEYS, &p, NSBN, &pp, NSBN );

      GPU_eps_diff_m1(nu,p,pp,i) = 1./(GPU_eps(nu,pp,i) - GPU_eps(nu,p,i));
    }

}

__global__ void cuda_compute_frechet( const discrMeshes *dm, double *_GPU_frechet, const double *_GPU_chi, const double *_GPU_surfdens, const double *_GPU_eps_diff_m1 )
{
  int global_index = blockIdx.x*blockDim.x + threadIdx.x;

  const int NSBN     = dm -> get_NSBN();
  const int NX       = dm -> get_X() -> get_N();
  const int NZ       = dm -> get_Z() -> get_N();

  if( global_index < NX*(NZ-2)*(NZ-2) )
    {
      int i, j, jj;
      GPU_map_1D_to_3D( global_index, &i, NX, &j, NZ-2, &jj, NZ-2 );

      ++j; ++jj;

      /*
	begin comment: LocMem_frechet
	date: 20230502
	author: Francesco Vecil

	For the need of statically-allocayed memory inside the kernel, 
	a parameter _NSBNMAX (together with other maximum numbers of points)
	has been introduced in 'cuda_solverparams.h'.
	The solverParams constructors will check on these values,
	and this kernel uses this value as a bound.
	For optimal memory usage, _NSBNMAX should be set to the actual
	value used in the simulations, and the code recompiled.
       */
      // double aux_chi_j [ _NVALLEYS * _NSBNMAX ];
      // double aux_chi_jj[ _NVALLEYS * _NSBNMAX ];
      /*
	end comment: LocMem_frechet
       */
      
      // for(int NU=0; NU<_NVALLEYS; ++NU)
      // 	for(int P=0; P<NSBN; ++P)
      // 	  {
      // 	    aux_chi_j[NU*NSBN + P] = GPU_chi(NU,P,i,j);
      // 	    aux_chi_jj[NU*NSBN + P] = GPU_chi(NU,P,i,jj);
      // 	  }
      
      double sum = 0;

      for(int nu=0; nu<_NVALLEYS; ++nu)
	for(int p=0; p<NSBN; ++p)
	  {
	    // const double aux = aux_chi_jj[nu*NSBN+p]*aux_chi_j[nu*NSBN+p];
	    const double aux = GPU_chi(nu,p,i,jj) * GPU_chi(nu,p,i,j);

	    for(int pp=0; pp<NSBN; ++pp)
	      {
		// sum += ( pp == p ? 0 : (GPU_surfdens(nu,p,i)-GPU_surfdens(nu,pp,i))*GPU_eps_diff_m1(nu,p,pp,i)*aux*aux_chi_jj[nu*NSBN+pp]*aux_chi_j[nu*NSBN+pp] );
		sum += ( pp == p ? 0 : (GPU_surfdens(nu,p,i)-GPU_surfdens(nu,pp,i))*GPU_eps_diff_m1(nu,p,pp,i)*aux*GPU_chi(nu,pp,i,jj)*GPU_chi(nu,pp,i,j) );
	      }
	  }
	    
      GPU_frechet(i,j,jj) = 2.*sum;
    }

}
#endif


void MOSFETProblemCuda::CPU_compute_frechet() // chi, surfdens, eps ---> frechet
{
  const int NSBN  = host_dm -> get_NSBN();
  const int NX    = host_dm -> get_X()   -> get_N();
  const int NZ    = host_dm -> get_Z()   -> get_N();
  const ContactMaterial CONTACTMATERIAL = host_physdesc->get_CONTACTMATERIAL();
  
  switch( _ep )
    {
    case INITCOND :
      {
	double fermilevel = 0;
	if( CONTACTMATERIAL == ALLUMINIUM )
	  fermilevel = host_phyco->__workfuncAl-host_phyco->__affinitySi;
	else if( CONTACTMATERIAL == TUNGSTEN )
	  fermilevel = host_phyco->__workfuncW-host_phyco->__affinitySi;
	else
	  {
	    // cerr << " Unknown contact material" << endl;
	    throw error_ITER_SPSTEP_FRECHET_MATERIAL();
	  }
	
#pragma omp parallel for
	for( int global_index=0; global_index<NX*(NZ-2)*(NZ-2); ++global_index )
	  {
	    int i, j, jj;
	    GPU_map_1D_to_3D( global_index, &i, NX, &j, NZ-2, &jj, NZ-2 );
	    ++j; ++jj;

	    double sum = 0;

	    for(int nu=0; nu<_NVALLEYS; ++nu)
	      for(int p=0; p<NSBN; ++p)
		sum += 2.*M_1_PI*sqrt(mass(Xdim,nu)*mass(Ydim,nu)) * exp(fermilevel-eps(nu,p,i))/(1.+exp(fermilevel-eps(nu,p,i))) * chi(nu,p,i,j) * chi(nu,p,i,j) * chi(nu,p,i,jj) * chi(nu,p,i,jj);

	    for(int nu=0; nu<_NVALLEYS; ++nu)
	      for(int p=0; p<NSBN; ++p)
		for(int pp=0; pp<NSBN; ++pp)
		  sum -= 2.*( pp != p ? 2.*surfdens(nu,p,i)/(eps(nu,p,i) - eps(nu,pp,i))*chi(nu,p,i,jj)*chi(nu,pp,i,jj)*chi(nu,p,i,j)*chi(nu,pp,i,j) : 0.0 );

	    frechet(i,j,jj) = sum;
	  }

	break;
      }

    case THERMEQUIL :
      {
#pragma omp parallel for
	for(int global_index=0; global_index<NX*(NZ-2)*(NZ-2); ++global_index)
	  {
	    int i, j, jj;
	    GPU_map_1D_to_3D( global_index, &i, NX, &j, NZ-2, &jj, NZ-2 );

	    ++j; ++jj;

	    double sum = 0;

	    for(int nu=0; nu<_NVALLEYS; ++nu)
	      for(int p=0; p<NSBN; ++p)
		sum += 2.*_intnd/_repartition_b*sqrt(mass(Xdim,nu)*mass(Ydim,nu))*exp(-eps(nu,p,i))*chi(nu,p,i,j)*chi(nu,p,i,j)*chi(nu,p,i,jj)*chi(nu,p,i,jj);

	    for(int nu=0; nu<_NVALLEYS; ++nu)
	      for(int p=0; p<NSBN; ++p)
		for(int pp=0; pp<NSBN; ++pp)
		  sum += 2.*( pp != p ? 2.*surfdens(nu,p,i)/(eps(nu,pp,i) - eps(nu,p,i))*chi(nu,p,i,jj)*chi(nu,pp,i,jj)*chi(nu,p,i,j)*chi(nu,pp,i,j) : 0.0 );

	    frechet(i,j,jj) = sum;
	  }
	break;
      }
    
    case POTENTIAL :
      {
#pragma omp parallel for
	for(int global_index=0; global_index<NX*(NZ-2)*(NZ-2); ++global_index)
	  {
	    int i, j, jj;
	    GPU_map_1D_to_3D( global_index, &i, NX, &j, NZ-2, &jj, NZ-2 );

	    ++j; ++jj;

	    frechet(i,j,jj) = 0;
	
	    for(int p=0; p<NSBN; ++p)
	      for(int pp=0; pp<NSBN; ++pp)
		if( pp != p )
		  for(int nu=0; nu<_NVALLEYS; ++nu)
		    frechet(i,j,jj) += (surfdens(nu,p,i)-surfdens(nu,pp,i)) / (eps(nu,pp,i) - eps(nu,p,i)) * chi(nu,p,i,jj) * chi(nu,pp,i,jj) * chi(nu,p,i,j) * chi(nu,pp,i,j);

	    frechet(i,j,jj) *= 2.;
	  }

	break;
      }
    }

  return;
}




/**
   PURPOSE:        

   FILE:           cuda_iter_spstep_frechet.cu

   NAME:           MOSFETProblem::compute_frechet

   PARAMETERS:

   RETURN VALUE:   none

   CALLS TO:       MOSFETProblem::start_time           (cuda_comptime.h - declared inline)
		   MOSFETProblem::stop_time            (cuda_comptime.h - declared inline)

   CALLED FROM:    MOSFETProblem::iter_2               (cuda_iter.cu)

   DATA MODIFIED:  

   METHOD:         

   LAST UPDATE:    2022 February 16
*/

// #define _CHECK_COMPUTE_FRECHET_INPUT_OUTPUT
void MOSFETProblemCuda::compute_frechet()
  /********************************************************************
   * This function computes the Newton-Raphson frechet.
   * 
   * Input: eps, chi, surfdens
   * Output: frechet
   ********************************************************************/
{
#ifdef _CHECK_COMPUTE_FRECHET_INPUT_OUTPUT
#warning   _CHECK_COMPUTE_FRECHET_INPUT_OUTPUT is activated
  check_compute_frechet_input();
#endif

  if( _ep == POTENTIAL )
    start_time( _PHASE_ITER_SPSTEP_FRECHET );

#ifdef __CUDACC__
  if( _ep != POTENTIAL )
    {
      CPU_compute_frechet();
    }
  else
    {
      /**
	 For debugging purposes, a #define is introduced, allowing the code to revert to 
	 computations on CPU to check whether the Cuda implementation fails.
	 If the following #define is commented, the code resumes normal funcioning.
      */
// #define _FRECHET_ON_CPU

#ifdef _FRECHET_ON_CPU
      cerr << " frechet on CPU...";
      const int NSBN  = host_dm -> get_NSBN();
      const int NX    = host_dm -> get_X()   -> get_N();
      const int NZ    = host_dm -> get_Z()   -> get_N();
      CUDA_SAFE_CALL( cudaMemcpy(_eps,      _GPU_eps,      _NVALLEYS*NSBN*NX*sizeof(double),      cudaMemcpyHostToDevice) );
      CUDA_SAFE_CALL( cudaMemcpy(_chi,      _GPU_chi,      _NVALLEYS*NSBN*NX*_SCHROED_MATRIX_SIZE_PAD*sizeof(double),      cudaMemcpyHostToDevice) );
      CUDA_SAFE_CALL( cudaMemcpy(_surfdens, _GPU_surfdens, _NVALLEYS*NSBN*NX*sizeof(double), cudaMemcpyHostToDevice) );
      CPU_compute_frechet();
      CUDA_SAFE_CALL( cudaMemcpy(_GPU_frechet, _frechet, NX*NZ*NZ*sizeof(double), cudaMemcpyHostToDevice) );
#else
      GPU_compute_frechet();
#endif
    }
#else
  CPU_compute_frechet();  
#endif

  if( _ep == POTENTIAL )
    stop_time( _PHASE_ITER_SPSTEP_FRECHET );

#ifdef _CHECK_COMPUTE_FRECHET_INPUT_OUTPUT
  check_compute_frechet_output();
#endif

  return;
}




#ifdef __CUDACC__
void MOSFETProblemCuda::GPU_compute_frechet()
{
  const int NSBN  = host_dm -> get_NSBN();
  const int NX    = host_dm -> get_X()   -> get_N();
  const int NZ    = host_dm -> get_Z()   -> get_N();

  {
    const int gridSize      = host_gridconfig -> cuda_compute_eps_diff_m1_config -> get_gridSize();
    const int blockSize     = host_gridconfig -> cuda_compute_eps_diff_m1_config -> get_blockSize();
    const int shmemSize     = host_gridconfig -> cuda_compute_eps_diff_m1_config -> get_shmemSize();
    const cudaFuncCache cfc = host_gridconfig -> cuda_compute_eps_diff_m1_config -> get_cfc();
    cudaDeviceSetCacheConfig( cfc );
    cuda_compute_eps_diff_m1 <<< gridSize, blockSize, shmemSize >>> ( device_dm, _GPU_eps_diff_m1, _GPU_eps );
  }

  {
    const int gridSize      = host_gridconfig -> cuda_compute_frechet_config -> get_gridSize();
    const int blockSize     = host_gridconfig -> cuda_compute_frechet_config -> get_blockSize();
    const int shmemSize     = host_gridconfig -> cuda_compute_frechet_config -> get_shmemSize();
    const cudaFuncCache cfc = host_gridconfig -> cuda_compute_frechet_config -> get_cfc();
    cudaDeviceSetCacheConfig( cfc );
    cuda_compute_frechet <<< gridSize, blockSize, shmemSize >>> ( device_dm, _GPU_frechet, _GPU_chi, _GPU_surfdens, _GPU_eps_diff_m1 );
  }



  // delete from here
  /*
    This part is inserted for debugging purposes.
   */
  // CUDA_SAFE_CALL( cudaMemcpy( _eps, _GPU_eps, _NVALLEYS*NSBN*NX*sizeof(double), cudaMemcpyDeviceToHost ) );
  // CUDA_SAFE_CALL( cudaMemcpy( _aux_chi, _GPU_chi, _NVALLEYS*NSBN*NX*_SCHROED_MATRIX_SIZE_PAD*sizeof(double), cudaMemcpyDeviceToHost ) );


  // to here

  
  return;
}
#endif



cuda_iter_spstep_frechet_kernels_config::cuda_iter_spstep_frechet_kernels_config(const discrMeshes *dm)
{
  const int NSBN  = dm -> get_NSBN();
  const int NX    = dm -> get_X()   -> get_N();
  const int NZ    = dm -> get_Z()   -> get_N();

  int blockdim, number_of_blocks;
  blockdim = 64;
  number_of_blocks = nblocks( NX*(NZ-2)*(NZ-2), blockdim );
  cuda_compute_frechet_config     = new kernelConfig(number_of_blocks, blockdim, NOSHMEM, cudaFuncCachePreferL1, "cuda_compute_frechet"    );

  blockdim = 64;
  number_of_blocks = nblocks( _NVALLEYS*NSBN*NSBN*NX, blockdim );
  cuda_compute_eps_diff_m1_config = new kernelConfig(number_of_blocks, blockdim, NOSHMEM, cudaFuncCachePreferL1, "cuda_compute_eps_diff_m1");
}




/*
  name          : 'check_compute_frechet_input', 'check_compute_frechet_output'
  last modified : 2023/06/06
  author        : Francesco VECIL

  description   : This functions check the input and the output of the 'compute_frechet' method.
                  As input  : (i) eps, (ii) chi
 */
void MOSFETProblemCuda::check_compute_frechet_input()
{
  cerr << "called '" << __func__ << "'...";

  const int NSBN  = host_dm -> get_NSBN();
  const int NX    = host_dm -> get_X()   -> get_N();
  const int NZ    = host_dm -> get_Z()   -> get_N();

  bool stopcheck = false;
  
  /*
    begin section : Check eps
   */
  // cerr << "check eps...";
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
  // cerr << "[ok]";
  /*
    end section : Check eps
   */
  

  
  /*
    begin section : Check chi
   */
  // cerr << "check chi...";
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
  // cerr << "[ok]";
  /*
    end section : Check chi
   */

  if( stopcheck == true )
    {
      cerr << " ERROR from function " << __func__ << ", from line " << __LINE__ << ", from file '" << __FILE__
	   << "' --- exiting..." << endl;
      exit(_ERR_CHECK_COMPUTE_FRECHET_INPUT_OUTPUT);
    }
  
  cerr << "[ok]" << endl;

  return;
}



/*
  name          : 'check_compute_frechet_input', 'check_compute_frechet_output'
  last modified : 2023/06/06
  author        : Francesco VECIL

  description   : This functions check the input and the output of the 'compute_frechet' method.
		  As output : (iii) frechet
 */
void MOSFETProblemCuda::check_compute_frechet_output()
{
  cerr << "called '" << __func__ << "'...";

  const int NX    = host_dm -> get_X()   -> get_N();
  const int NZ    = host_dm -> get_Z()   -> get_N();

  bool stopcheck = false;
  
  double *_aux_frechet = new double[ NX*NZ*NZ ];

#ifdef __CUDACC__
  CUDA_SAFE_CALL( cudaMemcpy( _aux_frechet, _GPU_frechet, NX*NZ*NZ*sizeof(double), cudaMemcpyDeviceToHost ) );
#else
  memcpy( _aux_frechet, _frechet, NX*NZ*NZ*sizeof(double) );
#endif
  
  for(int i=0; i<NX; ++i)
    {
      for(int j=0; j<NZ; ++j)
	{
	  for(int jj=0; jj<NZ; ++jj)
	    {
	      const double val = _aux_frechet[ indx_i_j_jj(i,j,jj) ];
	      if( isnan( val ) || isinf( val ) )
		{
		  cerr << " ERROR from function " << __func__ << ", from line " << __LINE__ << ", from file '" << __FILE__
		       << "' --- frechet(" << i << "," << j << "," << jj << ") = " << val << endl;
		  stopcheck = true;
		}
	    }
	}
    }
  delete [] _aux_frechet;

  if( stopcheck == true )
    {
      cerr << " ERROR from function " << __func__ << ", from line " << __LINE__ << ", from file '" << __FILE__
	   << "' --- exiting..." << endl;
      exit(_ERR_CHECK_COMPUTE_FRECHET_INPUT_OUTPUT);
    }
  
  cerr << "[ok]" << endl;

  return;
}
