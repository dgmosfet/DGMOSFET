#include "mosfetproblem.h"
#include "cuda_reductions_kernels.h"
#include "debug_flags.h"

/*
  name          : 'eigenstates'
  last modified : 2023/06/06
  author        : Francesco VECIL

  data flux     : This function has the following data flow
                     spphase [parameter], surfdens, pot[spphase] ---> eps[spphase], chi[spphase]

  notes         : (i) 
 */
// #define   _CHECK_EIGENSTATES_INPUT_OUTPUT
void MOSFETProblemCuda::eigenstates(const SPPhase spphase, const double eigvals_tolpar, const double eigvecs_tolpar)
{
#ifdef   _CHECK_EIGENSTATES_INPUT_OUTPUT
#warning   _CHECK_EIGENSTATES_INPUT_OUTPUT is activated
  check_eigenstates_input(spphase);
#endif
  
  if(_ep == POTENTIAL) 
    start_time( _PHASE_ITER_EIGENSTATES );

  /**
     For debugging purposes, a #define is introduced, allowing the code to revert to 
     computations on CPU to check whether the Cuda implementation fails.
     If the following #define is commented, the code resumes normal funcioning.
  */
  // #define _EIGENSTATES_ON_CPU

  // EXECUTION ON GPU
#ifdef __CUDACC__

#ifdef _EIGENSTATES_ON_CPU
  if( spphase == OLD )
    {
      CUDA_SAFE_CALL( cudaMemcpy( _pot_OLD, _GPU_pot_OLD, _POT_SIZE*sizeof(double), cudaMemcpyDeviceToHost ) );
    }
  else if( spphase == NEW )
    {
      CUDA_SAFE_CALL( cudaMemcpy( _pot, _GPU_pot, _POT_SIZE*sizeof(double), cudaMemcpyDeviceToHost ) );
    }
  else
    {
      cerr << " ERROR from function " << __func__ << "!" << endl;
      exit(123);
    }
  CPU_eigenstates_LAPACK(spphase);
  CUDA_SAFE_CALL( cudaMemcpy( _GPU_eps, _eps, (_NVALLEYS*NSBN*NX)*sizeof(double), cudaMemcpyHostToDevice ) );
  CUDA_SAFE_CALL( cudaMemcpy( _GPU_chi, _chi, _CHI_SIZE*sizeof(double), cudaMemcpyHostToDevice ) );
#else
  GPU_eigenstates(spphase, eigvals_tolpar, eigvecs_tolpar);
#endif
  
  // EXECUTION ON CPU
#else
  
  CPU_eigenstates(spphase, eigvals_tolpar, eigvecs_tolpar);

#endif

  if(_ep == POTENTIAL) 
    stop_time( _PHASE_ITER_EIGENSTATES );

#ifdef   _CHECK_EIGENSTATES_INPUT_OUTPUT
  check_eigenstates_output();
#endif

  return;
}




#ifdef __CUDACC__
void MOSFETProblemCuda::GPU_eigenstates(const SPPhase spphase, const double eigvals_tolpar, const double eigvecs_tolpar)
{
  /*******************************
   *  PRELIMINARY COMPUTATIONS   *
   *******************************/
  GPU_prepare_eigenstates( spphase );

  /*******************************
   *        EIGENVALUES          *
   *******************************/
  GPU_eigenvalues(eigvals_tolpar);
  
  /*******************************
   *        EIGENVECTORS         *
   *******************************/
  GPU_eigenvectors_Thomas(eigvecs_tolpar);
  
  return;
}
#endif



void MOSFETProblemCuda::CPU_eigenstates(const SPPhase spphase, const double eigvals_tolpar, const double eigvecs_tolpar)
{
  // WORK IN PROGRESS
  // multisection + Thomas is unstable, both on CPU and on GPU
  
  if( _ep != POTENTIAL )
  {
    CPU_eigenstates_LAPACK(spphase);
  }
  else
  {
    CPU_prepare_eigenstates( spphase );
    if( _step <= 1 )
      CPU_eigenvalues_ms(eigvals_tolpar);
    else
      CPU_eigenvalues_NR(eigvals_tolpar);

    CPU_tridiag_Thomas(eigvecs_tolpar);
  }


  // // STABLE VERSION
  // CPU_eigenstates_LAPACK(spphase);
  // // END OF STABLE VERSION


  // // WORK IN PROGRESS: testing Thomas. Use the STABLE version to be safe.  
  // if( _ep != POTENTIAL )
  //   {
  //     CPU_eigenstates_LAPACK(spphase);
  //   }
  // else
  //   {
  //     CPU_eigenvalues_LAPACK(spphase);
  //     CPU_prepare_eigenstates( spphase );
  //     CPU_tridiag_solve_Thomas();
  //   }
    
  return;
}












void MOSFETProblemCuda::CPU_prepare_eigenstates(const SPPhase spphase)
{
  const int NX    = host_dm -> get_X()   -> get_N();
  const int NZ    = host_dm -> get_Z()   -> get_N();

  if(_ep == POTENTIAL) 
    start_time( _PHASE_ITER_EIGENSTATES_PREPARE );
  
  memcpy( ___A, ___A_const, _NVALLEYS*NX*_SCHROED_ROW_SIZE*sizeof(double) );

  switch( spphase )
    {
    case NEW :
      {
#pragma omp parallel for
	for(int global_index = 0; global_index < _NVALLEYS*NX*(NZ-2); ++global_index)
	  {
	    int _i, _nu, j;
	    GPU_map_1D_to_3D( global_index, &_i, NX, &_nu, _NVALLEYS, &j, NZ-2 );
	    const int start = (_i*_NVALLEYS + _nu)*_SCHROED_ROW_SIZE;
	    ___A[start+j] -= _pot[_i*NZ+(j+1)];
	  }
	break;
      }
    case OLD :
      {
#pragma omp parallel for
	for(int global_index = 0; global_index < _NVALLEYS*NX*(NZ-2); ++global_index)
	  {
	    int _i, _nu, j;
	    GPU_map_1D_to_3D( global_index, &_i, NX, &_nu, _NVALLEYS, &j, NZ-2 );
	    const int start = (_i*_NVALLEYS + _nu)*_SCHROED_ROW_SIZE;
	    ___A[start+j] -= _pot_OLD[_i*NZ+(j+1)];
	  }
	break;
      }
    }

  if(_ep == POTENTIAL) 
    stop_time( _PHASE_ITER_EIGENSTATES_PREPARE );

  return;
}

#ifdef __CUDACC__
#define CUDA_SAFE_CALL( call ) {					\
    cudaError_t err = call;						\
    if( cudaSuccess != err ) {						\
      fprintf(stderr,"CUDA: error occurred in cuda routine. Exiting...\n"); \
      exit(err);							\
    } }



// #define   _CHECK_GPU_PREPARE_EIGENSTATES_INPUT_OUTPUT
void MOSFETProblemCuda::GPU_prepare_eigenstates(const SPPhase spphase)
{
  const int NX    = host_dm -> get_X()   -> get_N();
  const int NZ    = host_dm -> get_Z()   -> get_N();

  if(_ep == POTENTIAL) 
    start_time( _PHASE_ITER_EIGENSTATES_PREPARE );

#ifdef   _CHECK_GPU_PREPARE_EIGENSTATES_INPUT_OUTPUT
#warning   _CHECK_GPU_PREPARE_EIGENSTATES_INPUT_OUTPUT is activated
  check_GPU_prepare_eigenstates_input(spphase);
#endif
  
  CUDA_SAFE_CALL( cudaMemcpy( d_A, _GPU___A_const, _NVALLEYS*NX*_SCHROED_ROW_SIZE*sizeof(double), cudaMemcpyDeviceToDevice ) );

  // int blockdim = 32;
  // int blocks = nblocks( _NVALLEYS*NX*(NZ-2), blockdim );
  // cudaDeviceSetCacheConfig(cudaFuncCachePreferNone);
  const int gridSize      = host_gridconfig -> cuda_init_d_A_config -> get_gridSize();
  const int blockSize     = host_gridconfig -> cuda_init_d_A_config -> get_blockSize();
  const int shmemSize     = host_gridconfig -> cuda_init_d_A_config -> get_shmemSize();
  const cudaFuncCache cfc = host_gridconfig -> cuda_init_d_A_config -> get_cfc();
  cudaDeviceSetCacheConfig( cfc );
  switch( spphase )
    {
    case NEW :
      {
	cuda_init_d_A <<< gridSize, blockSize, shmemSize >>> ( device_dm, d_A, _GPU_pot );
	break;
      }
    case OLD :
      {
	cuda_init_d_A <<< gridSize, blockSize, shmemSize >>> ( device_dm, d_A, _GPU_pot_OLD );
	break;
      }
    }

  if(_ep == POTENTIAL) 
    stop_time( _PHASE_ITER_EIGENSTATES_PREPARE );

#ifdef   _CHECK_GPU_PREPARE_EIGENSTATES_INPUT_OUTPUT
  check_GPU_prepare_eigenstates_output();
#endif

  return;
}
#endif

#ifdef __CUDACC__
__global__ void cuda_init_d_A( const discrMeshes *dm, double *d_A, const double *aux )
{
  int global_index = blockIdx.x*blockDim.x + threadIdx.x;

  const int NX       = dm -> get_X() -> get_N();
  const int NZ       = dm -> get_X() -> get_N();

  if(global_index < _NVALLEYS*NX*(NZ-2))
    {
      int _i, _nu, j;
      GPU_map_1D_to_3D( global_index, &_i, NX, &_nu, _NVALLEYS, &j, NZ-2 );
      const int start = (_i*_NVALLEYS + _nu)*_SCHROED_ROW_SIZE;
      d_A[start+j] -= aux[_i*NZ+(j+1)];
    }
}
#endif

void MOSFETProblemCuda::CPU_eigenstates_LAPACK(const SPPhase spphase)
{
  const int NSBN  = host_dm -> get_NSBN();
  const int NX    = host_dm -> get_X()   -> get_N();
  const int NZ    = host_dm -> get_Z()   -> get_N();

#pragma omp parallel for
  for(int ind=0; ind<NX*_NVALLEYS; ++ind)
    {
      int i  = ind/_NVALLEYS;
      int nu = ind - i*_NVALLEYS;

      double _D   [NZ-2]; 
      double _E   [NZ-2];
      double _W   [NZ-2];
      double _Z   [(NZ-2)*(NZ-2)];
      int _ISUPPZ [2*(NZ-2)];
      double _WORK[20*(NZ-2)]; 
      int _IWORK  [10*(NZ-2)];

      char _JOBZ     = 'V'; // compute eigenvalues AND eigenvectors
      char _RANGE    = 'I'; // compute eigvalues and eigvectors in a given range
      int _N         = NZ-2; 
      double _VL; 
      double _VU; 
      int _IL        = 1; // lower bound for eig*
      int _IU        = NSBN; // upper bound for eig*
      double _ABSTOL = 0; // accuracy for the computation of the eigenvalues
      int _M; 
      int _LDZ       = _N;
      int _LWORK     = 20*_N; 
      int _LIWORK    = 10*_N; 
      int _INFO;

      // double DZM2 = 1./(_dz*_dz);
      // double DZ = _dz;
      double DZM2 = host_dm->get_Z()->get_delta_m2();
      double DZ   = host_dm->get_Z()->get_delta();

      // construction of the matrix representing the Schroedinger operator
      for(int j=1; j<NZ-1; ++j)
	_D[j-1]  = 0.5*host_adimpar->get_cs1()*DZM2*(.5/effmass(Zdim,nu,i,j-1)+1./effmass(Zdim,nu,i,j)+.5/effmass(Zdim,nu,i,j+1));

      switch( spphase )
	{
	case OLD :
	  {
	    for(int j=1; j<NZ-1; ++j)
	      _D[j-1] -= (pot_OLD(i,j)+vconf(i,j));
	    break;
	  }
	case NEW :
	  {
	    for(int j=1; j<NZ-1; ++j)
	      _D[j-1] -= (pot(i,j)+vconf(i,j));
	    break;
	  }
	}

      for(int j=1; j<_N; ++j)
	_E[j-1]  = -0.5*host_adimpar->get_cs1()*DZM2*(.5/effmass(Zdim,nu,i,j) + .5/effmass(Zdim,nu,i,j+1));

      dstegr_(&_JOBZ, &_RANGE, &_N, _D, _E, &_VL, &_VU, &_IL, &_IU, &_ABSTOL, &_M, _W, _Z, &_LDZ, _ISUPPZ, _WORK, &_LWORK, _IWORK, &_LIWORK, &_INFO);

      //normalize the eigenvectors
      for(int p=0; p<NSBN; ++p)
      	{
      	  double aux = 0;
      	  for(int j=0; j<NZ-2; ++j)
      	    aux += _Z[p*_N+j]*_Z[p*_N+j];
      	  aux *= DZ;
      	  aux=sqrt(aux); 
      	  // aux = _Z[p*_N]/(fabs(_Z[p*_N])*aux);
      	  aux = 1./aux;
      	  for(int j=0; j<NZ-2; ++j)
      	    _Z[p*_N+j] *= aux;
      	}

      /**********************************
       * STORE THE RESULT               *
       **********************************/
      memcpy( &eps(nu,0,i), _W, NSBN*sizeof(double) );
      for(int p=0; p<NSBN; ++p)
	memcpy( &chi(nu,p,i,1), &_Z[p*_N], _N*sizeof(double) );
    }

  return;
}


void MOSFETProblemCuda::CPU_eigenvalues_LAPACK(const SPPhase spphase)
{
  const int NSBN  = host_dm -> get_NSBN();
  const int NX    = host_dm -> get_X()   -> get_N();
  const int NZ    = host_dm -> get_Z()   -> get_N();

#pragma omp parallel for
  for(int ind=0; ind<NX*_NVALLEYS; ++ind)
    {
      int i  = ind/_NVALLEYS;
      int nu = ind - i*_NVALLEYS;

      double _D   [NZ-2]; 
      double _E   [NZ-2];
      double _W   [NZ-2];
      double _Z   [(NZ-2)*(NZ-2)];
      int _ISUPPZ [2*(NZ-2)];
      double _WORK[20*(NZ-2)]; 
      int _IWORK  [10*(NZ-2)];

      char _JOBZ     = 'V'; // compute eigenvalues AND eigenvectors
      char _RANGE    = 'I'; // compute eigvalues and eigvectors in a given range
      int _N         = NZ-2; 
      double _VL; 
      double _VU; 
      int _IL        = 1; // lower bound for eig*
      int _IU        = NSBN; // upper bound for eig*
      double _ABSTOL = 0; // accuracy for the computation of the eigenvalues
      int _M; 
      int _LDZ       = _N;
      int _LWORK     = 20*_N; 
      int _LIWORK    = 10*_N; 
      int _INFO;

      // double DZM2 = 1./(_dz*_dz);
      // double DZ = _dz;
      double DZM2 = host_dm->get_Z()->get_delta_m2();
      double DZ   = host_dm->get_Z()->get_delta();

      // construction of the matrix representing the Schroedinger operator
      for(int j=1; j<NZ-1; ++j)
	_D[j-1]  = 0.5*host_adimpar->get_cs1()*DZM2*(.5/effmass(Zdim,nu,i,j-1)+1./effmass(Zdim,nu,i,j)+.5/effmass(Zdim,nu,i,j+1));

      switch( spphase )
	{
	case OLD :
	  {
	    for(int j=1; j<NZ-1; ++j)
	      _D[j-1] -= (pot_OLD(i,j)+vconf(i,j));
	    break;
	  }
	case NEW :
	  {
	    for(int j=1; j<NZ-1; ++j)
	      _D[j-1] -= (pot(i,j)+vconf(i,j));
	    break;
	  }
	}

      for(int j=1; j<_N; ++j)
	_E[j-1]  = -0.5*host_adimpar->get_cs1()*DZM2*(.5/effmass(Zdim,nu,i,j) + .5/effmass(Zdim,nu,i,j+1));

      dstegr_(&_JOBZ, &_RANGE, &_N, _D, _E, &_VL, &_VU, &_IL, &_IU, &_ABSTOL, &_M, _W, _Z, &_LDZ, _ISUPPZ, _WORK, &_LWORK, _IWORK, &_LIWORK, &_INFO);

      // //normalize the eigenvectors
      // for(int p=0; p<_NSBN; ++p)
      // 	{
      // 	  double aux = 0;
      // 	  for(int j=0; j<_NZ-2; ++j)
      // 	    aux += _Z[p*_N+j]*_Z[p*_N+j];
      // 	  aux *= DZ;
      // 	  aux=sqrt(aux); 
      // 	  // aux = _Z[p*_N]/(fabs(_Z[p*_N])*aux);
      // 	  aux = 1./aux;
      // 	  for(int j=0; j<_NZ-2; ++j)
      // 	    _Z[p*_N+j] *= aux;
      // 	}

      /**********************************
       * STORE THE RESULT               *
       **********************************/
      memcpy( &eps(nu,0,i), _W, NSBN*sizeof(double) );
      // for(int p=0; p<_NSBN; ++p)
      // 	memcpy( &chi(nu,p,i,1), &_Z[p*_N], _N*sizeof(double) );
    }

  return;
}


cuda_iter_eigenstates_kernels_config::cuda_iter_eigenstates_kernels_config(const discrMeshes *dm)
{
  const int NX    = dm -> get_X()   -> get_N();
  const int NZ    = dm -> get_Z()   -> get_N();
  
  int blockdim = 32;
  int blocks = nblocks( _NVALLEYS*NX*(NZ-2), blockdim );
  cuda_init_d_A_config = new kernelConfig( blocks, blockdim, NOSHMEM, cudaFuncCachePreferNone, "cuda_init_d_A" );
}



/*
  name          : 'check_eigenstates_input'
  last modified : 2023/06/06
  author        : Francesco VECIL

  description   : This functions check the input of the 'eigenstates' method.
                  Namely, (i) surfdens, (ii) pot[spphase].
 */
void MOSFETProblemCuda::check_eigenstates_input(const SPPhase spphase)
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
  if( spphase == OLD )
    {
#ifdef __CUDACC__
      CUDA_SAFE_CALL( cudaMemcpy( _aux_pot, _GPU_pot_OLD, NX*NZ*sizeof(double), cudaMemcpyDeviceToHost ) );
#else
      memcpy( _aux_pot, _pot_OLD, NX*NZ*sizeof(double) );
#endif
    }
  else if( spphase == NEW )
    {
#ifdef __CUDACC__
      CUDA_SAFE_CALL( cudaMemcpy( _aux_pot, _GPU_pot, NX*NZ*sizeof(double), cudaMemcpyDeviceToHost ) );
#else
      memcpy( _aux_pot, _pot, NX*NZ*sizeof(double) );
#endif
    }
  else
    {
      cerr << " ERROR from function " << __func__ << ", from line " << __LINE__ << ", from file '" << __FILE__ << "'!" << endl;
      exit(_ERR_CHECK_EIGENSTATES_INPUT);
    }

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
      exit(_ERR_CHECK_EIGENSTATES_INPUT);
    }
  
  cerr << "[ok]" << endl;

  return;
}





/*
  name          : 'check_eigenstates_output'
  last modified : 2023/06/06
  author        : Francesco VECIL

  description   : This functions check the output of the 'eigenstates' method.
                  Namely, (iii) eps, (iv) chi.
 */
void MOSFETProblemCuda::check_eigenstates_output()
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
      exit(_ERR_CHECK_EIGENSTATES_INPUT);
    }
  
  cerr << "[ok]" << endl;

  return;
}




/*
  name          : 'check_GPU_prepare_eigenstates_input'
  last modified : 2023/06/06
  author        : Francesco VECIL

  description   : This functions check the input and the output of the 'GPU_prepare_eigenstates' method.
                  Namely, 'check_GPU_prepare_eigenstates_input' checks (i) pot[spphase].
 */
void MOSFETProblemCuda::check_GPU_prepare_eigenstates_input(const SPPhase spphase)
{
  cerr << "called '" << __func__ << "'...";

  const int NX    = host_dm -> get_X()   -> get_N();
  const int NZ    = host_dm -> get_Z()   -> get_N();

  bool stopcheck = false;

  /*
    begin section : Check pot[spphase]
   */
  double *_aux_pot = new double[ NX*NZ ];
  if( spphase == OLD )
    {
#ifdef __CUDACC__
      CUDA_SAFE_CALL( cudaMemcpy( _aux_pot, _GPU_pot_OLD, NX*NZ*sizeof(double), cudaMemcpyDeviceToHost ) );
#else
      memcpy( _aux_pot, _pot_OLD, NX*NZ*sizeof(double) );
#endif
    }
  else if( spphase == NEW )
    {
#ifdef __CUDACC__
      CUDA_SAFE_CALL( cudaMemcpy( _aux_pot, _GPU_pot, NX*NZ*sizeof(double), cudaMemcpyDeviceToHost ) );
#else
      memcpy( _aux_pot, _pot, NX*NZ*sizeof(double) );
#endif
    }
  else
    {
      cerr << " ERROR from function " << __func__ << ", from line " << __LINE__ << ", from file '" << __FILE__ << "'!" << endl;
      exit(_ERR_CHECK_GPU_PREPARE_EIGENSTATES_INPUT);
    }

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
  /*
    end section : Check pot[spphase]
   */

  if( stopcheck == true )
    {
      cerr << " ERROR from function " << __func__ << ", from line " << __LINE__ << ", from file '" << __FILE__
	   << "' --- exiting..." << endl;
      exit(_ERR_CHECK_GPU_PREPARE_EIGENSTATES_INPUT);
    }

  cerr << "[ok]" << endl;
  
  return;
}



/*
  name          : 'check_GPU_prepare_eigenstates_output'
  last modified : 2023/06/06
  author        : Francesco VECIL

  description   : This functions check the input and the output of the 'GPU_prepare_eigenstates' method.
                  Namely, 'check_GPU_prepare_eigenstates_output' checks (ii) matrix d_A.
 */
void MOSFETProblemCuda::check_GPU_prepare_eigenstates_output()
{
  cerr << "called '" << __func__ << "'...";

  const int NX    = host_dm -> get_X()   -> get_N();
  const int NZ    = host_dm -> get_Z()   -> get_N();

  bool stopcheck = false;

  double *_aux_d_A = new double[_NVALLEYS*NX*_SCHROED_ROW_SIZE];

  for(int idx=0; idx<_NVALLEYS*NX*_SCHROED_ROW_SIZE; ++idx)
    {
      const double val = _aux_d_A[idx];
      if( isnan( val ) || isinf( val ) )
	{
	  cerr << " ERROR from function " << __func__ << ", from line " << __LINE__ << ", from file '" << __FILE__
	       << "' --- d_A[" << idx << "] = " << val << endl;
	  stopcheck = true;
	}
    }
  
  delete [] _aux_d_A;

  if( stopcheck == true )
    {
      cerr << " ERROR from function " << __func__ << ", from line " << __LINE__ << ", from file '" << __FILE__
	   << "' --- exiting..." << endl;
      exit(_ERR_CHECK_GPU_PREPARE_EIGENSTATES_INPUT);
    }

  cerr << "[ok]" << endl;

  return;
}






