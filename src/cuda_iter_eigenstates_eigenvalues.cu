#include "mosfetproblem.h"
#include "cuda_reductions_kernels.h"
#include "debug_flags.h"


#define   _NMULTI   33    /* Parameter for the multi-section iterative algorithm. */


void MOSFETProblemCuda::CPU_gershgorin( const double *_A, double *_Y, double *_Z )
{
  const int NX    = host_dm->get_X()->get_N();
  const int NZ    = host_dm->get_Z()->get_N();

#pragma omp parallel for
  for(int global_index = 0; global_index < _NVALLEYS*NX; ++global_index)
    {
      const double *D = &_A[ global_index*_SCHROED_ROW_SIZE ]; 
      const double *S = D + _SCHROED_MATRIX_SIZE;

      /******************************************
       *            EIGENVALUES                 *
       ******************************************/
      /* Gershgorin circle */
      double Y = D[0] - fabs(S[0]);
      double Z = D[0] + fabs(S[0]);

      double a, b;
      for( int j = 1; j<_SCHROED_MATRIX_SIZE-1; j++ )
	{
	  a = D[j] - fabs(S[j]) - fabs(S[j-1]);
	  b = D[j] + fabs(S[j]) + fabs(S[j-1]);
	  Y = a < Y ? a : Y;
	  Z = b > Z ? b : Z;
	}
      a = D[_SCHROED_MATRIX_SIZE-1] - fabs(S[_SCHROED_MATRIX_SIZE-2]);
      b = D[_SCHROED_MATRIX_SIZE-1] + fabs(S[_SCHROED_MATRIX_SIZE-2]);
      Y = a < Y ? a : Y;
      Z = b > Z ? b : Z;

      _Y[global_index] = Y;
      _Z[global_index] = Z;
    }
}

void MOSFETProblemCuda::CPU_eigenvalues_ms(const double eigvals_tolpar)
{
  const int NSBN  = host_dm->get_NSBN();
  const int NX    = host_dm->get_X()->get_N();
  const int NZ    = host_dm->get_Z()->get_N();

  if(_ep == POTENTIAL) 
    start_time( _PHASE_ITER_EIGENSTATES_EIGENVALUES );

  double intlength;
  // Gershgorin circle
  double *_Y = new double[_NVALLEYS*NX];
  double *_Z = new double[_NVALLEYS*NX];
  CPU_gershgorin ( ___A, _Y, _Z );
  double min_Y = _Y[0]; double max_Z = _Z[0];
  for(int line=1; line<_NVALLEYS*NX; ++line)
    {
      min_Y = ( _Y[line] < min_Y ? _Y[line] : min_Y );
      max_Z = ( _Z[line] > max_Z ? _Z[line] : max_Z );
    }
  intlength = max_Z-min_Y;

  delete [] _Y;
  delete [] _Z;

  const double _EPS = eigvals_tolpar; // tolerance parameter
  if( _NMULTI != 33 )
    {
      cerr << " _NMULTI = 33 must be chosen!" << endl;
      throw error_ITER_EIGENSTATES_MULTISEC( _NMULTI );
    }
  const int number_of_iterations = int( log((intlength)/_EPS)/log(_NMULTI) ) + 1;
  const double step_ms = intlength/_NMULTI;
  const double nmultm1 = 1./_NMULTI;

#pragma omp parallel for
  for( int global_index = 0; global_index < _NVALLEYS*NSBN*NX; ++global_index )
    {
      int _i, _nu, _p;
      GPU_map_1D_to_3D( global_index, &_i, NX, &_nu, _NVALLEYS, &_p, NSBN );
      double step = step_ms;

      // initialize eps
      eps(_nu,_p,_i) = min_Y;

      const double *D = &___A[ (_i*_NVALLEYS + _nu)*_SCHROED_ROW_SIZE ];  // o el contrario??
      const double *S = D + _SCHROED_MATRIX_SIZE;

      /******************************************
       *            EIGENVALUES                 *
       ******************************************/
      double epsnupi = eps(_nu,_p,_i);
      double chsgn[ _NMULTI-1 ];
      for(int iter=0; iter<number_of_iterations; ++iter)
  	{
	  for(int _k=0; _k<_NMULTI-1; ++_k)
	    {
	      const double x = epsnupi + (_k+1)*step;

	      int neg_values = 0;
	      double _q = D[0] - x;
	      neg_values += ( _q < 0 );

	      for( int j = 2; j <= _SCHROED_MATRIX_SIZE; j++ )
		{
		  _q = (D[j-1] - x) - S[j-2]*S[j-2] / _q;
		  neg_values += ( _q < 0 );
		}
	      chsgn[_k] = ( neg_values < _p+1 ? _k : -1 );
	    }
	  
  	  int maximo = 0;
	  for(int _k=1; _k<_NMULTI-1; ++_k)
	    if(chsgn[_k] > chsgn[maximo])
	      maximo = _k;

	  epsnupi += ((double)chsgn[maximo]+1.)*step; // +1. o +1.5? O sea, cojo el borde izquierdo o el medio del intervalo? Yo pondria 1.5...

  	  step *= nmultm1;
  	}

      eps(_nu, _p, _i) = epsnupi;
    }

  if(_ep == POTENTIAL) 
    stop_time( _PHASE_ITER_EIGENSTATES_EIGENVALUES );

  return;
}


#ifdef __CUDACC__
#define CUDA_SAFE_CALL( call ) {                                         \
 cudaError_t err = call;                                                 \
 if( cudaSuccess != err ) {                                              \
   fprintf(stderr,"CUDA: error occurred in cuda routine. Exiting...\n"); \
   exit(err);                                                            \
 } }




// #define   _CHECK_GPU_EIGENVALUES_INPUT_OUTPUT
void MOSFETProblemCuda::GPU_eigenvalues(const double eigvals_tolpar)
{
#ifdef   _CHECK_GPU_EIGENVALUES_INPUT_OUTPUT
#warning   _CHECK_GPU_EIGENVALUES_INPUT_OUTPUT is activated
  check_GPU_eigenvalues_input();
#endif

  if( _step <= 1 )
    GPU_eigenvalues_ms(eigvals_tolpar);
  else
    GPU_eigenvalues_NR(eigvals_tolpar);

#ifdef _ITER_DEBUG
  const int NSBN  = host_dm -> get_NSBN();
  const int NX    = host_dm -> get_X()   -> get_N();

  if( are_there_nan_inf( _GPU_eps, _NVALLEYS * NSBN * NX, "testing eigenvalues" ) == false )
    {
      cerr << " From function " << __func__ << " : a problem has been detected!" << endl;
      exit(666);
    }
#endif
  
#ifdef   _CHECK_GPU_EIGENVALUES_INPUT_OUTPUT
  check_GPU_eigenvalues_output();
#endif

  return;
}

void MOSFETProblemCuda::GPU_eigenvalues_NR(const double eigvals_tolpar)
{
  const int NSBN  = host_dm -> get_NSBN();
  const int NX    = host_dm -> get_X()   -> get_N();

  const double EPSTOL = eigvals_tolpar;
  double *_eps_k = new double[ 2*_NVALLEYS*NSBN*NX ];
  CUDA_SAFE_CALL( cudaMemcpy( _eps_k, _GPU_eps, _NVALLEYS*NSBN*NX*sizeof(double), cudaMemcpyDeviceToHost ) );

  for(int iter=0; iter<999999; ++iter)
    {
      // int blockdim = 32;
      // int blocks = nblocks( _NVALLEYS*NSBN*NX, blockdim );
      // cudaDeviceSetCacheConfig(cudaFuncCachePreferNone);
      const int gridSize      = host_gridconfig -> cuda_eigenvalues_NR_config -> get_gridSize();
      const int blockSize     = host_gridconfig -> cuda_eigenvalues_NR_config -> get_blockSize();
      const int shmemSize     = host_gridconfig -> cuda_eigenvalues_NR_config -> get_shmemSize();
      const cudaFuncCache cfc = host_gridconfig -> cuda_eigenvalues_NR_config -> get_cfc();
      cudaDeviceSetCacheConfig( cfc );
      cuda_eigenvalues_NR <<< gridSize, blockSize, shmemSize >>> ( device_dm, d_A, _GPU_eps );
      CUDA_SAFE_CALL( cudaMemcpy( &_eps_k[ ((iter+1)%2)*_NVALLEYS*NSBN*NX ], _GPU_eps, _NVALLEYS*NSBN*NX*sizeof(double), cudaMemcpyDeviceToHost ) );
      double maxdiff = 0;
      for(int i=0; i<NX; ++i)
	for(int nu=0; nu<_NVALLEYS; ++nu)
	  for(int p=0; p<NSBN; ++p)
	    {
	      double val = fabs( _eps_k[i*_NVALLEYS*NSBN + nu*NSBN + p] - _eps_k[ _NVALLEYS*NSBN*NX + i*_NVALLEYS*NSBN + nu*NSBN + p] );
	      if(val > maxdiff)
		maxdiff = val;
	    }
      if(maxdiff < EPSTOL)
	break;
    }
  
  delete [] _eps_k;

  return;
}

void MOSFETProblemCuda::GPU_eigenvalues_ms(const double eigvals_tolpar)
{
  const int NSBN  = host_dm->get_NSBN();
  const int NX    = host_dm->get_X()->get_N();
  const int NZ    = host_dm->get_Z()->get_N();

  if(_ep == POTENTIAL) 
    start_time( _PHASE_ITER_EIGENSTATES_EIGENVALUES );

  double intlength;
  // Gershgorin circle
  double *_Y = new double[_NVALLEYS*NX];
  double *_Z = new double[_NVALLEYS*NX];
  double *_GPU_Y;
  double *_GPU_Z;
  checkCudaErrors( cudaMalloc((void **)&_GPU_Y, _NVALLEYS*NX*sizeof(double)) );          
  checkCudaErrors( cudaMalloc((void **)&_GPU_Z, _NVALLEYS*NX*sizeof(double)) );          

  // int blockdim = 12;
  // int blocks = nblocks( _NVALLEYS*NX, blockdim );
  // cudaDeviceSetCacheConfig(cudaFuncCachePreferNone);

  {
    const int gridSize      = host_gridconfig -> cuda_gershgorin_config -> get_gridSize();
    const int blockSize     = host_gridconfig -> cuda_gershgorin_config -> get_blockSize();
    const int shmemSize     = host_gridconfig -> cuda_gershgorin_config -> get_shmemSize();
    const cudaFuncCache cfc = host_gridconfig -> cuda_gershgorin_config -> get_cfc();
    cudaDeviceSetCacheConfig( cfc );
    cuda_gershgorin <<< gridSize, blockSize, shmemSize >>> ( device_dm, d_A, _GPU_Y, _GPU_Z );
  }

  CUDA_SAFE_CALL( cudaMemcpy( _Y, _GPU_Y, _NVALLEYS*NX*sizeof(double), cudaMemcpyDeviceToHost ) );
  CUDA_SAFE_CALL( cudaMemcpy( _Z, _GPU_Z, _NVALLEYS*NX*sizeof(double), cudaMemcpyDeviceToHost ) );
  double min_Y = _Y[0]; double max_Z = _Z[0];
  for(int line=1; line<_NVALLEYS*NX; ++line)
    {
      min_Y = ( _Y[line] < min_Y ? _Y[line] : min_Y );
      max_Z = ( _Z[line] > max_Z ? _Z[line] : max_Z );
    }

  intlength = max_Z-min_Y;
  delete [] _Y;
  delete [] _Z;
  checkCudaErrors( cudaFree(_GPU_Y) );
  checkCudaErrors( cudaFree(_GPU_Z) );
  
  // initialize eps
  // blockdim = 32;
  // blocks = nblocks( _NVALLEYS*NSBN*NX, blockdim );
  // cudaDeviceSetCacheConfig(cudaFuncCachePreferNone);
  {
    const int gridSize      = host_gridconfig -> cuda_initialize_eps_config -> get_gridSize();
    const int blockSize     = host_gridconfig -> cuda_initialize_eps_config -> get_blockSize();
    const int shmemSize     = host_gridconfig -> cuda_initialize_eps_config -> get_shmemSize();
    const cudaFuncCache cfc = host_gridconfig -> cuda_initialize_eps_config -> get_cfc();
    cudaDeviceSetCacheConfig( cfc );
    cuda_initialize_eps <<< gridSize, blockSize, shmemSize >>> (device_dm, _GPU_eps, min_Y);
  }

  const double _EPS = eigvals_tolpar;
  if( _NMULTI != 33 )
    {
      throw error_ITER_EIGENSTATES_MULTISEC( _NMULTI );
    }
  int number_of_iterations = int( log((intlength)/_EPS)/log(_NMULTI) ) + 1;
  double step_ms = intlength/_NMULTI;

  // blockdim = 32*6; // = _NMULTI-1;
  // blocks = nblocks( _NVALLEYS*NSBN*NX*(_NMULTI-1), blockdim );
  
  // int shmem_sign = blockdim;
  // int shmem_schr = _SCHROED_ROW_SIZE * ( (blockdim+(NSBN*(_NMULTI-1))-2)/(NSBN*(_NMULTI-1)) + 1 );
  // int shmem_size = shmem_schr + shmem_sign + _SCHROED_MATRIX_SIZE_PAD;
  // cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
  {
    const int gridSize      = host_gridconfig -> cuda_eigenvalues_ms_2_config -> get_gridSize();
    const int blockSize     = host_gridconfig -> cuda_eigenvalues_ms_2_config -> get_blockSize();
    const int shmemSize     = host_gridconfig -> cuda_eigenvalues_ms_2_config -> get_shmemSize();
    const cudaFuncCache cfc = host_gridconfig -> cuda_eigenvalues_ms_2_config -> get_cfc();
    cudaDeviceSetCacheConfig( cfc );
    cuda_eigenvalues_ms_2 <<< gridSize, blockSize, shmemSize >>> ( device_dm, d_A, step_ms, _NMULTI, _GPU_eps, number_of_iterations );
  }

  if(_ep == POTENTIAL) 
    stop_time( _PHASE_ITER_EIGENSTATES_EIGENVALUES );

  return;
}
#endif

#ifdef __CUDACC__
__global__ void cuda_eigenvalues_NR( const discrMeshes *dm, const double *A, double *_GPU_eps )
{
  int global_index = blockIdx.x*blockDim.x + threadIdx.x;

  const int NSBN     = dm->get_NSBN();
  const int NX       = dm->get_X()->get_N();
  const int NZ       = dm->get_Z()->get_N();

  if(global_index < _NVALLEYS*NSBN*NX)
    {
      int _i, _nu, _p;
      GPU_map_1D_to_3D( global_index, &_i, NX, &_nu, _NVALLEYS, &_p, NSBN );

      const int global_line = _i*_NVALLEYS + _nu;
      const double *D = &A[ global_line*_SCHROED_ROW_SIZE ]; 
      const double *S = D + _SCHROED_MATRIX_SIZE;

      // const double DZ = _dz;
      const double DZ = dm->get_Z()->get_delta();
      const double DZ2 = DZ * DZ;
      
      /******************************************
       *   NEWTON-RAPHSON ITERATIVE METHOD      *
       ******************************************/
      _GPU_eps[ _i*_NVALLEYS*NSBN + _nu*NSBN + _p ] *= DZ2;

      const double x = _GPU_eps[ _i*_NVALLEYS*NSBN + _nu*NSBN + _p ];
      double p = 1, pp = 0;
      double q = DZ2 * D[0] - x, qp = -1;
      double r, rp;
      for( int j = 1; j < _SCHROED_MATRIX_SIZE; j++ )
	{
	  r = ( DZ2 * D[j] - x ) * q - DZ2 * S[j-1] * DZ2 * S[j-1] * p;
	  rp = - q + ( DZ2 * D[j] - x ) * qp - DZ2 * S[j-1] * DZ2 * S[j-1] * pp;
	  p = q;
	  pp = qp;
	  q = r;
	  qp = rp;
	}

      _GPU_eps[ _i*_NVALLEYS*NSBN + _nu*NSBN + _p ] -= r/rp;
      
      _GPU_eps[ _i*_NVALLEYS*NSBN + _nu*NSBN + _p ] /= DZ2;
    }
}



__global__ void cuda_eigenvalues_ms_2( const discrMeshes *dm, const double *A, double step, const int _nmulti, double *_GPU_eps, const int _niters )
{
  int global_index = blockIdx.x*blockDim.x + threadIdx.x;
  int global_index0 = blockIdx.x*blockDim.x;

  extern __shared__ double shmem[];

  const int NSBN     = dm -> get_NSBN();
  const int NX       = dm -> get_X() -> get_N();
  const int NZ       = dm -> get_Z() -> get_N();

  // LOAD SCHROEDINGER MATRIX TO SHARED MEMORY
  const int linestart = (blockIdx.x * blockDim.x)/(NSBN*(_nmulti-1));
  const int number_of_lines = (blockDim.x+(NSBN*(_nmulti-1))-2)/(NSBN*(_nmulti-1)) + 1;
  int linestop  = linestart + number_of_lines - 1;
  linestop  = ( linestop < _NVALLEYS*NX ? linestop : _NVALLEYS*NX-1 );
  const int nelements = (linestop-linestart+1)*_SCHROED_ROW_SIZE;
  const int ncycles = (nelements+blockDim.x-1)/blockDim.x;
  for( int K=0; K<ncycles; ++K)
    {
      const int index = K*blockDim.x + threadIdx.x;
      if( index < nelements ) // sei tu if che mi fai calare al 50%?
  	shmem[ index ] = A[ _SCHROED_ROW_SIZE*linestart + index ];
    }
  __syncthreads();
  // lasciate ogni speranza o voi ch'intrate (with shared memory)
  const int global_first_line = (blockIdx.x * blockDim.x)/(NSBN*(_nmulti-1));
  const int global_line = global_index/(NSBN*(_nmulti-1));
  const int loc_line = global_line - global_first_line;
  const double *D = &shmem[ loc_line*_SCHROED_ROW_SIZE ]; 
  const double *S = D + _SCHROED_MATRIX_SIZE; // TASK. Es coerente con lo que pone con config, pero yo creo que habria que cambiarlo.
  // esto hay que reescribirlo mejor!!
  // __shared__ double Ssq[ _SCHROED_MATRIX_SIZE_PAD ];
  const int shmem_sign = blockDim.x;
  const int shmem_schr = _SCHROED_ROW_SIZE * ( (blockDim.x+(NSBN*(_nmulti-1))-2)/(NSBN*(_nmulti-1)) + 1 );
  double *Ssq = &shmem[ shmem_schr + shmem_sign ];

  const int _ncycles = _SCHROED_MATRIX_SIZE_PAD / 32;
  if( threadIdx.x < 32 )
    {
      for(int c = 0; c < _ncycles; ++c)
	Ssq[ threadIdx.x + 32*c ] = S[ threadIdx.x + 32*c ]*S[ threadIdx.x + 32*c ];
    }
  __syncthreads();
  
  // __shared__ volatile int chsgn[32*6]; // cosi sono sicuro di avere abbastanza spazio?
  __shared__ int chsgn[32*6]; // cosi sono sicuro di avere abbastanza spazio?

  if(global_index < _NVALLEYS*NSBN*NX*(_nmulti-1))
    {
      int _i, _nu, _p, _k;
      int index = global_index;
      GPU_map_1D_to_4D( index, &_i, NX, &_nu, _NVALLEYS, &_p, NSBN, &_k, (_nmulti-1) );
      
      int _i0, _nu0, _p0, _k0;
      int index0 = global_index0;
      GPU_map_1D_to_4D( index0, &_i0, NX, &_nu0, _NVALLEYS, &_p0, NSBN, &_k0, (_nmulti-1) );
      const int local_p = _p-_p0;

      /******************************************
       *            EIGENVALUES                 *
       ******************************************/
      const double nmultm1 = 1./_nmulti;
      double epsnupi = GPU_eps(_nu,_p,_i);
      for(int iter=0; iter<_niters; ++iter)
	{
	  const double x = epsnupi + (_k+1)*step;

	  int neg_values = 0;
	  double _q = D[0] - x;
	  neg_values += ( _q < 0 );

	  for( int j = 2; j <= _SCHROED_MATRIX_SIZE; j++ )
	    {
	      _q = (D[j-1] - x) - Ssq[j-2] / _q;
	      neg_values += ( _q < 0 );
	    }

	  chsgn[local_p*32 + _k] = ( neg_values < _p+1 ? _k : -1 );
	  int maximo;
	  if( local_p*32 <= threadIdx.x && threadIdx.x < (local_p+1)*32 )
	    maximo = reduce_max(&chsgn[32*local_p], threadIdx.x-local_p*32);

	  epsnupi += ((double)maximo+1.)*step; // +1. o +1.5? O sea, cojo el borde izquierdo o el medio del intervalo? Yo pondria 1.5...

	  step *= nmultm1;
	}

      if( threadIdx.x%32 == 0 )
	_GPU_eps[ _i*_NVALLEYS*NSBN + _nu*NSBN + _p ] = epsnupi;
    }
}




#endif


void MOSFETProblemCuda::CPU_eigenvalues_NR(const double eigvals_tolpar)
{
  const int NSBN  = host_dm -> get_NSBN();
  const int NX    = host_dm -> get_X()   -> get_N();
  const int NZ    = host_dm -> get_Z()   -> get_N();

  const double EPSTOL = eigvals_tolpar;
  double *_eps_k = new double[ 2*_NVALLEYS*NSBN*NX ];
  memcpy( _eps_k, _eps, _NVALLEYS*NSBN*NX*sizeof(double) );

  const double DZ = host_dm->get_Z()->get_delta();
  const double DZ2 = DZ * DZ;
  
  for(int iter=0; iter<999999; ++iter)
    {
      
      // NEWTON-RAPHSON ITERATIVE METHOD
#pragma omp parallel for
      for(int ind=0; ind<NX*_NVALLEYS*NSBN; ++ind)
	{
	  int _i, _nu, _p, index = ind;
	  _i    = index;
	  index = index/NSBN;
	  _p    = _i-index*NSBN;
	  _i    = index;
	  index = index/_NVALLEYS;
	  _nu   = _i-index*_NVALLEYS;
	  _i    = index;

	  _eps[ _i*_NVALLEYS*NSBN + _nu*NSBN + _p ] *= DZ2;

	  const int global_line = _i*_NVALLEYS + _nu;
	  const double *D = &___A[ global_line*_SCHROED_ROW_SIZE ]; 
	  const double *S = D + _SCHROED_MATRIX_SIZE;
	  
	  const double x = _eps[ _i*_NVALLEYS*NSBN + _nu*NSBN + _p ];
	  double p = 1, pp = 0;
	  double q = DZ2 * D[0] - x, qp = -1;
	  double r, rp;
	  for( int j = 1; j < _SCHROED_MATRIX_SIZE; j++ )
	    {
	      r = ( DZ2 * D[j] - x ) * q - DZ2 * S[j-1] * DZ2 * S[j-1] * p;
	      rp = - q + ( DZ2 * D[j] - x ) * qp - DZ2 * S[j-1] * DZ2 * S[j-1] * pp;
	      p = q;
	      pp = qp;
	      q = r;
	      qp = rp;
	    }
	  
	  _eps[ _i*_NVALLEYS*NSBN + _nu*NSBN + _p ] -= r/rp;

	  _eps[ _i*_NVALLEYS*NSBN + _nu*NSBN + _p ] /= DZ2;

	}
      
      // test convergence
      memcpy( &_eps_k[ ((iter+1)%2)*_NVALLEYS*NSBN*NX ], _eps, _NVALLEYS*NSBN*NX*sizeof(double) );
      double maxdiff = 0;
      for(int i=0; i<NX; ++i)
	for(int nu=0; nu<_NVALLEYS; ++nu)
	  for(int p=0; p<NSBN; ++p)
	    {
	      double val = fabs( _eps_k[i*_NVALLEYS*NSBN + nu*NSBN + p] - _eps_k[ _NVALLEYS*NSBN*NX + i*_NVALLEYS*NSBN + nu*NSBN + p] );
	      if(val > maxdiff)
		maxdiff = val;
	    }
      if(maxdiff < EPSTOL)
	break;
    }
  
  delete [] _eps_k;


  return;
}






/**
   PURPOSE:        

   FILE:           cuda_iter_eigenstates.cu

   NAME:           MOSFETProblem::eigenstates

   PARAMETERS:

   RETURN VALUE:   none

   CALLS TO:       MOSFETProblem::start_time           (cuda_comptime.h - declared inline)
		   MOSFETProblem::stop_time            (cuda_comptime.h - declared inline)
                   MOSFETProblem::show_eta             (cuda_testing.h - declared inline)
 
   CALLED FROM:    MOSFETProblem::iter_2               (cuda_iter.cu)

   DATA MODIFIED:  

   METHOD:         

   LAST UPDATE:    2022 February 16
*/


#ifdef __CUDACC__
__global__ void cuda_initialize_eps(const discrMeshes *dm, double *_GPU_eps, const double min_Y)
{
  int global_index = blockIdx.x*blockDim.x + threadIdx.x;

  const int NSBN     = dm -> get_NSBN();
  const int NX       = dm -> get_X() -> get_N();

  if(global_index < _NVALLEYS*NSBN*NX)
    {
      int _i, _nu, _p;
      GPU_map_1D_to_3D( global_index, &_i, NX, &_nu, _NVALLEYS, &_p, NSBN );

      GPU_eps(_nu,_p,_i) = min_Y;
    }
}



__global__ void cuda_gershgorin( const discrMeshes *dm, const double *A, double *_GPU_Y, double *_GPU_Z )
{
  int global_index = blockIdx.x*blockDim.x + threadIdx.x;

  const int NSBN     = dm -> get_NSBN();
  const int NX       = dm -> get_X() -> get_N();
  const int NZ       = dm -> get_Z() -> get_N();

  if(global_index < _NVALLEYS*NX)
    {
      const double *D = &A[ global_index*_SCHROED_ROW_SIZE ]; 
      const double *S = D + _SCHROED_MATRIX_SIZE; // TASK. Estas seguro? Tendria mas sentido SCHROED_MATRIX_SIZE_PAD, no? Pero asi es coerente con lo que pone en config.

      /******************************************
       *            EIGENVALUES                 *
       ******************************************/
      /* Gershgorin circle */
      double Y = D[0] - fabs(S[0]);
      double Z = D[0] + fabs(S[0]);

      double a, b;
      for( int j = 1; j<_SCHROED_MATRIX_SIZE-1; j++ )
	{
	  a = D[j] - fabs(S[j]) - fabs(S[j-1]);
	  b = D[j] + fabs(S[j]) + fabs(S[j-1]);
	  Y = a < Y ? a : Y;
	  Z = b > Z ? b : Z;
	}
      a = D[_SCHROED_MATRIX_SIZE-1] - fabs(S[_SCHROED_MATRIX_SIZE-2]);
      b = D[_SCHROED_MATRIX_SIZE-1] + fabs(S[_SCHROED_MATRIX_SIZE-2]);
      Y = a < Y ? a : Y;
      Z = b > Z ? b : Z;

      _GPU_Y[global_index] = Y;
      _GPU_Z[global_index] = Z;
    }
}

#endif

#ifdef __CUDACC__
template <class T> __device__ T reduce_sum(T *sdata, int tid)
{
  T res = sdata[tid];
  for(int offset=16; offset>0; offset/=2)
    res += __shfl_xor_sync( 0xFFFFFFFF, res, offset, 32 );
  return res;
}

template <class T> __device__ T reduce_sumsq(T *sdata, int tid)
{
  T res = sdata[tid]*sdata[tid];
  for(int offset=16; offset>0; offset/=2)
    res += __shfl_xor_sync( 0xFFFFFFFF, res, offset, 32 );
  return res;
}

template <class T> __device__ T reduce_max(T *sdata, int tid)
{
  T res = sdata[tid];
  for(int offset=16; offset>0; offset/=2)
    res = max( res, __shfl_xor_sync( 0xFFFFFFFF, res, offset, 32 ) );
  return res;
}
#endif



cuda_iter_eigenstates_eigenvalues_kernels_config::cuda_iter_eigenstates_eigenvalues_kernels_config( const discrMeshes *dm )
{
  const int NSBN  = dm -> get_NSBN();
  const int NX    = dm -> get_X()   -> get_N();
  const int NZ    = dm -> get_Z()   -> get_N();
  
  int blockdim = 32; int blocks = nblocks( _NVALLEYS*NSBN*NX, blockdim );
  cuda_initialize_eps_config   = new kernelConfig(blocks, blockdim, NOSHMEM                  , cudaFuncCachePreferNone  , "cuda_initialize_eps"  );

  blockdim = 12; blocks = nblocks( _NVALLEYS*NX, blockdim );
  cuda_gershgorin_config       = new kernelConfig(blocks, blockdim, NOSHMEM                  , cudaFuncCachePreferNone  , "cuda_gershgorin"      );

  blockdim = 32; blocks = nblocks( _NVALLEYS*NSBN*NX, blockdim );
  cuda_eigenvalues_NR_config   = new kernelConfig(blocks, blockdim, NOSHMEM                  , cudaFuncCachePreferNone  , "cuda_eigenvalues_NR"  );

  blockdim = (_NMULTI-1)*6;
  blocks = nblocks( _NVALLEYS*NSBN*NX*(_NMULTI-1), blockdim );
  int shmem_sign = blockdim;
  int shmem_schr = _SCHROED_ROW_SIZE * ( (blockdim+(NSBN*(_NMULTI-1))-2)/(NSBN*(_NMULTI-1)) + 1 );
  int shmem_size = shmem_schr + shmem_sign + _SCHROED_MATRIX_SIZE_PAD;
  cuda_eigenvalues_ms_2_config = new kernelConfig(blocks, blockdim, shmem_size*sizeof(double), cudaFuncCachePreferShared, "cuda_eigenvalues_ms_2");
}





/*
  name          : 'check_GPU_eigenvalues_input'
  last modified : 2023/06/06
  author        : Francesco VECIL

  description   : This functions check the input and the output of the 'GPU_eigenvalues' method.
                  Namely, 'check_GPU_eigenvalues_input' checks (i) matrix d_A.
 */
void MOSFETProblemCuda::check_GPU_eigenvalues_input()
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
      exit(_ERR_CHECK_GPU_EIGENVALUES_INPUT);
    }

  cerr << "[ok]" << endl;

  return;
}


/*
  name          : 'check_GPU_eigenvalues_output'
  last modified : 2023/06/06
  author        : Francesco VECIL

  description   : This functions check the input and the output of the 'GPU_eigenvalues' method.
                  Namely, 'check_GPU_eigenvalues_output' checks (ii) eps.
 */
void MOSFETProblemCuda::check_GPU_eigenvalues_output()
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
  

  
  if( stopcheck == true )
    {
      cerr << " ERROR from function " << __func__ << ", from line " << __LINE__ << ", from file '" << __FILE__
	   << "' --- exiting..." << endl;
      exit(_ERR_CHECK_GPU_EIGENVALUES_INPUT);
    }
  
  cerr << "[ok]" << endl;

  return;
}



