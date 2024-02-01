#include "mosfetproblem.h"
#include "cuda_reductions_kernels.h"
#include "debug_flags.h"


/*****************************************************************************************************************/

#ifdef __CUDACC__

// #define   _CHECK_GPU_EIGENVECTORS_THOMAS_INPUT_OUTPUT
void MOSFETProblemCuda::GPU_eigenvectors_Thomas(const double eigvecs_tolpar)
{
#ifdef   _CHECK_GPU_EIGENVECTORS_THOMAS_INPUT_OUTPUT
#warning   _CHECK_GPU_EIGENVECTORS_THOMAS_INPUT_OUTPUT is activated
  check_GPU_eigenvectors_Thomas_input();
#endif
  
  const int NSBN  = host_dm -> get_NSBN();
  const int NX    = host_dm -> get_X()   -> get_N();
  const int NZ    = host_dm -> get_Z()   -> get_N();

  if(_ep == POTENTIAL) 
    start_time( _PHASE_ITER_EIGENSTATES_EIGENVECTORS );

  const int gridSize      = host_gridconfig -> cuda_tridiag_Thomas_20230525_config -> get_gridSize();
  const int blockSize     = host_gridconfig -> cuda_tridiag_Thomas_20230525_config -> get_blockSize();
  const int shmemSize     = host_gridconfig -> cuda_tridiag_Thomas_20230525_config -> get_shmemSize();
  const cudaFuncCache cfc = host_gridconfig -> cuda_tridiag_Thomas_20230525_config -> get_cfc();
  cudaDeviceSetCacheConfig( cfc );
  cuda_tridiag_Thomas_20230525 <<< gridSize, blockSize, shmemSize >>> ( device_dm, device_solvpar, d_A, _GPU_eps, _GPU_chi, eigvecs_tolpar );

  if(_ep == POTENTIAL) 
    stop_time( _PHASE_ITER_EIGENSTATES_EIGENVECTORS );

#ifdef _ITER_DEBUG
  if( are_there_nan_inf( _GPU_chi, _NVALLEYS*NSBN*NX*_SCHROED_MATRIX_SIZE_PAD, "testing eigenvectors" ) == false )
    {
      cerr << " From function " << __func__ << " : a problem has been detected!" << endl;
      exit(666);
    }
#endif

#ifdef   _CHECK_GPU_EIGENVECTORS_THOMAS_INPUT_OUTPUT
  check_GPU_eigenvectors_Thomas_output();
#endif

  return;
}
#endif



#ifdef __CUDACC__
__global__ void cuda_tridiag_Thomas_20230525( const discrMeshes *dm, const solverParams *sp, const double *d_A, const double *_GPU_eps, double * _GPU_chi, const double eigvecs_tolpar )
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;

  const int NSBN     = dm->get_NSBN();
  const int NX       = dm->get_X()->get_N();
  const int NZ       = dm->get_Z()->get_N();
  const double DZ    = dm->get_Z()->get_delta();
  const double DZ2   = DZ * DZ;

  extern __shared__ double sm[]; /*
				   This data structure will contain the four local arrays
				   double d[size_x_d], l[size_x_d], x[size_x_d], xold[size_x_d]
				 */
  const int size_x_d = (((NZ-2)+31)/32)*32;
  double *d    = &sm[0];
  double *l    = &sm[  size_x_d];
  double *x    = &sm[2*size_x_d];
  double *xold = &sm[3*size_x_d];
  
  if( index < _NVALLEYS * NSBN * NX )
    {  
      // int N = _NZ-2;
      int N = NZ-2;
      int _nu, _p, _i;
      GPU_map_1D_to_3D( index, &_i, NX, &_nu, _NVALLEYS, &_p, NSBN );

      int row = _i*_NVALLEYS + _nu;
      const double *D = &d_A[ row*_SCHROED_ROW_SIZE ];
      const double *S = D + _SCHROED_MATRIX_SIZE;
      // const double alpha = GPU_eps(_nu, _p, _i) + 1.e-11;
      const double alpha = DZ2 * GPU_eps(_nu, _p, _i) + 1.e-8;
  
      for( int j = 0; j < N; j++ )
	x[j] = GPU_chi(_nu, _p, _i, j+1);

      const double epstol = eigvecs_tolpar;
      int k = 0;
      const int nmax = 9999;
      
      double linfdiff = 0;
      
      while( k < nmax )
	{
	  for( int j = 0; j < N; j++ )
	    xold[j] = x[j];

	  d[ 0 ] = ( DZ2 * D[ 0 ] - alpha );
	  for( int j = 0; j < N-1; j++ )
	    {
	      l[ j ] = DZ2 * S[ j ] / d[ j ];
	      d[j+1] = ( DZ2 * D[j+1] - alpha ) - l[j] * l[j] * d[j];
	    }

	  for( int j = 1; j < N; j++ )
	    x[j] = x[j] - l[j-1] * x[j-1];

	  x[N-1] = x[N-1] / d[N-1];
	  for( int j = N-2; j >= 0; j-- )
	    x[j] = x[j] / d[j] - l[j] * x[j+1];

	  /* Norm computation */
	  double a = 0.0;
	  for( int j = 0; j < N; j++ )
	    a += x[j]*x[j];
	  // double DZ = _dz;
	  a *= DZ;
	  a = 1./sqrt(a);
	  for( int j = 0; j < N; j++ )
	    x[j] *= a;

	  linfdiff = 0;
	  for( int j = 0; j < N; j++ )
	    linfdiff = ( fabs(fabs(x[j]) - fabs(xold[j])) > linfdiff ? fabs(fabs(x[j]) - fabs(xold[j])) : linfdiff );
	  if( linfdiff < epstol )
	    break;

	  ++k;
	}

      // if( k == nmax && threadIdx.x == 0 )
      if( k == nmax )
	{
	  printf("IPIM did not converge, for nu=%i, p=%i, i=%i, k=%i, linfdiff=%e.\n", _nu, _p, _i, k, linfdiff);
	  // asm("trap;");
	  assert(0);
	}

      // Store data 
      for( int j = 0; j < N; j++ )
	GPU_chi( _nu, _p, _i, j+1 ) = x[j];

    }
} 
#endif

/*****************************************************************************************************************/

#ifdef __CUDACC__
#define CUDA_SAFE_CALL( call ) {                                         \
 cudaError_t err = call;                                                 \
 if( cudaSuccess != err ) {                                              \
   fprintf(stderr,"CUDA: error occurred in cuda routine. Exiting...\n"); \
   exit(err);                                                            \
 } }
#endif



void MOSFETProblemCuda::CPU_tridiag_Thomas(const double eigvecs_tolpar)
{
  const int NSBN  = host_dm -> get_NSBN();
  const int NX    = host_dm -> get_X()   -> get_N();
  const int NZ    = host_dm -> get_Z()   -> get_N();

#pragma omp parallel for
  for( int index=0; index < _NVALLEYS * NSBN * NX; ++index )
    {  
      int N = NZ-2;
      int _nu, _p, _i;
      GPU_map_1D_to_3D( index, &_i, NX, &_nu, _NVALLEYS, &_p, NSBN );

      int row = _i*_NVALLEYS + _nu;
      const int size_x_d = (((NZ-2)+31)/32)*32;

      const double *D = &___A[ row*_SCHROED_ROW_SIZE ];
      const double *S = D + _SCHROED_MATRIX_SIZE;
      const double DZ = host_dm->get_Z()->get_delta();
      const double DZ2 = DZ * DZ;
      const double alpha = DZ2 * eps(_nu, _p, _i) + 1.e-8;
  
      double d[size_x_d], l[size_x_d], x[size_x_d], xold[size_x_d];
	  
      for( int j = 0; j < N; j++ )
	x[j] = chi(_nu, _p, _i, j+1);

      const double epstol = eigvecs_tolpar;
      int k = 0;
      const int nmax = 9999;
      
      double linfdiff = 0;
      
      while( k < nmax )
	{
	  for( int j = 0; j < N; j++ )
	    xold[j] = x[j];

	  d[ 0 ] = ( DZ2 * D[ 0 ] - alpha );
	  for( int j = 0; j < N-1; j++ )
	    {
	      l[ j ] = DZ2 * S[ j ] / d[ j ];
	      d[j+1] = ( DZ2 * D[j+1] - alpha ) - l[j] * l[j] * d[j];
	    }

	  for( int j = 1; j < N; j++ )
	    x[j] = x[j] - l[j-1] * x[j-1];

	  x[N-1] = x[N-1] / d[N-1];
	  for( int j = N-2; j >= 0; j-- )
	    x[j] = x[j] / d[j] - l[j] * x[j+1];

	  /* Norm computation */
	  double a = 0.0;
	  for( int j = 0; j < N; j++ )
	    a += x[j]*x[j];
	  const double DZ = host_dm->get_Z()->get_delta();
	  // double DZ = _dz;
	  a *= DZ;
	  a = 1./sqrt(a);
	  for( int j = 0; j < N; j++ )
	    x[j] *= a;

	  linfdiff = 0;
	  for( int j = 0; j < N; j++ )
	    linfdiff = ( fabs(fabs(x[j]) - fabs(xold[j])) > linfdiff ? fabs(fabs(x[j]) - fabs(xold[j])) : linfdiff );
	  if( linfdiff < epstol )
	    break;

	  ++k;
	}

      if( k == nmax )
	{
	  cerr << "Message from file '" __FILE__ << "', from function " << __func__ << ", from line " << __LINE__ << " :" << endl;
	  printf("IPIM did not converge, for nu=%i, p=%i, i=%i, k=%i, linfdiff=%e.\n", _nu, _p, _i, k, linfdiff);
	  exit(-1);
	}

      // Store data 
      for( int j = 0; j < N; j++ )
	chi( _nu, _p, _i, j+1 ) = x[j];
    }
} 



cuda_iter_eigenstates_eigenvectors_kernels_config::cuda_iter_eigenstates_eigenvectors_kernels_config( const discrMeshes *dm )
{
  const int NSBN  = dm -> get_NSBN();
  const int NX    = dm -> get_X()   -> get_N();
  const int NZ    = dm -> get_Z()   -> get_N();

  int blockdim = 1;
  int blocks = nblocks( _NVALLEYS * NSBN * NX, blockdim );
  const int size_x_d = (((NZ-2)+31)/32)*32;
  const int shmem_size = 4*size_x_d;

  cuda_tridiag_Thomas_20230525_config = new kernelConfig(blocks, blockdim, shmem_size*sizeof(double), cudaFuncCachePreferNone, "cuda_tridiag_Thomas_20230525");
}




/*
  name          : 'check_GPU_eigenvectors_Thomas_input'
  last modified : 2023/06/06
  author        : Francesco VECIL

  description   : This functions check the input and the output of the 'GPU_eigenvectors_Thomas' method.
                  Namely, 'check_GPU_eigenvectors_Thomas_input' checks (i) matrix d_A, (ii) eps.
 */
void MOSFETProblemCuda::check_GPU_eigenvectors_Thomas_input()
{
  cerr << "called '" << __func__ << "'...";

  const int NSBN  = host_dm -> get_NSBN();
  const int NX    = host_dm -> get_X()   -> get_N();
  const int NZ    = host_dm -> get_Z()   -> get_N();

  bool stopcheck = false;

  /*
    begin section : check d_A
   */
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
  /*
    end section : check d_A
   */

  /*
    begin section : check eps
   */
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
  /*
    end section : check eps
   */

  if( stopcheck == true )
    {
      cerr << " ERROR from function " << __func__ << ", from line " << __LINE__ << ", from file '" << __FILE__
	   << "' --- exiting..." << endl;
      exit(_ERR_CHECK_GPU_EIGENVECTORS_THOMAS_INPUT);
    }

  cerr << "[ok]" << endl;

  return;
}

/*
  name          : 'check_GPU_eigenvectors_Thomas_output'
  last modified : 2023/06/06
  author        : Francesco VECIL

  description   : This functions check the input and the output of the 'GPU_eigenvectors_Thomas' method.
                  Namely, 'check_GPU_eigenvectors_Thomas_output' checks (iii) chi.
 */
void MOSFETProblemCuda::check_GPU_eigenvectors_Thomas_output()
{
  cerr << "called '" << __func__ << "'...";

  const int NSBN  = host_dm -> get_NSBN();
  const int NX    = host_dm -> get_X()   -> get_N();
  const int NZ    = host_dm -> get_Z()   -> get_N();

  bool stopcheck = false;

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
      exit(_ERR_CHECK_GPU_EIGENVECTORS_THOMAS_INPUT);
    }

  cerr << "[ok]" << endl;

  return;
}
