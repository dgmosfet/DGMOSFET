#include "mosfetproblem.h"
#include "debug_flags.h"

#ifdef __CUDACC__
#include <cooperative_groups.h>
using namespace cooperative_groups;
#define Max(a,b) ((a) > (b) ? (a) : (b))
#define CUDA_SAFE_CALL( call ) {                                         \
 cudaError_t err = call;                                                 \
 if( cudaSuccess != err ) {                                              \
   fprintf(stderr,"CUDA: error occurred in cuda routine. Exiting...\n"); \
   exit(err);                                                            \
 } }
#endif


/**
   PURPOSE:        

   FILE:           cuda_iter_spstep_solvelinsys.cu

   NAME:           MOSFETProblem::solve_linsys

   PARAMETERS:

   RETURN VALUE:   none

   CALLS TO:       MOSFETProblem::start_time           (cuda_comptime.h - declared inline)
		   MOSFETProblem::stop_time            (cuda_comptime.h - declared inline)

   CALLED FROM:    MOSFETProblem::iter_2               (cuda_iter.cu)

   DATA MODIFIED:  

   METHOD:         

   LAST UPDATE:    2022 February 16
*/

// #define   _CHECK_SOLVE_LINSYS_INPUT_OUTPUT

int MOSFETProblemCuda::solve_linsys(const double linsys_tolpar)
  /********************************************************************
   * This function solves the linear system inside the
   * Newton-Raphson iterative method.
   * 
   * Input: matrix_2d, rhs
   * Output: pot
   ********************************************************************/
{
#ifdef   _CHECK_SOLVE_LINSYS_INPUT_OUTPUT
#warning   _CHECK_SOLVE_LINSYS_INPUT_OUTPUT is activated
  check_solve_linsys_input();
#endif
  
  if( _ep == POTENTIAL )
    start_time( _PHASE_ITER_SPSTEP_SOLVELINSYS );

#ifdef __CUDACC__
  int success = GPU_solve_linsys_SRJ(linsys_tolpar);
#else
  int success = CPU_solve_linsys_SRJ(linsys_tolpar);
#endif

  if( _ep == POTENTIAL )
    stop_time( _PHASE_ITER_SPSTEP_SOLVELINSYS );

#ifdef   _CHECK_SOLVE_LINSYS_INPUT_OUTPUT
  check_solve_linsys_output();
#endif

  return success;
}



int MOSFETProblemCuda::GPU_solve_linsys_SRJ(const double linsys_tolpar)
{
  const int NX    = host_dm -> get_X()   -> get_N();
  const int NZ    = host_dm -> get_Z()   -> get_N();

  double *_GPU_x_k;                                                                    
  checkCudaErrors( cudaMalloc((void **)&_GPU_x_k,  NX*NZ*sizeof(double)) );
  
  double *_GPU_x_kp1;      
  checkCudaErrors( cudaMalloc((void **)&_GPU_x_kp1,  NX*NZ*sizeof(double)) );

  double *_GPU_residual_vec; 
  checkCudaErrors( cudaMalloc((void **)&_GPU_residual_vec,  NX*NZ*sizeof(double)) );

  double device_residual = 1.e36;
  
  double *_x_k          = new double[ NX*NZ ];
  double *_residual_vec = new double[ NX*NZ ];

  // initial seed
  CUDA_SAFE_CALL( cudaMemcpy(_GPU_x_k, _GPU_pot, NX*NZ*sizeof(double), cudaMemcpyDeviceToDevice) );
  
  // JACOBI ITERATION 
  double omega;
  int iter;
  const int M = host_srj -> get_M();
  const int itermax = host_solvpar -> get_ITERMAX_LINSYS_SRJ();
  
  for( iter=0; device_residual > linsys_tolpar && iter < itermax; ++iter )
    {
      int j = iter % M;

      omega = host_srj -> get_relaxpar(j);
      // if external fixed-point iteration index is >1, switch to Jacobi
      if (iter_counter>1) omega=1.0;			     

      // MATRIX VECTOR PRODUCT 3
      {
	const int gridSize      = host_gridconfig -> cuda_matrix_vector_product_config -> get_gridSize();
	const int blockSize     = host_gridconfig -> cuda_matrix_vector_product_config -> get_blockSize();
	const int shmemSize     = host_gridconfig -> cuda_matrix_vector_product_config -> get_shmemSize();
	const cudaFuncCache cfc = host_gridconfig -> cuda_matrix_vector_product_config -> get_cfc();
	cudaDeviceSetCacheConfig( cfc );
	cuda_matrix_vector_product <<< gridSize, blockSize, shmemSize >>> ( device_dm, _GPU_residual_vec, _GPU_matrix_2d,_GPU_x_k );
      }

      {
	const int gridSize      = host_gridconfig -> cuda_update_x_config -> get_gridSize();
	const int blockSize     = host_gridconfig -> cuda_update_x_config -> get_blockSize();
	const int shmemSize     = host_gridconfig -> cuda_update_x_config -> get_shmemSize();
	const cudaFuncCache cfc = host_gridconfig -> cuda_update_x_config -> get_cfc();
	cudaDeviceSetCacheConfig( cfc );
	if( ((iter+1)%M==0 && iter_counter==1) || iter_counter>1 )
	  cuda_update_x <<< gridSize, blockSize, shmemSize >>> (device_dm, _GPU_x_kp1,_GPU_residual_vec, _GPU_rhs, _GPU_x_k,omega, true, shmemSize/sizeof(double));
	else
	  cuda_update_x <<< gridSize, blockSize, shmemSize >>> (device_dm, _GPU_x_kp1,_GPU_residual_vec, _GPU_rhs, _GPU_x_k,omega, false, shmemSize/sizeof(double));
      }

      if( ((iter+1)%M==0 && iter_counter==1) || iter_counter>1 )
	{
	  const int nelems = host_gridconfig -> cuda_update_x_config -> get_gridSize();
	  
	  // transfer vectors to host	       
	  CUDA_SAFE_CALL( cudaMemcpy(_residual_vec, _GPU_residual_vec, nelems*sizeof(double), cudaMemcpyDeviceToHost) );
	  CUDA_SAFE_CALL( cudaMemcpy(_x_k, _GPU_x_k, nelems*sizeof(double), cudaMemcpyDeviceToHost) );
	  
	  // COMPUTE RESIDUAL
	  device_residual = 0; double max_x_k = 0;
	  for(int line = 0; line < nelems; ++line)
	    {
	      device_residual = ( _residual_vec[line] > device_residual ? _residual_vec[line] : device_residual );
	      max_x_k = ( _x_k[line] > max_x_k ? _x_k[line] : max_x_k );
	    }

	  // residual is computed in relative sense
	  device_residual = device_residual / max_x_k;
	}
	
      // swap on GPU
      double *tmp = _GPU_x_k;
      _GPU_x_k    = _GPU_x_kp1;
      _GPU_x_kp1  = tmp;
    }

  // check if maximum number of iterations has been exceeded
  if( iter == itermax )
    {
      cerr << " from file '" << __FILE__ << "', from function '" << __func__ << "', from line " << __LINE__ << " --- SRJ method did not converge after " << itermax << " iterations!" << endl;
      return( _ERR_SRJ_NOCONV );
    }

  CUDA_SAFE_CALL( cudaMemcpy(_GPU_pot, _GPU_x_k, NX*NZ*sizeof(double), cudaMemcpyDeviceToDevice) );
    
  delete [] _x_k;
  delete [] _residual_vec;
  checkCudaErrors( cudaFree(_GPU_x_k) );
  checkCudaErrors( cudaFree(_GPU_x_kp1) );
  checkCudaErrors( cudaFree(_GPU_residual_vec) );

  return 0;
}



int MOSFETProblemCuda::CPU_solve_linsys_SRJ(const double linsys_tolpar)
{
  const int NX    = host_dm -> get_X()   -> get_N();
  const int NZ    = host_dm -> get_Z()   -> get_N();

  double *_x_k = new double[ NX*NZ ];
  double *_x_kp1 = new double[ NX*NZ ];
  double *_residual_vec = new double[ NX*NZ ];
  double *tmp;
  double residual = 1.e36;

  // initial seed
  memcpy( _x_k, _pot, NX*NZ*sizeof(double) );

  // JACOBI ITERATION 
  double omega;
  int iter;
  int M = host_srj -> get_M();
    
  const int itermax = host_solvpar -> get_ITERMAX_LINSYS_SRJ();
  for( iter=0; residual > linsys_tolpar && iter < itermax; ++iter )
    {
      int j = iter % M;

      omega = host_srj -> get_relaxpar(j);

      if (iter_counter>1) omega=1.0;			     

      // MATRIX VECTOR PRODUCT 3
      CPU_matrix_vector_product3( _residual_vec, _x_k );

      // UPDATE X AND RESIDUAL VECTOR					   	
      CPU_update_x ( _x_kp1, _residual_vec, _x_k, omega );

      if( ((iter+1)%M==0 && iter_counter==1) || iter_counter>1 )
	{
	  // COMPUTE RESIDUAL
	  residual = 0; double max_x_k = 0;
	  for(int line=0; line<NX*NZ; ++line)
	    {
	      residual = ( fabs(_residual_vec[line]) > residual ? fabs(_residual_vec[line]) : residual );
	      max_x_k = ( fabs(_x_k[line]) > max_x_k ? fabs(_x_k[line]) : max_x_k );
	    }

	  // residual is computed in relative sense
	  residual = residual/max_x_k;
	}

      // swap _x_kp1 and _x_k
      tmp    = _x_k;
      _x_k   = _x_kp1;
      _x_kp1 = tmp;
    }

  // check if maximum number of iterations has been exceeded
  if( iter == itermax )
    {
      cerr << " from file '" << __FILE__ << "', from function '" << __func__ << "', from line " << __LINE__ << " --- SRJ method did not converge after " << itermax << " iterations!" << endl;
      return( _ERR_SRJ_NOCONV );
    }

  memcpy( _pot, _x_k, NX*NZ*sizeof(double) );
    
  delete [] _x_k;
  delete [] _residual_vec;
  delete [] _x_kp1;
  // delete [] tmp;
  
  return 0;
}



#ifdef __CUDACC__

__global__ void cuda_matrix_vector_product( const discrMeshes *dm,
					    double *_GPU_residual_vec,
					    const double *_GPU_matrix_2d, 
					    const double * _GPU_x_k)
/***************************************************************************/
{
  extern __shared__ double sm[];

  const int NX       = dm -> get_X() -> get_N();
  const int NZ       = dm -> get_Z() -> get_N();

  // const int  NLINES  = BSIZE_MVP3/32;
  const int  NLINES  = blockDim.x/32;
  const int memsize  = 2*NZ + NLINES;

  double *sdata      = &sm[0];
  double *sdata2     = &sm[memsize];
  
  int id             = blockIdx.x*blockDim.x + threadIdx.x;
  int line           = id/32;
  
  /*********** Load to sdata *************/
  
  int first_line     = (blockIdx.x*blockDim.x)/32;
  int first_column   = max(first_line-NZ,0);
  int last_line      = ((blockIdx.x+1)*blockDim.x)/32-1;
  int ncolumns       = min(NX*NZ-1,last_line+NZ)-first_column + 1; // aqui habia que anadir un +1, correcto?
  
  for(int disp=threadIdx.x;disp<=ncolumns;disp+=blockDim.x)
    {
    if( first_column+disp < NX*NZ )
      {
	sdata[disp]=_GPU_x_k[first_column+disp];
      }
    }
  __syncthreads();     
  
  double sum=0;
  int tid=id-line*32;
  if (line<=NX*NZ-1)
    {    
      int disp=line*2*NZ+NZ;
      int start = max(line-NZ,0)+ tid +disp  ;    
      int stop = min(NX*NZ-1,line+NZ) +disp ;

      first_column+=disp;
      for(int column=start;column<=stop;column+=32)
	{
	  sum += _GPU_matrix_2d[column] * sdata[column-first_column];
	}

      for (int offset = 16; offset > 0; offset /= 2)
	{
	  sum += __shfl_down_sync(0xFFFFFFFF,sum, offset,32);
	}
    }
  
  // COALESCENT WRITING
  if (tid==0)
    {
      sdata2[threadIdx.x/32]=sum;
    }
  
  __syncthreads();    

  if ( threadIdx.x<blockDim.x/32 && first_line+threadIdx.x < NX*NZ )
    {
      _GPU_residual_vec[first_line+threadIdx.x] =sdata2[threadIdx.x];
    }
}

__global__ void cuda_update_x( const discrMeshes *dm, double *_GPU_x_kp1, double *_GPU_residual_vec, 
			       const double *_GPU_rhs, double *_GPU_x_k, const double omega, const bool residual, const int SHMEMSIZE )
/***************************************************************************/
{
  extern __shared__ volatile double shmem[];
  volatile double *sdata1 = &shmem[0];
  volatile double *sdata2 = &shmem[SHMEMSIZE/2];

  const int NX    = dm -> get_X() -> get_N();
  const int NZ    = dm -> get_Z() -> get_N();
  
  int line        = blockIdx.x*blockDim.x + threadIdx.x;
  int tid         = threadIdx.x;
  int n           = NX*NZ; 
  bool valid_line = (line < n); 
  double res;
  if (valid_line)
    {
      res=_GPU_rhs[line]-_GPU_residual_vec[line];
      _GPU_x_kp1[line]= _GPU_x_k[line] + omega*res;
    } 

  if (residual)
    {
      sdata1[tid] = (valid_line) ? fabs(omega*res)  : 0;
      sdata2[tid] = (valid_line) ? fabs(_GPU_x_k[line]): 0;
      __syncthreads();

      for (int s=blockDim.x/2; s>0; s>>=1)
	{
	  if (tid < s)
	    {
	      sdata1[tid] = Max(sdata1[tid],sdata1[tid+s]);
	    }
	  if  (tid>=(blockDim.x-s) )
	    {
	      sdata2[tid] = Max(sdata2[tid],sdata2[tid-s]);
	    }
	  __syncthreads();
	}

      if (tid==0)              _GPU_residual_vec[blockIdx.x]= sdata1[tid];   
      if (tid==blockDim.x-1)   _GPU_x_k[blockIdx.x]         = sdata2[tid];
    }          
}

#endif







int MOSFETProblemCuda::CPU_solve_linsys_LIS()
{
  const int NX    = host_dm -> get_X()   -> get_N();
  const int NZ    = host_dm -> get_Z()   -> get_N();

  lis_vector_duplicate( _matrix_2d_LIS, &_res_LIS );

  LIS_SOLVER _solver;
  lis_solver_create( &_solver );

  if( _ep == POTENTIAL )
    {
      char opts[] = "-i bicgstab -p ilut -tol 1.0e-8";
      lis_solver_set_option( opts, _solver);
    }
  else
    {
      char opts[] = "-i bicgstab -p ilut -tol 1.0e-12";
      lis_solver_set_option( opts, _solver);
    }

  lis_solve( _matrix_2d_LIS, _rhs_LIS, _res_LIS, _solver );

  LIS_SCALAR *_temp_pot;
  _temp_pot = (LIS_SCALAR *)malloc( NX*NZ*sizeof(LIS_SCALAR) );
  lis_vector_gather( _res_LIS, _temp_pot );

  // copy the result
  for(int i=0; i<NX; ++i)
    for(int j=0; j<NZ; ++j)
      pot(i,j) = _temp_pot[i*NZ+j];
  free( _temp_pot );

  lis_matrix_destroy( _matrix_2d_LIS );
  lis_vector_destroy( _res_LIS );
  lis_vector_destroy( _rhs_LIS );

  return 0;
}



/***************************************************************************/
void MOSFETProblemCuda::CPU_matrix_vector_product3( double *_residual_vec,
						const double * _x_k)
/***************************************************************************/
{
  const int NX    = host_dm -> get_X()   -> get_N();
  const int NZ    = host_dm -> get_Z()   -> get_N();

#pragma omp parallel for
  for(int line=0; line<NX*NZ; ++line)
    {
       int first_column=max(line-NZ,0);
       int last_column=min(line+NZ,NX*NZ-1);
  
       double sum=0;
       for(int k=first_column; k<=last_column; ++k)
	 sum += matrix_2d(line,k) * _x_k[k];
       _residual_vec[line] = sum;
    }

}



/***************************************************************************/
void MOSFETProblemCuda::CPU_update_x( double *_x_kp1, double *_residual_vec, 
				      double *_x_k, const double omega )
/***************************************************************************/
{
  cerr << " CPU : omega = " << omega << endl;
  
  const int NX    = host_dm -> get_X()   -> get_N();
  const int NZ    = host_dm -> get_Z()   -> get_N();

#pragma omp parallel for
  for(int line=0; line<NX*NZ; ++line)
    {
      double res = _rhs[line]-_residual_vec[line];
      // // delete from here
      // if(blocco*64 <= line && line < (blocco+1)*64)
      // 	{
      // 	  cerr << " CPU --- line = " << line << " --- res = " << res << endl;
      // 	}
      // // to here
      _residual_vec[line] = omega*res;
      _x_kp1[line]= _x_k[line] + _residual_vec[line];
    } 
}




cuda_iter_spstep_solvelinsys_kernels_config::cuda_iter_spstep_solvelinsys_kernels_config(const discrMeshes *dm)
{
  const int NX    = dm -> get_X()   -> get_N();
  const int NZ    = dm -> get_Z()   -> get_N();

  {
    const int BSIZE_MVP3       = 128;
    const unsigned bsize3      = BSIZE_MVP3;
    const int number_of_blocks = nblocks( NX*NZ*32, bsize3 );
    const int NLINES           = BSIZE_MVP3/32;
    const int memsize          = 2*NZ + NLINES;
    const int shmem_size       = NLINES + memsize;
    cuda_matrix_vector_product_config = new kernelConfig(number_of_blocks, bsize3, shmem_size*sizeof(double), cudaFuncCachePreferShared, "cuda_matrix_vector_product");
  }

  {
    const int BSIZE_UPDATE_X   = 64;
    const int bsize            = BSIZE_UPDATE_X;
    const int number_of_blocks = nblocks( NX*NZ, bsize );
    const int shmem_size       = 2*bsize;
    cuda_update_x_config              = new kernelConfig(number_of_blocks, bsize,  shmem_size*sizeof(double), cudaFuncCachePreferNone,   "cuda_update_x"             );
  }
}




/*
  name          : 'check_solve_linsys_input', 'check_solve_linsys_output'
  last modified : 2023/06/06
  author        : Francesco VECIL

  description   : This functions check the input and the output of the 'solve_linsys' method.
                  As input  : (i) matrix_2d, (ii) rhs, (iii) pot [for initialization only]
		  As output : (iv) pot
 */
void MOSFETProblemCuda::check_solve_linsys_input()
{
  cerr << "called '" << __func__ << "'...";

  const int NSBN  = host_dm -> get_NSBN();
  const int NX    = host_dm -> get_X()   -> get_N();
  const int NZ    = host_dm -> get_Z()   -> get_N();

  bool stopcheck = false;
  
  /*
    begin section : check matrix_2d
   */
  // cerr << "check matrix_2d...";
  double *_aux_matrix_2d = new double[NX*NZ*(2*NZ+1)];
#ifdef __CUDACC__
  CUDA_SAFE_CALL( cudaMemcpy( _aux_matrix_2d, _GPU_matrix_2d, NX*NZ*(2*NZ+1)*sizeof(double), cudaMemcpyDeviceToHost ) );
#else
  memcpy( _aux_matrix_2d, _matrix_2d, NX*NZ*(2*NZ+1)*sizeof(double) );
#endif
  for(int idx=0; idx<NX*NZ*(2*NZ+1); ++idx)
    {
      const double val = _aux_matrix_2d[ idx ];
      if( isnan( val ) || isinf( val ) )
	{
	  cerr << " ERROR from function " << __func__ << ", from line " << __LINE__ << ", from file '" << __FILE__
	       << "' --- matrix_2d[" << idx << "] = " << val << endl;
	  stopcheck = true;
	}
    }
  delete [] _aux_matrix_2d;
  // cerr << "[ok]";
  /*
    end section : check matrix_2d
   */

  /*
    begin section : check rhs
   */
  // cerr << "check rhs...";
  double *_aux_rhs = new double[ NX*NZ ];
#ifdef __CUDACC__
  CUDA_SAFE_CALL( cudaMemcpy( _aux_rhs, _GPU_rhs, NX*NZ*sizeof(double), cudaMemcpyDeviceToHost ) );
#else
  memcpy( _aux_rhs, _rhs, NX*NZ*sizeof(double) );
#endif
  for(int i=0; i<NX; ++i)
    {
      for(int j=0; j<NZ; ++j)
	{
	  const double val = _aux_rhs[ indx_i_j(i,j) ];
	  if( isnan( val ) || isinf( val ) )
	    {
	      cerr << " ERROR from function " << __func__ << ", from line " << __LINE__ << ", from file '" << __FILE__
		   << "' --- rhs(" << i << "," << j << ") = " << val << endl;
	      stopcheck = true;
	    }
	}
    }
  delete [] _aux_rhs;
  // cerr << "[ok]";
  /*
    end section : check rhs
   */

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
    end section : Check pot
  */

  if( stopcheck == true )
    {
      cerr << " ERROR from function " << __func__ << ", from line " << __LINE__ << ", from file '" << __FILE__
	   << "' --- exiting..." << endl;
      exit(_ERR_CHECK_SOLVE_LINSYS_INPUT_OUTPUT);
    }

  cerr << "[ok]" << endl;

  return;
}



/*
  name          : 'check_solve_linsys_input', 'check_solve_linsys_output'
  last modified : 2023/06/06
  author        : Francesco VECIL

  description   : This functions check the input and the output of the 'solve_linsys' method.
                  As input  : (i) matrix_2d, (ii) rhs, (iii) pot [for initialization only]
		  As output : (iv) pot
 */
void MOSFETProblemCuda::check_solve_linsys_output()
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
    end section : Check pot
  */

  if( stopcheck == true )
    {
      cerr << " ERROR from function " << __func__ << ", from line " << __LINE__ << ", from file '" << __FILE__
	   << "' --- exiting..." << endl;
      exit(_ERR_CHECK_SOLVE_LINSYS_INPUT_OUTPUT);
    }

  cerr << "[ok]" << endl;

  return;
}



bool MOSFETProblemCuda::is_matrix_2d_diagonally_dominant()
{
  const int NX    = host_dm -> get_X()   -> get_N();
  const int NZ    = host_dm -> get_Z()   -> get_N();

#ifdef __CUDACC__
  CUDA_SAFE_CALL( cudaMemcpy(_matrix_2d, _GPU_matrix_2d, NX*NZ*(2*NZ+1)*sizeof(double), cudaMemcpyDeviceToHost) );
#endif
  
  for(int r=0; r<NX*NZ; ++r)
    {
      double diagelem = fabs(matrix_2d(r, r));
      double nondiagelem = 0;
      for(int s=r-NZ; s<=r+NZ; ++s)
	{
	  nondiagelem += fabs(matrix_2d(r, s));
	}
      nondiagelem -= diagelem;
      // if(nondiagelem >= diagelem)
      if(nondiagelem-1.e-2 >= diagelem)
	{
	  cerr << " linea r = " << r << " , diagelem = " << diagelem << ", nondiagelem = " << nondiagelem << "...";
	  return false;
	}
    }
  return true;
}



#define   cmo(i,j)   (i)+(j)*N
double MOSFETProblemCuda::compute_matrix_2d_spectral_radius()
{
  const int NX    = host_dm -> get_X()   -> get_N();
  const int NZ    = host_dm -> get_Z()   -> get_N();
  int N     = NX*NZ;
  
  cerr << "1)...";
  
#ifdef __CUDACC__
  CUDA_SAFE_CALL( cudaMemcpy(_matrix_2d, _GPU_matrix_2d, NX*NZ*(2*NZ+1)*sizeof(double), cudaMemcpyDeviceToHost) );
#endif
  
  double *aux_mat = new double[ N*N ];
  for(int i=0; i<N; ++i)
    {
      for(int j=0; j<N; ++j)
	{
	  aux_mat[ cmo(i,j) ] = 0;
	}
    }

  for(int r=0; r<N; ++r)
    {
      for(int s=max(r-NZ,0); s<=min(r+NZ, N-1); ++s)
	{
	  aux_mat[ cmo(r,s) ] = matrix_2d(r,s);
	}
    }
    
  cerr << "2)...";

  double *_L = new double[ N*N ];
  double *_D = new double[ N*N ];
  double *_U = new double[ N*N ];

  for(int i=0; i<N; ++i)
    {
      for(int j=0; j<N; ++j)
	{
	  const double val = aux_mat[ cmo(i,j) ];
	  if( j<i )
	    {
	      _L[ cmo(i,j) ] = val;
	      _D[ cmo(i,j) ] = 0;
	      _U[ cmo(i,j) ] = 0;
	    }
	  if( j == i )
	    {
	      _L[ cmo(i,j) ] = 0;
	      _D[ cmo(i,j) ] = val;
	      _U[ cmo(i,j) ] = 0;
	    }
	  if( j > i )
	    {
	      _L[ cmo(i,j) ] = 0;
	      _D[ cmo(i,j) ] = 0;
	      _U[ cmo(i,j) ] = val;
	    }
	}
    }

  cerr << "3)...";

  double *_M = new double[ N*N ];
  double *_N = new double[ N*N ];
  double *_invM = new double[ N*N ];
  double *_itermatrix = new double[ N*N ];
  for(int i=0; i<N; ++i)
    {
      for(int j=0; j<N; ++j)
	{
	  _M[ cmo(i,j) ] = _D[ cmo(i,j) ];
	  _N[ cmo(i,j) ] = -_L[ cmo(i,j) ]-_U[ cmo(i,j) ];
	}
    }

  for(int i=0; i<N; ++i)
    {
      for(int j=0; j<N; ++j)
	{
	  _invM[ cmo(i,j) ] = 0;
	}
    }
  for(int i=0; i<N; ++i)
    {
      _invM[ cmo(i,i) ] = 1./_M[ cmo(i,i) ];
    }

  cerr << "4)...";

  for(int i=0; i<N; ++i)
    {
      for(int j=0; j<N; ++j)
	{
	  double aux = 0;
	  for(int k=0; k<N; ++k)
	    {
	      aux += _invM[ cmo(i,k) ] * _N[ cmo(k,j) ];
	    }
	  _itermatrix[ cmo(i,j) ] = aux;
	}
    }
  
  cerr << "5)...";

  // now I want the spectral radius of itermatrix

  char JOBVL = 'N';
  char JOBVR = 'N';
  int LDA = N;
  double *WR = new double[N];
  double *WI = new double[N];
  double *VL = new double[N*N];
  int LDVL = N;
  double *VR = new double[N*N];
  int LDVR = N;
  double *WORK = new double[3*N];
  int LWORK = 3*N;
  int INFO;

  dgeev_( &JOBVL, &JOBVR, &N, _itermatrix, &LDA, WR, WI, VL, &LDVL, VR, &LDVR, WORK, &LWORK, &INFO);

  cerr << "6)...";

  if( INFO != 0 )
    {
      cerr << " dgeev failed." << endl;
      exit(-1);
    }

  double maxval = 0;
  for(int i=0; i<N; ++i)
    {
      double val = sqrt( WR[i]*WR[i] + WI[i]*WI[i] );
      if( val > maxval )
	{
	  maxval = val;
	}
    }
  
  cerr << "7)...";

  delete [] WORK;
  delete [] VR;
  delete [] VL;
  delete [] WI;
  delete [] WR;
	  
  delete [] _itermatrix;
  delete [] _invM;
  delete [] _N;
  delete [] _M;
  delete [] _U;
  delete [] _D;
  delete [] _L;
  delete [] aux_mat;

  return maxval;
}


