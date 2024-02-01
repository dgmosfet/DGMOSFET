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
__global__ void cuda_constr_linsys_NewtonRaphson( const discrMeshes *dm, const adimParams *adimpar, double * _GPU_matrix_2d, double *_GPU_rhs, const double *_GPU_frechet, const double *_GPU_totvoldens_OLD, const double *_GPU_nd, const double *_GPU_pot_OLD )
{
  int global_index = blockIdx.x*blockDim.x + threadIdx.x;

  const int NX       = dm->get_X()->get_N();
  const int NZ       = dm->get_Z()->get_N();

  if(global_index < (NX-2)*(NZ-2))
    {
      const double DZ = dm->get_Z()->get_delta();
      const double CP = adimpar->get_cp();
      
      int i, j;
      GPU_map_1D_to_2D( global_index, &i, NX-2, &j, NZ-2 );
      ++i; ++j;

      int r = i*NZ+j;

      // coefficient matrix
      int JX;
      int JZ;
      int s;
      int jp;
      {
	jp = 0;
	JX = i;
	JZ = jp;
	s = JX*NZ+JZ;
	GPU_matrix_2d(r,s) += .5*CP*DZ*GPU_frechet(i,j,JZ);
      }
      for(jp=1; jp<NZ-1; ++jp)
	{
	  JX = i;
	  JZ = jp;
	  s = JX*NZ+JZ;
	  GPU_matrix_2d(r,s) += CP*DZ*GPU_frechet(i,j,JZ);
	}
      {
	jp = NZ-1;
	JX = i;
	JZ = jp;
	s = JX*NZ+JZ;
	GPU_matrix_2d(r,s) += .5*CP*DZ*GPU_frechet(i,j,JZ);
      }
	  
      // right hand side
      _GPU_rhs[r] += -CP*(GPU_totvoldens_OLD(i,j)-GPU_nd(i,j));
      _GPU_rhs[r] += .5*CP*DZ*GPU_frechet(i,j,0)*GPU_pot_OLD(i,0);
      for(jp=1; jp<NZ-1; ++jp)
	{
	  _GPU_rhs[r] += CP*DZ*GPU_frechet(i,j,jp)*GPU_pot_OLD(i,jp);
	}
      _GPU_rhs[r] += .5*CP*DZ*GPU_frechet(i,j,NZ-1)*GPU_pot_OLD(i,NZ-1);

      // preconditioning
      double factor = 1./GPU_matrix_2d(r, r);
      for(int s=r-NZ; s<=r+NZ; ++s)
      	GPU_matrix_2d(r,s) *= factor;
      _GPU_rhs[r] *= factor;
    }
}



// __global__ void cuda_constr_linsys_NewtonRaphson( const discrMeshes *dm, const adimParams *adimpar, double * _GPU_matrix_2d, double *_GPU_rhs, const double *_GPU_frechet, const double *_GPU_totvoldens_OLD, const double *_GPU_nd, const double *_GPU_pot_OLD )
// {
//   int global_index = blockIdx.x*blockDim.x + threadIdx.x;

//   const int NX       = dm->get_X()->get_N();
//   const int NZ       = dm->get_Z()->get_N();

//   if(global_index < (NX-2)*(NZ-2))
//     {
//       const double DZ = dm->get_Z()->get_delta();
//       const double CP = adimpar->get_cp();
      
//       int i, j;
//       GPU_map_1D_to_2D( global_index, &i, NX-2, &j, NZ-2 );
//       ++i; ++j;

//       int r = i*NZ+j;

//       // coefficient matrix
//       int JX;
//       int JZ;
//       int s;
//       for(int jp=1; jp<NZ-1; ++jp)
// 	{
// 	  JX = i;
// 	  JZ = jp;
// 	  s = JX*NZ+JZ;
// 	  GPU_matrix_2d(r,s) += CP*DZ*GPU_frechet(i,j,JZ);
// 	}
	  
//       // right hand side
//       _GPU_rhs[r] += -CP*(GPU_totvoldens_OLD(i,j)-GPU_nd(i,j));
//       for(int jp=1; jp<NZ-1; ++jp)
// 	{
// 	  _GPU_rhs[r] += CP*DZ*GPU_frechet(i,j,jp)*GPU_pot_OLD(i,jp);
// 	}

//       // preconditioning
//       double factor = 1./GPU_matrix_2d(r, r);
//       for(int s=r-NZ; s<=r+NZ; ++s)
//       	GPU_matrix_2d(r,s) *= factor;
//       _GPU_rhs[r] *= factor;
//     }
// }



__global__ void cuda_constr_linsys_Gummel( const discrMeshes *dm, const adimParams *adimpar, double * _GPU_matrix_2d, double *_GPU_rhs, const double *_GPU_frechet, const double *_GPU_totvoldens_OLD, const double *_GPU_nd, const double *_GPU_pot_OLD )
{
  int global_index = blockIdx.x*blockDim.x + threadIdx.x;

  const int NX       = dm->get_X()->get_N();
  const int NZ       = dm->get_Z()->get_N();

  if(global_index < (NX-2)*(NZ-2))
    {
      const double DZ = dm->get_Z()->get_delta();
      const double CP = adimpar->get_cp();
      const double CG = adimpar->get_cg();
      
      int i, j;
      GPU_map_1D_to_2D( global_index, &i, NX-2, &j, NZ-2 );
      ++i; ++j;

      int r = i*NZ+j;

      // Gummel
      GPU_matrix_2d(r,r) += CG * GPU_totvoldens_OLD(i,j);

      // right hand side
      _GPU_rhs[r] += -CP * (GPU_totvoldens_OLD(i,j)-GPU_nd(i,j));

      // Gummel
      _GPU_rhs[r] += CG * GPU_totvoldens_OLD(i,j) * GPU_pot_OLD(i,j);

      // preconditioning
      double factor = 1./GPU_matrix_2d(r, r);
      for(int s=r-NZ; s<=r+NZ; ++s)
      	GPU_matrix_2d(r,s) *= factor;
      _GPU_rhs[r] *= factor;
    }
}
#endif





#ifdef __CUDACC__
void MOSFETProblemCuda::GPU_constr_linsys_SRJ( const FixedPointType fpt )
{
  const int NX    = host_dm -> get_X()   -> get_N();
  const int NZ    = host_dm -> get_Z()   -> get_N();

  CUDA_SAFE_CALL( cudaMemcpy(_GPU_matrix_2d, &_GPU_matrix_2dconst[2*NX*NZ*(2*NZ+1)], NX*NZ*(2*NZ+1)*sizeof(double),cudaMemcpyDeviceToDevice) );
  CUDA_SAFE_CALL( cudaMemcpy(_GPU_rhs, _GPU_rhs_const, NX*NZ*sizeof(double),cudaMemcpyDeviceToDevice) );
  const int gridSize      = host_gridconfig -> cuda_constr_linsys_config -> get_gridSize();
  const int blockSize     = host_gridconfig -> cuda_constr_linsys_config -> get_blockSize();
  const int shmemSize     = host_gridconfig -> cuda_constr_linsys_config -> get_shmemSize();
  const cudaFuncCache cfc = host_gridconfig -> cuda_constr_linsys_config -> get_cfc();
  cudaDeviceSetCacheConfig( cfc );
  if( fpt == NEWTON_RAPHSON )
    {
      cuda_constr_linsys_NewtonRaphson <<< gridSize, blockSize, shmemSize >>> ( device_dm, device_adimpar, _GPU_matrix_2d, _GPU_rhs, _GPU_frechet, _GPU_totvoldens_OLD, _GPU_nd, _GPU_pot_OLD );
    }
  else if( fpt == GUMMEL )
    {
      cuda_constr_linsys_Gummel <<< gridSize, blockSize, shmemSize >>> ( device_dm, device_adimpar, _GPU_matrix_2d, _GPU_rhs, _GPU_frechet, _GPU_totvoldens_OLD, _GPU_nd, _GPU_pot_OLD );
    }
  else
    {
      cerr << " ERROR from function " << __func__ << ", from line " << __LINE__ << ", from file '" << __FILE__ << " : ERROR! Unrecognized fixed-point iteration type!" << endl;
      exit(-1);
    }

  return;
}
#endif







/**
   PURPOSE:        

   FILE:           cuda_iter_spstep_constrlinsys.cu

   NAME:           MOSFETProblem::constr_linsys

   PARAMETERS:

   RETURN VALUE:   none

   CALLS TO:       MOSFETProblem::start_time           (cuda_comptime.h - declared inline)
		   MOSFETProblem::stop_time            (cuda_comptime.h - declared inline)

   CALLED FROM:    MOSFETProblem::iter_2               (cuda_iter.cu)

   DATA MODIFIED:  

   METHOD:         

   LAST UPDATE:    2022 February 16
*/
// #define  _CHECK_CONSTR_LINSYS_INPUT_OUTPUT

void MOSFETProblemCuda::constr_linsys( const FixedPointType fpt )
  /********************************************************************
   * This function constructs the matrix and rhs of the linear
   * system involved inside the Newton-Raphson iterative method.
   * 
   * Constants: nd, 
   * Input: pot_OLD, totvoldens_OLD, frechet
   * Output: matrix_2d, rhs
   ********************************************************************/
{
#ifdef   _CHECK_CONSTR_LINSYS_INPUT_OUTPUT
#warning   _CHECK_CONSTR_LINSYS_INPUT_OUTPUT is activated
  check_constr_linsys_input();
#endif


  if( _ep == POTENTIAL )
    start_time( _PHASE_ITER_SPSTEP_CONSTRLINSYS );
 
  /**
     For debugging purposes, a #define is introduced, allowing the code to revert to 
     computations on CPU to check whether the Cuda implementation fails.
     If the following #define is commented, the code resumes normal funcioning.
  */
// #define _CONSTRLINSYS_ON_CPU
#ifdef   _CONSTRLINSYS_ON_CPU
#warning   _CONSTRLINSYS_ON_CPU is activated. Have you activated _SOLVELINSYS_ON_CPU too?
#endif

  // EXECUTION ON GPU
#ifdef __CUDACC__

#ifdef _CONSTRLINSYS_ON_CPU
  cerr << " constructing linsys on CPU...";
  CUDA_SAFE_CALL( cudaMemcpy( _frechet,         _GPU_frechet,         _FRECHET_SIZE*sizeof(double),         cudaMemcpyDeviceToHost ) );
  CUDA_SAFE_CALL( cudaMemcpy( _totvoldens_OLD, _GPU_totvoldens_OLD, _TOTVOLDENS_SIZE*sizeof(double),     cudaMemcpyDeviceToHost ) );
  CUDA_SAFE_CALL( cudaMemcpy( _pot_OLD,        _GPU_pot_OLD,        _POT_SIZE*sizeof(double),            cudaMemcpyDeviceToHost ) );
  CPU_constr_linsys_LIS( fpt );
#else
  GPU_constr_linsys_SRJ( fpt );
#endif
  
  // EXECUTION ON CPU
#else
  // CPU_constr_linsys_LIS();
  CPU_constr_linsys_SRJ( fpt );
#endif

  if( _ep == POTENTIAL )
    stop_time( _PHASE_ITER_SPSTEP_CONSTRLINSYS );
  
#ifdef   _CHECK_CONSTR_LINSYS_INPUT_OUTPUT
  check_constr_linsys_output();
#endif

  return;
}


void MOSFETProblemCuda::CPU_constr_linsys_LIS( const FixedPointType fpt )
{
  /************************************************************************************
   *                                  construct the matrix                            *
   ************************************************************************************/
  const int NX    = host_dm -> get_X()   -> get_N();
  const int NZ    = host_dm -> get_Z()   -> get_N();

  LIS_INT *ptr, *index;
  LIS_SCALAR *value;
  int n = NX*NZ; 
  int k;
  const double DZ = host_dm->get_Z()->get_delta();
  const double CP = host_adimpar->get_cp();

  const int A_nnz =        // ATENCION: A ESTE A_nnz LE SOBRAN ALGUNOS PUNTOS, NO ES OPTIMO, PERO TAMPOCO LES SOBRAN DEMASIADOS.
  2*NZ                     // condiciones de frontera x=0 (se preven dos puntos por si se usa Neumann; solo uno haria falta para Dirichlet)
  +2*(NX-2)                // condiciones de frontera z=0 (se preven dos puntos por Neumann; solo uno hace falta para Dirichlet)
  +(NX-2)*(NZ-2)*5         // puntos interiores: para cada uno de ellos hay un laplaciano 2D
  +(NX-2)*(NZ-2)*NZ        // puntos interiores: para cada uno de ellos hay una integral a lo largo de z
  +2*(NX-2)                // condiciones de frontera z=L_z (se preven dos puntos por Neumann; solo uno hace falta para Dirichlet)
  +2*NZ;                   // condiciones de frontera x=L_x (se preven dos puntos por si se usa Neumann; solo uno haria falta para Dirichlet)
											    
  lis_matrix_malloc_csr(n,A_nnz,&ptr,&index,&value);
  lis_matrix_create( LIS_COMM_WORLD, &_matrix_2d_LIS );
  lis_matrix_set_size( _matrix_2d_LIS, 0, NX*NZ );

  ptr[0] = 0;
  k=0;
  for(int i=0; i<NX; ++i)
    for(int j=0; j<NZ; ++j)
      {
  	int r = i*NZ+j;
  	int JX;
  	int JZ;
  	int s;
  	if(i == 0)
  	  {
	    if(0 <= j && j <= NZ-1)
  	      {
  		JX = 0; JZ = j; s = JX*NZ+JZ; index[k] = s; value[k] = matrix_2dconst(_ep,r,s); ++k;
  		JX = 1; JZ = j; s = JX*NZ+JZ; index[k] = s; value[k] = matrix_2dconst(_ep,r,s); ++k;
  	      }
  	  }
  	else if(1 <= i && i <= NX-2)
  	  {
  	    if(j == 0)
  	      {
  		JX = i; JZ = j;   s = JX*NZ+JZ; index[k] = s; value[k] = matrix_2dconst(_ep,r,s); ++k;
  		JX = i; JZ = j+1; s = JX*NZ+JZ; index[k] = s; value[k] = matrix_2dconst(_ep,r,s); ++k;
  	      }
  	    else if(1 <= j && j <= NZ-2)
  	      {
  		JX = i-1; JZ = j; s = JX*NZ+JZ; index[k] = s; value[k] = matrix_2dconst(_ep,r,s);      ++k;
  		JX = i;   JZ = 0;  s = JX*NZ+JZ; index[k] = s; value[k] = .5*CP*DZ*frechet(i,j,JZ) + (j-1<=JZ&&JZ<=j+1 ? matrix_2dconst(_ep,r,s) : 0); ++k;
  		for(int jp=1; jp<NZ-1; ++jp)
  		  {
  		    JX = i; JZ = jp; s = JX*NZ+JZ; index[k] = s; value[k] = CP*DZ*frechet(i,j,JZ) + (j-1<=JZ&&JZ<=j+1 ? matrix_2dconst(_ep,r,s) : 0); ++k;
  		  }
  		JX = i;   JZ = NZ-1; s = JX*NZ+JZ; index[k] = s; value[k] = .5*CP*DZ*frechet(i,j,JZ) + (j-1<=JZ&&JZ<=j+1 ? matrix_2dconst(_ep,r,s) : 0); ++k;
  		JX = i+1; JZ = j;     s = JX*NZ+JZ; index[k] = s; value[k] = matrix_2dconst(_ep,r,s);      ++k;
  	      }
  	    else if(j == NZ-1)
  	      {
  		JX = i; JZ = j-1; s = JX*NZ+JZ; index[k] = s; value[k] = matrix_2dconst(_ep,r,s); ++k;
  		JX = i; JZ = j;   s = JX*NZ+JZ; index[k] = s; value[k] = matrix_2dconst(_ep,r,s); ++k;
  	      }
  	  }
  	else if(i == NX-1)
  	  {
	    if(0 <= j && j <= NZ-1)
  	      {
  		JX = NX-2; JZ = j; s = JX*NZ+JZ; index[k] = s; value[k] = matrix_2dconst(_ep,r,s); ++k;
  		JX = NX-1; JZ = j; s = JX*NZ+JZ; index[k] = s; value[k] = matrix_2dconst(_ep,r,s); ++k;
  	      }
  	  }
  	ptr[r-0+1] = k;
      }

  lis_matrix_set_csr( A_nnz, ptr, index, value, _matrix_2d_LIS );
  lis_matrix_assemble( _matrix_2d_LIS );

  /************************************************************************************
   *                            construct the right-hand side                         *
   ************************************************************************************/
  lis_vector_duplicate( _matrix_2d_LIS, &_rhs_LIS );

  for(int i=0; i<NX; ++i)
    for(int j=0; j<NZ; ++j)
      {
  	int line = i*NZ+j;

  	// boundary conditions
  	if( j==0 && 1<=i && i<=NX-2 )
  	  {
  	    switch( bc_confined(_ep,i,j).condtype )
  	      {
  	      case DIRICHLET :
  		lis_vector_set_value( LIS_ADD_VALUE, line, bc_confined(_ep,i,j).point_value, _rhs_LIS );
  		break;
	      
  	      case NEUMANN :
  		lis_vector_set_value( LIS_ADD_VALUE, line, 0, _rhs_LIS );
  		break;
	      
  	      default : 
  		// cerr << "For POISSON problems, only DIRICHLET, NEUMANN boundary conditions are allowed."
  		//      << "Please modify the boundary condition." << endl;
  		throw error_ITER_CONSTRLINSYS_BOUNDCOND(i, j, bc_confined(_ep,i,j), _ep);
  		break;
  	      }	
  	  }
      
  	if( i==NX-1 && 0<=j && j<=NZ-1 )
  	  {
  	    switch( bc_confined(_ep,i,j).condtype )
  	      {
  	      case DIRICHLET :
  		lis_vector_set_value( LIS_ADD_VALUE, line, bc_confined(_ep,i,j).point_value, _rhs_LIS );
  		break;
	      
  	      case NEUMANN :
  		lis_vector_set_value( LIS_ADD_VALUE, line, 0, _rhs_LIS );
  		break;
	      
  	      default : 
  		// cerr << "For POISSON problems, only DIRICHLET, NEUMANN boundary conditions are allowed."
  		//      << "Please modify the boundary condition." << endl;
  		throw error_ITER_CONSTRLINSYS_BOUNDCOND(i, j, bc_confined(_ep,i,j), _ep);
  		break;
  	      }
  	  }
      
  	if( j==NZ-1 && 1<=i && i<=NX-2 )
  	  {
  	    switch( bc_confined(_ep,i,j).condtype )
  	      {
  	      case DIRICHLET :
  		lis_vector_set_value( LIS_ADD_VALUE, line, bc_confined(_ep,i,j).point_value, _rhs_LIS );
  		break;
	      
  	      case NEUMANN :
  		lis_vector_set_value( LIS_ADD_VALUE, line, 0, _rhs_LIS );
  		break;
	      
  	      default : 
  		// cerr << "For POISSON problems, only DIRICHLET, NEUMANN boundary conditions are allowed."
  		//      << "Please modify the boundary condition." << endl;
  		throw error_ITER_CONSTRLINSYS_BOUNDCOND(i, j, bc_confined(_ep,i,j), _ep);
  		break;
  	      }
  	  }
      
  	if( i==0 && 0<=j && j<=NZ-1 )
  	  {
  	    switch( bc_confined(_ep,i,j).condtype )
  	      {
  	      case DIRICHLET :
  		lis_vector_set_value( LIS_ADD_VALUE, line, bc_confined(_ep,i,j).point_value, _rhs_LIS );
  		break;
	      
  	      case NEUMANN :
  		lis_vector_set_value( LIS_ADD_VALUE, line, 0, _rhs_LIS );
  		break;
	      
  	      default : 
  		// cerr << "For POISSON problems, only DIRICHLET, NEUMANN boundary conditions are allowed."
  		//      << "Please modify the boundary condition." << endl;
  		throw error_ITER_CONSTRLINSYS_BOUNDCOND(i, j, bc_confined(_ep,i,j), _ep);
  		break;
  	      }
  	  }
      
      }

  for(int i=0; i<NX; ++i)
    for(int j=0; j<NZ; ++j)
      {
  	int line = i*NZ+j;
	
  	// inner points
  	if( 0 < i && i < NX-1 && 0 < j && j < NZ-1 )
  	  {
  	    lis_vector_set_value( LIS_ADD_VALUE, line, -CP*(totvoldens_OLD(i,j)-nd(i,j)), _rhs_LIS );
	    
  	    lis_vector_set_value( LIS_ADD_VALUE, line, .5*CP*DZ*frechet(i,j,0)*pot_OLD(i,0), _rhs_LIS );
  	    for(int jp=1; jp<NZ-1; ++jp)
  	      lis_vector_set_value( LIS_ADD_VALUE, line, CP*DZ*frechet(i,j,jp)*pot_OLD(i,jp), _rhs_LIS );
  	    lis_vector_set_value( LIS_ADD_VALUE, line, .5*CP*DZ*frechet(i,j,NZ-1)*pot_OLD(i,NZ-1), _rhs_LIS );
  	  }
      }

  return;
}

void MOSFETProblemCuda::CPU_constr_linsys_SRJ( const FixedPointType fpt )
{
  const int NX    = host_dm -> get_X()   -> get_N();
  const int NZ    = host_dm -> get_Z()   -> get_N();
  const double DZ = host_dm->get_Z()->get_delta();
  const double CP = host_adimpar->get_cp();

  memcpy( _matrix_2d, &_matrix_2dconst[2*NX*NZ*(2*NZ+1)], NX*NZ*(2*NZ+1)*sizeof(double) );
  memcpy( _rhs, _rhs_const, NX*NZ*sizeof(double) );

#pragma omp parallel for
  for(int global_index=0; global_index < (NX-2)*(NZ-2); ++global_index)
    {
      int i, j;

      GPU_map_1D_to_2D( global_index, &i, NX-2, &j, NZ-2 );
      ++i; ++j;

      int r = i*NZ+j;

      double aux_frechet[NZ];
      for(int jj=1; jj<NZ-1; ++jj)
  	     aux_frechet[jj] = frechet(i,j,jj);

      // coefficient matrix
      int JX;
      int JZ;
      int s;
      for(int jp=1; jp<NZ-1; ++jp)
  	  {
  	     JX = i; JZ = jp; s = JX*NZ+JZ; matrix_2d(r,s) += CP*DZ*aux_frechet[JZ];
  	  }
	  
      // right hand side
      _rhs[r] += -CP*(totvoldens_OLD(i,j)-nd(i,j));
      for(int jp=1; jp<NZ-1; ++jp)                               
  	    _rhs[r] += CP*DZ*aux_frechet[jp]*pot_OLD(i,jp);
    }
}




cuda_iter_spstep_constrlinsys_kernels_config::cuda_iter_spstep_constrlinsys_kernels_config(const discrMeshes *dm)
{
  const int NX    = dm -> get_X()   -> get_N();
  const int NZ    = dm -> get_Z()   -> get_N();

  const int threads_per_block = 32;
  const int number_of_blocks = nblocks( (NX-2)*(NZ-2), threads_per_block);
  cuda_constr_linsys_config = new kernelConfig(number_of_blocks, threads_per_block, NOSHMEM, cudaFuncCachePreferL1, "cuda_constr_linsys");
}



/*
  name          : 'check_constr_linsys_input', 'check_constr_linsys_output'
  last modified : 2023/06/06
  author        : Francesco VECIL

  description   : This functions check the input and the output of the 'constr_linsys' method.
                  As input  : (i) frechet, (ii) tovoldens_OLD, (iii) pot_OLD
 */
void MOSFETProblemCuda::check_constr_linsys_input()
{
  cerr << "called '" << __func__ << "'...";

  const int NSBN  = host_dm -> get_NSBN();
  const int NX    = host_dm -> get_X()   -> get_N();
  const int NZ    = host_dm -> get_Z()   -> get_N();

  bool stopcheck = false;
  
  /*
    begin section : check frechet
   */
  // cerr << "check frechet...";
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
  // cerr << "[ok]";
  /*
    end section : check frechet
   */

  /*
    begin section : check totvoldens_OLD
   */
  // cerr << "check totvoldens_OLD...";
  double *_aux_totvoldens_OLD = new double[ NX*NZ ];
#ifdef __CUDACC__
  CUDA_SAFE_CALL( cudaMemcpy( _aux_totvoldens_OLD, _GPU_totvoldens_OLD, NX*NZ*sizeof(double), cudaMemcpyDeviceToHost ) );
#else
  memcpy( _aux_totvoldens_OLD, _totvoldens_OLD, NX*NZ*sizeof(double) );
#endif
  for(int i=0; i<NX; ++i)
    {
      for(int j=0; j<NZ; ++j)
	{
	  const double val = _aux_totvoldens_OLD[ indx_i_j(i,j) ];
	  if( isnan( val ) || isinf( val ) )
	    {
	      cerr << " ERROR from function " << __func__ << ", from line " << __LINE__ << ", from file '" << __FILE__
		   << "' --- totvoldens_OLD(" << i << "," << j << ") = " << val << endl;
	      stopcheck = true;
	    }
	}
    }
  delete [] _aux_totvoldens_OLD;
  // cerr << "[ok]";
  /*
    end section : check frechet
   */

  /*
    begin section : Check pot
   */
  // cerr << "check pot OLD...";
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
  // cerr << "[ok]";
  /*
    end section : Check pot[spphase]
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
  name          : 'check_constr_linsys_input', 'check_constr_linsys_output'
  last modified : 2023/06/06
  author        : Francesco VECIL

  description   : This functions check the input and the output of the 'constr_linsys' method.
		  As output : (iv) matrix_2d, (v) rhs
 */
void MOSFETProblemCuda::check_constr_linsys_output()
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

  if( stopcheck == true )
    {
      cerr << " ERROR from function " << __func__ << ", from line " << __LINE__ << ", from file '" << __FILE__
	   << "' --- exiting..." << endl;
      exit(_ERR_CHECK_COMPUTE_FRECHET_INPUT_OUTPUT);
    }

  cerr << "[ok]" << endl;

  return;
}



