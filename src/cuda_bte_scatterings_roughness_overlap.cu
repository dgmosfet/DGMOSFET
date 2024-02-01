#include "mosfetproblem.h"
#include "debug_flags.h"

/**
   PURPOSE:        

   FILE:           cuda_bte_scatterings_roughness.cu

   NAME:           MOSFETProblem::compute_I_SR

   PARAMETERS:

   RETURN VALUE:   none

   CALLS TO:       MOSFETProblem::start_time                    (cuda_comptime.h - declared inline)
		   MOSFETProblem::stop_time                     (cuda_comptime.h - declared inline)
		   MOSFETProblem::stretch_totvoldens            (cuda_bte_scatterings_roughness.cu)
		   MOSFETProblem::solve_poisson_ext             (cuda_bte_scatterings_roughness.cu)
		   MOSFETProblem::compute_Deltapot              (cuda_bte_scatterings_roughness.cu)
		   MOSFETProblem::compute_overlap_SR            (cuda_bte_scatterings_roughness.cu)
 
   CALLED FROM:    MOSFETProblem::scatterings                   (cuda_time_integration.cu)

   DATA MODIFIED:  

   METHOD:         

   LAST UPDATE:    2022 February 16
*/
void MOSFETProblemCuda::compute_I_SR( const int stage )
{
  start_time( _PHASE_BTE_SCATTERINGS_ROUGHNESS_ISR );

  /****************************************************
   *      integracion de la rugosidad superficial     *
   ****************************************************/
  // primer paso: estirar la densidad para perturbar el sistema suponiendo que el cuerpo en silicio es mas grande de lo que estaba disenado
  stretch_totvoldens();
  
  // segundo paso: se resuelve Poisson para calcular el potencial perturbado; no hace falta el acoplamiento con Schroedinger para este calculo, puesto que la densidad la decidimos
  solve_poisson_ext();
  
  // // tercer paso: calculo de la diferencia entre el potencial no perturbado y el potencial perturbado
  compute_Deltapot();
  
  // cuarto paso: calculo de la cantidad que sustituye la integral de solapamiento, que es al final la cantidad que se usa de verdad en la expresion de ganancia y perdida
  compute_overlap_SR();

  stop_time( _PHASE_BTE_SCATTERINGS_ROUGHNESS_ISR );

  return;
}



#ifdef __CUDACC__
__global__ void kernel_compute_overlap_SR(const discrMeshes *dm, const double DZ, const double *_GPU_chi, const double *_GPU_Deltapot_upper, const double *_GPU_Deltapot_lower, double *_GPU_I_SR)
{
  extern __shared__ double aux[];

  int global_index = blockIdx.x*blockDim.x + threadIdx.x;
  
  const int NSBN     = dm -> get_NSBN();
  const int NX       = dm -> get_X()   -> get_N();
  const int NZ       = dm -> get_Z()   -> get_N();
  const int NW       = dm -> get_W()   -> get_N();
  const int NPHI     = dm -> get_PHI() -> get_N();

  if( global_index < _NVALLEYS * NSBN * NX )
    {
      // double aux[_NZMAX];
      for(int j=0; j<NZ; ++j)
	aux[j] = 0;
      double sum;

      // nu > i > p as ordering, but it does not seem suitable
      
      int index = global_index;
      int nu = index;
      index  = index/NSBN;
      int p  = nu-index*NSBN;
      nu     = index;
      index  = index/NX;
      int i  = nu-index*NX;
      nu     = index;

      GPU_I_SR(nu,p,i) = 0;

      for(int j=1; j<NZ-1; ++j)
	aux[j] = GPU_chi(nu,p,i,j)*GPU_chi(nu,p,i,j)*GPU_Deltapot_upper(i,j);
      sum = 0;
      for(int j=1; j<NZ-1; ++j)
	sum += aux[j];
      sum *= DZ;
      GPU_I_SR(nu,p,i) += sum*sum;

      for(int j=1; j<NZ-1; ++j)
	aux[j] = GPU_chi(nu,p,i,j)*GPU_chi(nu,p,i,j)*GPU_Deltapot_lower(i,j);
      sum = 0;
      for(int j=1; j<NZ-1; ++j)
	sum += aux[j];
      sum *= DZ;
      GPU_I_SR(nu,p,i) += sum*sum;
    }
}

void MOSFETProblemCuda::GPU_compute_overlap_SR()
{
  const int NSBN  = host_dm->get_NSBN();
  const int NX    = host_dm->get_X()->get_N();
  const int NZ    = host_dm->get_Z()->get_N();
  const double DZ = host_dm->get_Z()->get_delta();

  // const int blocksize = 1;
  // const int gridsize = ceil( (float)(_NVALLEYS*NSBN*NX)/ blocksize );
  const int gridSize      = host_gridconfig -> kernel_compute_overlap_SR_config -> get_gridSize();
  const int blockSize     = host_gridconfig -> kernel_compute_overlap_SR_config -> get_blockSize();
  const int shmemSize     = host_gridconfig -> kernel_compute_overlap_SR_config -> get_shmemSize();
  const cudaFuncCache cfc = host_gridconfig -> kernel_compute_overlap_SR_config -> get_cfc();
  cudaDeviceSetCacheConfig( cfc );
  kernel_compute_overlap_SR <<< gridSize, blockSize, shmemSize >>> (device_dm, DZ, _GPU_chi, _GPU_Deltapot_upper, _GPU_Deltapot_lower, _GPU_I_SR);
}
#else
void MOSFETProblemCuda::CPU_compute_overlap_SR()
{
  const int NSBN  = host_dm->get_NSBN();
  const int NX    = host_dm->get_X()->get_N();
  const int NZ    = host_dm->get_Z()->get_N();
  const int NW    = host_dm->get_W()->get_N();
  const int NPHI  = host_dm->get_PHI()->get_N();
  const double DZ = host_dm->get_Z()->get_delta();
  
  double *aux;
  aux = new double[NZ];
  for(int j=0; j<NZ; ++j)
    aux[j] = 0;

  for(int global_index=0; global_index<_NVALLEYS*NSBN*NX; ++global_index)
    {
      // nu > i > p as ordering, but it does not seem suitable
      
      int index = global_index;
      int nu = index;
      index  = index/NSBN;
      int p  = nu-index*NSBN;
      nu     = index;
      index  = index/NX;
      int i  = nu-index*NX;
      nu     = index;

      I_SR(nu,p,i) = 0;

      for(int j=1; j<NZ-1; ++j)
	aux[j] = chi(nu,p,i,j)*chi(nu,p,i,j)*Deltapot_upper(i,j);
      I_SR(nu,p,i) += pow(integrate_R1( aux, NZ, DZ, TRAPEZOIDS ),2);

      for(int j=1; j<NZ-1; ++j)
	aux[j] = chi(nu,p,i,j)*chi(nu,p,i,j)*Deltapot_lower(i,j);
      I_SR(nu,p,i) += pow(integrate_R1( aux, NZ, DZ, TRAPEZOIDS ),2);
    }
  delete [] aux;
}
#endif

/**
   PURPOSE:        

   FILE:           cuda_bte_scatterings_roughness.cu

   NAME:           MOSFETProblem::compute_overlap_SR

   PARAMETERS:

   RETURN VALUE:   none

   CALLS TO:       MOSFETProblem::start_time           (cuda_comptime.h - declared inline)
		   MOSFETProblem::stop_time            (cuda_comptime.h - declared inline)
 
   CALLED FROM:    MOSFETProblem::compute_I_SR         (cuda_bte_scatterings_roughness.cu)

   DATA MODIFIED:  

   METHOD:         

   LAST UPDATE:    2022 February 16
*/
void MOSFETProblemCuda::compute_overlap_SR()
{
  start_time( _PHASE_BTE_SCATTERINGS_ROUGHNESS_ISR_OVERLAPSR );

#ifdef __CUDACC__
  GPU_compute_overlap_SR();
#else
  CPU_compute_overlap_SR();
#endif
  
  stop_time( _PHASE_BTE_SCATTERINGS_ROUGHNESS_ISR_OVERLAPSR );

  return;
}

#ifdef __CUDACC__
__global__ void kernel_compute_Deltapot(const discrMeshes *dm, const solverParams *sp, const physDevice *pd, const double DZEXTM1, const double DZEXT, const double DZ, const double *_GPU_pot, const double *_GPU_pot_ext, double *_GPU_Deltapot_upper, double *_GPU_Deltapot_lower)
{
  int index = blockIdx.x*blockDim.x + threadIdx.x;

  const int NSBN     = dm -> get_NSBN();
  const int NX       = dm -> get_X()   -> get_N();
  const int NZ       = dm -> get_Z()   -> get_N();
  const int NW       = dm -> get_W()   -> get_N();
  const int NZEXT    = dm -> get_ZEXT()-> get_N();
  const int NPHI     = dm -> get_PHI() -> get_N();
  const double ZWIDTH = pd -> get_ZWIDTH();

  if( index<NX*NZ )
    {
      int i = index/NZ;
      int j = index-i*NZ;

      double zj = j*DZ;
      int jext = int(zj*DZEXTM1);
      double zextjext = jext*DZEXT;
      double zextjextp1 = (jext+1)*DZEXT;
      double slope = (GPU_pot_ext(i,jext+1)-GPU_pot_ext(i,jext))*DZEXTM1;
      double intercept = (GPU_pot_ext(i,jext)*zextjextp1-GPU_pot_ext(i,jext+1)*zextjext)*DZEXTM1;
      GPU_Deltapot_upper(i,j) = GPU_pot(i,j) - (slope*zj+intercept);

      zj = j*DZ + sp->get_DELTA_SR()/ZWIDTH;
      jext = int(zj*DZEXTM1);
      zextjext = jext*DZEXT;
      zextjextp1 = (jext+1)*DZEXT;
      slope = (GPU_pot_ext(i,jext+1)-GPU_pot_ext(i,jext))*DZEXTM1;
      intercept = (GPU_pot_ext(i,jext)*zextjextp1-GPU_pot_ext(i,jext+1)*zextjext)*DZEXTM1;
      GPU_Deltapot_lower(i,j) = GPU_pot(i,j) - (slope*zj+intercept);
    }
}

void MOSFETProblemCuda::GPU_compute_Deltapot()
{
  const int NX    = host_dm -> get_X()   -> get_N();
  const int NZ    = host_dm -> get_Z()   -> get_N();

  // const double DZEXTM1 = 1./_dzext;
  // const double DZEXT = _dzext;
  const double DZEXTM1 = host_dm -> get_ZEXT() -> get_delta_m1();
  const double DZEXT   = host_dm -> get_ZEXT() -> get_delta();
  const double DZ      = host_dm -> get_Z()    -> get_delta();

  const int gridSize      = host_gridconfig -> kernel_compute_Deltapot_config -> get_gridSize();
  const int blockSize     = host_gridconfig -> kernel_compute_Deltapot_config -> get_blockSize();
  const int shmemSize     = host_gridconfig -> kernel_compute_Deltapot_config -> get_shmemSize();
  const cudaFuncCache cfc = host_gridconfig -> kernel_compute_Deltapot_config -> get_cfc();
  cudaDeviceSetCacheConfig( cfc );
  kernel_compute_Deltapot <<< gridSize, blockSize, shmemSize >>> (device_dm, device_solvpar, device_physdesc, DZEXTM1, DZEXT, DZ, _GPU_pot, _GPU_pot_ext, _GPU_Deltapot_upper, _GPU_Deltapot_lower);
}
#else
void MOSFETProblemCuda::CPU_compute_Deltapot()
{
  const int NSBN  = host_dm->get_NSBN();
  const int NX    = host_dm->get_X()->get_N();
  const int NZ    = host_dm->get_Z()->get_N();
  const int NW    = host_dm->get_W()->get_N();
  const int NPHI  = host_dm->get_PHI()->get_N();

  const double ZWIDTH  = host_physdesc -> get_ZWIDTH();

  // tercer paso: calculo de la diferencia entre el potencial no perturbado y el potencial perturbado
  const double DZEXTM1 = 1./_dzext;
  const double DZEXT = _dzext;
  const double DZ = host_dm->get_Z()->get_delta();

  for(int index=0; index<NX*NZ; ++index)
    {
      int i = index/NZ;
      int j = index-i*NZ;

      double zj = j*DZ;
      int jext = int(zj*DZEXTM1);
      double zextjext = jext*DZEXT;
      double zextjextp1 = (jext+1)*DZEXT;
      double slope = (pot_ext(i,jext+1)-pot_ext(i,jext))*DZEXTM1;
      double intercept = (pot_ext(i,jext)*zextjextp1-pot_ext(i,jext+1)*zextjext)*DZEXTM1;
      Deltapot_upper(i,j) = pot(i,j) - (slope*zj+intercept);

      zj = j*DZ + host_solvpar->get_DELTA_SR()/ZWIDTH;
      jext = int(zj*DZEXTM1);
      zextjext = jext*DZEXT;
      zextjextp1 = (jext+1)*DZEXT;
      slope = (pot_ext(i,jext+1)-pot_ext(i,jext))*DZEXTM1;
      intercept = (pot_ext(i,jext)*zextjextp1-pot_ext(i,jext+1)*zextjext)*DZEXTM1;
      Deltapot_lower(i,j) = pot(i,j) - (slope*zj+intercept);
    }
}
#endif

/**
   PURPOSE:        

   FILE:           cuda_bte_scatterings_roughness.cu

   NAME:           MOSFETProblem::compute_Deltapot

   PARAMETERS:

   RETURN VALUE:   none

   CALLS TO:       MOSFETProblem::start_time           (cuda_comptime.h - declared inline)
		   MOSFETProblem::stop_time            (cuda_comptime.h - declared inline)
 
   CALLED FROM:    MOSFETProblem::compute_I_SR         (cuda_bte_scatterings_roughness.cu)

   DATA MODIFIED:  

   METHOD:         

   LAST UPDATE:    2022 February 16
*/
void MOSFETProblemCuda::compute_Deltapot()
{
  start_time( _PHASE_BTE_SCATTERINGS_ROUGHNESS_ISR_DELTAPOT );

#ifdef __CUDACC__
  GPU_compute_Deltapot();
#else
  CPU_compute_Deltapot();
#endif
  
  stop_time( _PHASE_BTE_SCATTERINGS_ROUGHNESS_ISR_DELTAPOT );

  return;
}


#ifdef __CUDACC__
__global__ void kernel_construct_rhs_ext(const discrMeshes *dm, const double CP, const double *_GPU_totvoldens_ext, double *_GPU_rhs_ext)
{
  int index = blockIdx.x*blockDim.x + threadIdx.x;
  const int NSBN     = dm -> get_NSBN();
  const int NX       = dm -> get_X()   -> get_N();
  const int NZ       = dm -> get_Z()   -> get_N();
  const int NW       = dm -> get_W()   -> get_N();
  const int NZEXT    = dm -> get_ZEXT()-> get_N();
  if( index < (NX-2)*(NZEXT-2) )
    {
      int IX = index/(NZEXT-2);
      int IZ = index-IX*(NZEXT-2);
      ++IX; ++IZ;
      _GPU_rhs_ext[IX*NZEXT+IZ] += -CP*(GPU_totvoldens_ext(IX,IZ));
    }
}

void MOSFETProblemCuda::GPU_construct_rhs_ext()
{
  const int NX    = host_dm -> get_X()   -> get_N();
  const int NZEXT = host_dm -> get_ZEXT()-> get_N();

  const double CP = host_adimpar->get_cp();
  checkCudaErrors( cudaMemcpy(_GPU_rhs_ext, _GPU_rhs_ext_const, NX*NZEXT*sizeof(double), cudaMemcpyDeviceToDevice) );
  // const int blocksize = 64;
  // const int gridsize = ceil( (float)(NX*NZEXT)/ blocksize );
  const int gridSize      = host_gridconfig -> kernel_construct_rhs_ext_config -> get_gridSize();
  const int blockSize     = host_gridconfig -> kernel_construct_rhs_ext_config -> get_blockSize();
  const int shmemSize     = host_gridconfig -> kernel_construct_rhs_ext_config -> get_shmemSize();
  const cudaFuncCache cfc = host_gridconfig -> kernel_construct_rhs_ext_config -> get_cfc();
  cudaDeviceSetCacheConfig( cfc );
  kernel_construct_rhs_ext <<< gridSize, blockSize, shmemSize >>> (device_dm, CP, _GPU_totvoldens_ext, _GPU_rhs_ext);
}

void MOSFETProblemCuda::GPU_solve_poisson_ext()
{
  GPU_construct_rhs_ext();

  const int NX    = host_dm -> get_X()   -> get_N();
  const int NZEXT = host_dm -> get_ZEXT()-> get_N();

  // compute _GPU_pot_ext
  const int gridSize      = host_gridconfig -> multiply_config -> get_gridSize();
  const int blockSize     = host_gridconfig -> multiply_config -> get_blockSize();
  const int shmemSize     = host_gridconfig -> multiply_config -> get_shmemSize();
  const cudaFuncCache cfc = host_gridconfig -> multiply_config -> get_cfc();
  cudaDeviceSetCacheConfig( cfc );
  multiply <<< gridSize, blockSize, shmemSize >>> ( device_dm, _GPU_matrix_2dconst_ext_inv, _GPU_rhs_ext, _GPU_pot_ext );
  return;
}

__global__ void multiply( const discrMeshes *dm,
			  const double *_GPU_matrix_2dconst_ext_inv,
			  const double *_GPU_rhs_ext, 
			  double *_GPU_pot_ext)
{
  int global_index = blockIdx.x*blockDim.x + threadIdx.x;

  const int NX    = dm -> get_X()   -> get_N();
  const int NZEXT = dm -> get_ZEXT()-> get_N();

  if( global_index < NX*NZEXT )
    {
      double res = 0;
      for(int k=0; k<NX*NZEXT; ++k)
	res += GPU_matrix_2dconst_ext_inv(global_index, k) * _GPU_rhs_ext[k];
      _GPU_pot_ext[global_index] = res;
    }
}

#else
void MOSFETProblemCuda::CPU_solve_poisson_ext()
{
  const int NX    = host_dm -> get_X()   -> get_N();
  const int NZ    = host_dm -> get_Z()   -> get_N();
  const int NZEXT = host_dm -> get_ZEXT()-> get_N();

  const double VLOWERGATE = host_physdesc -> get_VLOWERGATE();
  const double VUPPERGATE = host_physdesc -> get_VUPPERGATE();
  const double VBIAS      = host_physdesc -> get_VBIAS();
  
  // Poisson solver for the extended domain
  LIS_MATRIX _matrix_2d_ext;
  LIS_INT *ptr_ext,*index_ext;
  LIS_SCALAR *value_ext;
  int n = NX*NZEXT; 
  int k;
  const double CP = host_adimpar->get_cp();

  const int A_nnz_ext = 
    2*NZEXT                      
    +2*(NX-2)                 
    +(NX-2)*(NZEXT-2)*5         
    +(NX-2)*(NZEXT-2)*NZEXT       
    +2*(NX-2)                 
    +2*NZEXT;                    

  lis_matrix_malloc_csr(n,A_nnz_ext,&ptr_ext,&index_ext,&value_ext);
  lis_matrix_create( LIS_COMM_WORLD, &_matrix_2d_ext );
  lis_matrix_set_size( _matrix_2d_ext, 0, NX*NZEXT );

  ptr_ext[0] = 0;
  k=0;
  for(int i=0; i<NX; ++i)
    for(int j=0; j<NZEXT; ++j)
      {
  	int r = i*NZEXT+j;
  	int JX;
  	int JZ;
  	int s;
  	if(i == 0)
  	  {
	    if(0 <= j && j <= NZEXT-1)
  	      {
  		JX = 0; JZ = j; s = JX*NZEXT+JZ; index_ext[k] = s; value_ext[k] = 1.0; ++k;
  	      }
	    // if(0 <= j && j <= _NZEXT-1)
  	    //   {
  	    // 	JX = 0; JZ = j; s = JX*_NZEXT+JZ; index_ext[k] = s; value_ext[k] = matrix_2dconst_ext(r,s); ++k;
  	    // 	JX = 1; JZ = j; s = JX*_NZEXT+JZ; index_ext[k] = s; value_ext[k] = matrix_2dconst_ext(r,s); ++k;
  	    //   }
  	  }
  	else if(1 <= i && i <= NX-2)
  	  {
  	    if(j == 0)
  	      {
  		JX = i; JZ = j;   s = JX*NZEXT+JZ; index_ext[k] = s; value_ext[k] = matrix_2dconst_ext(r,s); ++k;
  		JX = i; JZ = j+1; s = JX*NZEXT+JZ; index_ext[k] = s; value_ext[k] = matrix_2dconst_ext(r,s); ++k;
  	      }
  	    else if(1 <= j && j <= NZEXT-2)
  	      {
  		JX = i-1; JZ = j; s = JX*NZEXT+JZ; index_ext[k] = s; value_ext[k] = matrix_2dconst_ext(r,s);      ++k;
  		JX = i;   JZ = 0; s = JX*NZEXT+JZ; index_ext[k] = s; value_ext[k] = (j-1<=JZ && JZ<=j+1 ? matrix_2dconst_ext(r,s) : 0); ++k;
  		for(int jp=1; jp<NZEXT-1; ++jp)
  		  {
  		    JX = i; JZ = jp; s = JX*NZEXT+JZ; index_ext[k] = s; value_ext[k] = (j-1<=JZ && JZ<=j+1 ? matrix_2dconst_ext(r,s) : 0); ++k;
  		  }
  		JX = i;   JZ = NZEXT-1; s = JX*NZEXT+JZ; index_ext[k] = s; value_ext[k] = (j-1<=JZ && JZ<=j+1 ? matrix_2dconst_ext(r,s) : 0); ++k;
  		JX = i+1; JZ = j;     s = JX*NZEXT+JZ; index_ext[k] = s; value_ext[k] = matrix_2dconst_ext(r,s);      ++k;
  	      }
  	    else if(j == NZEXT-1)
  	      {
  		JX = i; JZ = j-1; s = JX*NZEXT+JZ; index_ext[k] = s; value_ext[k] = matrix_2dconst_ext(r,s); ++k;
  		JX = i; JZ = j;   s = JX*NZEXT+JZ; index_ext[k] = s; value_ext[k] = matrix_2dconst_ext(r,s); ++k;
  	      }
  	  }
  	else if(i == NX-1)
  	  {
	    // if(0 <= j && j <= _NZEXT-1)
  	    //   {
  	    // 	JX = _NX-2; JZ = j; s = JX*_NZEXT+JZ; index_ext[k] = s; value_ext[k] = matrix_2dconst_ext(r,s); ++k;
  	    // 	JX = _NX-1; JZ = j; s = JX*_NZEXT+JZ; index_ext[k] = s; value_ext[k] = matrix_2dconst_ext(r,s); ++k;
  	    //   }
	    if(0 <= j && j <= NZEXT-1)
  	      {
  		JX = NX-1; JZ = j; s = JX*NZEXT+JZ; index_ext[k] = s; value_ext[k] = 1.0; ++k;
  	      }
  	  }
  	ptr_ext[r-0+1] = k;
      }

  lis_matrix_set_csr( A_nnz_ext, ptr_ext, index_ext, value_ext, _matrix_2d_ext );
  lis_matrix_assemble( _matrix_2d_ext );

  LIS_VECTOR _rhs_ext;
  lis_vector_duplicate( _matrix_2d_ext, &_rhs_ext );

  for(int IX = 0; IX < NX; ++IX)
    for(int IZ = 0; IZ < NZEXT; ++IZ )
    {
      int line = IX*NZEXT + IZ;

      // inner points
      if( 0 < IX && IX < NX-1 && 0 < IZ && IZ < NZEXT-1 )
  	{
  	  lis_vector_set_value( LIS_ADD_VALUE, line, -CP*(totvoldens_ext(IX,IZ)-nd_ext(IX,IZ)), _rhs_ext );
  	}

      // boundary conditions
      if( 0 < IX && IX < NX-1 && IZ == 0 )
  	{
  	  if(isgate(IX,0))
  	    lis_vector_set_value( LIS_INS_VALUE, line, VLOWERGATE/host_rescpar->get_potstar(), _rhs_ext );
  	  else
  	    lis_vector_set_value( LIS_INS_VALUE, line, 0, _rhs_ext );
  	}

      if( 0 < IX && IX < NX-1 && IZ == NZEXT-1 )
  	{
  	  if(isgate(IX,NZ-1))
  	    lis_vector_set_value( LIS_INS_VALUE, line, VUPPERGATE/host_rescpar->get_potstar(), _rhs_ext );
  	  else
  	    lis_vector_set_value( LIS_INS_VALUE, line, 0, _rhs_ext );
  	}
      
      if( 0 <= IZ && IZ <= NZEXT-1 && IX == 0 )
  	{
  	  lis_vector_set_value( LIS_INS_VALUE, line, pot_b_ext(IZ), _rhs_ext );
  	}

      if( 0 <= IZ && IZ <= NZEXT-1 && IX == NX-1 )
  	{
  	  lis_vector_set_value( LIS_INS_VALUE, line, pot_b_ext(IZ) + VBIAS/host_rescpar->get_potstar(), _rhs_ext );
  	}
    }

  LIS_VECTOR _res_ext;
  lis_vector_duplicate( _matrix_2d_ext, &_res_ext );

  LIS_SOLVER _solver;
  lis_solver_create( &_solver );
  char opts[] = "-i bicgstab -p ilut -tol 1.0e-8";
  lis_solver_set_option( opts, _solver);
  lis_solve( _matrix_2d_ext, _rhs_ext, _res_ext, _solver );

  double *_temp_pot_ext;
  _temp_pot_ext = new double [NX*NZEXT];
  lis_vector_gather( _res_ext, _temp_pot_ext );
  // copy the result
  for(int i=0; i<NX; ++i)
    for(int j=0; j<NZEXT; ++j)
      pot_ext(i,j) = _temp_pot_ext[i*NZEXT+j];
  delete [] _temp_pot_ext;

  lis_matrix_destroy( _matrix_2d_ext );
  lis_vector_destroy( _res_ext );
  lis_vector_destroy( _rhs_ext );

  return;
}
#endif

/**
   PURPOSE:        

   FILE:           cuda_bte_scatterings_roughness.cu

   NAME:           MOSFETProblem::solve_poisson_ext

   PARAMETERS:

   RETURN VALUE:   none

   CALLS TO:       MOSFETProblem::start_time           (cuda_comptime.h - declared inline)
		   MOSFETProblem::stop_time            (cuda_comptime.h - declared inline)
 
   CALLED FROM:    MOSFETProblem::compute_I_SR         (cuda_bte_scatterings_roughness.cu)

   DATA MODIFIED:  

   METHOD:         

   LAST UPDATE:    2022 February 16
*/
void MOSFETProblemCuda::solve_poisson_ext()
{
  start_time( _PHASE_BTE_SCATTERINGS_ROUGHNESS_ISR_SOLVEPOISSEXT );
  
#ifdef __CUDACC__
  GPU_solve_poisson_ext();
#else
  CPU_solve_poisson_ext();
#endif
  
  stop_time( _PHASE_BTE_SCATTERINGS_ROUGHNESS_ISR_SOLVEPOISSEXT );
  
  return;
}










#ifdef __CUDACC__
__global__ void kernel_stretch_totvoldens( const discrMeshes *dm, const double *_GPU_totvoldens, const double DZM1, const double *_GPU_sigma, const double *_GPU_Sigma, double *_GPU_totvoldens_ext )
{
  int index = blockIdx.x*blockDim.x + threadIdx.x;
  const int NX       = dm -> get_X()   -> get_N();
  const int NZ       = dm -> get_Z()   -> get_N();
  const int NZEXT    = dm -> get_ZEXT()-> get_N();
  const double DZ  = dm->get_Z()->get_delta();
  if( index < NX*NZEXT )
    {
      const int i = index/NZEXT;
      const int jext = index-i*NZEXT;

      const int j = _GPU_Sigma[jext];
      const double zj = j*DZ;
      const double zjp1 = (j+1)*DZ;
      const double slope = (_GPU_totvoldens[i*NZ+j+1]-_GPU_totvoldens[i*NZ+j])*DZM1;
      const double intercept = (_GPU_totvoldens[i*NZ+j]*zjp1-_GPU_totvoldens[i*NZ+j+1]*zj)*DZM1;
      _GPU_totvoldens_ext[i*NZEXT+jext] = slope*_GPU_sigma[jext] + intercept;
    }

  return;
}
void MOSFETProblemCuda::GPU_stretch_totvoldens()
{
  const int NX    = host_dm -> get_X()   -> get_N();
  const int NZEXT = host_dm -> get_ZEXT()-> get_N();

  // const double DZM1 = 1./_dz;
  const double DZM1 = host_dm->get_Z()->get_delta_m1();
  // const int blocksize = 64;
  // const int gridsize = ceil( (float)(NX*NZEXT)/ blocksize );
  const int gridSize      = host_gridconfig -> kernel_stretch_totvoldens_config -> get_gridSize();
  const int blockSize     = host_gridconfig -> kernel_stretch_totvoldens_config -> get_blockSize();
  const int shmemSize     = host_gridconfig -> kernel_stretch_totvoldens_config -> get_shmemSize();
  const cudaFuncCache cfc = host_gridconfig -> kernel_stretch_totvoldens_config -> get_cfc();
  cudaDeviceSetCacheConfig( cfc );
  kernel_stretch_totvoldens <<< gridSize, blockSize, shmemSize >>> ( device_dm, _GPU_totvoldens, DZM1, _GPU_sigma, _GPU_Sigma, _GPU_totvoldens_ext );

  return;
}
#else

void MOSFETProblemCuda::CPU_stretch_totvoldens()
{
  const int NX    = host_dm -> get_X()   -> get_N();
  const int NZEXT = host_dm -> get_ZEXT()-> get_N();
  const double dzm1 = host_dm->get_Z()->get_delta_m1();

#pragma omp parallel for
  for( int index = 0; index < NX*NZEXT; ++index )
    {
      int i = index/NZEXT;
      int jext = index-i*NZEXT;

      int j = _Sigma[jext];
      double slope = (totvoldens(i,j+1)-totvoldens(i,j))*dzm1;
      double intercept = (totvoldens(i,j)*host_dm->get_Z()->mesh(j+1)-totvoldens(i,j+1)*host_dm->get_Z()->mesh(j+1))*dzm1;
      totvoldens_ext(i,jext) = slope*_sigma[jext] + intercept;
    }

  return;
}
#endif



/**
   PURPOSE:        

   FILE:           cuda_bte_scatterings_roughness.cu

   NAME:           MOSFETProblem::stretch_totvoldens

   PARAMETERS:

   RETURN VALUE:   none

   CALLS TO:       MOSFETProblem::start_time           (cuda_comptime.h - declared inline)
		   MOSFETProblem::stop_time            (cuda_comptime.h - declared inline)
 
   CALLED FROM:    MOSFETProblem::compute_I_SR         (cuda_bte_scatterings_roughness.cu)

   DATA MODIFIED:  

   METHOD:         

   LAST UPDATE:    2022 February 16
*/
void MOSFETProblemCuda::stretch_totvoldens()
{
  start_time( _PHASE_BTE_SCATTERINGS_ROUGHNESS_ISR_STRETCHTOTVOLDENS );

#ifdef __CUDACC__
  GPU_stretch_totvoldens();
#else
  CPU_stretch_totvoldens();
#endif
  
  stop_time( _PHASE_BTE_SCATTERINGS_ROUGHNESS_ISR_STRETCHTOTVOLDENS );

  return;
}


