#include "mosfetproblem.h"
#include "debug_flags.h"





/**
   PURPOSE:        

   FILE:           cuda_bte_scatterings_roughness.cu

   NAME:           MOSFETProblem::roughness_gain

   PARAMETERS:

   RETURN VALUE:   none

   CALLS TO:       MOSFETProblem::start_time           (cuda_comptime.h - declared inline)
		   MOSFETProblem::stop_time            (cuda_comptime.h - declared inline)
 
   CALLED FROM:    MOSFETProblem::scatterings          (cuda_time_integration.cu)

   DATA MODIFIED:  

   METHOD:         

   LAST UPDATE:    2023/06/15
*/
void MOSFETProblemCuda::roughness_gain( const int STAGE )
{
  start_time( _PHASE_BTE_SCATTERINGS_ROUGHNESS_ROUGHNESSGAIN );

#ifdef __CUDACC__
  const int NSBN  = host_dm -> get_NSBN();
  const int NX    = host_dm -> get_X()   -> get_N();
  const int NW    = host_dm -> get_W()   -> get_N();
  const int NPHI  = host_dm -> get_PHI() -> get_N();

  checkCudaErrors( cudaMemcpyToSymbol(_GPU_sqrtmassXY,  _sqrtmassXY,  _NVALLEYS*sizeof(double),             0, cudaMemcpyHostToDevice) );
  checkCudaErrors( cudaMemcpyToSymbol(_GPU_kane,        _kane,        _NVALLEYS*sizeof(double),             0, cudaMemcpyHostToDevice) );
  checkCudaErrors( cudaMemcpyToSymbol(_GPU_mass,        _mass,        9*sizeof(double),                     0, cudaMemcpyHostToDevice) );

  const int gridSize      = host_gridconfig -> kernel_roughness_gain_20230616_config -> get_gridSize();
  const int blockSize     = host_gridconfig -> kernel_roughness_gain_20230616_config -> get_blockSize();
  const int shmemSize     = host_gridconfig -> kernel_roughness_gain_20230616_config -> get_shmemSize();
  const cudaFuncCache cfc = host_gridconfig -> kernel_roughness_gain_20230616_config -> get_cfc();
  cudaDeviceSetCacheConfig( cfc );
  kernel_roughness_gain_20230616 <<< gridSize, blockSize, shmemSize >>> ( device_dm, device_rescpar, _GPU_pdf, _GPU_I_SR, _c_SR, _GPU_rhs_pdf,  _GPU_denom_SR, STAGE, _GPU_test_gain );
#else
  CPU_roughness_gain ( STAGE );
#endif

  stop_time( _PHASE_BTE_SCATTERINGS_ROUGHNESS_ROUGHNESSGAIN );

  return;
}

/**
   PURPOSE:        

   FILE:           cuda_bte_scatterings_roughness.cu

   NAME:           MOSFETProblem::roughness_loss

   PARAMETERS:

   RETURN VALUE:   none

   CALLS TO:       MOSFETProblem::start_time           (cuda_comptime.h - declared inline)
		   MOSFETProblem::stop_time            (cuda_comptime.h - declared inline)
 
   CALLED FROM:    MOSFETProblem::scatterings          (cuda_time_integration.cu)

   DATA MODIFIED:  

   METHOD:         

   LAST UPDATE:    2022 February 16
*/
void MOSFETProblemCuda::roughness_loss( const int STAGE )
{
  start_time( _PHASE_BTE_SCATTERINGS_ROUGHNESS_ROUGHNESSLOSS );

#ifdef __CUDACC__
  const int NSBN  = host_dm -> get_NSBN();
  const int NX    = host_dm -> get_X()   -> get_N();
  const int NW    = host_dm -> get_W()   -> get_N();
  const int NPHI  = host_dm -> get_PHI() -> get_N();

  checkCudaErrors( cudaMemcpyToSymbol(_GPU_mass,        _mass,        9*sizeof(double),                     0, cudaMemcpyHostToDevice) );
  checkCudaErrors( cudaMemcpyToSymbol(_GPU_kane,        _kane,        _NVALLEYS*sizeof(double),             0, cudaMemcpyHostToDevice) );

  const int gridSize      = host_gridconfig -> kernel_roughness_loss_config -> get_gridSize();
  const int blockSize     = host_gridconfig -> kernel_roughness_loss_config -> get_blockSize();
  const int shmemSize     = host_gridconfig -> kernel_roughness_loss_config -> get_shmemSize();
  const cudaFuncCache cfc = host_gridconfig -> kernel_roughness_loss_config -> get_cfc();
  cudaDeviceSetCacheConfig( cfc );
  kernel_roughness_loss <<< gridSize, blockSize, shmemSize >>> ( device_dm, device_rescpar, _GPU_pdf, _GPU_integrateddenom_SR, _GPU_I_SR, _c_SR, _GPU_rhs_pdf, _GPU_denom_SR, STAGE , _GPU_test_loss );
#else
  CPU_roughness_loss ( STAGE );
#endif

  stop_time( _PHASE_BTE_SCATTERINGS_ROUGHNESS_ROUGHNESSLOSS );

  return;
}


#ifdef __CUDACC__
/*
  name        : 'kernel_roughness_gain_20230616'
  author      : Jose Miguel MANTAS RUIZ, Francesco VECIL
  date        : 2023/06/16
  modifies    : 2023/06/16

  description : HIERARCHY.
                This kernel implements the gain part for the roughness scattering operator
                using the following hierarchy: nu > l > i > p > m
		Each thread will take care of one integration of pdf againt denom_SR,
		i.e. each thread will compute one value for rhs(nu,p,i,l,m).

		GRID AND BLOCK SIZE.
		In order to do that, we need a total of NVALLEYS*NSBN*NX*NW*NPHI threads.
		Therefore, if blockSize is the block size, then the grid size will be
		gridSize = ceil( NVALLEYS*NSBN*NX*NW*NPHI / blockSize).

		ARRAY ORDERING.
		INPUT ---- Vector     I_SR   is ordered        i > nu >  p
		INPUT ---- Vector denom_SR   is ordered            nu      > l > m > mp           and is a precomputed constant vector
		INPUT ---- Vector      pdf   is ordered   s >  i > nu >  p > l     > mp  
		OUTPUT --- Vector      rhs   is ordered        i > nu >  p > l > m   
		
		SHARED MEMORY.
		We have to allocate   ceil( (blockDim.x-1)/NPHI ) + 1   lines of PHI-values.
  notes       :
 */
__global__ void kernel_roughness_gain_20230616( const discrMeshes *dm,
						const rescalingParams *rescpar,
						const double *_GPU_pdf,
						const double *_GPU_I_SR, 
						const double CSR, 
						double *_GPU_rhs_pdf, 
						const double *_GPU_denom_SR, 
						const int STAGE,
						double *_GPU_test_gain )
{
  const int NSBN     = dm -> get_NSBN();
  const int NX       = dm -> get_X()   -> get_N();
  const int NW       = dm -> get_W()   -> get_N();
  const int NPHI     = dm -> get_PHI() -> get_N();

  extern __shared__ double shmem[];

  int global_index = blockIdx.x*blockDim.x + threadIdx.x;

  if( global_index < NX*_NVALLEYS*NSBN*NW*NPHI )
    {
      int i, nu, p, l, m;
      GPU_map_1D_to_5D( global_index, &i, NX, &nu, _NVALLEYS, &p, NSBN, &l, NW, &m, NPHI );

      const int block_first_gl_index = blockIdx.x * blockDim.x;
      const int block_last_gl_index  = (blockIdx.x+1) * blockDim.x - 1;

      int i0, nu0, p0, l0, m0;
      GPU_map_1D_to_5D( block_first_gl_index, &i0, NX, &nu0, _NVALLEYS, &p0, NSBN, &l0, NW, &m0, NPHI );

      const int first_i_nu_p_l_index = block_first_gl_index / NPHI; // = indx_i_nu_p_l_m(nu,p,i,l,0) ?
      const int last_i_nu_p_l_index  = block_last_gl_index  / NPHI;
      const int number_of_PHI_rows   = last_i_nu_p_l_index - first_i_nu_p_l_index + 1;

      // LOAD pdf TO SHARED MEMORY
      const int load_index_start     = STAGE*NX*_NVALLEYS*NSBN*NW*NPHI + first_i_nu_p_l_index*NPHI;
      const int nelems_to_load       = number_of_PHI_rows * NPHI;
      const int read_cycles          = (nelems_to_load + blockDim.x - 1) / blockDim.x;
      for( int u=0; u<read_cycles; ++u )
	{
	  const int idx = u*blockDim.x + threadIdx.x;
	  if( idx < nelems_to_load )
	    {
	      // shmem[ idx ] = _GPU_pdf[ load_index_start + idx ];
	      shmem[ idx ] = _GPU_pdf[ load_index_start + idx ];
	    }
	}

      __syncthreads();
      const int shmem_entry_point_for_this_thread = (global_index/NPHI - first_i_nu_p_l_index)*NPHI;
      
      // AND NOW THE COMPUTATIONS
      const double WL        = dm -> get_W() -> mesh(l);
      const double EPSSTAR   = rescpar->get_epsstar();
      const double DPHI      = dm -> get_PHI() -> get_delta();
      const double s_chcoord = GPU_sqrtmassXY(nu)*(1.+2.*GPU_kane(nu)*EPSSTAR*WL);
      const double intfact   = CSR*GPU_I_SR(nu,p,i)*s_chcoord*DPHI;

      // INTEGRATE
      double ganancia = 0;
      for(int mp=0; mp<NPHI; ++mp)
	{
	  // ganancia += GPU_pdf(nu,p,i,l,mp,STAGE) * GPU_denom_SR(nu,l,m,mp);
	  ganancia += shmem[ shmem_entry_point_for_this_thread + mp ] * GPU_denom_SR(nu,l,m,mp);
	}
      ganancia *= intfact;
      GPU_rhs_pdf(nu,p,i,l,m) += ganancia;
      
      // // comment for non-debugging run
      // GPU_test_gain(nu,p,i,l,m) = ganancia;
    }
}



#else
void MOSFETProblemCuda::CPU_roughness_gain( const int STAGE )
{
  const double DPHI = host_dm->get_PHI()->get_delta();

  const int NSBN = host_dm->get_NSBN();
  const int NX   = host_dm->get_X()->get_N();
  const int NZ   = host_dm->get_Z()->get_N();
  const int NW   = host_dm->get_W()->get_N();
  const int NPHI = host_dm->get_PHI()->get_N();

#pragma omp parallel for
  for( int globidx=0; globidx<_NVALLEYS*NW*NX*NSBN; ++globidx )
    {
      // int index = globidx;
      int nu, p, i, l;
      GPU_map_1D_to_4D( globidx, &nu, _NVALLEYS, &l, NW, &i, NX, &p, NSBN );

      // AND NOW THE COMPUTATIONS
      const double WL = host_dm->get_W()->mesh(l);
      const double EPSSTAR = rescpar->get_epsstar();
      const double s_chcoord = sqrtmassXY(nu)*(1.+2.*_kane[nu]*EPSSTAR*WL);
      const double intfact = _c_SR*I_SR(nu,p,i)*s_chcoord*DPHI;
	    
      // INTEGRATE
      for(int m=0; m<NPHI; ++m)
	{
	  double ganancia = 0;
	  for(int mp=0; mp<NPHI; ++mp)
	    ganancia += pdf(nu,p,i,l,mp,STAGE) * denom_SR(nu,l,m,mp);
	  ganancia *= intfact;
	  rhs_pdf(nu,p,i,l,m) += ganancia;

	  // test_gain(nu,p,i,l,m) = ganancia;
	}
	    
    }
}
#endif

#ifdef __CUDACC__
__global__ void kernel_roughness_loss( const discrMeshes *dm,
				       const rescalingParams *rescpar,
				       const double *_GPU_pdf,
				       const double *_GPU_integrateddenom_SR, 
				       const double *_GPU_I_SR, 
				       const double CSR, 
				       double *_GPU_rhs_pdf, 
				       const double *_GPU_denom_SR, 
				       const int STAGE,
				       double *_GPU_test_loss )
{
  int global_index = blockIdx.x*blockDim.x + threadIdx.x;

  const int NSBN     = dm -> get_NSBN();
  const int NX       = dm -> get_X()   -> get_N();
  const int NW       = dm -> get_W()   -> get_N();
  const int NPHI     = dm -> get_PHI() -> get_N();

  if( global_index < _NVALLEYS*NSBN*NX*NW*NPHI )
    {
      int nu,p,i,l,m;
      GPU_map_1D_to_5D( global_index, &i, NX, &nu, _NVALLEYS, &p, NSBN, &l, NW, &m, NPHI );

      const double WL = dm->get_W()->mesh(l);
      const double EPSSTAR = rescpar->get_epsstar();
      const double GPU_s_chcoord = sqrt(GPU_mass(Xdim,nu)*GPU_mass(Ydim,nu))*(1.+2.*GPU_kane(nu)*EPSSTAR*WL);
      const double aux_I_SR = GPU_I_SR(nu,p,i);

      GPU_rhs_pdf(nu,p,i,l,m) -= CSR*aux_I_SR*GPU_s_chcoord*GPU_pdf(nu,p,i,l,m,STAGE)*GPU_integrateddenom_SR(nu,l,m);

      // // comment for non-debugging run
      // GPU_test_loss(nu,p,i,l,m) = CSR*aux_I_SR*GPU_s_chcoord*GPU_pdf(nu,p,i,l,m,STAGE)*GPU_integrateddenom_SR(nu,l,m);
    }
}
#else
void MOSFETProblemCuda::CPU_roughness_loss( const int STAGE )
{
  const int NSBN  = host_dm -> get_NSBN();
  const int NX    = host_dm -> get_X()   -> get_N();
  const int NZ    = host_dm -> get_Z()   -> get_N();
  const int NW    = host_dm -> get_W()   -> get_N();
  const int NPHI  = host_dm -> get_PHI() -> get_N();

#pragma omp parallel for
  for( int globidx=0; globidx<_NVALLEYS*NSBN*NX*NW*NPHI; ++globidx )
    {
      // int index = globidx;
      int nu, p, i, l, m;
      GPU_map_1D_to_5D( globidx, &i, NX, &nu, _NVALLEYS, &p, NSBN, &l, NW, &m, NPHI );

      const double WL = host_dm->get_W()->mesh(l);
      const double EPSSTAR = host_rescpar->get_epsstar();
      const double s_chcoord = sqrt(mass(Xdim,nu)*mass(Ydim,nu))*(1.+2.*_kane[nu]*EPSSTAR*WL);
      const double aux_I_SR = I_SR(nu,p,i);
      
      rhs_pdf(nu,p,i,l,m) -= _c_SR*aux_I_SR*s_chcoord*pdf(nu,p,i,l,m,STAGE)*integrateddenom_SR(nu,l,m);
      // test_loss(nu,p,i,l,m) = _c_SR*aux_I_SR*s_chcoord*pdf(nu,p,i,l,m,STAGE)*integrateddenom_SR(nu,l,m);
    }
}
#endif




















// __global__ void kernel_roughness_gain_20230616_2( const discrMeshes *dm,
// 						  const rescalingParams *rescpar,
// 						  const solverParams *solvpar,
// 						  const double *_GPU_pdf,
// 						  const double *_GPU_I_SR, 
// 						  const double CSR, 
// 						  double *_GPU_rhs_pdf, 
// 						  const double *_GPU_denom_SR, 
// 						  const int STAGE,
// 						  double *_GPU_test_gain )
// {
//   const int NSBN     = dm -> get_NSBN();
//   const int NX       = dm -> get_X()   -> get_N();
//   const int NW       = dm -> get_W()   -> get_N();
//   const int NPHI     = dm -> get_PHI() -> get_N();

//   extern __shared__ double shmem[];

//   int global_index = blockIdx.x*blockDim.x + threadIdx.x;

//   if( global_index < NX*_NVALLEYS*NSBN*NW*NPHI )
//     {
//       int i, nu, p, l, m;
//       GPU_map_1D_to_5D( global_index, &i, NX, &nu, _NVALLEYS, &p, NSBN, &l, NW, &m, NPHI );

//       const int block_first_gl_index = blockIdx.x * blockDim.x;
//       const int block_last_gl_index  = (blockIdx.x+1) * blockDim.x - 1;

//       int i0, nu0, p0, l0, m0;
//       GPU_map_1D_to_5D( block_first_gl_index, &i0, NX, &nu0, _NVALLEYS, &p0, NSBN, &l0, NW, &m0, NPHI );

//       const int first_i_nu_p_l_index = block_first_gl_index / NPHI; // = indx_i_nu_p_l_m(nu,p,i,l,0) ?
//       const int last_i_nu_p_l_index  = block_last_gl_index  / NPHI;
//       const int number_of_PHI_rows   = last_i_nu_p_l_index - first_i_nu_p_l_index + 1;

//       // LOAD pdf TO SHARED MEMORY
//       const int load_index_start     = STAGE*NX*_NVALLEYS*NSBN*NW*NPHI + first_i_nu_p_l_index*NPHI;
//       const int nelems_to_load       = number_of_PHI_rows * NPHI;
//       const int read_cycles          = (nelems_to_load + blockDim.x - 1) / blockDim.x;
//       for( int u=0; u<read_cycles; ++u )
// 	{
// 	  const int idx = u*blockDim.x + threadIdx.x;
// 	  if( idx < nelems_to_load )
// 	    {
// 	      shmem[ idx ] = _GPU_pdf[ load_index_start + idx ];
// 	    }
// 	}

//       __syncthreads();
//       const int shmem_entry_point_for_this_thread = (global_index/NPHI - first_i_nu_p_l_index)*NPHI;
      
//       // AND NOW THE COMPUTATIONS
//       const double WL        = dm -> get_W() -> mesh(l);
//       const double EPSSTAR   = rescpar->get_epsstar();
//       const double DPHI      = dm -> get_PHI() -> get_delta();
//       const double s_chcoord = GPU_sqrtmassXY(nu)*(1.+2.*GPU_kane(nu)*EPSSTAR*WL);
//       const double intfact   = CSR*GPU_I_SR(nu,p,i)*s_chcoord*DPHI;

//       // INTEGRATE
//       double ganancia = 0;
//       for(int mp=0; mp<NPHI; ++mp)
// 	{
// 	  const double AUX_CST     = 4.*rescpar->get_kstar()*rescpar->get_kstar()*solvpar->get_LENGTH_SR()*solvpar->get_LENGTH_SR()*WL*(1+_GPU_kane[nu]*rescpar->get_epsstar()*WL);
// 	  const double SIN_MINUS   = sin( .5*(dm->get_PHI()->mesh(m)-dm->get_PHI()->mesh(mp)) );
// 	  const double SIN_MINUS_2 = SIN_MINUS * SIN_MINUS;
// 	  const double SIN_PLUS    = sin( .5*(dm->get_PHI()->mesh(m)+dm->get_PHI()->mesh(mp)) );
// 	  const double SIN_PLUS_2  = SIN_PLUS * SIN_PLUS;
// 	  const double COS_PLUS    = cos( .5*(dm->get_PHI()->mesh(m)+dm->get_PHI()->mesh(mp)) );
// 	  const double COS_PLUS_2  = COS_PLUS * COS_PLUS;
// 	  const double denom       = 1.+AUX_CST*SIN_MINUS_2*(GPU_mass(Xdim,nu)*SIN_PLUS_2+GPU_mass(Ydim,nu)*COS_PLUS_2);
// 	  const double d_SR        = 1./pow( denom,1.5 );

// 	  // if( fabs(GPU_denom_SR(nu,l,m,mp) - d_SR) > 1.E-8 )
// 	  //   {
// 	  //     printf("Nooooo!!\n");
// 	  //   }
	  
// 	  ganancia += shmem[ shmem_entry_point_for_this_thread + mp ] * d_SR;
// 	}
//       ganancia *= intfact;
//       GPU_rhs_pdf(nu,p,i,l,m) += ganancia;
      
//       // // comment for non-debugging run
//       // GPU_test_gain(nu,p,i,l,m) = ganancia;
//     }
// }



// __global__ void kernel_roughness_gain_7( const discrMeshes *dm,
// 					 const rescalingParams *rescpar,
// 					 const double *_GPU_pdf,
// 					 const double *_GPU_integrateddenom_SR, 
// 					 const double *_GPU_I_SR, 
// 					 const double CSR, 
// 					 double *_GPU_rhs_pdf, 
// 					 const double *_GPU_denom_SR, 
// 					 const int STAGE,
// 					 double *_GPU_test_gain )
// {
//   const int NSBN     = dm -> get_NSBN();
//   const int NX       = dm -> get_X()   -> get_N();
//   const int NW       = dm -> get_W()   -> get_N();
//   const int NPHI     = dm -> get_PHI() -> get_N();

//   extern __shared__ double shmem[];

//   int global_index = blockIdx.x*blockDim.x + threadIdx.x;

//   if( global_index < NX*_NVALLEYS*NSBN*NW )
//     {
//       int i, nu, p, l;
//       GPU_map_1D_to_4D( global_index, &nu, _NVALLEYS, &l, NW, &i, NX, &p, NSBN );

//       // load denom_SR to shared memory
//       const int H_0 = nu*NW*NPHI*NPHI + l*NPHI*NPHI;
//       const int ncycles = (NPHI*NPHI+blockDim.x-1)/blockDim.x;
//       for( int u=0; u<ncycles; ++u )
//       	{
//       	  const int idx = u*blockDim.x + threadIdx.x;
// 	  if( idx < NPHI*NPHI )
// 	    {
// 	      shmem[ idx ] = _GPU_denom_SR[ H_0 + idx ];
// 	    }
//       	}

//       __syncthreads();

//       // AND NOW THE COMPUTATIONS
//       const double WL        = dm -> get_W() -> mesh(l);
//       const double EPSSTAR   = rescpar->get_epsstar();
//       const double DPHI      = dm -> get_PHI() -> get_delta();
//       const double s_chcoord = GPU_sqrtmassXY(nu)*(1.+2.*GPU_kane(nu)*EPSSTAR*WL);
//       const double intfact   = CSR*GPU_I_SR(nu,p,i)*s_chcoord*DPHI;

//       // INTEGRATE
//       for(int m=0; m<NPHI; ++m)
// 	{
// 	  double ganancia = 0;
// 	  for(int mp=0; mp<NPHI; ++mp)
// 	    ganancia += GPU_pdf(nu,p,i,l,mp,STAGE) * shmem[ m*NPHI + mp ];
// 	  ganancia *= intfact;
// 	  GPU_rhs_pdf(nu,p,i,l,m) += ganancia;

// 	  // comment for non-debugging run
// 	  GPU_test_gain(nu,p,i,l,m) = ganancia;
// 	}
//     }
// }

