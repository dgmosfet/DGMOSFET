#include "mosfetproblem.h"
#include "debug_flags.h"



#ifdef __CUDACC__
__global__ void cuda_pdftilde( const discrMeshes *dm, double *_GPU_integrated_pdf_energy, const double * _GPU_pdf, const int STAGE )
{
  extern __shared__ double sdata[];

  int global_index = blockIdx.x*blockDim.x + threadIdx.x;

  const int NSBN     = dm->get_NSBN();
  const int NX       = dm->get_X()->get_N();
  const int NW       = dm->get_W()->get_N();
  const int NPHI     = dm->get_PHI()->get_N();
  const double DPHI  = dm->get_PHI()->get_delta();

  if( global_index < _NVALLEYS*NSBN*NX*NW )
    {
      const int start_idx = (STAGE)*NX*_NVALLEYS*NSBN*NW*NPHI + blockIdx.x*blockDim.x*NPHI;
      for( int m=0; m<NPHI; ++m )
	{
	  const int despl = threadIdx.x*NPHI + m;
	  sdata[ despl ] = _GPU_pdf[ start_idx + despl ];
	}
      __syncthreads();

      double res = 0;
      const int K = threadIdx.x - (threadIdx.x/32)*32;
      for( int m=0; m<NPHI; ++m )
      	{
      	  int access = m+K;
      	  access -= (access/NPHI)*NPHI;
      	  res += sdata[ threadIdx.x*NPHI + access ];
      	}
      res *= DPHI;

      _GPU_integrated_pdf_energy[ global_index ] = res;
    }
}

#else
void MOSFETProblemCuda::CPU_pdftilde( const int STAGE )
{
  const int NSBN    = host_dm->get_NSBN();
  const int NX      = host_dm->get_X()->get_N();
  const int NW      = host_dm->get_W()->get_N();
  const int NPHI    = host_dm->get_PHI()->get_N();
  const double DPHI = host_dm->get_PHI()->get_delta();

#pragma omp parallel for
  for( int global_index=0; global_index < _NVALLEYS*NSBN*NX*NW; ++global_index )
    {
      int index = global_index;
      int i        = index;
      index = index/NW;
      int l        = i-index*NW;
      i            = index;
      index = index/NSBN;
      const int p  = i-index*NSBN;
      i            = index;
      index = index/_NVALLEYS;
      const int nu = i-index*_NVALLEYS;
      i             = index;

      double res = 0;
      for( int m=0; m<NPHI; ++m )
	res += pdf( nu,p,i,l,m,STAGE );
      res *= DPHI;

      _integrated_pdf_energy[ global_index ] = res;
    }

  return;
}
#endif




#ifdef __CUDACC__
__global__ void cuda_surfdens_2( const discrMeshes *dm, const double *_GPU_pdf, const int stage, double *_GPU_surfdens )
{
  int global_index = blockIdx.x*blockDim.x + threadIdx.x;

  const int NSBN     = dm->get_NSBN();
  const int NX       = dm->get_X()->get_N();
  const int NW       = dm->get_W()->get_N();
  const int NPHI     = dm->get_PHI()->get_N();

  if(global_index < _NVALLEYS*NSBN*NX)
    {
      int i, nu, p;
      GPU_map_1D_to_3D( global_index, &i, NX, &nu, _NVALLEYS, &p, NSBN );

      const double DW   = dm->get_W()->get_delta();
      const double DPHI = dm->get_PHI()->get_delta();

      double int_surfdens = 0;

      for( int l=0; l<NW; ++l )
	for( int m=0; m<NPHI; ++m )
	  int_surfdens += GPU_pdf(nu,p,i,l,m,stage);
      int_surfdens *= DW * DPHI;

      GPU_surfdens( nu,p,i ) = int_surfdens;

      // delete from here
      if( nu == _NVALLEYS-1 && p == NSBN-1 && i == NX-1 )
	{
	  if( isnan(GPU_surfdens(nu, p, i)) == true )
	    {
	      printf("From kernel 'cuda_surfdens_2' : GPU_surfdens(%i, %i, %i) = %e\n", nu, p, i, GPU_surfdens(nu, p, i));
	    }
	}
      // to here
    }
}
#else
void MOSFETProblemCuda::CPU_integrate_phitilde()
{
  const int NSBN  = host_dm -> get_NSBN();
  const int NX    = host_dm -> get_X()   -> get_N();
  const int NW    = host_dm -> get_W()   -> get_N();
  const double DW = host_dm->get_W()->get_delta();
#pragma omp parallel for
  for( int global_index=0; global_index < _NVALLEYS*NSBN*NX; ++global_index )
    {
      int index = global_index;
      int i, nu, p;
      i    = index;
      index  = index/NSBN;
      p    = i-index*NSBN;
      i    = index;
      index  = index/_NVALLEYS;
      nu   = i-index*_NVALLEYS;
      i    = index;
      
      double int_surfdens = 0;
      
      for( int l=0; l<NW; ++l )
	int_surfdens += integrated_pdf_energy(nu,p,i,l);

      int_surfdens *= DW;
      surfdens( nu,p,i ) = int_surfdens;
    }
}
#endif



void MOSFETProblemCuda::dens( const int STAGE )
{
#ifdef _SHOW_ETA
  ostringstream message;
  message << "SURFDENS phase at RK stage " << STAGE << "/3";
  show_eta(message.str());
#endif
  
start_time( _PHASE_DENS );

#ifdef __CUDACC__
  GPU_dens( STAGE );
#else
  CPU_dens( STAGE );
#endif

  stop_time( _PHASE_DENS );

#ifdef _SHOW_ETA
  cerr << "[ok] (function " << __func__ << ", line " << __LINE__ << ")" << endl;
#endif

  return;
}


#ifdef __CUDACC__
void MOSFETProblemCuda::GPU_dens( const int STAGE )
{
  const int NSBN = host_dm->get_NSBN();
  const int NX   = host_dm->get_X()->get_N();
  const int NW   = host_dm->get_W()->get_N();
  const int NPHI = host_dm->get_PHI()->get_N();

  {
    const int gridSize      = host_gridconfig -> cuda_pdftilde_config -> get_gridSize();
    const int blockSize     = host_gridconfig -> cuda_pdftilde_config -> get_blockSize();
    const int shmemSize     = host_gridconfig -> cuda_pdftilde_config -> get_shmemSize();
    const cudaFuncCache cfc = host_gridconfig -> cuda_pdftilde_config -> get_cfc();
    cudaDeviceSetCacheConfig( cfc );
    cuda_pdftilde <<< gridSize, blockSize, shmemSize >>> ( device_dm, _GPU_integrated_pdf_energy, _GPU_pdf, STAGE );
  }

  {
    const int gridSize      = host_gridconfig -> cuda_surfdens_2_config -> get_gridSize();
    const int blockSize     = host_gridconfig -> cuda_surfdens_2_config -> get_blockSize();
    const int shmemSize     = host_gridconfig -> cuda_surfdens_2_config -> get_shmemSize();
    const cudaFuncCache cfc = host_gridconfig -> cuda_surfdens_2_config -> get_cfc();
    cudaDeviceSetCacheConfig( cfc );
    cuda_surfdens_2 <<< gridSize, blockSize, shmemSize >>> ( device_dm, _GPU_pdf, STAGE, _GPU_surfdens );
  }

  return;
}
#else
void MOSFETProblemCuda::CPU_dens( const int STAGE )
{
  CPU_pdftilde ( STAGE );
  CPU_integrate_phitilde ();

  return;
}
#endif




cuda_dens_kernels_config::cuda_dens_kernels_config( const discrMeshes *dm )
{  
  const int NSBN  = dm -> get_NSBN();
  const int NX    = dm -> get_X()   -> get_N();
  const int NW    = dm -> get_W()   -> get_N();
  const int NPHI  = dm -> get_PHI() -> get_N();

  const int intpdf_TPB = 54;
  cuda_pdftilde_config   = new kernelConfig( nblocks( _NVALLEYS*NSBN*NX*NW, intpdf_TPB ), intpdf_TPB, intpdf_TPB*NPHI*sizeof(double), cudaFuncCachePreferShared, "cuda_pdftilde"   );
  
  cuda_surfdens_2_config = new kernelConfig( nblocks( _NVALLEYS*NSBN*NX, 32 ),            32,         NOSHMEM,                        cudaFuncCachePreferL1,     "cuda_surfdens_2" );
}
