#include "mosfetproblem.h"
#include "debug_flags.h"

#ifdef __CUDACC__
__global__ void cuda_compute_voldens(const discrMeshes *dm, double *_GPU_voldens, const double *_GPU_surfdens, const double *_GPU_chi)
{
  int global_index = blockIdx.x*blockDim.x + threadIdx.x;

  const int NSBN     = dm -> get_NSBN();
  const int NX       = dm -> get_X() -> get_N();
  const int NZ       = dm -> get_Z() -> get_N();

  if( global_index < NX*_NVALLEYS*NSBN*(NZ-2) ) // this is the order to have coalescent reading
    {
      int i, j, p, nu;
      GPU_map_1D_to_4D( global_index, &i, NX, &nu, _NVALLEYS, &p, NSBN, &j, NZ-2 );
      ++j;

      GPU_voldens(nu,p,i,j) = GPU_surfdens(nu,p,i) * GPU_chi(nu,p,i,j) * GPU_chi(nu,p,i,j);
    }

  return;
}

__global__ void cuda_compute_totvoldens_OLD(const discrMeshes *dm, double *_GPU_totvoldens_OLD, const double *_GPU_voldens)
{
  // NOW COMPUTE THE VOLUME DENSITIES

  int global_index = blockIdx.x*blockDim.x + threadIdx.x;

  const int NSBN     = dm -> get_NSBN();
  const int NX       = dm -> get_X() -> get_N();
  const int NZ       = dm -> get_Z() -> get_N();

  if( global_index < NX*(NZ-2) ) // this is the order to have coalescent reading
    {
      int i, j;
      GPU_map_1D_to_2D( global_index, &i, NX, &j, NZ-2 );
      ++j;

      double sum = 0;
      for(int nu=0; nu<_NVALLEYS; ++nu)
	for(int p=0; p<NSBN; ++p)
	  sum += GPU_voldens(nu,p,i,j);
      GPU_totvoldens_OLD(i,j) = 2*sum;
    }
  
  return;
}

__global__ void cuda_compute_totvoldens(const discrMeshes *dm, double *_GPU_totvoldens, const double *_GPU_voldens)
{

  int global_index = blockIdx.x*blockDim.x + threadIdx.x;

  const int NSBN     = dm -> get_NSBN();
  const int NX       = dm -> get_X() -> get_N();
  const int NZ       = dm -> get_Z() -> get_N();

  if( global_index < NX*(NZ-2) ) // this is the order to have coalescent reading
    {
      int i, j;
      GPU_map_1D_to_2D( global_index, &i, NX, &j, NZ-2 );
      ++j;
      double sum = 0;
      for(int nu=0; nu<_NVALLEYS; ++nu)
	for(int p=0; p<NSBN; ++p)
	  sum += GPU_voldens(nu,p,i,j);
      GPU_totvoldens(i,j) = 2*sum;
    }
  
  return;
}
#endif


void MOSFETProblemCuda::CPU_voldens_totvoldens(const SPPhase spphase)
{
  const int NSBN  = host_dm -> get_NSBN();
  const int NX    = host_dm -> get_X()   -> get_N();
  const int NZ    = host_dm -> get_Z()   -> get_N();

  // NOW COMPUTE THE VOLUME DENSITIES

  for( int global_index=0; global_index<NX*(NZ-2); ++global_index )
    {
      int i, j;
      GPU_map_1D_to_2D( global_index, &i, NX, &j, NZ-2 );
      ++j;

      for(int nu=0; nu<_NVALLEYS; ++nu)
	for(int p=0; p<NSBN; ++p)
	  voldens(nu,p,i,j) = surfdens(nu,p,i) * chi(nu,p,i,j) * chi(nu,p,i,j);
    }

  if(spphase == OLD)
    {
      for( int global_index=0; global_index<NX*(NZ-2); ++global_index )
	{
	  int i, j;
	  GPU_map_1D_to_2D( global_index, &i, NX, &j, NZ-2 );
	  ++j;
	  double sum = 0;
	  for(int nu=0; nu<_NVALLEYS; ++nu)
	    for(int p=0; p<NSBN; ++p)
	      sum += voldens(nu,p,i,j);
	  totvoldens_OLD(i,j) = 2*sum;
	}
    }
  else if (spphase == NEW)
    {
      for( int global_index=0; global_index<NX*(NZ-2); ++global_index )
	{
	  int i, j;
	  GPU_map_1D_to_2D( global_index, &i, NX, &j, NZ-2 );
	  ++j;
	  double sum = 0;
	  for(int nu=0; nu<_NVALLEYS; ++nu)
	    for(int p=0; p<NSBN; ++p)
	      sum += voldens(nu,p,i,j);
	  totvoldens(i,j) = 2*sum;
	}
    }
  else
    {
      throw error_REDUCTION_WRONG_SPPHASE();
    }
  
  return;
}

void MOSFETProblemCuda::voldens_totvoldens(const SPPhase spphase)
  /********************************************************************
   * This function computes the volume densities and the
   * total volume density.
   * 
   * Input: surfdens, chi
   * Output: voldens, totvoldens/totvoldens_OLD
   ********************************************************************/
{
  // NOW COMPUTE THE VOLUME DENSITIES

  const int NSBN  = host_dm -> get_NSBN();
  const int NX    = host_dm -> get_X()   -> get_N();
  const int NZ    = host_dm -> get_Z()   -> get_N();

  if( _ep != POTENTIAL )
    {
      CPU_voldens_totvoldens(spphase);
    }
  else
    {
#ifdef __CUDACC__
      const int gridSize      = host_gridconfig -> cuda_compute_voldens_config -> get_gridSize();
      const int blockSize     = host_gridconfig -> cuda_compute_voldens_config -> get_blockSize();
      const int shmemSize     = host_gridconfig -> cuda_compute_voldens_config -> get_shmemSize();
      const cudaFuncCache cfc = host_gridconfig -> cuda_compute_voldens_config -> get_cfc();
      cudaDeviceSetCacheConfig( cfc );
      cuda_compute_voldens <<< gridSize, blockSize, shmemSize >>>(device_dm, _GPU_voldens, _GPU_surfdens, _GPU_chi);
      
      if(spphase == OLD)
	{
	  const int gridSize      = host_gridconfig -> cuda_compute_totvoldens_OLD_config -> get_gridSize();
	  const int blockSize     = host_gridconfig -> cuda_compute_totvoldens_OLD_config -> get_blockSize();
	  const int shmemSize     = host_gridconfig -> cuda_compute_totvoldens_OLD_config -> get_shmemSize();
	  const cudaFuncCache cfc = host_gridconfig -> cuda_compute_totvoldens_OLD_config -> get_cfc();
	  cudaDeviceSetCacheConfig( cfc );
	  cuda_compute_totvoldens_OLD <<< gridSize, blockSize, shmemSize >>> (device_dm, _GPU_totvoldens_OLD, _GPU_voldens);

#ifdef _ITER_DEBUG
	  if( are_there_nan_inf( _GPU_voldens, _NVALLEYS*NSBN*NX*NZ, "testing voldens" ) == false )
	    {
	      cerr << " From function " << __func__ << " line " << __LINE__ << " : a problem has been detected!" << endl;
	      exit(666);
	    }
	  if( are_there_nan_inf( _GPU_totvoldens_OLD, NX*NZ, "testing totvoldens_OLD" ) == false )
	    {
	      cerr << " From function " << __func__ << " line " << __LINE__ << " : a problem has been detected!" << endl;
	      exit(666);
	    }
#endif
	}
      else if (spphase == NEW)
	{
	  const int gridSize      = host_gridconfig -> cuda_compute_totvoldens_config -> get_gridSize();
	  const int blockSize     = host_gridconfig -> cuda_compute_totvoldens_config -> get_blockSize();
	  const int shmemSize     = host_gridconfig -> cuda_compute_totvoldens_config -> get_shmemSize();
	  const cudaFuncCache cfc = host_gridconfig -> cuda_compute_totvoldens_config -> get_cfc();
	  cudaDeviceSetCacheConfig( cfc );
	  cuda_compute_totvoldens <<< gridSize, blockSize, shmemSize >>> (device_dm, _GPU_totvoldens, _GPU_voldens);

#ifdef _ITER_DEBUG
	  if( are_there_nan_inf( _GPU_voldens, _NVALLEYS*NSBN*NX*NZ, "testing voldens" ) == false )
	    {
	      cerr << " From function " << __func__ << " line " << __LINE__ << " : a problem has been detected!" << endl;
	      exit(666);
	    }
	  if( are_there_nan_inf( _GPU_totvoldens, NX*NZ, "testing totvoldens" ) == false )
	    {
	      cerr << " From function " << __func__ << " line " << __LINE__ << " : a problem has been detected!" << endl;
	      exit(666);
	    }
#endif
	}
      else
	{
	  throw error_REDUCTION_WRONG_SPPHASE();
	}

#else
      CPU_voldens_totvoldens(spphase);
#endif
    }
  
  return;
}


void MOSFETProblemCuda::CPU_compute_surfdens()
{
  const int NSBN  = host_dm -> get_NSBN();
  const int NX    = host_dm -> get_X()   -> get_N();
  const ContactMaterial CONTACTMATERIAL = host_physdesc->get_CONTACTMATERIAL();

  // first of all compute the surface densities
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
	    throw error_REDUCTION_MATERIAL();
	  }

	for(int i=0; i<NX; ++i)
	  for(int nu=0; nu<_NVALLEYS; ++nu)
	    for(int p=0; p<NSBN; ++p)
	      {
		if( log(1.+exp(fermilevel-eps(nu,p,i))) != 0 )
		  surfdens(nu,p,i) = M_1_PI*sqrt(mass(Xdim,nu)*mass(Ydim,nu)) * log(1.+exp(fermilevel-eps(nu,p,i)));
		else
		  surfdens(nu,p,i) = M_1_PI*sqrt(mass(Xdim,nu)*mass(Ydim,nu)) * exp(fermilevel-eps(nu,p,i));
	      }

	break;
      }

    case THERMEQUIL :
      {
	for(int i=0; i<NX; ++i)
	  for(int nu=0; nu<_NVALLEYS; ++nu)
	    for(int p=0; p<NSBN; ++p)
	      surfdens(nu,p,i) = _intnd/_repartition_b * sqrt(mass(Xdim,nu)*mass(Ydim,nu)) * exp(-eps(nu,p,i));
	// surfdens(nu,p,i) = _totsurfdens_eq[0]/_repartition_b * sqrt(mass(X,nu)*mass(Y,nu)) * exp(-eps(nu,p,i));

	break;
      }

    case POTENTIAL :
      {
	// cerr << " Aqui no tengo por que pasar!" << endl;
	throw error_REDUCTION_WRONG_EIGENPROBLEM();

	break;
      }
    }
  
  return;
}




void MOSFETProblemCuda::currdens_voldens( const int s )
{
#ifdef __CUDACC__
  const int NSBN  = host_dm -> get_NSBN();
  const int NX    = host_dm -> get_X()   -> get_N();

  const int gridSize      = host_gridconfig -> cuda_currdens_voldens_config -> get_gridSize();
  const int blockSize     = host_gridconfig -> cuda_currdens_voldens_config -> get_blockSize();
  const int shmemSize     = host_gridconfig -> cuda_currdens_voldens_config -> get_shmemSize();
  const cudaFuncCache cfc = host_gridconfig -> cuda_currdens_voldens_config -> get_cfc();
  cudaDeviceSetCacheConfig( cfc );
  cuda_currdens_voldens <<< gridSize, blockSize, shmemSize >>> ( device_dm, _GPU_pdf, _GPU_a1, _GPU_chi, _GPU_surfdens, _GPU_currdens, _GPU_voldens, s );
#else
  CPU_currdens_voldens ( s );
#endif
  
  return;
}
#ifdef __CUDACC__
__global__ void cuda_currdens_voldens( const discrMeshes *dm, const double *_GPU_pdf, const double *_GPU_a1, const double *_GPU_chi, const double *_GPU_surfdens, double *_GPU_currdens, double *_GPU_voldens, const int STAGE )
{
  int global_index = blockIdx.x*blockDim.x + threadIdx.x;

  const int NSBN     = dm->get_NSBN();
  const int NX       = dm->get_X()->get_N();
  const int NZ       = dm->get_Z()->get_N();
  const int NW       = dm->get_W()->get_N();
  const int NPHI     = dm->get_PHI()->get_N();

  if(global_index < _NVALLEYS*NSBN*NX)
    {
      int i, nu, p;
      GPU_map_1D_to_3D( global_index, &i, NX, &nu, _NVALLEYS, &p, NSBN ); // TASK. Sure this is the good order??

      GPU_currdens(nu,p,i) = 0;

      double int_currdens = 0;

      const double DW   = dm->get_W()->get_delta();
      const double DPHI = dm->get_PHI()->get_delta();

      for(int l=0;l<NW;++l)
	{
	  double aux_x1 = 0;
	  for(int m=0;m<NPHI;++m)
	    aux_x1 += GPU_a1( nu,l,m )*GPU_pdf(nu,p,i,l,m,STAGE);
	  aux_x1 *= DPHI;

	  int_currdens += aux_x1;
	}

      int_currdens*=DW;

      GPU_currdens( nu,p,i ) = int_currdens;

      GPU_voldens(nu,p,i,0) = 0;
      for(int j=1; j<NZ-1; ++j)
	GPU_voldens(nu,p,i,j) = GPU_surfdens(nu,p,i)*GPU_chi(nu,p,i,j)*GPU_chi(nu,p,i,j);
      GPU_voldens(nu,p,i,NZ-1) = 0;
    }
}
#else
void MOSFETProblemCuda::CPU_currdens_voldens( const int STAGE )
{
  const int NSBN    = host_dm           ->get_NSBN();
  const int NX      = host_dm->get_X()  ->get_N();
  const int NZ      = host_dm->get_Z()  ->get_N();
  const int NW      = host_dm->get_W()  ->get_N();
  const int NPHI    = host_dm->get_PHI()->get_N();
  const double DW   = host_dm->get_W()  ->get_delta();
  const double DPHI = host_dm->get_PHI()->get_delta();

#pragma omp parallel for
  for( int global_index=0; global_index < _NVALLEYS*NSBN*NX; ++global_index )
    {
      int i, nu, p;
      GPU_map_1D_to_3D( global_index, &i, NX, &nu, _NVALLEYS, &p, NSBN ); // TASK. Sure this is the good order??

      currdens(nu,p,i) = 0;

      double int_currdens = 0;

      for( int l=0; l<NW; ++l )
	for( int m=0; m<NPHI; ++m )
	  {
	    int_currdens += a1( nu,l,m )*pdf(nu,p,i,l,m,STAGE);
	  }

      int_currdens *= DW*DPHI;

      currdens( nu,p,i ) = int_currdens;

      voldens(nu,p,i,0) = 0;
      for(int j=1; j<NZ-1; ++j)
	voldens(nu,p,i,j) = surfdens(nu,p,i)*chi(nu,p,i,j)*chi(nu,p,i,j);
      voldens(nu,p,i,NZ-1) = 0;
    }
  return;
}
#endif



double MOSFETProblemCuda::integrate_R1( double *f, int n1, double d1, IntegrType it1)
{
  double res=0;

  switch( it1 )
    {
    case TOPLEFT_RECTANGLES :
      {
	for(int i1=0;i1<n1-1;++i1)
	  res += f[ i1 ];
	res*=d1;
	break;
      }
    case TOPRIGHT_RECTANGLES :
      {
	for(int i1=1;i1<n1;++i1)
	  res += f[ i1 ];
	res*=d1;
	break;
      }
    case TRAPEZOIDS :
      {
	for(int i1=0;i1<n1-1;++i1)
	  res += f[ i1 ] + f[ i1+1 ];
	res*=0.5*d1;
	break;
      }
    case PHISTYLE :
      {
	for(int i1=0;i1<n1-1;++i1)
	  res += .5*(f[ i1 ]+f[ i1+1 ]);
	res += .5*(f[ n1-1 ]+f[ 0 ]);
	res*=d1;
	break;
      }
    }

  return res;
}



double MOSFETProblemCuda::integrate_R2( double *f, int n1, double d1,  IntegrType it1, int n2, double d2,IntegrType it2)
{
  double res=0;

  double *aux_x1;
  aux_x1 = new double [ n1 ];
      
  switch( it2 )
    {
    case TOPLEFT_RECTANGLES :
      {
	for(int i1=0;i1<n1;++i1)
	  {
	    aux_x1[ i1 ] = 0;
	    for(int i2=0;i2<n2-1;++i2)
	      aux_x1[ i1 ] += f[ i1*n2 + i2 ];
	    aux_x1[ i1 ] *= d2;
	  }
	break;
      }

    case TOPRIGHT_RECTANGLES :
      {
	for(int i1=0;i1<n1;++i1)
	  {
	    aux_x1[ i1 ] = 0;
	    for(int i2=1;i2<n2;++i2)
	      aux_x1[ i1 ] += f[ i1*n2 + i2 ];
	    aux_x1[ i1 ] *= 0.5*d2;
	  }
	break;
      }

    case TRAPEZOIDS :
      {
	for(int i1=0;i1<n1;++i1)
	  {
	    aux_x1[ i1 ] = 0;
	    for(int i2=0;i2<n2-1;++i2)
	      aux_x1[ i1 ] += f[ i1*n2 + i2 ] + f[ i1*n2 + (i2+1) ];
	    aux_x1[ i1 ] *= 0.5*d2;
	  }
	break;
      }
    case PHISTYLE :
      {
	for(int i1=0;i1<n1;++i1)
	  {
	    aux_x1[ i1 ] = 0;
	    for(int i2=0;i2<n2;++i2)
	      aux_x1[ i1 ] += f[ i1*n2 + i2 ];
	    aux_x1[ i1 ] *= d2;
	  }
	break;
      }
    }
      
  switch( it1 )
    {
    case TOPLEFT_RECTANGLES :
      {
	for(int i1=0;i1<n1-1;++i1)
	  res += aux_x1[ i1 ];
	res*=d1;
	break;
      }
    case TOPRIGHT_RECTANGLES :
      {
	for(int i1=1;i1<n1;++i1)
	  res += aux_x1[ i1 ];
	res*=d1;
	break;
      }
    case TRAPEZOIDS :
      {
	for(int i1=0;i1<n1-1;++i1)
	  res += aux_x1[ i1 ] + aux_x1[ i1+1 ];
	res*=0.5*d1;
	break;
      }
    case PHISTYLE :
      {
	for(int i1=0;i1<n1;++i1)
	  res += aux_x1[ i1 ];
	res*=d1;
	break;
      }
    }
      
  delete [] aux_x1;
  
  return res;
}






void MOSFETProblemCuda::macro( const int s )
{
  const int NSBN  = host_dm -> get_NSBN();
  const int NX    = host_dm -> get_X()   -> get_N();
  const int NZ    = host_dm -> get_Z()   -> get_N();
  const int NW    = host_dm -> get_W()   -> get_N();
  const int NPHI  = host_dm -> get_PHI() -> get_N();
  const double XLENGTH = host_physdesc -> compute_XLENGTH();
  const double DX = host_dm->get_X()->get_delta();

  currdens_voldens(s);
#ifdef __CUDACC__
  checkCudaErrors( cudaMemcpy(_currdens, _GPU_currdens, _NVALLEYS*NSBN*NX*sizeof(double),     cudaMemcpyDeviceToHost) );
  checkCudaErrors( cudaMemcpy(_voldens,  _GPU_voldens,  _NVALLEYS*NSBN*NX*NZ*sizeof(double), cudaMemcpyDeviceToHost) );
  checkCudaErrors( cudaMemcpy(_surfdens, _GPU_surfdens, _NVALLEYS*NSBN*NX*sizeof(double),     cudaMemcpyDeviceToHost) );
#endif

#pragma omp parallel for
  for(int i=0; i<NX; ++i)
    {
      // CLASSICAL STATES: SURFACE DENSITIES
      double int_totsurfdens = 0;
      double int_totcurrdens = 0;

      for( int nu=0; nu<_NVALLEYS; ++nu )
	for( int p=0; p<NSBN; ++p )
	  {
	    int_totsurfdens += 2.*surfdens( nu,p,i );
	    int_totcurrdens += 2.*currdens( nu,p,i );
	  }
      _totsurfdens[ i ] = int_totsurfdens;
      _totcurrdens[ i ] = int_totcurrdens;
	
      // MIXED STATES: VOLUME DENSITIES
      for(int j=0; j<NZ; ++j)
	{
	  double int_totvoldens = 0;
	  for(int nu=0; nu<_NVALLEYS; ++nu)
	    for(int p=0; p<NSBN; ++p)
	      int_totvoldens += 2.*voldens(nu,p,i,j);
	  totvoldens(i,j) = int_totvoldens;
	}
      // CLASSICAL STATES: AVERAGE DRIFT VELOCITY
      _avgdriftvel[ i ] = _totcurrdens[ i ]/_totsurfdens[ i ];
    }

#ifdef __CUDACC__
  checkCudaErrors( cudaMemcpy(_GPU_totsurfdens, _totsurfdens, NX*sizeof(double),     cudaMemcpyHostToDevice) );
  checkCudaErrors( cudaMemcpy(_GPU_totcurrdens, _totcurrdens, NX*sizeof(double),     cudaMemcpyHostToDevice) );
  checkCudaErrors( cudaMemcpy(_GPU_avgdriftvel, _avgdriftvel, NX*sizeof(double),     cudaMemcpyHostToDevice) );
  checkCudaErrors( cudaMemcpy(_GPU_totvoldens,  _totvoldens,  NX*NZ*sizeof(double), cudaMemcpyHostToDevice) );
#endif
  
  // // SCALAR MAGNITUDES
  _totmass = 0;
  _avgcurr = 0;
  for(int i=0; i<NX-1; ++i)
    {
      _totmass += _totsurfdens[i];
      _avgcurr += _totcurrdens[i];
    }
  for(int i=1; i<NX; ++i)
    {
      _totmass += _totsurfdens[i];
      _avgcurr += _totcurrdens[i];
    }
  _totmass *= .5;
  _avgcurr *= .5;

  _totmass *= XLENGTH*DX*host_rescpar->get_rhostar();
  _avgcurr *= host_rescpar->get_jstar()*XLENGTH*DX;

  _avgdriftvel_scalar = 0;
  _avgdriftvel_min = fabs(_avgdriftvel[0]);
  _avgdriftvel_max = fabs(_avgdriftvel[0]);
  for(int i=0; i<NX; ++i)
    {
      _avgdriftvel_scalar += _avgdriftvel[i];
      _avgdriftvel_min = min( fabs(_avgdriftvel[i]), _avgdriftvel_min );
      _avgdriftvel_max = max( fabs(_avgdriftvel[i]), _avgdriftvel_max );
    }
  _avgdriftvel_scalar /= NX;
      
  return;
}




cuda_reductions_kernels_config::cuda_reductions_kernels_config(const discrMeshes *dm)
{  
  const int NSBN  = dm -> get_NSBN();
  const int NX    = dm -> get_X()   -> get_N();
  const int NZ    = dm -> get_Z()   -> get_N();

  int blockdim = 64;
  int number_of_blocks = nblocks( NX*_NVALLEYS*NSBN*(NZ-2), blockdim );
  cuda_compute_voldens_config        = new kernelConfig(number_of_blocks,                 blockdim, NOSHMEM, cudaFuncCachePreferL1, "cuda_compute_voldens"       );
  
  blockdim = 64;
  number_of_blocks = nblocks( NX*(NZ-2), blockdim );
  cuda_compute_totvoldens_OLD_config = new kernelConfig(number_of_blocks,                 blockdim, NOSHMEM, cudaFuncCachePreferL1, "cuda_compute_totvoldens_OLD");
  cuda_compute_totvoldens_config     = new kernelConfig(number_of_blocks,                 blockdim, NOSHMEM, cudaFuncCachePreferL1, "cuda_compute_totvoldens"    );

  cuda_currdens_voldens_config       = new kernelConfig(nblocks( _NVALLEYS*NSBN*NX, 32 ), 32,       NOSHMEM, cudaFuncCachePreferL1, "cuda_currdens_voldens"      );
}
