#ifndef _CUDA_BTE_WENO_KERNELS_H
#define _CUDA_BTE_WENO_KERNELS_H

#include "discrmeshes.h"
#include "rescalingparams.h"

__global__ void cuda_WENO_X_20230525( const discrMeshes *dm, const double * _GPU_pdf, double *_GPU_rhs_pdf, const double *_GPU_surfdens, const double *_GPU_surfdens_eq, const int STAGE, double *test_fh, const double *_GPU_a1, const double *_GPU_vel );
__global__ void cuda_WENO_PHI( const discrMeshes *dm, const double *_GPU_pdf, double *_GPU_rhs_pdf, const double *_GPU_deps_dx, const double MAX_A3, const int STAGE, const double *_GPU_a3, double *test_fh=NULL );
__global__ void cuda_WENO_W_20230524( const discrMeshes *dm, const rescalingParams *rescpar, const double *_GPU_pdf, double *_GPU_rhs_pdf, const double *_GPU_deps_dx, const int STAGE );

#include "kernelconfig.h"
class cuda_bte_weno_kernels_config
{
public:
  cuda_bte_weno_kernels_config(const discrMeshes*);
  
  kernelConfig *cuda_WENO_X_20230525_config;
  kernelConfig *cuda_WENO_PHI_config;
  kernelConfig *cuda_WENO_W_20230524_config;
};

#endif
