#ifndef _CUDA_ITER_SPSTEP_FRECHET_KERNELS_H
#define _CUDA_ITER_SPSTEP_FRECHET_KERNELS_H

#include "discrmeshes.h"

__global__ void cuda_compute_frechet( const discrMeshes *dm, double *_GPU_frechet, const double *_GPU_chi, const double *_GPU_surfdens, const double *_GPU_eps );
__global__ void cuda_compute_eps_diff_m1( const discrMeshes *dm, double *_GPU_eps_diff_m1, const double *_GPU_eps );

#include "kernelconfig.h"
class cuda_iter_spstep_frechet_kernels_config
{
public:
  cuda_iter_spstep_frechet_kernels_config(const discrMeshes *dm);
  
  kernelConfig *cuda_compute_frechet_config;
  kernelConfig *cuda_compute_eps_diff_m1_config;
};

#endif
