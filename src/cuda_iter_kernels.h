#ifndef _CUDA_ITER_KERNELS_H
#define _CUDA_ITER_KERNELS_H

__global__ void cuda_compute_deps_dx( const discrMeshes *dm, double *_GPU_deps_dx, const double *_GPU_eps );

#include "kernelconfig.h"
class cuda_iter_kernels_config
{
public:
  cuda_iter_kernels_config(const discrMeshes *dm);
  
  kernelConfig *cuda_compute_deps_dx_config;
};

#endif
