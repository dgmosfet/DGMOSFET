#ifndef _CUDA_ITER_EIGENSTATES_KERNELS_H
#define _CUDA_ITER_EIGENSTATES_KERNELS_H

#include "discrmeshes.h"

__global__ void cuda_init_d_A( const discrMeshes *dm, double *d_A, const double *aux );

#include "kernelconfig.h"
class cuda_iter_eigenstates_kernels_config
{
public:
  cuda_iter_eigenstates_kernels_config( const discrMeshes *dm );
  
  kernelConfig *cuda_init_d_A_config;
};

#endif
