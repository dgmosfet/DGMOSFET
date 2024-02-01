#ifndef _CUDA_ITER_SPSTEP_SOLVELINSYS_KERNELS_H
#define _CUDA_ITER_SPSTEP_SOLVELINSYS_KERNELS_H

#include "discrmeshes.h"

__global__ void cuda_matrix_vector_product( const discrMeshes *dm, double *_GPU_residual_vec, const double *_GPU_matrix_2d, const double * _GPU_x_k);

__global__ void cuda_update_x( const discrMeshes *dm, double *_GPU_x_kp1, double *_GPU_residual_vec, const double *_GPU_rhs, double *_GPU_x_k, const double omega, const bool residual, const int SHMEMSIZE );

#include "kernelconfig.h"
class cuda_iter_spstep_solvelinsys_kernels_config
{
public:
  cuda_iter_spstep_solvelinsys_kernels_config(const discrMeshes *dm);

  kernelConfig *cuda_matrix_vector_product_config;
  kernelConfig *cuda_update_x_config;
};

#endif
