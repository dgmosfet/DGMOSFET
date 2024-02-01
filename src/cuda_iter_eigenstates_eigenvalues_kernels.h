#ifndef _CUDA_ITER_EIGENSTATES_EIGENVALUES_KERNELS_H
#define _CUDA_ITER_EIGENSTATES_EIGENVALUES_KERNELS_H

#include "discrmeshes.h"

__global__ void cuda_initialize_eps(const discrMeshes *dm, double *_GPU_eps, const double min_Y);
__global__ void cuda_gershgorin( const discrMeshes *dm, const double *A, double *_GPU_Y, double *_GPU_Z );
__global__ void cuda_eigenvalues_NR( const discrMeshes *dm, const double *A, double *_GPU_eps );
__global__ void cuda_eigenvalues_ms_2( const discrMeshes *dm, const double *A, double step, const int _NMULTI, double *_GPU_eps, const int _niters );

template <class T> __device__ T reduce_sum(T *sdata, int tid);
template <class T> __device__ T reduce_sumsq(T *sdata, int tid);
template <class T> __device__ T reduce_max(T *sdata, int tid);

#include "kernelconfig.h"
class cuda_iter_eigenstates_eigenvalues_kernels_config
{
public:
  cuda_iter_eigenstates_eigenvalues_kernels_config( const discrMeshes *dm );
  
  kernelConfig *cuda_initialize_eps_config;
  kernelConfig *cuda_gershgorin_config;
  kernelConfig *cuda_eigenvalues_NR_config;
  kernelConfig *cuda_eigenvalues_ms_2_config;
};

#endif
