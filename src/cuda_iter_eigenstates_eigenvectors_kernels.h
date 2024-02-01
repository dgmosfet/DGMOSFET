#ifndef _CUDA_ITER_EIGENSTATES_EIGENVECTORS_KERNELS_H
#define _CUDA_ITER_EIGENSTATES_EIGENVECTORS_KERNELS_H

#include "discrmeshes.h"
#include "solverparams.h"

__global__ void cuda_tridiag_Thomas_20230525( const discrMeshes *dm, const solverParams *sp, const double *d_A, const double *_GPU_eps, double * _GPU_chi, const double eigvecs_tolpar );

#include "kernelconfig.h"
class cuda_iter_eigenstates_eigenvectors_kernels_config
{
public:
  cuda_iter_eigenstates_eigenvectors_kernels_config( const discrMeshes *dm );
  
  kernelConfig *cuda_tridiag_Thomas_20230525_config;
};

#endif
