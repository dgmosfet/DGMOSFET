#ifndef _CUDA_TESTING_KERNELS_H
#define _CUDA_TESTING_KERNELS_H

/*
  date: 2023/01/30
  last modified: 2023/01/30
  who: Francesco Vecil

  Description
  -----------
  It can be useful to perform some tests directly on the GPU, without copying
  data from D-RAM to RAM.
  It has been observed that an unstable code becomes stable, though possibly computing
  wrong results, just by introducing some tests.
 */

#include "solverparams.h"

__global__ void cuda_test_data( const solverParams *sp, const double *start_address, const int N, const char *msg );
__global__ void cuda_test_pdf( const solverParams *sp, const double *_GPU_pdf, const double *_GPU_rhs_pdf, const int stage );
__global__ void cuda_test_rhs( const solverParams *sp, const double *_GPU_rhs );
__global__ void cuda_compare( const solverParams *sp, const double *_vec1, const double *_vec2, const int N );

#include "kernelconfig.h"
class cuda_testing_kernels_config
{
public:
  cuda_testing_kernels_config(const discrMeshes *dm);

  kernelConfig *cuda_test_data_config;
  kernelConfig *cuda_test_pdf_config;
  kernelConfig *cuda_test_rhs_config;
  kernelConfig *cuda_compare_config;
};

#endif
