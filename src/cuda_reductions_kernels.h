#ifndef _CUDA_REDUCTIONS_KERNELS_H
#define _CUDA_REDUCTIONS_KERNELS_H

#include "discrmeshes.h"

__global__ void cuda_compute_voldens(const discrMeshes *dm, double *_GPU_voldens, const double *_GPU_surfdens, const double *_GPU_chi);
__global__ void cuda_compute_totvoldens_OLD(const discrMeshes *dm, double *_GPU_totvoldens_OLD, const double *_GPU_voldens);
__global__ void cuda_compute_totvoldens(const discrMeshes *dm, double *_GPU_totvoldens, const double *_GPU_voldens);
__global__ void cuda_currdens_voldens( const discrMeshes *dm, const double *_GPU_pdf, const double *_GPU_a1, const double *_GPU_chi, const double *_GPU_surfdens, double *_GPU_currdens, double *_GPU_voldens, const int STAGE );

#include "kernelconfig.h"
class cuda_reductions_kernels_config
{
public:
  cuda_reductions_kernels_config(const discrMeshes *dm);
  
  kernelConfig *cuda_compute_voldens_config;
  kernelConfig *cuda_compute_totvoldens_OLD_config;
  kernelConfig *cuda_compute_totvoldens_config;
  kernelConfig *cuda_currdens_voldens_config;
};

#endif
