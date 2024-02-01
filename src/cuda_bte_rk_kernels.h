#ifndef _CUDA_BTE_RK_KERNELS_H
#define _CUDA_BTE_RK_KERNELS_H

#include "discrmeshes.h"
#include <cuda.h>

__global__ void            cuda_perform_RK_1_3( const discrMeshes *dm, double *_GPU_pdf,     const double *_GPU_rhs_pdf, const double DT             );
__global__ void            cuda_perform_RK_2_3( const discrMeshes *dm, double *_GPU_pdf,     const double *_GPU_rhs_pdf, const double DT             );
__global__ void            cuda_perform_RK_3_3( const discrMeshes *dm, double *_GPU_pdf,     const double *_GPU_rhs_pdf, const double DT             );
__global__ void             cuda_set_fluxes_a3( const discrMeshes *dm, double *_GPU_a3,      const double *_GPU_deps_dx, const double *_GPU_a3_const );
__global__ void cuda_set_fluxes_a2_testversion( const discrMeshes *dm, double *_GPU_a2,      const double *_GPU_deps_dx, const double *_GPU_vel      );
__global__ void           cuda_set_rhs_to_zero( const discrMeshes *dm, double *_GPU_rhs_pdf                                                          );

#include "kernelconfig.h"
class cuda_bte_rk_kernels_config
{
public:
  cuda_bte_rk_kernels_config( const discrMeshes *dm );
  
  kernelConfig *cuda_perform_RK_1_3_config;
  kernelConfig *cuda_perform_RK_2_3_config;
  kernelConfig *cuda_perform_RK_3_3_config;
  kernelConfig *cuda_set_fluxes_a3_config;
  kernelConfig *cuda_set_fluxes_a2_testversion_config;
  kernelConfig *cuda_set_rhs_to_zero_config;
};

#endif
