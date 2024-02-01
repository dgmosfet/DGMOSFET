#ifndef _CUDA_DENS_KERNELS_H
#define _CUDA_DENS_KERNELS_H

#include "discrmeshes.h"

__global__ void cuda_pdftilde( const discrMeshes *dm, double *_GPU_integrated_pdf_energy, const double * _GPU_pdf, const int STAGE );
__global__ void cuda_surfdens_2( const discrMeshes *dm, const double *_GPU_pdf, const int stage, double *_GPU_surfdens );

#include "kernelconfig.h"
class cuda_dens_kernels_config
{
public:
  cuda_dens_kernels_config( const discrMeshes *dm );
  
  kernelConfig *cuda_pdftilde_config;
  kernelConfig *cuda_surfdens_2_config;
};

#endif
