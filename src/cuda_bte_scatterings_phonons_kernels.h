#ifndef _CUDA_BTE_SCATTERINGS_PHONONS_KERNELS_H
#define _CUDA_BTE_SCATTERINGS_PHONONS_KERNELS_H

#include "discrmeshes.h"
#include "rescalingparams.h"
#include "physconsts.h"

__global__ void cuda_phonons_loss( const discrMeshes *dm, const rescalingParams *rescpar, const physConsts *pc, const double* _GPU_pdf, const double* _GPU_Wm1, double* _GPU_rhs_pdf, const double* _GPU_rhs_pdf_gain, const double* _GPU_eps, const int STAGE, double *_GPU_test_loss=NULL );
__global__ void cuda_phonons_gain( const discrMeshes *dm, const rescalingParams *rescpar, const physConsts *pc, const double *_GPU_integrated_pdf_energy, const double *_GPU_Wm1, double *_GPU_rhs_pdf_gain, const double *_GPU_eps, const int STAGE, double *_GPU_test_gain=NULL );
__global__ void cuda_compute_Wm1( const discrMeshes *dm, double *_GPU_Wm1, const double *_GPU_chi );

#include "kernelconfig.h"
class cuda_bte_scatterings_phonons_kernels_config
{
public:
  cuda_bte_scatterings_phonons_kernels_config( const discrMeshes *dm );

  kernelConfig *cuda_phonons_loss_config;
  kernelConfig *cuda_phonons_gain_config;
  kernelConfig *cuda_compute_Wm1_config;
};

#endif
