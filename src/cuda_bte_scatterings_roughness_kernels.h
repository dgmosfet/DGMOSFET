#ifndef _CUDA_BTE_SCATTERINGS_ROUGHNESS_KERNELS_H
#define _CUDA_BTE_SCATTERINGS_ROUGHNESS_KERNELS_H

#include "discrmeshes.h"
#include "solverparams.h"
#include "rescalingparams.h"
#include "physdevice.h"



/*
  KERNELS OF cuda_bte_scatterings.cu
 */
__global__ void kernel_roughness_gain_7( const discrMeshes *dm, const rescalingParams *rescpar, const double *_GPU_pdf, const double *_GPU_integrateddenom_SR, const double *_GPU_I_SR, const double CSR, double *_GPU_rhs_pdf, const double *_GPU_denom_SR, const int STAGE, double *_GPU_test_gain );

__global__ void kernel_roughness_gain_20230616( const discrMeshes *dm, const rescalingParams *rescpar, const double *_GPU_pdf, const double *_GPU_I_SR, const double CSR, double *_GPU_rhs_pdf, const double *_GPU_denom_SR, const int STAGE, double *_GPU_test_gain );

__global__ void kernel_roughness_gain_20230616_2( const discrMeshes *dm, const rescalingParams *rescpar, const solverParams *solvpar, const double *_GPU_pdf, const double *_GPU_I_SR, const double CSR, double *_GPU_rhs_pdf, const double *_GPU_denom_SR, const int STAGE, double *_GPU_test_gain );

__global__ void kernel_roughness_loss( const discrMeshes *dm, const rescalingParams *rescpar, const double *_GPU_pdf, const double *_GPU_integrateddenom_SR, const double *_GPU_I_SR, const double CSR, double *_GPU_rhs_pdf, const double *_GPU_denom_SR, const int STAGE, double *_GPU_test_loss );



/*
  KERNELS OF cuda_bte_scatterings_overlap.cu
 */
__global__ void kernel_stretch_totvoldens( const discrMeshes *dm, const double *_GPU_totvoldens, const double DZM1, const double *_GPU_sigma, const double *_GPU_Sigma, double *_GPU_totvoldens_ext );

__global__ void kernel_construct_rhs_ext(const discrMeshes *dm, const double CP, const double *_GPU_totvoldens_ext, double *_GPU_rhs_ext);
__global__ void multiply( const discrMeshes *dm, const double *_GPU_matrix_2dconst_ext_int, const double *_GPU_rhs_ext, double *_GPU_pot_ext);

__global__ void kernel_compute_Deltapot(const discrMeshes *dm, const solverParams *sp, const physDevice *pd, const double DZEXTM1, const double DZEXT, const double DZ, const double *_GPU_pot_ext, double *_GPU_Deltapot_upper, double *_GPU_Deltapot_lower);

__global__ void kernel_compute_overlap_SR(const discrMeshes *dm, const double DZ, const double *_GPU_chi, const double *_GPU_Deltapot_upper, const double *_GPU_Deltapot_lower, double *_GPU_I_SR);



/*
  KERNELS CONFIGURATION
 */
#include "kernelconfig.h"
class cuda_bte_scatterings_roughness_kernels_config
{
public:
  cuda_bte_scatterings_roughness_kernels_config( const discrMeshes *dm );
  
  /*
    KERNELS OF cuda_bte_scatterings.cu
  */
  kernelConfig *kernel_roughness_gain_7_config;
  kernelConfig *kernel_roughness_gain_20230616_config;
  kernelConfig *kernel_roughness_gain_20230616_2_config;
  kernelConfig *kernel_roughness_loss_config;
  /*
    KERNELS OF cuda_bte_scatterings_overlap.cu
  */
  kernelConfig *kernel_stretch_totvoldens_config;
  kernelConfig *kernel_construct_rhs_ext_config;
  kernelConfig *kernel_compute_Deltapot_config;
  kernelConfig *kernel_compute_overlap_SR_config;
  kernelConfig *multiply_config;
};

#endif
