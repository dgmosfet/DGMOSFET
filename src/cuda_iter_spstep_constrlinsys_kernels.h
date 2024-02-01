#ifndef _CUDA_ITER_SPSTEP_CONSTRLINSYS_KERNELS_H
#define _CUDA_ITER_SPSTEP_CONSTRLINSYS_KERNELS_H

#include "discrmeshes.h"
#include "adimparams.h"

__global__ void cuda_constr_linsys_NewtonRaphson( const discrMeshes *dm, const adimParams *adimpar, double * _GPU_matrix_2d, double *_GPU_rhs, const double *_GPU_kernel, const double *_GPU_totvoldens_OLD, const double *_GPU_nd, const double *_GPU_pot_OLD );
__global__ void cuda_constr_linsys_Gummel( const discrMeshes *dm, const adimParams *adimpar, double * _GPU_matrix_2d, double *_GPU_rhs, const double *_GPU_kernel, const double *_GPU_totvoldens_OLD, const double *_GPU_nd, const double *_GPU_pot_OLD );

#include "kernelconfig.h"
class cuda_iter_spstep_constrlinsys_kernels_config
{
public:
  cuda_iter_spstep_constrlinsys_kernels_config(const discrMeshes *dm);

  kernelConfig *cuda_constr_linsys_config;
};

#endif
