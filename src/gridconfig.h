#ifndef _GRIDCONFIG_H
#define _GRIDCONFIG_H

#include "cuda_bte_rk_kernels.h"
#include "cuda_bte_scatterings_phonons_kernels.h"
#include "cuda_bte_scatterings_roughness_kernels.h"
#include "cuda_bte_weno_kernels.h"
#include "cuda_dens_kernels.h"
#include "cuda_iter_eigenstates_eigenvalues_kernels.h"
#include "cuda_iter_eigenstates_eigenvectors_kernels.h"
#include "cuda_iter_eigenstates_kernels.h"
#include "cuda_iter_kernels.h"
#include "cuda_iter_spstep_constrlinsys_kernels.h"
#include "cuda_iter_spstep_frechet_kernels.h"
#include "cuda_iter_spstep_solvelinsys_kernels.h"
#include "cuda_reductions_kernels.h"
#include "cuda_testing_kernels.h"

class cudaGridConfig :
  public cuda_bte_rk_kernels_config,
  public cuda_bte_scatterings_phonons_kernels_config,
  public cuda_bte_scatterings_roughness_kernels_config,
  public cuda_bte_weno_kernels_config,
  public cuda_dens_kernels_config,
  public cuda_iter_eigenstates_eigenvalues_kernels_config,
  public cuda_iter_eigenstates_eigenvectors_kernels_config,
  public cuda_iter_eigenstates_kernels_config,
  public cuda_iter_kernels_config,
  public cuda_iter_spstep_constrlinsys_kernels_config,
  public cuda_iter_spstep_frechet_kernels_config,
  public cuda_iter_spstep_solvelinsys_kernels_config,
  public cuda_reductions_kernels_config,
  public cuda_testing_kernels_config
{
 public:
  inline cudaGridConfig( const discrMeshes *dm ) :
    cuda_bte_rk_kernels_config( dm ),
    cuda_bte_scatterings_phonons_kernels_config( dm ),
    cuda_bte_scatterings_roughness_kernels_config( dm ),
    cuda_bte_weno_kernels_config( dm ),
    cuda_dens_kernels_config( dm ),
    cuda_iter_eigenstates_eigenvalues_kernels_config( dm ),
    cuda_iter_eigenstates_eigenvectors_kernels_config( dm ),
    cuda_iter_eigenstates_kernels_config( dm ),
    cuda_iter_kernels_config( dm ),
    cuda_iter_spstep_constrlinsys_kernels_config( dm ),
    cuda_iter_spstep_frechet_kernels_config( dm ),
    cuda_iter_spstep_solvelinsys_kernels_config( dm ),
    cuda_reductions_kernels_config( dm ),
    cuda_testing_kernels_config( dm )
  {
  }

  void printDataFields();
};

#endif
