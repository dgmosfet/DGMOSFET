/************************************************
 *               TO-DO LIST                     *
 ************************************************
1) Hacer el kernel integrate_pdf_density (en dens.cu) que lea de memoria global de manera coalescente, y que sea flexible 
   con respecto al tamano de bloque.
2) Reescribir el fichero mappings.h para factorizar funciones de acceso, como ya se ha hecho para unos muy pocos vectores.
*/



// #undef __CUDACC__
// #define __ITER_ON_CPU__ // in order to execute the code on GPU except the 'iter' computational phase



#ifndef CUDA_MOSFETPROBLEM_H
#define CUDA_MOSFETPROBLEM_H

// Cuda part
#ifdef __CUDACC__
#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include <cublas_v2.h>
#include <cusparse.h>
#include <cusolverSp.h>
#endif

#define   _NVALLEYS   3

/*************************************************************
 *         CONSTANT MAGNITUDES USED BY THE GPU DEVICE        *
 *************************************************************/
#ifdef __CUDACC__
__constant__ double        _GPU_mass[3*_NVALLEYS];
__constant__ double    _GPU_sqrtmass[3*_NVALLEYS];
__constant__ double  _GPU_sqrtmassXY[  _NVALLEYS];
__constant__ double        _GPU_kane[  _NVALLEYS];
__constant__ double _GPU_occupations[8*_NVALLEYS*_NVALLEYS];
__constant__ double      _GPU_cscatt[8];
__constant__ double       _GPU_omega[8];
#endif

/*************************************************************
 *************************************************************
 *                    MOSFETProblem class                    *
 *************************************************************
 *************************************************************/
class MOSFETProblemCuda
{
  /*********************************************************
   * CONSTRUCTOR AND DESTRUCTOR                            *
   *********************************************************/
 public:
  MOSFETProblemCuda(const int devId, int _argc, char** _argv);
  ~MOSFETProblemCuda();

  // For the constructor
  void allocate_memory();
  void compute_initcond();

  /*********************************************************
   * DATA                                                  *
   *********************************************************/
  // CONSTANT DATA
#include"cuda_constdata.h"

  // DATA STRUCTURES
#include"cuda_datastructs.h"
  
  /*********************************************************
   * METHODS OF THE CLASS                                  *
   *********************************************************/
  // ACCESS FUNCTIONS
#include"cuda_mappings.h"

  // CONFIGURATION AND INITIALIZATION
#include"cuda_config_methods.h"
 
  // REDUCTIONS
#include"cuda_reductions_methods.h"

  // TOP-LEVEL SOLVER METHOD
  /* void solve(); */
  void solve_2();
  bool stopcondition();
  
  // TIME INTEGRATION
#define RK_ORDER (3)
#include"cuda_time_integration_methods.h"
  
  // BTE
#include"cuda_bte_methods.h"
#include"cuda_bte_weno_methods.h"
#include"cuda_bte_rk_methods.h"
#include"cuda_bte_scatterings_methods.h"
#include"cuda_bte_scatterings_phonons_methods.h"
#include"cuda_bte_scatterings_roughness_methods.h"

  // DENS
#include"cuda_dens_methods.h"

  // ITER
#include"cuda_iter_methods.h"
#include"cuda_iter_spstep_methods.h"
#include"cuda_iter_spstep_constrlinsys_methods.h"
#include"cuda_iter_spstep_frechet_methods.h"
#include"cuda_iter_spstep_solvelinsys_methods.h"
#include"cuda_iter_eigenstates_methods.h"
#include"cuda_iter_eigenstates_eigenvalues_methods.h"
#include"cuda_iter_eigenstates_eigenvectors_methods.h"

  /*********************************************************
   * FILE STORAGE AND POST-PROCESSING, PLUS OTHER STUFF    *
   *********************************************************/
#include"cuda_filestorage.h"
#include"cuda_comptime.h"
#include"cuda_testing.h"
};

/*************************************************************
 *************************************************************
 *                        CUDA kernels                       *
 *************************************************************
 *************************************************************/
#ifdef __CUDACC__
#include"cuda_config_kernels.h"
#include"cuda_reductions_kernels.h"
#include"cuda_bte_rk_kernels.h"
#include"cuda_bte_weno_kernels.h"
#include"cuda_bte_scatterings_phonons_kernels.h"
#include"cuda_bte_scatterings_roughness_kernels.h"
#include"cuda_dens_kernels.h"
#include"cuda_iter_kernels.h"
#include"cuda_iter_spstep_kernels.h"
#include"cuda_iter_spstep_constrlinsys_kernels.h"
#include"cuda_iter_spstep_frechet_kernels.h"
#include"cuda_iter_spstep_solvelinsys_kernels.h"
#include"cuda_iter_eigenstates_kernels.h"
#include"cuda_iter_eigenstates_eigenvalues_kernels.h"
#include"cuda_iter_eigenstates_eigenvectors_kernels.h"
#include"cuda_testing_kernels.h"

#endif
// rimettere dentro lo ifdef
#include"cuda_kernels.h"



#endif

