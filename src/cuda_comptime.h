// ANALYSIS OF THE COMPUTATIONAL TIMES

inline string ReplaceAll(string str, const string& from, const string& to)
{
  size_t start_pos = 0;
  while((start_pos = str.find(from, start_pos)) != string::npos)
    {
      str.replace(start_pos, from.length(), to);
      start_pos += to.length(); // Handles case where 'to' is a substring of 'from'
    }
  return str;
}


#define _COMPTIMES

#define    _PHASE_STEP                                                0  // 
          
#define    _PHASE_BTE                                                 1  // 
#define    _PHASE_BTE_SETFLUXES                                       2  //
#define    _PHASE_BTE_SCATTERINGS                                     3  //
#define    _PHASE_BTE_SCATTERINGS_PHONONS                             4  //
#define    _PHASE_BTE_SCATTERINGS_ROUGHNESS                           5  //
#define    _PHASE_BTE_SCATTERINGS_PHONONS_WM1                         6  //
#define    _PHASE_BTE_SCATTERINGS_PHONONS_PHONONSGAIN                 7  //
#define    _PHASE_BTE_SCATTERINGS_PHONONS_PHONONSLOSS                 8  //
#define    _PHASE_BTE_SCATTERINGS_ROUGHNESS_ISR                       9  //
#define    _PHASE_BTE_SCATTERINGS_ROUGHNESS_ISR_STRETCHTOTVOLDENS    10  //
#define    _PHASE_BTE_SCATTERINGS_ROUGHNESS_ISR_SOLVEPOISSEXT        11  //
#define    _PHASE_BTE_SCATTERINGS_ROUGHNESS_ISR_DELTAPOT             12  //
#define    _PHASE_BTE_SCATTERINGS_ROUGHNESS_ISR_OVERLAPSR            13  //
#define    _PHASE_BTE_SCATTERINGS_ROUGHNESS_ROUGHNESSGAIN            14  //
#define    _PHASE_BTE_SCATTERINGS_ROUGHNESS_ROUGHNESSLOSS            15  //
#define    _PHASE_BTE_WENOX                                          16  //
#define    _PHASE_BTE_WENOW                                          17  //
#define    _PHASE_BTE_WENOPHI                                        18  //
#define    _PHASE_BTE_RK                                             19  //

#define    _PHASE_DENS                                               20  // 

#define    _PHASE_ITER                                               21  // 
#define    _PHASE_ITER_EIGENSTATES                                   22  // 
#define    _PHASE_ITER_EIGENSTATES_PREPARE                           23  // 
#define    _PHASE_ITER_EIGENSTATES_EIGENVALUES                       24  // 
#define    _PHASE_ITER_EIGENSTATES_EIGENVECTORS                      25  // 
#define    _PHASE_ITER_SPSTEP                                        26  //
#define    _PHASE_ITER_SPSTEP_FRECHET                                27  // 
#define    _PHASE_ITER_SPSTEP_CONSTRLINSYS                           28  // 
#define    _PHASE_ITER_SPSTEP_SOLVELINSYS                            29  //

#define    _NUM_CT_FIELDS                                            30 
          
void config_comptime();
double gettime();
void analysis_comptime();
void save_times();
void save_histograms();
void save_piecharts();
void save_piecharts_2( const int *, int, const string, const int=_PHASE_STEP );
void save_hostograms();
void save_histograms_2( const int *, int, const string, const int=_PHASE_STEP );
void save_histograms_3( const int *, int, const string, const int=_PHASE_STEP );
void save_speedups();
void save_speedups_2( const string, const int=_PHASE_STEP );
void save_speedups_3();

/**
   PURPOSE:        

   FILE:           cuda_comptime.cu

   NAME:           MOSFETProblem::start_time

   PARAMETERS:

   RETURN VALUE:   none

   CALLS TO:       MOSFETProblem::gettime                       (cuda_comptime.cu)

   CALLED FROM:    MOSFETProblem::perform_RK_1_3                (cuda_bte_rk.cu)
                   MOSFETProblem::perform_RK_2_3                (cuda_bte_rk.cu)
                   MOSFETProblem::perform_RK_3_3                (cuda_bte_rk.cu)
                   MOSFETProblem::set_fluxes_a3                 (cuda_bte_rk.cu)
                   MOSFETProblem::set_fluxes_a2                 (cuda_bte_rk.cu)
                   MOSFETProblem::scatterings                   (cuda_bte_scatterings.cu)
                   MOSFETProblem::phonons_loss                  (cuda_bte_scatterings_phonons.cu)
                   MOSFETProblem::phonons_gain                  (cuda_bte_scatterings_phonons.cu)
                   MOSFETProblem::compute_Wm1                   (cuda_bte_scatterings_phonons.cu)
                   MOSFETProblem::compute_I_SR                  (cuda_bte_scatterings_roughness.cu)
                   MOSFETProblem::roughness_gain                (cuda_bte_scatterings_roughness.cu)
                   MOSFETProblem::roughness_loss                (cuda_bte_scatterings_roughness.cu)
                   MOSFETProblem::compute_overlap_SR            (cuda_bte_scatterings_roughness.cu)
                   MOSFETProblem::compute_Deltapot              (cuda_bte_scatterings_roughness.cu)
                   MOSFETProblem::solve_poisson_ext             (cuda_bte_scatterings_roughness.cu)
                   MOSFETProblem::weno_X                        (cuda_bte_weno.cu)
                   MOSFETProblem::weno_W                        (cuda_bte_weno.cu)
                   MOSFETProblem::weno_PHI                      (cuda_bte_weno.cu)
                   MOSFETProblem::dens                          (cuda_dens.cu)
                   MOSFETProblem::iter_2                        (cuda_iter.cu)
                   MOSFETProblem::GPU_eigenvectors_ingenieros   (cuda_iter_eigenvectors.cu)
                   MOSFETProblem::CPU_prepare_eigenstates       (cuda_iter_eigenvectors.cu)
                   MOSFETProblem::CPU_eigenvalues_multisection  (cuda_iter_eigenvectors.cu)
                   MOSFETProblem::GPU_prepare_eigenstates       (cuda_iter_eigenvectors.cu)
                   MOSFETProblem::GPU_eigenvalues_multisection  (cuda_iter_eigenvectors.cu)
                   MOSFETProblem::eigenstates                   (cuda_iter_eigenvectors.cu)
                   MOSFETProblem::constr_linsys                 (cuda_iter_spstep_constrlinsys.cu)
                   MOSFETProblem::compute_frechet               (cuda_iter_spstep_frechet.cu)
                   MOSFETProblem::solve_linsys                  (cuda_iter_spstep_solvelinsys.cu)
                   MOSFETProblem::perform_step_2                (cuda_time_integration.cu)

   DATA MODIFIED:  

   METHOD:         

   LAST UPDATE:    2022 February 16
*/
inline void start_time(const int measure)
{
#ifdef _COMPTIMES
#pragma omp barrier
#ifdef __CUDACC__
  cudaDeviceSynchronize();
#endif
  start_val[measure] = gettime();
#endif

  return;
}

/* void start_time(const int); */

/**
   PURPOSE:        

   FILE:           cuda_comptime.cu

   NAME:           MOSFETProblem::stop_time

   PARAMETERS:

   RETURN VALUE:   none

   CALLS TO:       MOSFETProblem::gettime                       (cuda_comptime.cu)

   CALLED FROM:    MOSFETProblem::perform_RK_1_3                (cuda_bte_rk.cu)
                   MOSFETProblem::perform_RK_2_3                (cuda_bte_rk.cu)
                   MOSFETProblem::perform_RK_3_3                (cuda_bte_rk.cu)
                   MOSFETProblem::set_fluxes_a3                 (cuda_bte_rk.cu)
                   MOSFETProblem::set_fluxes_a2                 (cuda_bte_rk.cu)
                   MOSFETProblem::scatterings                   (cuda_bte_scatterings.cu)
                   MOSFETProblem::phonons_loss                  (cuda_bte_scatterings_phonons.cu)
                   MOSFETProblem::phonons_gain                  (cuda_bte_scatterings_phonons.cu)
                   MOSFETProblem::compute_Wm1                   (cuda_bte_scatterings_phonons.cu)
                   MOSFETProblem::compute_I_SR                  (cuda_bte_scatterings_roughness.cu)
                   MOSFETProblem::roughness_gain                (cuda_bte_scatterings_roughness.cu)
                   MOSFETProblem::roughness_loss                (cuda_bte_scatterings_roughness.cu)
                   MOSFETProblem::compute_overlap_SR            (cuda_bte_scatterings_roughness.cu)
                   MOSFETProblem::compute_Deltapot              (cuda_bte_scatterings_roughness.cu)
                   MOSFETProblem::solve_poisson_ext             (cuda_bte_scatterings_roughness.cu)
                   MOSFETProblem::weno_X                        (cuda_bte_weno.cu)
                   MOSFETProblem::weno_W                        (cuda_bte_weno.cu)
                   MOSFETProblem::weno_PHI                      (cuda_bte_weno.cu)
                   MOSFETProblem::dens                          (cuda_dens.cu)
                   MOSFETProblem::iter_2                        (cuda_iter.cu)
                   MOSFETProblem::GPU_eigenvectors_ingenieros   (cuda_iter_eigenvectors.cu)
                   MOSFETProblem::CPU_prepare_eigenstates       (cuda_iter_eigenvectors.cu)
                   MOSFETProblem::CPU_eigenvalues_multisection  (cuda_iter_eigenvectors.cu)
                   MOSFETProblem::GPU_prepare_eigenstates       (cuda_iter_eigenvectors.cu)
                   MOSFETProblem::GPU_eigenvalues_multisection  (cuda_iter_eigenvectors.cu)
                   MOSFETProblem::eigenstates                   (cuda_iter_eigenvectors.cu)
                   MOSFETProblem::constr_linsys                 (cuda_iter_spstep_constrlinsys.cu)
                   MOSFETProblem::compute_frechet               (cuda_iter_spstep_frechet.cu)
                   MOSFETProblem::solve_linsys                  (cuda_iter_spstep_solvelinsys.cu)
                   MOSFETProblem::perform_step_2                (cuda_time_integration.cu)

   DATA MODIFIED:  

   METHOD:         

   LAST UPDATE:    2022 February 16
*/

inline void stop_time(const int measure)
{
#ifdef _COMPTIMES
#pragma omp barrier
#ifdef __CUDACC__
  cudaDeviceSynchronize();
#endif
  stop_val[measure] = gettime();
  _ct[ measure ] += stop_val[measure] - start_val[measure];
  ++_nb[ measure ];
#endif

  return;
}

/* void stop_time(const int); */

