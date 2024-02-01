#include "gridconfig.h"

void cudaGridConfig::printDataFields()
{
  cuda_perform_RK_1_3_config            -> printDataFields();
  cuda_perform_RK_2_3_config            -> printDataFields();
  cuda_perform_RK_3_3_config            -> printDataFields();
  cuda_set_fluxes_a3_config             -> printDataFields();
  cuda_set_fluxes_a2_testversion_config -> printDataFields();
  cuda_set_rhs_to_zero_config           -> printDataFields();

  cuda_phonons_loss_config              -> printDataFields();
  cuda_phonons_gain_config              -> printDataFields();
  cuda_compute_Wm1_config               -> printDataFields();

  kernel_roughness_gain_7_config        -> printDataFields();
  kernel_roughness_loss_config          -> printDataFields();
  kernel_stretch_totvoldens_config      -> printDataFields();
  kernel_construct_rhs_ext_config       -> printDataFields();
  kernel_compute_Deltapot_config        -> printDataFields();
  kernel_compute_overlap_SR_config      -> printDataFields();
  multiply_config                       -> printDataFields();

  cuda_WENO_X_20230525_config           -> printDataFields();
  cuda_WENO_PHI_config                  -> printDataFields();
  cuda_WENO_W_20230524_config           -> printDataFields();

  cuda_pdftilde_config                  -> printDataFields();
  cuda_surfdens_2_config                -> printDataFields();

  cuda_initialize_eps_config            -> printDataFields();
  cuda_gershgorin_config                -> printDataFields();
  cuda_eigenvalues_NR_config            -> printDataFields();
  cuda_eigenvalues_ms_2_config          -> printDataFields();

  cuda_tridiag_Thomas_20230525_config   -> printDataFields();

  cuda_init_d_A_config                  -> printDataFields();

  cuda_compute_deps_dx_config           -> printDataFields();

  cuda_constr_linsys_config             -> printDataFields();

  cuda_compute_frechet_config           -> printDataFields();
  cuda_compute_eps_diff_m1_config       -> printDataFields();

  cuda_matrix_vector_product_config     -> printDataFields();
  cuda_update_x_config                  -> printDataFields();

  cuda_compute_voldens_config           -> printDataFields();
  cuda_compute_totvoldens_OLD_config    -> printDataFields();
  cuda_compute_totvoldens_config        -> printDataFields();
  cuda_currdens_voldens_config          -> printDataFields();

  // cuda_test_data_config                 -> printDataFields();
  // cuda_test_pdf_config                  -> printDataFields();
  // cuda_test_rhs_config                  -> printDataFields();
  // cuda_compare_config                   -> printDataFields();
  
  return;
}
