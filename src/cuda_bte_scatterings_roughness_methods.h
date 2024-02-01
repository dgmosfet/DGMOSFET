/*
  METHODS OF cuda_bte_scattering_roughness.cu
 */
void config_roughness();

double resampling_extended_to_original( const double Z );
double interpolate_pot_b( const double Z );



/*
  METHODS OF cuda_bte_scattering_roughness.cu
 */
void roughness_gain( const int STAGE );
void roughness_loss( const int STAGE );

void CPU_roughness_gain( const int STAGE );
void CPU_roughness_loss( const int STAGE );



/*
  METHODS OF cuda_bte_scattering_roughness_overlap.cu
 */
void compute_I_SR( const int stage );

void stretch_totvoldens();
void solve_poisson_ext();
void compute_Deltapot();
void compute_overlap_SR();

void CPU_stretch_totvoldens();
void GPU_stretch_totvoldens();

void CPU_solve_poisson_ext();
void GPU_solve_poisson_ext();
void GPU_construct_rhs_ext();

void CPU_compute_Deltapot();
void GPU_compute_Deltapot();

void CPU_compute_overlap_SR();
void GPU_compute_overlap_SR();
