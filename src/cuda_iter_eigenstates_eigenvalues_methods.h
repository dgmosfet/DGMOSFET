void GPU_eigenvalues_NR(const double eigvals_tolpar);
void GPU_eigenvalues_ms(const double eigvals_tolpar);
void GPU_eigenvalues(const double eigvals_tolpar);

void CPU_gershgorin( const double *A, double *_GPU_Y, double *_GPU_Z );
void CPU_eigenvalues_ms(const double eigvals_tolpar);
void CPU_eigenvalues_NR(const double eigvals_tolpar);



/*
  name          : 'check_GPU_eigenvalues_input', 'check_GPU_eigenvalues_output'
  last modified : 2023/06/06
  author        : Francesco VECIL

  description   : This functions check the input and the output of the 'GPU_eigenvalues' method.
                  Namely, 'check_GPU_eigenvalues_input' checks (i) matrix d_A
		  while 'check_GPU_eigenvalues_output' checks (ii) eps.
 */
#define _ERR_CHECK_GPU_EIGENVALUES_INPUT   151
void check_GPU_eigenvalues_input();
void check_GPU_eigenvalues_output();



