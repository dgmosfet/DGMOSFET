/* void GPU_eigenvectors_subs(); */
/* void GPU_eigenvectors_PCR(); */
/* void GPU_eigenvectors_PCR_2(); */
/* void GPU_eigenvectors_invpow(); */
void GPU_eigenvectors_Thomas(const double eigvect_tolpar);
/* void GPU_eigenvectors_PCR_3(); */
/* void test_PCR(); */

/* void test_tridiag_Thomas( const double*, const double*, double* ); */

/* void CPU_eigenvectors(); */
/* void CPU_tridiag_solve_Thomas(); */
void CPU_tridiag_Thomas(const double eigvect_tolpar);



/*
  name          : 'check_GPU_eigenvectors_Thomas_input', 'check_GPU_eigenvectors_Thomas_output'
  last modified : 2023/06/06
  author        : Francesco VECIL

  description   : This functions check the input and the output of the 'GPU_eigenvectors_Thomas' method.
                  Namely, 'check_GPU_eigenvectors_Thomas_input' checks (i) matrix d_A, (ii) eps
		  while 'check_GPU_eigenvectors_Thomas_output' checks (iii) chi.
 */
#define   _ERR_CHECK_GPU_EIGENVECTORS_THOMAS_INPUT   152
void check_GPU_eigenvectors_Thomas_input();
void check_GPU_eigenvectors_Thomas_output();
