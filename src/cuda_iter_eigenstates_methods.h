void eigenstates(const SPPhase spphase, const double eigvals_tolpar, const double eigvecs_tolpar);

void CPU_prepare_eigenstates(const SPPhase spphase);

void CPU_eigenstates(const SPPhase spphase, const double eigvals_tolpar, const double eigvecs_tolpar);
void GPU_eigenstates(const SPPhase spphase, const double eigvals_tolpar, const double eigvecs_tolpar);
void GPU_prepare_eigenstates(const SPPhase spphase);
void GPU_eigenstates_test(const SPPhase spphase);
void CPU_eigenstates_LAPACK(const SPPhase spphase);
void CPU_eigenvalues_LAPACK(const SPPhase spphase);

/*
  name          : 'check_eigenstates_input', 'check_eigenstates_output'
  last modified : 2023/06/06
  author        : Francesco VECIL

  description   : This functions check the input and the output of the 'eigenstates' method.
                  Namely, 'check_eigenstates_input' checks (i) surfdens, (ii) pot[spphase]
		  while 'check_eigenstates_output' checks (iii) eps, (iv) chi[spphase].
 */
#define  _ERR_CHECK_EIGENSTATES_INPUT   149
void check_eigenstates_input(const SPPhase spphase);
void check_eigenstates_output();



/*
  name          : 'check_GPU_prepare_eigenstates_input', 'check_GPU_prepare_eigenstates_output'
  last modified : 2023/06/06
  author        : Francesco VECIL

  description   : This functions check the input and the output of the 'GPU_prepare_eigenstates' method.
                  Namely, 'check_GPU_prepare_eigenstates_input' checks (i) pot[spphase]
		  while 'check_GPU_prepare_eigenstates_output' checks (ii) matrix d_A.
 */
#define   _ERR_CHECK_GPU_PREPARE_EIGENSTATES_INPUT   150
void check_GPU_prepare_eigenstates_input(const SPPhase spphase);
void check_GPU_prepare_eigenstates_output();



