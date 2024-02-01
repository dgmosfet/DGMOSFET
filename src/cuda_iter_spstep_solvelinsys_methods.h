/*
  begin section : SLSRT_01
  creation      : 2023/06/09
  modified      : 2023/06/09
  author        : Francesco VECIL

  description   : Function 'solve_linsys()' has been modified as to return an integer value,
                  representing whether it succeeded in computing the solution of the linear system or not.
		  More into details, it returns the integer returned by 
		  either GPU_solve_linsys_SRJ, or CPU_solve_linsys_SRJ, or CPU_solve_linsys_LIS,
		  which is
		  0               --- if the method succeeded in computing the linear system
		  _ERR_SRJ_NOCONV --- if SRJ exceeded the maximum number of iterations
		  If CPU_solve_linsys_LIS fails, it exits execution before sending back an integer value.
 */
int solve_linsys(const double linsys_tolpar);
/*
  end section : SLSRT_01
 */

/*
  begin section : SLSRT_02
  creation      : 2023/06/09
  modified      : 2023/06/09
  author        : Francesco VECIL

  description   : Function 'CPU_solve_linsys_LIS()' uses the Library of Iterative Solvers 
                  and has been modified as to return an integer value,
                  representing whether it succeeded in computing the solution of the linear system or not.
		  More into details, it returns
		  0               --- if the method succeeded in computing the linear system
		  If CPU_solve_linsys_LIS fails, it exits execution before sending back an integer value.
 */
int CPU_solve_linsys_LIS();
/*
  end section : SLSRT_02
 */

/*
  begin section : SLSRT_03
  creation      : 2023/06/09
  modified      : 2023/06/09
  author        : Francesco VECIL

  description   : Function 'GPU_solve_linsys_SRJ()' implements the Scheduled Relaxed Jacobi on GPU,
                  and has been modified as to return an integer value,
                  representing whether it succeeded in computing the solution of the linear system or not.
		  More into details, it returns
		  0               --- if the method succeeded in computing the linear system
		  _ERR_SRJ_NOCONV --- if SRJ exceeded the maximum number of iterations
 */
int GPU_solve_linsys_SRJ(const double linsys_tolpar);
/*
  end section : SLSRT_03
 */

/*
  begin section : SLSRT_04
  creation      : 2023/06/09
  modified      : 2023/06/09
  author        : Francesco VECIL

  description   : Function 'GPU_solve_linsys_SRJ()' implements the Scheduled Relaxed Jacobi on CPU,
                  and has been modified as to return an integer value,
                  representing whether it succeeded in computing the solution of the linear system or not.
		  More into details, it returns
		  0               --- if the method succeeded in computing the linear system
		  _ERR_SRJ_NOCONV --- if SRJ exceeded the maximum number of iterations
 */
int CPU_solve_linsys_SRJ(const double linsys_tolpar);
/*
  end section : SLSRT_04
 */
void CPU_matrix_vector_product3( double *_residual_vec, const double * _x_k);
void CPU_update_x( double *_x_kp1, double *_residual_vec, double *_x_k, const double omega );

bool is_matrix_2d_diagonally_dominant();
double compute_matrix_2d_spectral_radius();


/*
  name          : 'check_solve_linsys_input', 'check_solve_linsys_output'
  last modified : 2023/06/06
  author        : Francesco VECIL

  description   : This functions check the input and the output of the 'solve_linsys' method.
                  As input  : (i) matrix_2d, (ii) rhs, (iii) pot [for initialization only]
		  As output : (iv) pot
 */
#define  _ERR_CHECK_SOLVE_LINSYS_INPUT_OUTPUT   170
void check_solve_linsys_input();
void check_solve_linsys_output();


#define   _ERR_GPU_SOLVE_LINSYS_SRJ_FINETRACKING   192

#define   _ERR_SRJ_NOCONV                          462
