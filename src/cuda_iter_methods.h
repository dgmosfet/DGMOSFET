/*
  begin section : ITER_09
  creation      : 2023/06/09
  modified      : 2023/06/09
  author        : Francesco VECIL

  description   : Function 'iter_2()' has been modified as to return an integer value,
                  representing whether it succeeded or not.
		  See into the .cu file for more details on the integer return code.
 */
int iter_2( const FixedPointType, const FixedPointTest, const double fp_tolpar, const double eigvals_tolpar, const double eigvecs_tolpar, const double linsys_tolpar );
/*
  end section : ITER_09
*/



/*
  begin section : ITER_10
  creation      : 2023/06/09
  modified      : 2023/06/09
  author        : Francesco VECIL

  description   : Function 'update_potential_2(ftp)' can use either NEWTON_RAPHSON or GUMMEL,
                  and has been modified as to return an integer value,
                  representing whether it succeeded or not.
		  More into details, it returns the value returned by 'solve_linsys'.
 */
int update_potential( const FixedPointType fpt, const double linsys_tolpar );
/*
  end section : ITER_10
*/




void initcond_on_cuda();
void thermequil_on_cuda();
#ifdef __CUDACC__
void iter_on_CPU();
#endif

#ifdef __CUDACC__
void GPU_compute_deps_dx();
#endif
void CPU_compute_deps_dx();
void compute_deps_dx();

#ifdef __CUDACC__
void GPU_compute_max_deps_dx();
#endif
void CPU_compute_max_deps_dx();
void compute_max_deps_dx();

#ifdef __CUDACC__
void GPU_reverse_into_OLD();
#endif
void reverse_into_OLD();
void CPU_reverse_into_OLD();

#ifdef __CUDACC__
bool GPU_test_conv_POTENTIAL(const double tolpar);
bool GPU_test_conv_POTENTIAL_ABSOLUTE(const double tolpar);
#endif
bool CPU_test_conv_POTENTIAL_ABSOLUTE(const double tolpar);
bool CPU_test_conv_POTENTIAL(const double tolpar);
bool test_conv_POTENTIAL(const double tolpar);
bool test_conv_POTENTIAL_ABSOLUTE(const double tolpar);

void compute_THERMEQUIL();
void compute_INITCOND();
bool CPU_test_conv_INITCOND_THERMEQUIL();

#ifdef __CUDACC__
template <class T> __device__ __host__ T reduce_maxabs(volatile T *sdata, int tid);
#endif

void store_corrupt_data(const double*, const double*, const double*, const double*, const double*);



/*
  name          : 'check_iter_2_input', 'check_iter_2_output'
  last modified : 2023/06/06
  author        : Francesco VECIL

  description   : This functions check the input and the output of the 'iter_2' method.
                  As input  : (i) surfdens, (ii) pot [as initialization]
		  As output : (iii) eps, (iv) deps_dx, (v) chi, (vi) pot
 */
#define  _ERR_CHECK_ITER_2_INPUT_OUTPUT   154
void check_iter_2_input();
void check_iter_2_output();



/*
  name          : 'check_update_potential_input', 'check_update_potential_output'
  last modified : 2023/06/06
  author        : Francesco VECIL

  description   : This functions check the input and the output of the 'update_potential' method.
                  As input  : (i) eps, (ii) chi, (iii) pot[OLD]
		  As output : (iv) pot[NEW]
 */
#define  _ERR_CHECK_UPDATE_POTENTIAL_INPUT_OUTPUT   157
void check_update_potential_input();
void check_update_potential_output();
