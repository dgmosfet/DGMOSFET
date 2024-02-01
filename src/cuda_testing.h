/***************************************************
 *         SWITCH ON AND OFF TESTING ROUTINES      *
 ***************************************************/
/* #define _TEST_EIGENSTATES */
/* #define _SHOW_ETA */

/***************************************************
 *                  GENERAL TESTING                *
 ***************************************************/
template <class T> void test( T *dmem, const int N, const string="filename" );
void test_vec(double *, int, char*);
int compare_host_device(const double*, const double *, const int, const string="compare.txt");
int compare_device_device(const double*, const double *, const int, const string="compare.txt");

/*
  function      : compute_eta
  last modified : 2023/06/05
  author        : Francesco VECIL

  notes         : (i) Modified on 2023/06/05 in order to take into account 
                      for the computation of the ETA
                      not only the final simulation time
                      but also the maximum number of steps.
 */
void compute_eta()
{
  const double time_now = gettime();
  const double TIME   = _time*host_rescpar->get_tstar();

  const double slope  = TIME / ( time_now-_time_init );
  const double eft    = _time_init + host_solvpar->get_TMAX() / slope; /* estimated final time */

  const double slope2 = ( time_now-_time_init ) / (double)(_step);
  const double eft2   = _time_init + host_solvpar->get_STEPMAX() * slope2;

  // cerr << " slope2 = " << slope2 << " --- eft2 = " << eft2 << " --- eft = " << eft << " --- time_now = " << time_now << endl;
  
  _ETA = min(eft, eft2) - time_now;
  // _ETA =eft - time_now;
  
  seconds = (int)_ETA % 60;
  int totminutes = ((int)_ETA - seconds)/60;
  minutes = totminutes % 60;
  hours = (totminutes-minutes)/60;

  return;
}


void show_eta(string message)
{
  cerr << "ETA = " << setw(12) << fmax(hours,0) << "h" << setw(2) << fmax(minutes,0) << "'" << setw(2) << fmax(seconds,0) << "\""
       << ", step = "  << setw(12) << _step << " / " << host_solvpar->get_STEPMAX()
       << ", time = " << setw(12) << (_time)*host_rescpar->get_tstar()/1.e-12 << " ps" << " / " << host_solvpar->get_TMAX()/1.e-12 << " ps"
       << ", dt = " << setw(12) << _dt*host_rescpar->get_tstar()/1.e-12 << " ps"
       << ": " << message;

  return;
}

bool are_there_nan_inf(double *data, const int N, const string message="");


/***************************************************
 *         TESTING THE ITERATIVE PART              *
 ***************************************************/
void test_eigenstates_GPU();
void test_tridiag_solve_2();
void test_cuda_gershgorin( double*, double* );
void test_eigenvalues_device();
void test_eigenvalues_host();
void test_eigenvalues_matrices();
int compare_rhs_host_device(const string="compare.txt");
int compare_chi_host_device(const double*, const double *, const int, const string="compare.txt");
void compare_iter();
int compare_linsys_host_device(const string="compare.txt");
void test_rhs( const string="filename" );

/***************************************************
 *       TESTING THE DISTRIBUTION FUNCTION         *
 ***************************************************/
double max_pdf_border(const int s);
void test_pdf( const int STAGE, const string="filename" );

/***************************************************
 *                  FILE SAVING                    *
 ***************************************************/
/* template <class T> int save_device_array(T *pt, const int N, const string filename="prova.dat"); */
int save_device_array(double *pt, const int N, const string filename="prova.dat");
int save_host_array(double *pt, const int N, const string filename="prova.dat");
int save_GPU_matrix_2d(const string filename="prova.dat");
int save_GPU_pot(const string filename="prova.dat");
template <class T> void print_to_file( T *dmem, const int N, const string filename );

