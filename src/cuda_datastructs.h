double *___A;                       // 1

double *___A_const;                 // 2
#ifdef __CUDACC__
double *_GPU___A_const;             // 2
#endif

/* double _adim_cp;                    // 2.1 */

adimParams *host_adimpar;
adimParams *device_adimpar;

double _analyticnormfact[3];        // 3

// For debugging purposes
int aux_int_1;
int aux_int_2;
int aux_int_3;
int aux_int_4;
int aux_double_1;
int aux_double_2;
int aux_double_3;
int aux_double_4;

double _avgcurr;                    // 4

double *_avgdriftvel;               // 5
#ifdef __CUDACC__
double *_GPU_avgdriftvel;            
#endif

double _avgdriftvel_min;            // 5.1
double _avgdriftvel_max;            // 5.1
double _avgdriftvel_scalar;

double *_a1;                        // 6
#ifdef __CUDACC__
double *_GPU_a1;                    
#endif

#ifdef __CUDACC__
double *_GPU_a2;                    // 7
#endif

double *_a3;                        // 8
double *_a3_const;
#ifdef __CUDACC__
double *_GPU_a3;                    
double *_GPU_a3_const;                    
#endif

BoundCond *_bc_confined;            // 5

double *_chi;                       // 6
#ifdef __CUDACC__
double *_GPU_chi;                   // 5
#endif
#define _CHI_SIZE   (_NVALLEYS*_NSBN*_NX*_SCHROED_MATRIX_SIZE_PAD)

string comment;

double *_cosphi;                    // 7

double *_cscatt;                    // 8

double _c_SR;

double _ct[100];

int _nb[100];

int _jacobi_iters;

string CPU_INFO;

int CUDA_DEVICE; /*
		   This integer corresponds to the GPU device id.
		   It is set, if the GPU is used for the simulations,
		   by the main. Default value is defined in the main,
		   this value being subject to change by inline command
		   ./detmosfet --GPU id
		  */

double *_currdens;                  // 9
#ifdef __CUDACC__
double *_GPU_currdens;              // 6
#endif

#ifdef __CUDACC__
double *d_A;
#endif

double *_Deltapot_lower;            // 10
#ifdef __CUDACC__
double *_GPU_Deltapot_lower;            // 10
#endif

double *_Deltapot_upper;            // 11
#ifdef __CUDACC__
double *_GPU_Deltapot_upper;            // 10
#endif

double _defpot[8];

double *_denom_SR;                  // 12
#ifdef __CUDACC__
double *_GPU_denom_SR;              // 7
#endif

double *_deps_dx;                   // 13
#ifdef __CUDACC__
double *_GPU_deps_dx;               // 8
#endif

/*
  begin section : PD29
  date          : 2023/05/17
  description   : These structures will contain all the discretization meshes information, individually.
 */
discrDim *host_dd_X;
discrDim *device_dd_X;

discrDim *host_dd_Z;
discrDim *device_dd_Z;

discrDim *host_dd_W;
discrDim *device_dd_W;

discrDim *host_dd_PHI;
discrDim *device_dd_PHI;

discrDim *host_dd_ZEXT;
discrDim *device_dd_ZEXT;
/*
  end section : PD29
 */

/*
  begin section : PD29
  date          : 2023/05/16
  description   : These structures will contain all the discretization meshes information.
 */
discrMeshes *host_dm;
discrMeshes *device_dm;
/*
  end section : PD29
 */

double _dt;

double _dtmax;

double *_effmass;                   // 14

EigenProblem _ep;

double *_eps;                       // 15
#ifdef __CUDACC__
double *_GPU_eps;                   // 9
#endif

double *_eps_diff_m1;   
#ifdef __CUDACC__
double *_GPU_eps_diff_m1;
#endif

double *_epskin;                    // 16

double *_epsr_ext;                  // 17

double _ETA;

string fieldname[100];

cudaGridConfig *host_gridconfig;

double hours;

double *_integrateddenom_SR;        // 18
#ifdef __CUDACC__
double *_GPU_integrateddenom_SR;    // 10
#endif

double *_integrated_pdf_energy;     // 18.1
#ifdef __CUDACC__
double *_GPU_integrated_pdf_energy; // 11
#endif

double *_integrated_pdf_nu_p_i_m;   // 18.2

double _intnd;

bool *_ischannel;                   // 19

bool *_isdrain;                     // 20

bool *_issource;                    // 21

bool *_isdrain_ext;                 // 22

bool *_isoxide_ext;                 // 23

bool *_issource_ext;                // 24

double *_I_SR;                      // 25
#ifdef __CUDACC__
double *_GPU_I_SR;                  // 12
#endif

int iter_counter;

#ifdef __CUDACC__
int *_GPU_J_to_j;
int *_GPU_J_to_jj;
#endif

double _kane[_NVALLEYS];

double *_frechet;                    // 26
#ifdef __CUDACC__
double *_GPU_frechet;                // 24
#endif
/* #define _FRECHET_SIZE                (_NX*_NZ*_NZ) */

double _linfV, _l2V, _linfN, _l2N;

LIS_MATRIX _matrix_2d_LIS;
double *_matrix_2d;                 // 
#ifdef __CUDACC__
double *_GPU_matrix_2d;             // 12.0
#endif

double _mass[3*_NVALLEYS];

double *_matrix_2dconst;            // 27
double *_matrix_2dconst_ext_inv;
#ifdef __CUDACC__
double *_GPU_matrix_2dconst;        // 12.1
double *_GPU_matrix_2dconst_ext_inv;
#endif

double *_matrix_2dconst_ext;        // 28
#ifdef __CUDACC__
int *    _GPU_csrRowIndex_matrix_2dconst_ext;
int *    _GPU_csrColIndex_matrix_2dconst_ext;
double * _GPU_csrVal_matrix_2dconst_ext;
#endif

double *_maxw;                      // 29
#ifdef __CUDACC__
double *_GPU_maxw;                  // 13
#endif

double _max_currdens;

double _max_deps_dx;

double _max_driftvel;

double _max_test;

double _max_eps;

double _max_potential;

double _max_totsurfdens;

double _max_totvoldens;

double _max_a1;

double _max_a2;

double _max_a3;

double _max_a3const;

double minutes;

double _min_currdens;

double _min_deps_dx;

double _min_driftvel;

double _min_test;

double _min_eps;

int nnz_ext;
  
double _min_potential;

double _min_totsurfdens;

double _min_totvoldens;

double *_nd;                        // 30
#ifdef __CUDACC__
double *_GPU_nd;                    // 13.1
#endif

double *_nd_ext;                    // 31

double _normfact[3];

double *_occupations;               // 32

double *_omega;                     // 33

double _oxidemass;

double *_pdf;                       // 34
#ifdef __CUDACC__
double *_GPU_pdf;                   // 14
#endif

double _phtemp[8];

physConsts *host_phyco;
physConsts *device_phyco;

/* physDevice physdesc; */
/*
  begin section : PD23
  date          : 2023/05/11
  description   : I am adding these two lines, as I starting today
                  to insert the usage of this class inside the code.
 */
physDevice *host_physdesc;
physDevice *device_physdesc;
/*
  end section   : PD23
 */

double *_pot;                       // 35
#ifdef __CUDACC__
double *_GPU_pot;                   // 14.1
#endif
/* #define _POT_SIZE    (_NX*_NZ) */

double *_pot_b;                     // 36
#ifdef __CUDACC__
double *_GPU_pot_b;                 // 14.1.1
#endif

double *_pot_b_ext;                 // 37

double *_pot_ext;                   // 38
#ifdef __CUDACC__
double *_GPU_pot_ext;                   // 38
#endif

double *_pot_OLD;                   // 39
#ifdef __CUDACC__
double *_GPU_pot_OLD;               // 14.2
#endif

#ifdef __CUDACC__
cudaDeviceProp props; /* 
			 This is a structure that contains info about the GPU that is being used, the full description being available at
			 https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html
			 In particular, 
			 char cudaDeviceProp::name       describes the name of the GPU
			 int  cudaDeviceProp::major      describes the compute capability
			 int  cudaDeviceProp::minor      describes the compute capability
		      */
#endif

double _repartition_b;

rescalingParams *host_rescpar;
rescalingParams *device_rescpar;

double *_result_ext;                // 40

double *_rhs;                       // 
#ifdef __CUDACC__
double *_GPU_rhs;                   // 14.2.1   
#endif

double *_rhs_const;                 // 41
#ifdef __CUDACC__
double *_GPU_rhs_const;             // 14.3
double *_GPU_rhs_ext; 
double *_GPU_rhs_ext_const; 
#endif

double *_rhs_pdf;                   // 41.1
#ifdef __CUDACC__
double *_GPU_rhs_pdf;               // 15
#endif

#ifdef __CUDACC__
double *_GPU_rhs_pdf_gain;          // 16
#endif

double *_test_gain;                 // 41.1.1

double *_test_loss;                 // 41.1.2

LIS_VECTOR _res_LIS;                // 41.1.2.1

LIS_VECTOR _rhs_LIS;                // 41.1.3

double *_rhs_pdf_gain;              // 41.2

double *_righthandside_ext;         // 42

enum ScattType  _scatttype[8];

double seconds;

double *_sigma;
#ifdef __CUDACC__
double *_GPU_sigma;
#endif

double *_Sigma;
#ifdef __CUDACC__
double *_GPU_Sigma;
#endif

double *_sinphi;                    // 43

/* solverParams solvpar; */
solverParams *host_solvpar;
solverParams *device_solvpar;

double _sqrtmass[3*_NVALLEYS];

double _sqrtmassXY[_NVALLEYS];

srjParams *host_srj;

int _stage_ep;

double start_val[100];
double stop_val[100];

int _step;

int success;

double *_surfdens;                  // 44 
#ifdef __CUDACC__
double *_GPU_surfdens;              // 17
#endif
/* #define _SURFDENS_SIZE    (_NVALLEYS*_NSBN*_NX) */

double *_surfdens_eq;               // 45
#ifdef __CUDACC__
double *_GPU_surfdens_eq;           // 18
#endif

#ifdef __CUDACC__
double *_GPU_test_gain;             // 15.1
#endif

#ifdef __CUDACC__
double *_GPU_test_loss;             // 15.2
#endif

double *_totsurfdens_eq;            // 45.1

double _time;
double _time_init;

double *_totcurrdens;               // 46
#ifdef __CUDACC__
double *_GPU_totcurrdens;           // 19
#endif

double _totmass;

double *_totsurfdens;               // 47
#ifdef __CUDACC__
double *_GPU_totsurfdens;           // 20
#endif

double *_totvoldens;                // 48
#ifdef __CUDACC__
double *_GPU_totvoldens;            // 21
#endif
/* #define _TOTVOLDENS_SIZE    (_NX*_NZ) */

double *_totvoldens_ext;
#ifdef __CUDACC__
double *_GPU_totvoldens_ext;
#endif

double *_totvoldens_OLD;            // 50
#ifdef __CUDACC__
double *_GPU_totvoldens_OLD;        // 21.1
#endif

double *_vconf;                     // 51

double *_vel;           
#ifdef __CUDACC__
double *_GPU_vel;       
#endif

double *_vgate;                     // 52

double *_voldens;                   // 53
#ifdef __CUDACC__
double *_GPU_voldens;               // 22
#endif

double *_Wm1;                       // 54
#ifdef __CUDACC__
double *_GPU_Wm1;                   // 23
#endif


