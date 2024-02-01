#include "mosfetproblem.h"

/**
   PURPOSE:        This function performs all the computations needed to set up 
                   the problem. Namely:
                   1) It shows some relevant parameters so that the user can check that they are running the desired simulation.
		   2) It creates the files and directories where the results are going to be stored.
		   3) It allocates the memory on CPU and GPU for all the magnitudes that are going to be used.
		   4) It configures all the information that is constant along the execution of the code.
		   5) It computes the initial condition.

   FILE:           cuda_mosfetproblem.cu

   NAME:           MOSFETProblem::MOSFETProblem (constructor)

   PARAMETERS:

   RETURN VALUE:   none (constructor)

   CALLS TO:       MOSFETProblem::allocate_memory       (cuda_config.cu)
                   MOSFETProblem::config                (cuda_config.cu)
                   MOSFETProblem::compute_initcond      (cuda_config.cu)

   CALLED FROM:    main                                 (main.cu)

   DATA MODIFIED:  

   METHOD:         

   LAST UPDATE:    2023/04/24
*/
MOSFETProblemCuda::MOSFETProblemCuda(const int devId, int _argc, char** _argv)
{
  /*
    begin section : constr_01
    date          : ??
    description   : The first thing that MOSFETProblemCuda must do is to fill its solverParams class data structure.
   */
  host_solvpar = new solverParams();
  host_solvpar->readInline( _argc, _argv );

  // delete from here
  host_solvpar->printDataFields();
  // to here

#ifdef __CUDACC__
  checkCudaErrors( cudaMalloc((void **)&device_solvpar, sizeof(solverParams)) );                                
  checkCudaErrors( cudaMemcpy( device_solvpar, host_solvpar, sizeof(solverParams), cudaMemcpyHostToDevice) );
#endif
  /*
    end section   : constr_01
   */
  
  /*
    begin section : constr_02
    date          : 2023/05/11
    description   : The second thing that MOSFETProblemCuda must do is to fill its physDevice class data structure.
                    The object is called 'physdesc' instead of 'physdevice', the names are not 
		    100% consistent, unlike the solverParams class, but it is not that important.
   */
  host_physdesc = new physDevice();
  host_physdesc->readInline( _argc, _argv );

  // delete from here
  host_physdesc->printDataFields();
  // to here

#ifdef __CUDACC__
  checkCudaErrors( cudaMalloc((void **)&device_physdesc, sizeof(physDevice)) );                                
  checkCudaErrors( cudaMemcpy( device_physdesc, host_physdesc, sizeof(physDevice), cudaMemcpyHostToDevice) );
#endif
  /*
    end section   : constr_02
   */
  
  /*
    begin section : constr_03
    date          : 2023/05/15
    description   : Construction of the physConsts object phyco
   */
  host_phyco = new physConsts();

  // // delete from here
  // cerr << " Print physical constants" << endl;
  // host_phyco-> printConsts();
  // // to here

#ifdef __CUDACC__
  checkCudaErrors( cudaMalloc((void **)&device_phyco, sizeof(physConsts)) );                                
  checkCudaErrors( cudaMemcpy( device_phyco, host_phyco, sizeof(physConsts), cudaMemcpyHostToDevice) );
#endif
  /*
    end section   : constr_03
   */

  /*
    begin section : constr_04
    date          : 2023/05/15
    description   : Construction of the rescalingParams object rescpar
   */
  host_rescpar = new rescalingParams(host_physdesc, host_phyco);

  // // delete from here
  // cerr << " Print rescaling parameters" << endl;
  // host_rescpar->printParams();
  // // to here

#ifdef __CUDACC__
  checkCudaErrors( cudaMalloc((void **)&device_rescpar, sizeof(rescalingParams)) );                                
  checkCudaErrors( cudaMemcpy( device_rescpar, host_rescpar, sizeof(rescalingParams), cudaMemcpyHostToDevice) );
#endif
  /*
    end section   : constr_04
   */

  /*
    begin section : constr_05
    date          : 2023/05/15
    description   : Construction of the adimParams object rescpar
   */
  host_adimpar = new adimParams(host_physdesc, host_phyco, host_rescpar);

  // // delete from here
  // cerr << " Print adimensionalized parameters" << endl;
  // host_adimpar->printParams();
  // // to here

#ifdef __CUDACC__
  checkCudaErrors( cudaMalloc((void **)&device_adimpar, sizeof(adimParams)) );                                
  checkCudaErrors( cudaMemcpy( device_adimpar, host_adimpar, sizeof(adimParams), cudaMemcpyHostToDevice) );
#endif
  /*
    end section   : constr_05
   */

  /*
    begin section : constr_07
    date          : 2023/05/17
    description   : Construction of the discrDim objects, one for each dimension of the problem

    notes         : Here, the real meshes will be introduced. More into details, for the sake 
                    of this code we will need full information about

		    (i)   number of subbands :           host_solvpar->get_NSBN()

		    (ii)  x-dimension discretization :   N = host_solvpar->get_NX()
		                                         min = 0 
						         max = host_physdesc->compute_XLENGTH() / host_rescpar->get_xstar()
							 mt = std_mesh

		    (iii) z-dimension discretization :   N = host_solvpar->get_NZ()
		                                         min = 0 
						         max = host_physdesc->get_ZWIDTH() / host_rescpar->get_zstar()
							 mt = std_mesh

		    (iv)  w-dimension discretization :   N = host_solvpar->get_NW()
		                                         min = 0 
						         max = wmax     WHERE     wmax = host_solvpar->get_BARN() * omega_ref * host_phyco->__hbar / host_rescpar->get_epsstar()     AND     omega_ref = 0.063 * host_phyco->__eV / host_phyco->__hbar
							 mt = W_mesh

		    (v)   phi-dimension discretization : N = host_solvpar->get_NPHI()
		                                         min = 0 
						         max = 2.*M_PI
							 mt = PHI_mesh

		    (vi)  zext-dimension discretization : N = host_solvpar->get_NZEXT()
		                                          min = 0 
					      	          max = ( host_physdesc->get_ZWIDTH() + host_solvpar->get_DELTA_SR() ) / host_rescpar->get_zstar()
					 		  mt = std_mesh
   */
  const double __omega_ref = 0.063 * host_phyco->__eV / host_phyco->__hbar;
  const double __wmax      = host_solvpar->get_BARN() * __omega_ref * host_phyco->__hbar / host_rescpar->get_epsstar();

  host_dd_X    = new discrDim( host_solvpar->get_NX(),    0, host_physdesc->compute_XLENGTH() / host_rescpar->get_xstar()                              , std_mesh );
  host_dd_Z    = new discrDim( host_solvpar->get_NZ(),    0, host_physdesc->get_ZWIDTH() / host_rescpar->get_zstar()                                   , std_mesh );
  host_dd_W    = new discrDim( host_solvpar->get_NW(),    0, __wmax                                                                                    , W_mesh   );
  host_dd_PHI  = new discrDim( host_solvpar->get_NPHI(),  0, 2*M_PI                                                                                    , PHI_mesh );
  host_dd_ZEXT = new discrDim( host_solvpar->get_NZEXT(), 0, ( host_physdesc->get_ZWIDTH() + host_solvpar->get_DELTA_SR() ) / host_rescpar->get_zstar(), std_mesh );
  
#ifdef __CUDACC__
  checkCudaErrors( cudaMalloc((void **)&device_dd_X, sizeof(discrDim)) );                                
  checkCudaErrors( cudaMemcpy( device_dd_X, host_dd_X, sizeof(discrDim), cudaMemcpyHostToDevice) );

  checkCudaErrors( cudaMalloc((void **)&device_dd_Z, sizeof(discrDim)) );                                
  checkCudaErrors( cudaMemcpy( device_dd_Z, host_dd_Z, sizeof(discrDim), cudaMemcpyHostToDevice) );

  checkCudaErrors( cudaMalloc((void **)&device_dd_W, sizeof(discrDim)) );                                
  checkCudaErrors( cudaMemcpy( device_dd_W, host_dd_W, sizeof(discrDim), cudaMemcpyHostToDevice) );

  checkCudaErrors( cudaMalloc((void **)&device_dd_PHI, sizeof(discrDim)) );                                
  checkCudaErrors( cudaMemcpy( device_dd_PHI, host_dd_PHI, sizeof(discrDim), cudaMemcpyHostToDevice) );

  checkCudaErrors( cudaMalloc((void **)&device_dd_ZEXT, sizeof(discrDim)) );                                
  checkCudaErrors( cudaMemcpy( device_dd_ZEXT, host_dd_ZEXT, sizeof(discrDim), cudaMemcpyHostToDevice) );
#endif

  /*
    end section   : constr_07
   */

  /*
    begin section : constr_06
    date          : 2023/05/16
    modified      : 2023/05/19
    description   : Construction of the discrMeshes object, containing the discretization meshes

    notes         : Here, the real meshes will be introduced. More into details, for the sake 
                    of this code we will need full information about

		    (i)   number of subbands :           host_solvpar->get_NSBN()

		    (ii)  x-dimension discretization :   N = host_solvpar->get_NX()
		                                         min = 0 
						         max = host_physdesc->compute_XLENGTH() / host_rescpar->get_xstar()
							 mt = std_mesh

		    (iii) z-dimension discretization :   N = host_solvpar->get_NZ()
		                                         min = 0 
						         max = host_physdesc->get_ZWIDTH() / host_rescpar->get_zstar()
							 mt = std_mesh

		    (iv)  w-dimension discretization :   N = host_solvpar->get_NW()
		                                         min = 0 
						         max = wmax     WHERE     wmax = host_solvpar->get_BARN() * omega_ref * host_phyco->__hbar / host_rescpar->get_epsstar()     AND     omega_ref = 0.063 * host_phyco->__eV / host_phyco->__hbar
							 mt = W_mesh

		    (v)   phi-dimension discretization : N = host_solvpar->get_NPHI()
		                                         min = 0 
						         max = 2.*M_PI
							 mt = PHI_mesh
   */
  host_dm = new discrMeshes( host_solvpar->get_NSBN(), host_dd_X, host_dd_Z, host_dd_W, host_dd_PHI, host_dd_ZEXT );
#ifdef __CUDACC__
  discrMeshes* aux_dm = new discrMeshes( host_solvpar->get_NSBN(), device_dd_X, device_dd_Z, device_dd_W, device_dd_PHI, device_dd_ZEXT );
  checkCudaErrors( cudaMalloc((void **)&device_dm, sizeof(discrMeshes)) );                                
  checkCudaErrors( cudaMemcpy( device_dm, aux_dm, sizeof(discrMeshes), cudaMemcpyHostToDevice) );
#endif

  /*
    end section   : constr_06
   */

  /*
    begin section : constr_07
    date          : 2023/05/30
    modified      : 2023/05/30
    description   : 
   */
  host_gridconfig = new cudaGridConfig( host_dm );
  // // delete from here
  // host_gridconfig -> printDataFields();
  // // exit(0);
  // to here
  /*
    end section   : constr_07
   */



  /*
    begin section : constr_08
    date          : 2023/06/07
    modified      : 2023/06/08
    description   : 
   */
  // host_srj = new srjParams( 32, 7 );
  host_srj = new srjParams( 64, 7, "srj_parameters/P7_64.srj" );
  // host_srj = new srjParams( 32, 11, "srj_parameters/P11_32.srj" );
  // host_srj = new srjParams( 32, 14, "srj_parameters/P14_32.srj" );
  // delete from here
  // host_srj -> printDataFields();
  // to here
  /*
    end section   : constr_08
   */

  cerr << endl
       << " ************************************************************************************* " << endl
       << " *                       INITIALIZATION AND PRECOMPUTATIONS                          * " << endl
       << " ************************************************************************************* " << endl << endl;
  
  if(system( "mkdir films_and_graphs 1>/dev/null 2>/dev/null" ));
  if(system( "mkdir comptimes_analysis 1>/dev/null 2>/dev/null" ));
  if(system( "mkdir scripts 1>/dev/null 2>/dev/null" ));
  if(system( "mkdir results 1>/dev/null 2>/dev/null" ));
  if(system( "rm SRJ.txt 1>/dev/null 2>/dev/null" ));
  ostringstream sstr;
  sstr << "results/scalars.dat";
  string filename = sstr.str();
  ofstream str;
  str.open( filename.c_str(),ios_base::out );
  str << setprecision(12) 
      << setw(25) << "#1:time"                 // 1
      << setw(25) << "#2:total_mass"           // 2
      << setw(25) << "#3:avg_current"          // 3
      << setw(25) << "#4:step"                 // 4
      << setw(25) << "#5:dt"                   // 5
      << setw(25) << "#6:max_pdf_border(0)"    // 6
      << setw(25) << "#7:totsurfd_sx-_intnd"   // 7
      << setw(25) << "#8:_totsurfd_dx-intnd"   // 8
      << setw(25) << "#9:elec_neutr_sx"        // 9
      << setw(25) << "#10:elec_neutr_dx"       // 10
      << setw(25) << "#11:iter_counter"        // 11
      << setw(25) << "#12:avgdriftvel_scalar"  // 12
      << setw(25) << "#13:avgdriftvel_max"     // 13
      << endl;
  str.close();
  ofstream ostr;
  ostr.open("SRJ.txt", ios_base::app);
  ostr << "#"
       << setw(15) << "_step"
       << setw(16) << "iter_counter"
       << setw(16) << "iter"
       << setw(16) << "residual"
       << endl;
  ostr.close();

  /************************************
   *      WARNING FOR DEBUG MODES     *
   ************************************/
#ifdef _ITER_DEBUG
  cerr << " !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! " << endl
       << " ! WARNING : _ITER_DEBUG is active                 ! " << endl
       << " ! Are you sure you wish to execute in debug mode? ! " << endl
       << " !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! " << endl;
#endif

  /************************************
   *        MEMORY ALLOCATION         *
   ************************************/
  allocate_memory();

  /************************************
   *          CONFIGURATION           *
   ************************************/
  config(devId);

  /*****************************************************
   *        INITIALIZATION                             *
   *****************************************************/
  compute_initcond();

  cerr << endl
       << " ************************************************************************************* " << endl
       << " *                                  TIME INTEGRATION                                 * " << endl
       << " ************************************************************************************* " << endl << endl;
}



/**
   PURPOSE:        This function frees all the memory at the end of the execution.

   FILE:           cuda_mosfetproblem.cu

   NAME:           MOSFETProblem::~MOSFETProblem (desstructor)

   PARAMETERS:

   RETURN VALUE:   none (desstructor)

   CALLS TO:       

   CALLED FROM:    

   DATA MODIFIED:  

   METHOD:         

   LAST UPDATE:    2022 March 28
*/
MOSFETProblemCuda::~MOSFETProblemCuda()
{
  enterfunc;

  // host memory
  free(___A);                        // 1
  free(___A_const);                  // 2
  delete [] _avgdriftvel;            // 3
  delete [] _a1;                     // 4
  delete [] _a3;                     // 4.1
  delete [] _a3_const;                   
  delete []_bc_confined;             // 5
  cudaFreeHost( _chi );              // 6
  delete [] _cosphi;                 // 7
  delete [] _cscatt;                 // 8
  delete [] _currdens;               // 9
  delete [] _Deltapot_upper;         // 10
  delete [] _Deltapot_lower;         // 11
  delete [] _denom_SR;               // 12
  cudaFreeHost( _deps_dx );          // 13
  delete [] _effmass;                // 14
  cudaFreeHost( _eps );              // 15
  delete [] _eps_diff_m1;
  delete [] _epskin;                 // 16
  delete [] _epsr_ext;               // 17
  delete [] _integrateddenom_SR;     // 18
  delete [] _integrated_pdf_energy;  // 18.1
  delete [] _integrated_pdf_nu_p_i_m;  // 18.2
  delete [] _ischannel;              // 19
  delete [] _isdrain;                // 20
  delete [] _issource;               // 21
  delete [] _isdrain_ext;            // 22
  delete [] _isoxide_ext;            // 23
  delete [] _issource_ext;           // 24
  delete [] _I_SR;                   // 25
  delete [] _frechet;                // 26
  delete [] _matrix_2dconst;         // 27
  delete [] _matrix_2dconst_ext_inv;         
  delete [] _matrix_2dconst_ext;     // 28
  delete [] _maxw;                   // 29
  delete [] _nd;                     // 30
  delete [] _nd_ext;                 // 31
  delete [] _occupations;            // 32
  delete [] _omega;                  // 33
  delete [] _pdf;                    // 34
  delete [] _pot;                    // 35
  delete [] _pot_b;                  // 36
  delete [] _pot_b_ext;              // 37
  delete [] _pot_ext;                // 38
  delete [] _pot_OLD;                // 39
  delete [] _result_ext;             // 40
  delete [] _rhs_const;              // 41
  delete [] _rhs_pdf;                // 41.1
  delete [] _test_gain;              // 41.1.1
  delete [] _test_loss;              // 41.1.2
  delete [] _rhs_pdf_gain;           // 41.2
  delete [] _righthandside_ext;      // 42
  delete [] _sigma;               
  delete [] _Sigma;               
  delete [] _sinphi;                 // 43
  cudaFreeHost( _surfdens );         // 44
  delete [] _surfdens_eq;            // 45
  delete [] _totsurfdens_eq;         // 45.1
  delete [] _totcurrdens;            // 46
  delete [] _totsurfdens;            // 47
  delete [] _totvoldens;             // 48
  delete [] _totvoldens_ext;         // 49
  delete [] _totvoldens_OLD;         // 50
  delete [] _vconf;                  // 51
  delete [] _vgate;                  // 52
  delete [] _voldens;                // 53
  delete [] _Wm1;                    // 54

#ifdef __CUDACC__
  // device memory
  checkCudaErrors( cudaFree(_GPU___A_const) ); 
  checkCudaErrors( cudaFree(_GPU_avgdriftvel) );            // 1
  checkCudaErrors( cudaFree(_GPU_a1) );                     // 2
  checkCudaErrors( cudaFree(_GPU_a2) );                     // 3
  checkCudaErrors( cudaFree(_GPU_a3) );                     // 4
  checkCudaErrors( cudaFree(_GPU_a3_const) );    
  checkCudaErrors( cudaFree(_GPU_chi) );                    // 5
  checkCudaErrors( cudaFree(_GPU_currdens) );               // 6
  checkCudaErrors( cudaFree(d_A) );
  checkCudaErrors( cudaFree(_GPU_Deltapot_lower) );
  checkCudaErrors( cudaFree(_GPU_Deltapot_upper) );
  checkCudaErrors( cudaFree(_GPU_denom_SR) );               // 7
  checkCudaErrors( cudaFree(_GPU_deps_dx) );                // 8
  checkCudaErrors( cudaFree(_GPU_eps) );                    // 9
  checkCudaErrors( cudaFree(_GPU_eps_diff_m1) );  
  checkCudaErrors( cudaFree(_GPU_I_SR) );                   // 10
  checkCudaErrors( cudaFree(_GPU_J_to_j) ); 
  checkCudaErrors( cudaFree(_GPU_J_to_jj) ); 
  checkCudaErrors( cudaFree(_GPU_integrateddenom_SR) );     // 11
  checkCudaErrors( cudaFree(_GPU_integrated_pdf_energy) );  // 12
  checkCudaErrors( cudaFree(_GPU_matrix_2d) );              // 12.0
  checkCudaErrors( cudaFree(_GPU_matrix_2dconst) );         // 12.1
  checkCudaErrors( cudaFree(_GPU_matrix_2dconst_ext_inv) );
  checkCudaErrors( cudaFree(_GPU_maxw) );                   // 13
  checkCudaErrors( cudaFree(_GPU_pdf) );                    // 14
  checkCudaErrors( cudaFree(_GPU_pot) );                    // 14.1
  checkCudaErrors( cudaFree(_GPU_pot_b) );                  // 14.1.1
  checkCudaErrors( cudaFree(_GPU_pot_ext) );           
  checkCudaErrors( cudaFree(_GPU_rhs) );                    // 14.1.2
  checkCudaErrors( cudaFree(_GPU_rhs_ext) );        
  checkCudaErrors( cudaFree(_GPU_rhs_ext_const) );        
  checkCudaErrors( cudaFree(_GPU_rhs_pdf) );                // 15
  checkCudaErrors( cudaFree(_GPU_test_gain) );              // 15.1
  checkCudaErrors( cudaFree(_GPU_test_loss) );              // 15.2
  checkCudaErrors( cudaFree(_GPU_rhs_const) );              // 15.3
  checkCudaErrors( cudaFree(_GPU_rhs_pdf_gain) );           // 16
  checkCudaErrors( cudaFree(_GPU_sigma) );             
  checkCudaErrors( cudaFree(_GPU_Sigma) );             
  checkCudaErrors( cudaFree(_GPU_surfdens) );               // 17
  checkCudaErrors( cudaFree(_GPU_surfdens_eq) );            // 18
  checkCudaErrors( cudaFree(_GPU_totcurrdens) );            // 19
  checkCudaErrors( cudaFree(_GPU_totsurfdens) );            // 20
  checkCudaErrors( cudaFree(_GPU_totvoldens) );             // 21
  checkCudaErrors( cudaFree(_GPU_totvoldens_ext) );      
  checkCudaErrors( cudaFree(_GPU_totvoldens_OLD) );         // 21.1
  checkCudaErrors( cudaFree(_GPU_voldens) );                // 22 
  checkCudaErrors( cudaFree(_GPU_Wm1) );                    // 23
#endif

  exitfunc;
}



/**
   PURPOSE:        This function allocates the memory for all the magnitudes involved in the simulation
                   both for the CPU and for the GPU.

   FILE:           cuda_mosfetproblem.cu

   NAME:           MOSFETProblem::allocate_memory

   PARAMETERS:

   RETURN VALUE:   none

   CALLS TO:       MOSFETProblem::show_eta              (cuda_testing.h --- declared inline)

   CALLED FROM:    MOSFETProblem::cuda_config           (cuda_config.cu)

   DATA MODIFIED:  

   METHOD:         

   LAST UPDATE:    2022 March 28
*/
void MOSFETProblemCuda::allocate_memory()
{
  enterfunc;

  const int NSBN  = host_dm -> get_NSBN();          
  const int NX    = host_dm -> get_X()   -> get_N();
  const int NZ    = host_dm -> get_Z()   -> get_N();
  const int NW    = host_dm -> get_W()   -> get_N();
  const int NPHI  = host_dm -> get_PHI() -> get_N();
  const int NZEXT = host_dm -> get_ZEXT()-> get_N();

  /*****************************************
   *            host memory                *
   *****************************************/
  ___A                   = (double*) calloc(_NVALLEYS*NX*_SCHROED_ROW_SIZE,sizeof(double));                             // 1
  ___A_const             = (double*) calloc(_NVALLEYS*NX*_SCHROED_ROW_SIZE,sizeof(double));                             // 2
  _avgdriftvel           = new double[NX];                                                                              // 3
  _a1                    = new double[_NVALLEYS*NW*NPHI ];                                                             // 4
  _a3                    = new double[_NVALLEYS*NSBN*NX*NW*NPHI ];                                                   // 4.1
  _a3_const              = new double[_NVALLEYS*NW*NPHI ];                                        
  _bc_confined           = new BoundCond[ 4*NX*NZ ];                                                                   // 5
  cudaHostAlloc( (void**)&_chi, _NVALLEYS*NSBN*NX*_SCHROED_MATRIX_SIZE_PAD*sizeof(double), cudaHostAllocDefault );     // 6
  _cosphi                = new double[ NPHI ];                                                                          // 7
  _cscatt                = new double[8];                                                                                // 8
  _currdens              = new double[ _NVALLEYS*NSBN*NX ];                                                            // 9
  _Deltapot_lower        = new double[ NX*NZ ];                                                                        // 10 
  _Deltapot_upper        = new double[ NX*NZ ];                                                                        // 11
  _denom_SR              = new double[ _NVALLEYS*NW*NPHI*NPHI ];                                                      // 12
  cudaHostAlloc( (void**)&_deps_dx, _NVALLEYS*NSBN*NX*sizeof(double), cudaHostAllocDefault );                          // 13
  _effmass               = new double[ 3*_NVALLEYS*NX*NZ ];                                                           // 14
  cudaHostAlloc( (void**)&_eps, _NVALLEYS*NSBN*NX*sizeof(double), cudaHostAllocDefault );                             // 15
  _eps_diff_m1           = new double[_NVALLEYS*NSBN*NSBN*NX];
  _epskin                = new double[_NVALLEYS*NW];                                                                   // 16
  _epsr_ext              = new double[NX*NZEXT];                                                                      // 17
  _integrateddenom_SR    = new double[ _NVALLEYS*NW*NPHI ];                                                           // 18
  _integrated_pdf_energy = new double[ _NVALLEYS*NSBN*NX*NW ];                                                       // 18.1
  _integrated_pdf_nu_p_i_m = new double[ NW ];                                                                         // 18.2
  _ischannel             = new bool[ NX*NZ ];                                                                         // 19
  _isdrain               = new bool[ NX*NZ ];                                                                         // 20
  _issource              = new bool[ NX*NZ ];                                                                         // 21
  _isdrain_ext           = new bool[NX*NZEXT];                                                                        // 22
  _isoxide_ext           = new bool[NX*NZEXT];                                                                        // 23
  _issource_ext          = new bool[NX*NZEXT];                                                                        // 24
  _I_SR                  = new double[ _NVALLEYS*NSBN*NX ];                                                           // 25
  _frechet               = new double[NX*NZ*NZ];                                                                     // 26
  _matrix_2d             = new double[NX*NZ*(2*NZ+1)];                                                               //
  _matrix_2dconst        = new double [ 3*NX*NZ*(2*NZ+1) ];                                                          // 27
  _matrix_2dconst_ext    = new double [ NX*NZEXT*(2*NZEXT+1) ];                                                      // 28
  _matrix_2dconst_ext_inv= new double [ NX*NZEXT*NX*NZEXT ];                                                   
  _maxw                  = new double[_NVALLEYS*NW];                                                                   // 29
  _nd                    = new double[ NX*NZ ];                                                                   // 30
  _nd_ext                = new double[NX*NZEXT];                                                                  // 31
  _occupations           = new double[8*_NVALLEYS*_NVALLEYS];                                                       // 32
  _omega                 = new double[8];                                                                           // 33
  _pdf                   = new double[_NVALLEYS*NSBN*NX*NW*NPHI*4];                                             // 34
  _pot                   = new double[NX*NZ];                                                                     // 35
  _pot_b                 = new double[NZ];                                                                         // 36
  _pot_b_ext             = new double[NZEXT];                                                                      // 37
  _pot_ext               = new double[ NX*NZEXT ];                                                                // 38
  _pot_OLD               = new double[NX*NZ];                                                                     // 39
  _result_ext            = new double [NX*NZEXT];                                                                 // 40
  _rhs                   = new double[NX*NZ];
  _rhs_const             = new double[ NX*NZ ];                                                                   // 41
  _rhs_pdf               = new double[ _NVALLEYS*NSBN*NX*NW*NPHI ];                                             // 41.1
  _test_gain             = new double[ _NVALLEYS*NSBN*NX*NW*NPHI ];                                             // 41.1.1
  _test_loss             = new double[ _NVALLEYS*NSBN*NX*NW*NPHI ];                                             // 41.1.2
  _rhs_pdf_gain          = new double[ _NVALLEYS*NSBN*NX*NW ];                                                   // 41.2
  _righthandside_ext     = new double [NX*NZEXT];                                                                 // 42
  _sigma                 = new double[ NZEXT ];                                                                    
  _Sigma                 = new double[ NZEXT ];                                                                    
  _sinphi                = new double[ NPHI ];                                                                     // 43
  cudaHostAlloc( (void**)&_surfdens, NX*_NVALLEYS*NSBN*sizeof(double), cudaHostAllocDefault );                    // 44
  _surfdens_eq           = new double[ _NVALLEYS*NSBN*NX ];                                                       // 45
  _totsurfdens_eq        = new double[ NX ];                                                                       // 45.1
  _totsurfdens           = new double[ NX ];                                                                       // 46
  _totcurrdens           = new double[ NX ];                                                                       // 47
  _totvoldens            = new double[ NX*NZ ];                                                                   // 48
  _totvoldens_ext        = new double[ NX*NZEXT ];                                                                // 49
  _totvoldens_OLD        = new double[ NX*NZ ];                                                                   // 50
  _vconf                 = new double[ NX*NZ ];                                                                   // 51
  _vel                   = new double[ 2*_NVALLEYS*NW*NPHI ];
  _vgate                 = new double[ NX*NZ ];                                                                   // 52
  _voldens               = new double[ _NVALLEYS*NSBN*NX*NZ ];                                                   // 53
  _Wm1                   = new double[ _NVALLEYS*NSBN*_NVALLEYS*NSBN*NX ];                                       // 54

  for(int i=0; i<NX; ++i)
    for(int nu=0; nu<_NVALLEYS; ++nu)
      for(int p=0; p<NSBN; ++p)
	for(int j=0; j<NZ; ++j)
	  voldens(nu,p,i,j) = 0;
  for(int i=0; i<NX; ++i)
    for(int j=0; j<NZ; ++j)
      totvoldens(i,j) = 0;
  for(int i=0; i<NX; ++i)
    for(int j=0; j<NZ; ++j)
      totvoldens_OLD(i,j) = 0;
  for(int i=0; i<NX; ++i)
    for(int j=0; j<NZ; ++j)
      for(int jj=0; jj<NZ; ++jj)
	frechet(i,j,jj) = 0;

#ifdef __CUDACC__
  /*****************************************
   *            device memory              *
   *****************************************/
  checkCudaErrors( cudaMalloc((void **)&_GPU___A_const,             _NVALLEYS*NX*_SCHROED_ROW_SIZE*sizeof(double)) );                                
  checkCudaErrors( cudaMalloc((void **)&_GPU_avgdriftvel,           NX*sizeof(double)) );                                 // 1
  checkCudaErrors( cudaMalloc((void **)&_GPU_a1,                    _NVALLEYS*NW*NPHI*sizeof(double)) );                 // 2
  checkCudaErrors( cudaMalloc((void **)&_GPU_a2,                    _NVALLEYS*NSBN*NX*NW*NPHI*sizeof(double)) );       // 3   
  checkCudaErrors( cudaMalloc((void **)&_GPU_a3,                    _NVALLEYS*NSBN*NX*NW*NPHI*sizeof(double)) );       // 4
  checkCudaErrors( cudaMalloc((void **)&_GPU_a3_const,              _NVALLEYS*NW*NPHI*sizeof(double)) );  
  checkCudaErrors( cudaMalloc((void **)&_GPU_chi,                   _NVALLEYS*NSBN*NX*_SCHROED_MATRIX_SIZE_PAD*sizeof(double)) ); // 5
  checkCudaErrors( cudaMalloc((void **)&_GPU_currdens,              _NVALLEYS*NSBN*NX*sizeof(double)) );                 // 6
  checkCudaErrors( cudaMalloc((void **)&d_A,                        _NVALLEYS*NX*_SCHROED_ROW_SIZE*sizeof(double)) ); 
  checkCudaErrors( cudaMalloc((void **)&_GPU_Deltapot_lower,        NX*NZ*sizeof(double)) );          
  checkCudaErrors( cudaMalloc((void **)&_GPU_Deltapot_upper,        NX*NZ*sizeof(double)) );          
  checkCudaErrors( cudaMalloc((void **)&_GPU_denom_SR,              _NVALLEYS*NW*NPHI*NPHI*sizeof(double)) );           // 7 
  checkCudaErrors( cudaMalloc((void **)&_GPU_deps_dx,               _NVALLEYS*NSBN*NX*sizeof(double)) );                 // 8 
  checkCudaErrors( cudaMalloc((void **)&_GPU_eps,                   _NVALLEYS*NSBN*NX*sizeof(double)) );                 // 9
  checkCudaErrors( cudaMalloc((void **)&_GPU_eps_diff_m1,           _NVALLEYS*NSBN*NSBN*NX*sizeof(double)) );       
  checkCudaErrors( cudaMalloc((void **)&_GPU_integrateddenom_SR,    _NVALLEYS*NW*NPHI*sizeof(double)) );                 // 11
  checkCudaErrors( cudaMalloc((void **)&_GPU_integrated_pdf_energy, _NVALLEYS*NSBN*NX*NW*sizeof(double)) );             // 12
  checkCudaErrors( cudaMalloc((void **)&_GPU_I_SR,                  _NVALLEYS*NSBN*NX*sizeof(double)) );                 // 10
  checkCudaErrors( cudaMalloc((void **)&_GPU_J_to_j,                (NZ-2)*(NZ-1)/2*sizeof(int)) ); 
  checkCudaErrors( cudaMalloc((void **)&_GPU_J_to_jj,               (NZ-2)*(NZ-1)/2*sizeof(int)) ); 
  checkCudaErrors( cudaMalloc((void **)&_GPU_matrix_2d,             NX*NZ*(2*NZ+1)*sizeof(double)) );                   // 12.0
  checkCudaErrors( cudaMalloc((void **)&_GPU_matrix_2dconst,        _NVALLEYS*NX*NZ*(2*NZ+1)*sizeof(double)) );         // 12.1 // TASK: _NVALLEYS ?? En serio?
  checkCudaErrors( cudaMalloc((void **)&_GPU_matrix_2dconst_ext_inv,NX*NZEXT*NX*NZEXT*sizeof(double)) ); 
  checkCudaErrors( cudaMalloc((void **)&_GPU_maxw,                  _NVALLEYS*NW*sizeof(double)) );                       // 13
  checkCudaErrors( cudaMalloc((void **)&_GPU_nd,                    NX*NZ*sizeof(double)) );                             // 13.1
  checkCudaErrors( cudaMalloc((void **)&_GPU_pdf,                   _NVALLEYS*NSBN*NX*NW*NPHI*4*sizeof(double)) );     // 14
  checkCudaErrors( cudaMalloc((void **)&_GPU_pot,                   NX*NZ*sizeof(double)) );                             // 14.1
  checkCudaErrors( cudaMalloc((void **)&_GPU_pot_b,                 NZ*sizeof(double)) );                                 // 14.1.1
  checkCudaErrors( cudaMalloc((void **)&_GPU_pot_ext,               NX*NZEXT*sizeof(double)) );           
  checkCudaErrors( cudaMalloc((void **)&_GPU_pot_OLD,               NX*NZ*sizeof(double)) );                             // 14.2
  checkCudaErrors( cudaMalloc((void **)&_GPU_rhs,                   NX*NZ*sizeof(double)) );                             // 14.2.1
  checkCudaErrors( cudaMalloc((void **)&_GPU_rhs_pdf,               _NVALLEYS*NSBN*NX*NW*NPHI*sizeof(double)) );       // 15
  checkCudaErrors( cudaMalloc((void **)&_GPU_test_gain,             _NVALLEYS*NSBN*NX*NW*NPHI*sizeof(double)) );       // 15.1
  checkCudaErrors( cudaMalloc((void **)&_GPU_test_loss,             _NVALLEYS*NSBN*NX*NW*NPHI*sizeof(double)) );       // 15.2
  checkCudaErrors( cudaMalloc((void **)&_GPU_rhs_const,             NX*NZ*sizeof(double)) );                             // 15.3
  checkCudaErrors( cudaMalloc((void **)&_GPU_rhs_ext,               NX*NZEXT*sizeof(double)) );           
  checkCudaErrors( cudaMalloc((void **)&_GPU_rhs_ext_const,         NX*NZEXT*sizeof(double)) );           
  checkCudaErrors( cudaMalloc((void **)&_GPU_rhs_pdf_gain,          _NVALLEYS*NSBN*NX*NW*sizeof(double)) );             // 16
  checkCudaErrors( cudaMalloc((void **)&_GPU_sigma,                 NZEXT*sizeof(double)) );
  checkCudaErrors( cudaMalloc((void **)&_GPU_Sigma,                 NZEXT*sizeof(double)) );
  checkCudaErrors( cudaMalloc((void **)&_GPU_surfdens,              _NVALLEYS*NSBN*NX*sizeof(double)) );                 // 17
  checkCudaErrors( cudaMalloc((void **)&_GPU_surfdens_eq,           _NVALLEYS*NSBN*NX*sizeof(double)) );                 // 18
  checkCudaErrors( cudaMalloc((void **)&_GPU_totcurrdens,           NX*sizeof(double)) );                                 // 19
  checkCudaErrors( cudaMalloc((void **)&_GPU_totsurfdens,           (NX+1)*sizeof(double)) );                             // 20
  checkCudaErrors( cudaMalloc((void **)&_GPU_totvoldens,            NX*NZ*sizeof(double)) );                             // 21
  checkCudaErrors( cudaMalloc((void **)&_GPU_totvoldens_ext,        NX*NZEXT*sizeof(double)) );           
  checkCudaErrors( cudaMalloc((void **)&_GPU_totvoldens_OLD,        NX*NZ*sizeof(double)) );                             // 21.1
  checkCudaErrors( cudaMalloc((void **)&_GPU_vel,                   2*_NVALLEYS*NW*NPHI*sizeof(double)) );           
  checkCudaErrors( cudaMalloc((void **)&_GPU_voldens,               _NVALLEYS*NSBN*NX*NZ*sizeof(double)) );             // 22
  checkCudaErrors( cudaMalloc((void **)&_GPU_Wm1,                   _NVALLEYS*NSBN*_NVALLEYS*NSBN*NX*sizeof(double)) ); // 23
  checkCudaErrors( cudaMalloc((void **)&_GPU_frechet,               NX*NZ*NZ*sizeof(double)) );                         // 26
#endif

  exitfunc;
  
  return;
}



/**
   PURPOSE:        

   FILE:           cuda_mosfetproblem.cu

   NAME:           MOSFETProblem::compute_initcond

   PARAMETERS:

   RETURN VALUE:   none

   CALLS TO:       MOSFETProblem::initcond_on_cuda    (cuda_iter.cu)
                   MOSFETProblem::thermequil_on_cuda  (cuda_iter.cu)
		   MOSFETProblem::config_roughness    (cuda_bte_scatterings_roughness.cu)
		   MOSFETProblem::CPU_initialize_pdf  (cuda_config.cu)
		   MOSFETProblem::dens                (cuda_dens.cu)
		   MOSFETProblem::macro               (cuda_reductions.cu)

   CALLED FROM:    MOSFETProblem::MOSFETProblem       (cuda_mosfetproblem.cu)

   DATA MODIFIED:  

   METHOD:         

   LAST UPDATE:    2022 February 16
*/
void MOSFETProblemCuda::compute_initcond()
{
  const int NSBN  = host_dm -> get_NSBN();
  const int NX    = host_dm -> get_X()   -> get_N();
  const int NW    = host_dm -> get_W()   -> get_N();
  const int NPHI  = host_dm -> get_PHI() -> get_N();

  initcond_on_cuda();
  thermequil_on_cuda();
  config_roughness();

  CPU_initialize_pdf();
#ifdef __CUDACC__
  checkCudaErrors( cudaMemcpy( _GPU_pdf, _pdf, _NVALLEYS*NSBN*NX*NW*NPHI*sizeof(double), cudaMemcpyHostToDevice) );
#endif

  dens(0);
  macro( 0 );

  return;
}




