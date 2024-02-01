/*****************************************************************
 *                     TOPMOST LEVEL FUNCTION                    *
 *                     ======================                    *
 *                                                               *
 * Read below, or execute with the --help flag for more details. *
 *****************************************************************/

/**
  Include all the structures and methods to perform
  the simulations. A class of type MOSFETProblem
  will be created, which contains a solve function, called
  by the main after constructing the problem.
*/
#include "mosfetproblem.h" 

#ifdef __CUDACC__
/**
  This kernel does not perform any specific
  computation. Its only goal is to "warm-up"
  the graphic card before starting real computations.
*/
__global__ void cuda_warmup()
{
  double A[10];

  for(int i=0; i<10; ++i)
    A[i] = (double)i+1;

  for(int i=0; i<10; ++i)
    ++A[i];
}
#endif

/**
  The Library for Iterative Solvers (LIS) is used
  in the computation of the initial condition.
  Namely, it is used for the computation of eigenvalues,
  eigenvectors and the solution of linear systems.
  It is also used in the solution of the Poisson
  equation for the integration of the roughness scattering
  mechanism.
  TASK: replace all of this with Cuda implementations.
*/
int main( int argc, char **argv )
{
  lis_initialize(&argc, &argv);

  /* 
     Here follows a list of parameters that can be set by the user
     when executing the solver. Their meanings
     are detailed by the --help flag and can be read here below.
  */
  int devId = 1;
  int NUMTHREADS = 16;
  string comment = "no_comment";
  
  /*
    Read the optional arguments passed at execution time.
  */
  for( int a=1; a<argc; ++a )
    {
      if( strcmp( argv[a], "--help" ) == 0 )
	{
	  cerr << endl;
	  cerr << "The executable file 'detmosfet' performs a numerical simulation of a Double-Gate Metal Oxide Semiconductor Field-Effect Transistor (DG MOSFET)." << endl;
	  cerr << "The model is 2D: the x dimension is the unconfined dimension, the z dimension is the confined one." << endl;
	  cerr << "It implements a deterministic solver, based on Botlzmann Transport Equations (BTEs) for the transport" << endl;
	  cerr << "of electrons along the x dimension, and the Schroedinger-Poisson block for the computations" << endl;
	  cerr << "of the electrostatic field and the eigenproperties due to the confinement." << endl;
	  cerr << "" << endl;
	  cerr << "INPUT" << endl;
	  cerr << "The physical description of the device and the applied voltages are set inside file 'cuda_physdevice.h'." << endl;
	  cerr << "The parameters for the solver are set inside file 'solverparams.h'." << endl;
	  cerr << "Other parameters, described here below in the 'usage' section, are passed inline." << endl;
	  cerr << endl;
	  cerr << "OUTPUT" << endl;
	  cerr << "The code creates several outputs:" << endl;
	  cerr << "- File 'times.DAT'                       containing the computation cost of all the phases listed in the first line of the code," << endl;
	  cerr << "                                         provided that the #define _COMPTIMES has not been commented, in file 'src/cuda_comptimes.h'." << endl;
	  cerr << "                                         When used for real-life simulations, that flag should be undefined, because time measurements are costly." << endl;
	  cerr << "- Folder 'results/'                      containing all the raw results in '.dat' text file format. Namely it contains, in alphabetical order," << endl;
	  cerr << "                                         (i) currdens.12345678.dat     representing the total current density (column 2) against position (column 1) at step 12345678;" << endl;
	  cerr << "                                                                       the other columns (3-) represent the current density inside each valley and energy level" << endl;
	  cerr << "                                         (ii) deps_dx.12345678.dat     representing the force field (the x-derivative of the energy levels eps)" << endl;
	  cerr << "                                                                       for each valley and energy level (columns 2-), against position (column 1), at step 12345678" << endl;
	  cerr << "                                         (iii) driftvel.12345678.dat   representing the average drift velocity of the electrons (column 2)" << endl;
	  cerr << "                                                                       against position (column 1), at step 12345678" << endl;
	  cerr << "                                         (iv) eps.12345678.dat         representing the energy levels for each valley and energy level (columns 2-), against position (column 1), at step 12345678" << endl;
	  cerr << "                                         (v) potential.12345678.dat    representing the (x,z)-potential energy (column 3), against x-position (column 1) and z-position (column 2), at step 12345678;" << endl;
	  cerr << "                                                                       when the roughness scattering mechanism is added, columns 4 and 5 represent the potential difference and the upper and lowe gate respectively" << endl;
	  cerr << "                                         (vi) scalars.dat              containing all the scalar quantities; one line is added to this file at each time step;" << endl;
	  cerr << "                                                                       the meaning of the different columns is specified in the first line of the file" << endl;
	  cerr << "                                         (vi) currdens.12345678.dat    representing the total surface density (column 2) against position (column 1) at step 12345678;" << endl;
	  cerr << "                                                                       the other columns (3-) represent the surface density inside each valley and energy level" << endl;
	  cerr << "                                         (vii) test.12345678.dat       representing the (nu, p, x, phi)-integrated distribution function (column 2) against energy w (column 1) at step 12345678;" << endl;
	  cerr << "                                                                       it is meant as a check that the w-domain has been chosen large enough for the electrons" << endl;
	  cerr << "                                                                       not to escape the w-border; what we want to see is that the values are 'very small' when w is large" << endl;
	  cerr << "                                         (viii) voldens.12345678.dat   representing the total volume density (column 3), against x-position (column 1) and z-position (column 2), at step 12345678;" << endl;
	  cerr << "                                                                       in the following columns (4-) the (valley, energy level) volume density is stored" << endl;
	  cerr << "- Folder 'scripts/'                      containing several gnuplot scripts, that generate either films (encoded via mencoder) or pdf files. (TASK: complete description.)" << endl;
	  cerr << "                                         film_* and graph_* scripts generate an output inside folder 'films_and_graphs/'" << endl;
	  cerr << "                                         histogram_* and speedups_* scripts generate an output inside folder 'comptime_analysis/' reading data from file 'times.DAT'" << endl;
	  cerr << "                                         piechart_* scripts generate an output inside folder 'comptime_analysis/'" << endl;
	  cerr << "- File 'generate_films_and_graphs.sh'    which calls all the film_* and graph_* scripts inside folder 'scripts/' (output is in folder 'films_and_graphs/')" << endl;
	  cerr << "- File 'generate_comptime_analysis.sh'   which calls all the histogram_*, piechart_* and speedups_* scripts inside folder 'scripts/' (output is in folder 'comptime_analysis/')" << endl;
	  cerr << "" << endl;
	  cerr << "TASK: improve this description." << endl;
	  cerr << "" << endl;
	  cerr << "Usage: detmosfet [OPTION]..." << endl;
	  cerr << "" << endl;
	  cerr << left << setw(32) << "  --GPU CUDA_DEVICE"
	       << setw(4) << " "
	       << "sets CUDA_DEVICE (integer) as the GPU device used to perform the simulations" << endl
	       << setw(36) << " " << "default value is set to 1 for 'historical' reasons" << endl
	       << setw(36) << " " << endl;
	  cerr << left << setw(32) << "  --comment \"text\""
	       << setw(4) << " "
	       << "sets text (string) as the message which will appear" << endl
	       << setw(36) << " " << "in the first column of file 'times.DAT', which is meant to be" << endl
	       << setw(36) << " " << "a comment about the parameters used for the simulation;" << endl
	       << setw(36) << " " << "for example, use --comment \"6x65x65x300x48\"" << endl
	       << setw(36) << " " << "DO NOT USE SPACES IN THE COMMENT, or the number of columns inside 'times.DAT' won't be constant" << endl
	       << setw(36) << " " << "default value is \"no_comment\"" << endl
	       << setw(36) << " " << endl;
	  cerr << left << setw(32) << "  --ompthreads NUMTHREADS"
	       << setw(4) << " "
	       << "sets text NUMTHREADS (integer) as the number of parallel threads used on CPU for OpenMP environments;" << endl
	       << setw(36) << " " << "default value is 16" << endl
	       << setw(36) << " " << endl;
	  cerr << left << setw(32) << "  --gpuinfo id"
	       << setw(4) << " "
	       << "prints on screen, through nvidia-smi, the name of the graphic card corresponding to identification id;" << endl
	       << setw(36) << " " << endl;
	  cerr << left << setw(32) << "  --help"
	       << setw(4) << " "
	       << "to display this message" << endl
	       << setw(36) << " " << endl;
	  cerr << "Exit status:" << endl
	       << " --- TASK: part to be written --- " << endl
	       << "" << endl;
	  cerr << "**********************" << endl;
	  cerr << " HOW TO USE THIS CODE " << endl;
	  cerr << "**********************" << endl;
	  cerr << "1) Open file 'src/cuda_physdevice.h' and set the physical description of the device." << endl;
	  cerr << "2) Open file 'src/cuda_solverparams.h' and set the numerical parameters for the simulation." << endl;
	  cerr << "3) Type 'make' and wait for the code to compile." << endl;
	  cerr << "4) Execute by './detmosfet' and wait for the code to complete the computations." << endl;
	  cerr << "5) Type './generate_films_and_graphs.sh' to produce films and graphs to be found in folder 'films_and_graphs/'." << endl;
	  cerr << "" << endl;
	  exit(0);
	}
      if( strcmp( argv[a], "--GPU" ) == 0 )
	{
	  devId = atoi( argv[a+1] );
	}
      if( strcmp( argv[a], "--comment" ) == 0 )
	{
	  comment = argv[a+1];
	}
      if( strcmp( argv[a], "--ompthreads" ) == 0 )
	{
	  NUMTHREADS = atoi(argv[a+1]);
	}
      if( strcmp( argv[a], "--gpuinfo" ) == 0 )
	{
	  int devID=atoi(argv[a+1]);
	  cudaDeviceProp props;
	  cudaError_t err;
	  err=cudaSetDevice(devID);
	  err=cudaGetDevice(&devID);
	  if (err!=cudaSuccess)
	    {
	      cout<<"ERRORRR"<<endl;
	    }
	  cudaGetDeviceProperties(&props, devID);
	  printf("Device %d: \"%s\" with Compute %d.%d capability\n",
		 devID, props.name, props.major, props.minor);
	  
	  exit(0);
	}
    }

  omp_set_num_threads( NUMTHREADS );   /* Create the parallel environment for OpenMP. */

#ifdef __CUDACC__
  /* 
     If the code is executed on the GPU, choose the device and perform warm-up.
     This is also used to check that the device is available and working.
  */
  cudaSetDevice(devId);          
  cuda_warmup <<< 1, 256 >>> ();
  if( cudaSuccess != cudaGetLastError() )
    {
      cerr << " Exiting at line " << __LINE__ << " in function '" << __FUNCTION__ << "' !" << endl;
      throw error_WARMUP();
    }
#endif

  MOSFETProblemCuda mp_cuda(devId, argc, argv);   /* 
						     Initialize the solver by creating the MOSFETProblem object. 
						     Note: most parameters are introduced by using #define directives
						     instead of being class members. In such a way, they are
						     known at compile-time. By one side, this forces recompiling
						     the code each time a parameter is modified, but at the same time
						     this makes the code more readable.
						  */

  mp_cuda.comment = comment;   /* Set the comment message for 'times.DAT' file. */

  mp_cuda.solve_2();   /* Call the solver. */
  
  lis_finalize();   /*
		      Finalize the LIS environment.
		      TASK: once LIS evicted from the code, this line will be deleted.
		    */
}
