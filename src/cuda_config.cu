#include "mosfetproblem.h"
#include "debug_flags.h"


#ifdef __CUDACC__
#define CUDA_SAFE_CALL( call ) {                                         \
 cudaError_t err = call;                                                 \
 if( cudaSuccess != err ) {                                              \
   fprintf(stderr,"CUDA: error occurred in cuda routine. Exiting...\n"); \
   exit(err);                                                            \
 } }
#endif

/**
   PURPOSE:        This functions sets up all the magnitudes that can be precomputed.
                   In particular: the band structure, the scattering parameters, 
		   all the constants, the data structures, the constant parts of the
		   ITER computational section, the constant parts of the integration of the
		   scattering mechanisms, and the magnitudes which will be used to
		   estimate the computational performances.

   FILE:           cuda_config.cu

   NAME:           MOSFETProblem::config

   PARAMETERS:

   RETURN VALUE:   none

   CALLS TO:       MOSFETProblem::config_bandstruct      (cuda_config.cu)
                   MOSFETProblem::config_scatterings     (cuda_config.cu)
                   MOSFETProblem::config_constdata       (cuda_config.cu)
                   MOSFETProblem::config_datastructs     (cuda_config.cu)
                   MOSFETProblem::config_spsolvers       (cuda_config.cu)
                   MOSFETProblem::config_scintegrate     (cuda_config.cu)
                   MOSFETProblem::config_comptime        (cuda_comptime.cu)

   CALLED FROM:    MOSFETProblem::cuda_mosfetproblem     (cuda_mosfetproblem.cu)

   DATA MODIFIED:  

   METHOD:         

   LAST UPDATE:    2022 March 2022
*/
void MOSFETProblemCuda::config(const int devId)
{
  /* 
     begin section CPUINFO_01
     created: 2023/03/01
     last modified: 2023/03/01
     author: Francesco Vecil

     This section stores into 'string CPU_INFO' the info about the
     CPU that is actually being used.
  */

  system("lscpu | grep 'Model name' > cpu.txt"); /* write into cpu.txt the info */
  // now read the info from the file
  ifstream istr;
  istr.open("cpu.txt", ios_base::in);
  string rl;
  getline(istr, rl);
  rl.erase(0, 33);
  istr.close();
  CPU_INFO = ReplaceAll(rl, string(" "), string("_"));
  
  /* 
     end section CPUINFO_01
  */

  cerr << endl << endl;
  cerr << "----------------------------------------------------------------------------------------------------" << endl;
  cerr << "                                           HARDWARE IN USE" << endl;
  cerr << "----------------------------------------------------------------------------------------------------" << endl;

  cerr << "the CPU being used for the current execution is the following one:" << endl << endl
       << CPU_INFO
       << endl;

  
#ifdef __CUDACC__
  /* 
     begin section GPUINFO_01
     created: 2023/03/01
     last modified: 2023/03/01
     author: Francesco Vecil

     This section stores into 'cudaDeviceProp props' the info about the
     GPU that is actually being used.
     Then, it sends the name of the GPU and its compute capability to stderr.
  */
  CUDA_DEVICE = devId;
  cudaError_t err;
  err = cudaGetDevice(&CUDA_DEVICE);
  if (err!=cudaSuccess)
    {
      cerr << "From file '" << __FILE__ << "', from function '" << __func__ <<  "', error message from line " << __LINE__ << endl;
      exit(972);
    }
  cudaGetDeviceProperties(&props, CUDA_DEVICE);
  cerr << endl
       << "the GPU being used for the current execution is the following one:" << endl << endl
       << "Device " << CUDA_DEVICE << ": \"" << props.name << "\" with compute capability " << props.major << "." << props.minor << ""
       << endl;
  /* 
     end section GPUINFO_01
  */

  cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
  cudaDeviceSetSharedMemConfig( cudaSharedMemBankSizeEightByte );
#endif
  cerr << "----------------------------------------------------------------------------------------------------" << endl;
  cerr << endl;
  
  config_bandstruct();
  config_scatterings();
  config_constdata();
  config_datastructs();
  config_spsolvers();
  config_scintegrate();

  for(int i=0; i<100; ++i)
  {
    _ct[i] = 0; // ANALYSIS OF THE COMPUTATIONAL TIMES
    // _ct_best[i] = 1.e20; // 
    _nb[i] = 0; // ANALYSIS OF THE COMPUTATIONAL TIMES
  }
  _jacobi_iters = 0;
  _ETA = 0;
  seconds = 0;
  minutes = 0;
  hours = 0;
  config_comptime();
  
  return;
}



/**
   PURPOSE:        This function sets all the constants concerning the band structure,
                   namely the Kane distorsion factors, the effective masses, 
		   the silicon-oxide effective mass, and some precomputable square roots
		   involving effective masses.

   FILE:           cuda_config.cu

   NAME:           MOSFETProblem::config_bandstruct

   PARAMETERS:

   RETURN VALUE:   none

   CALLS TO:       MOSFETProblem::show_eta            (cuda_testing.h --- declared inline)

   CALLED FROM:    MOSFETProblem::cuda_config         (cuda_config.cu)

   DATA MODIFIED:  

   METHOD:         

   LAST UPDATE:    2022 March 28
*/
void MOSFETProblemCuda::config_bandstruct()
{
  enterfunc;

  // DEFAULT VALUES
  _kane[0]  = 0.5/host_phyco->__eV;
  mass(Xdim,0) = 0.98;
  mass(Ydim,0) = 0.19;
  mass(Zdim,0) = 0.19;

  _kane[1]  = 0.5/host_phyco->__eV;
  mass(Xdim,1) = 0.19;
  mass(Ydim,1) = 0.98;
  mass(Zdim,1) = 0.19;

  _kane[2]  = 0.5/host_phyco->__eV;
  mass(Xdim,2) = 0.19;
  mass(Ydim,2) = 0.19;
  mass(Zdim,2) = 0.98;

  _oxidemass = 0.50;

  sqrtmass(Xdim,0) = sqrt(mass(Xdim,0));
  sqrtmass(Xdim,1) = sqrt(mass(Xdim,1));
  sqrtmass(Xdim,2) = sqrt(mass(Xdim,2));
  sqrtmass(Ydim,0) = sqrt(mass(Ydim,0));
  sqrtmass(Ydim,1) = sqrt(mass(Ydim,1));
  sqrtmass(Ydim,2) = sqrt(mass(Ydim,2));
  sqrtmass(Zdim,0) = sqrt(mass(Zdim,0));
  sqrtmass(Zdim,1) = sqrt(mass(Zdim,1));
  sqrtmass(Zdim,2) = sqrt(mass(Zdim,2));

  sqrtmassXY(0) = sqrt( mass(Xdim,0) * mass(Ydim,0) );
  sqrtmassXY(1) = sqrt( mass(Xdim,1) * mass(Ydim,1) );
  sqrtmassXY(2) = sqrt( mass(Xdim,1) * mass(Ydim,2) );

  exitfunc;
  
  return;
}



/**
   PURPOSE:        This function sets all the physical constants involving the scattering
                   mechanisms, namely: the kind of scattering mechanism, the phonons
		   temperature, the deformation potential.

   FILE:           cuda_config.cu

   NAME:           MOSFETProblem::config_scatterings

   PARAMETERS:

   RETURN VALUE:   none

   CALLS TO:       MOSFETProblem::show_eta            (cuda_testing.h --- declared inline)

   CALLED FROM:    MOSFETProblem::cuda_config         (cuda_config.cu)

   DATA MODIFIED:  

   METHOD:         

   LAST UPDATE:    2022 March 28
*/
void MOSFETProblemCuda::config_scatterings()
{
  enterfunc;

  scatttype(0)  = PHONON_ELASTIC;
  phtemp(0)     = 0;
  defpot(0)     = 9.0;

  scatttype(1)  = PHONON_GTYPE;
  phtemp(1)     = 140;
  defpot(1)     = 0.5;

  scatttype(2)  = PHONON_GTYPE;
  phtemp(2)     = 215;
  defpot(2)     = 0.8;

  scatttype(3)  = PHONON_GTYPE;
  phtemp(3)     = 720;
  defpot(3)     = 11.0;

  scatttype(4)  = PHONON_FTYPE;
  phtemp(4)     = 220;
  defpot(4)     = 0.3;

  scatttype(5)  = PHONON_FTYPE;
  phtemp(5)     = 500;
  defpot(5)     = 2.0;

  scatttype(6)  = PHONON_FTYPE;
  phtemp(6)     = 685;
  defpot(6)     = 2.0;

  scatttype(7)  = ROUGHNESS;

  exitfunc;
  
  return;
}



/**
   PURPOSE:        This function sets many constants, related to the numerical parameters,
                   the geometrical description, Maxwellian, fluxes, confining potential,
		   potential at gates and boundary conditions. Moreover, it copies
		   to GPU, in case, the quantities involved.

   FILE:           cuda_config.cu

   NAME:           MOSFETProblem::config_constdata

   PARAMETERS:

   RETURN VALUE:   none

   CALLS TO:       MOSFETProblem::show_eta            (cuda_testing.h --- declared inline)

   CALLED FROM:    MOSFETProblem::cuda_config         (cuda_config.cu)

   DATA MODIFIED:  

   METHOD:         

   LAST UPDATE:    2022 February 17
*/
void MOSFETProblemCuda::config_constdata()
{
  enterfunc;

  const int NSBN  = host_dm -> get_NSBN();
  const int NX    = host_dm -> get_X()   -> get_N();
  const int NZ    = host_dm -> get_Z()   -> get_N();
  const int NW    = host_dm -> get_W()   -> get_N();
  const int NPHI  = host_dm -> get_PHI() -> get_N();
  const double DZ = host_dm -> get_Z()   -> get_delta();
  
  const double LSOURCE  = host_physdesc -> get_LSOURCE();
  const double LCHANNEL = host_physdesc -> get_LCHANNEL();
  const double LDRAIN   = host_physdesc -> get_LDRAIN();
  const double XLENGTH = host_physdesc -> compute_XLENGTH();
  const double ZWIDTH  = host_physdesc -> get_ZWIDTH();
  const double NDHIGH = host_physdesc -> get_NDHIGH();
  const double NDLOW = host_physdesc -> get_NDLOW();
  const double VCONF_WELL = host_physdesc -> get_VCONF_WELL();
  const double VUPPERGATE = host_physdesc -> get_VUPPERGATE();
  const double VLOWERGATE = host_physdesc -> get_VLOWERGATE();

  const double DPHI = host_dm->get_PHI()->get_delta();
  
  // adimensionalization parameters
  // _adim_cp = ((poscharge*host_rescpar->get_nstar()*ZWIDTH*ZWIDTH)/(host_rescpar->get_potstar()*eps0));

  if( NPHI%2 == 1 )
    {
      cerr<<" ERROR: nphi must be even!"<<endl;
      throw error_CONFIG_NPHI_ODD();
    }
  
  _dt = 1.E-18/host_rescpar->get_tstar(); 

  // GEOMETRICAL DESCFIPTION
  for( int i=0; i<NX; ++i )
    for( int j=0; j<NZ; ++j )
      {
	issource( i,j )  = false;
	ischannel( i,j ) = false;
	isdrain( i,j )   = false;
      }
  
  // SOURCE CONTACT
  for( int i=0; i<NX; ++i )
    for( int j=0; j<NZ; ++j )
      {
	if( isoxide( i,j ) == false )
	  {
	    const double xi = host_dm->get_X()->mesh(i);
	    // if( 0 <= x(i) && x(i) <= LSOURCE/XLENGTH )
	    if( 0 <= xi && xi <= LSOURCE/XLENGTH )
	      issource( i,j ) = true;
	    else
	      issource( i,j ) = false;
	  }
	else
	  issource( i,j ) = false;
      }
  
  // CHANNEL
  for( int i=0; i<NX; ++i )
    for( int j=0; j<NZ; ++j )
      {
	if( isoxide( i,j ) == false )
	  {
	    const double xi = host_dm->get_X()->mesh(i);
	    // if( LSOURCE/XLENGTH <= x(i) && x(i) <= (LSOURCE+LCHANNEL)/XLENGTH )
	    if( LSOURCE/XLENGTH <= xi && xi <= (LSOURCE+LCHANNEL)/XLENGTH )
	      ischannel( i,j ) = true;
	    else
	      ischannel( i,j ) = false;
	  }
	else
	  ischannel( i,j ) = false;
      }
  
  // DRAIN CONTACT
  for( int i=0; i<NX; ++i )
    for( int j=0; j<NZ; ++j )
      {
	if( isoxide( i,j ) == false )
	  {
	    const double xi = host_dm->get_X()->mesh(i);
	    // if( (LSOURCE+LCHANNEL)/XLENGTH-1.e-12 <= x(i) && x(i) <= (LSOURCE+LCHANNEL+LDRAIN)/XLENGTH )
	    if( (LSOURCE+LCHANNEL)/XLENGTH-1.e-12 <= xi && xi <= (LSOURCE+LCHANNEL+LDRAIN)/XLENGTH )
	      isdrain( i,j ) = true;
	    else
	      isdrain( i,j ) = false;
	  }
	else
	  isdrain( i,j ) = false;
      }

  // EFFECTIVE MASSES
  for( int i=0; i<NX; ++i )
    for( int j=0; j<NZ; ++j )
      if( !isoxide(i,j) )
	for( int nu=0; nu<_NVALLEYS; ++nu )
	  {
	    effmass(Xdim,nu,i,j ) = mass(Xdim,nu );
	    effmass(Ydim,nu,i,j ) = mass(Ydim,nu );
	    effmass(Zdim,nu,i,j ) = mass(Zdim,nu );
	  }	      
      else
	for( int nu=0; nu<_NVALLEYS; ++nu )
	  {
	    effmass(Xdim,nu,i,j ) = _oxidemass;
	    effmass(Ydim,nu,i,j ) = _oxidemass;
	    effmass(Zdim,nu,i,j ) = _oxidemass;
	  }
  
  // introduce kinetic energies,maxwellians and velocities
  for( int nu=0; nu<_NVALLEYS; ++nu )
    for( int l=0; l<NW; ++l )
      {
	// epskin( nu,l ) = w(l);
	epskin( nu,l ) = host_dm->get_W()->mesh(l);
	maxw( nu,l ) = exp( -epskin( nu,l ) );
      }
  
  _max_a1=0;
  for( int nu=0; nu<_NVALLEYS; ++nu )
    for( int l=0; l<NW; ++l )
      for( int m=0; m<NPHI; ++m )
	{
	  a1( nu,l,m ) = sqrt(2.*host_dm->get_W()->mesh(l)*(1.+host_rescpar->get_epsstar()*_kane[nu]*host_dm->get_W()->mesh(l)))*cos(host_dm->get_PHI()->mesh(m))/(sqrt(mass(Xdim,nu))*(1.+2.*host_rescpar->get_epsstar()*_kane[nu]*host_dm->get_W()->mesh(l)));
	  _max_a1 = max( _max_a1,fabs(a1(nu,l,m)) );
	}

#ifdef __CUDACC__  
  checkCudaErrors( cudaMemcpy(_GPU_a1, _a1, _NVALLEYS*NW*NPHI*sizeof(double), cudaMemcpyHostToDevice) );
#endif

  _max_a3const=0;
  for( int nu=0; nu<_NVALLEYS; ++nu )
    for( int l=0; l<NW; ++l )
      for( int m=0; m<NPHI; ++m )
	{
	  const double wl = host_dm->get_W()->mesh(l);
	  const double phim = m*DPHI;

	  double val = sin(phim)/( sqrt(mass(Xdim,nu))*sqrt(2.*wl*(1+_kane[nu]*host_rescpar->get_epsstar()*wl)) );
	  a3_const(nu,l,m) = val;
	  _max_a3const = max( _max_a3const, fabs(val) );
	}
  
#ifdef __CUDACC__
  checkCudaErrors( cudaMemcpy(_GPU_a3_const, _a3_const, _NVALLEYS*NW*NPHI*sizeof(double), cudaMemcpyHostToDevice) );
#endif

  // normalize Maxwellians
  double *aux;
  aux = new double [ (NW)*(NPHI) ];
  for( int nu=0; nu<_NVALLEYS; ++nu )
    {
      for(int l=0; l<NW; ++l )
	for( int m=0;m<NPHI; ++m )
	  aux[ l*NPHI+m ] = s_chcoord( nu,host_dm->get_W()->mesh(l) )*maxw( nu,l );
      
      _normfact[nu]       = host_rescpar->get_kstar()*host_rescpar->get_kstar()*integrate_R2( aux,NW,host_dm->get_W()->get_delta(),PHISTYLE,NPHI,DPHI,PHISTYLE );
      _analyticnormfact[nu] = 2.*M_PI*(host_phyco->__me*host_phyco->__kb*host_phyco->__latticetemp)/(host_phyco->__hbar*host_phyco->__hbar)*sqrt(mass(Xdim,nu)*mass(Ydim,nu))*(1.+2.*_kane[nu]*host_phyco->__kb*host_phyco->__latticetemp);
      
      if( fabs(_normfact[nu]-_analyticnormfact[nu])/fabs(_analyticnormfact[nu]) > .01 )
	{
	  ofstream str;
	  str.open( "errormessages.txt", ios_base::app );
 	  str << " WARNING from 'ConstData' constructor : wave-vector mesh is rough." << endl
	      << "         the numerical integration of the non-normalized Maxwellian gives (adim. units) ---> " << _normfact[nu] 
	      << endl
	      << "         while its analytical integration would give                      (adim. units) ---> " << _analyticnormfact[nu]
	      << endl;
	  str.close();
	}
      
      for(int l=0; l<NW; ++l)
	maxw( nu,l ) /= (host_rescpar->get_maxwstar()*_normfact[nu]);
    }
  delete [] aux;

#ifdef __CUDACC__
  checkCudaErrors( cudaMemcpy(_GPU_maxw, _maxw, _NVALLEYS*NW*sizeof(double), cudaMemcpyHostToDevice) );
#endif
	
  // nd
  for(int i=0; i<NX; ++i)
    for(int j=0; j<NZ; ++j)
      {
	nd( i,j ) = 0.0;
	
	if( !isoxide(i,j) )
	  {
	    if( issource(i,j) || isdrain(i,j) )
	      nd(i,j) = NDHIGH/host_rescpar->get_nstar();
	    else
	      nd(i,j) = NDLOW/host_rescpar->get_nstar();
	  }
      }
  
#ifdef __CUDACC__
  checkCudaErrors( cudaMemcpy(_GPU_nd, _nd, NX*NZ*sizeof(double), cudaMemcpyHostToDevice) );
#endif

  // vconf
  for( int i=0; i<NX; ++i )
    for( int j=0; j<NZ; ++j )
      {
	if( !isoxide(i,j) )
	  vconf(i,j) = 0.0;
	else
	  vconf(i,j) = VCONF_WELL/host_rescpar->get_potstar();
      }
  
  // vgate
  for( int i=0; i<NX; ++i )
    for( int j=0; j<NZ; ++j )
      {
	if( j==0 && isgate(i,j) )
	  vgate(i,j) = VLOWERGATE/host_rescpar->get_potstar();
	else if( j==NZ-1 && isgate(i,j) )
	  vgate(i,j) = VUPPERGATE/host_rescpar->get_potstar();
	else
	  vgate(i,j) = NAN; // must be uninfluent
      }
  
  // \int_0^{L_z} N_D(0,z) dz
  double *aux_R1;
  aux_R1 = new double [ NZ ];
  for( int j=0; j<NZ; ++j )
    aux_R1[j] = nd( 0,j );
  _intnd = integrate_R1( aux_R1,NZ,DZ,TRAPEZOIDS );

  delete [] aux_R1;

  // BOUNDARY CONDITIONS FOR THE POISSON EQUATION

  for(int i=0; i<NX; ++i)
    for(int j=0; j<NZ; ++j)
      {
	bc_confined(INITCOND,i,j).condtype = INTERNAL;
	bc_confined(THERMEQUIL,i,j).condtype = INTERNAL;
	bc_confined(POTENTIAL,i,j).condtype = INTERNAL;

	if( 0<=j && j<=NZ-1)
	  {
	    if( i==0 )
	      {
		bc_confined(INITCOND,i,j).condtype         = NEUMANN;
		bc_confined(THERMEQUIL,i,j).condtype       = DIRICHLET;
		bc_confined(POTENTIAL,i,j).condtype        = DIRICHLET;
	      }
	    else if( i==NX-1 )
	      {
		bc_confined(INITCOND,i,j).condtype         = NEUMANN;
		bc_confined(THERMEQUIL,i,j).condtype       = DIRICHLET;
		bc_confined(POTENTIAL,i,j).condtype        = DIRICHLET;
	      }
	  }
      
	// REMARK: the point values cannot be stored at the
	// moment of construction,because an eigenproblem
	// needs be solved to compute them      
      }
  
  for(int i=0; i<NX; ++i)
    for(int j=0; j<NZ; ++j)
      {
	if( 1<=i && i<NX-1 )
	  {
	    if( j==0 )
	      {
		if( isgate(i,j) == true )
		  {
		    bc_confined(INITCOND,i,j).condtype    = DIRICHLET;
		    bc_confined(INITCOND,i,j).point_value = vgate(i,j);
	  
		    bc_confined(THERMEQUIL,i,j).condtype    = DIRICHLET;
		    bc_confined(THERMEQUIL,i,j).point_value = vgate(i,j);
	  
		    bc_confined(POTENTIAL,i,j).condtype    = DIRICHLET;
		    bc_confined(POTENTIAL,i,j).point_value = vgate(i,j);
		  }
		else
		  {
		    bc_confined(INITCOND,i,j).condtype         = NEUMANN;
	  
		    bc_confined(THERMEQUIL,i,j).condtype         = NEUMANN;
	  
		    bc_confined(POTENTIAL,i,j).condtype         = NEUMANN;
		  }
	      }
	    else if( j==NZ-1 )
	      {
		if( isgate(i,j) == true )
		  {
		    bc_confined(INITCOND,i,j).condtype    = DIRICHLET;
		    bc_confined(INITCOND,i,j).point_value = vgate(i,j);
	  
		    bc_confined(THERMEQUIL,i,j).condtype    = DIRICHLET;
		    bc_confined(THERMEQUIL,i,j).point_value = vgate(i,j);
	  
		    bc_confined(POTENTIAL,i,j).condtype    = DIRICHLET;
		    bc_confined(POTENTIAL,i,j).point_value = vgate(i,j);
		  }
		else
		  {
		    bc_confined(INITCOND,i,j).condtype         = NEUMANN;
	  
		    bc_confined(THERMEQUIL,i,j).condtype         = NEUMANN;
	  
		    bc_confined(POTENTIAL,i,j).condtype         = NEUMANN;
		  }
	      }
	  }
      }
  
  // sinus of the angle
  for(int m=0; m<NPHI; ++m)
    {
      const double phim = host_dm->get_PHI()->mesh(m);
      _sinphi[m] = sin(phim);
      _cosphi[m] = cos(phim);
    }

#ifdef __CUDACC__
  // mapping for cuda_compute_kernel_3
  int N = NZ-2;
  int *aux_j  = new int [N*(N+1)/2];
  int *aux_jj = new int [N*(N+1)/2];

  for(int J=0; J<N*(N+1)/2; ++J)
    {
      int r = int( (2*N+1 - sqrt((double)(4*N*N + 4*N + 1 - 8*J)))/2 );
      int c = (r*r - (2*N-1)*r + 2*J)/2;

      aux_j[J] = r+1;
      aux_jj[J] = c+1;
    }

  CUDA_SAFE_CALL( cudaMemcpy( _GPU_J_to_j, aux_j, N*(N+1)/2*sizeof(int), cudaMemcpyHostToDevice ) );
  CUDA_SAFE_CALL( cudaMemcpy( _GPU_J_to_jj, aux_jj, N*(N+1)/2*sizeof(int), cudaMemcpyHostToDevice ) );
  
  delete [] aux_j;
  delete [] aux_jj;
#endif

  const double DW = host_dm->get_W()->get_delta();
  const double EPSSTAR = host_rescpar->get_epsstar();
  for( int nu=0; nu<_NVALLEYS; ++nu )
    for( int l=0; l<NW; ++l )
      for( int m=0; m<NPHI; ++m )
	{
	  const double wl = (l+0.5)*DW;
	  const double phim = host_dm->get_PHI()->mesh(m);
	  
	  vel( Xdim,nu,l,m ) = sqrt(2.*wl*(1.+EPSSTAR*_kane[nu]*wl))*cos(phim)/(sqrt(mass(Xdim,nu))*(1.+2.*EPSSTAR*_kane[nu]*wl));
	  vel( Ydim,nu,l,m ) = sqrt(2.*wl*(1.+EPSSTAR*_kane[nu]*wl))*sin(phim)/(sqrt(mass(Ydim,nu))*(1.+2.*EPSSTAR*_kane[nu]*wl));
	}

#ifdef __CUDACC__
  CUDA_SAFE_CALL( cudaMemcpy( _GPU_vel, _vel, 2*_NVALLEYS*NW*NPHI*sizeof(double), cudaMemcpyHostToDevice ) );
#endif

  exitfunc;
  
  return;
}



/**
   PURPOSE:        This function just sets the initial step at 0, as well as the initial time.

   FILE:           cuda_config.cu

   NAME:           MOSFETProblem::config_datastructs

   PARAMETERS:

   RETURN VALUE:   none

   CALLS TO:       MOSFETProblem::show_eta     (cuda_testing.h --- declared inline)

   CALLED FROM:    MOSFETProblem::cuda_config  (cuda_config.cu)

   DATA MODIFIED:  

   METHOD:         

   LAST UPDATE:    2022 March 28
*/
void MOSFETProblemCuda::config_datastructs()
{
  enterfunc;

  _step = 0;
  _time = 0;

  exitfunc;
  
  return;
}



/**
   PURPOSE:        This function sets all the constant part for the ITER computational section.
                   In particular, it sets the constant part for the Vlasov-like equation
		   used to advancing the guess for the potential, taking into account
		   the boundary conditions, and the constant part of the Schroedinger matrix
		   for the computation of eigenvalues/eigenvectors.

   FILE:           cuda_config.cu

   NAME:           MOSFETProblem::config_spsolvers

   PARAMETERS:

   RETURN VALUE:   none

   CALLS TO:       MOSFETProblem::show_eta            (cuda_testing.cu)

   CALLED FROM:    MOSFETProblem::cuda_config         (cuda_config.cu)

   DATA MODIFIED:  

   METHOD:         

   LAST UPDATE:    2022 March 28
*/
void MOSFETProblemCuda::config_spsolvers()
{
  enterfunc;

  const int NSBN  = host_dm -> get_NSBN();
  const int NX    = host_dm -> get_X()   -> get_N();
  const int NZ    = host_dm -> get_Z()   -> get_N();
  const int NW    = host_dm -> get_W()   -> get_N();
  const int NPHI  = host_dm -> get_PHI() -> get_N();
  const double ZWIDTH  = host_physdesc -> get_ZWIDTH();
  const double LSiO2  = host_physdesc -> get_LSiO2();

  /**********************************************
   *    CONSTANT PART OF THE LINEAR SYSTEM      *
   **********************************************/
  // initialize with zeros
  for( int r=0; r<NX*NZ; ++r )
    for(int s=r-NZ; s<=r+NZ; ++s)
      for(int EP = 0; EP<=2; ++EP)
	matrix_2dconst(EP,r,s) = 0.0;
  
  // double DXM2 = 1./(_dx*_dx);
  // double DZM2 = 1./(_dz*_dz);
  double DXM2 = host_dm->get_X()->get_delta_m2();
  double DZM2 = host_dm->get_Z()->get_delta_m2();

  // Laplacian
  for(int EP = 0; EP<=2; ++EP)
    {
      for(int i=0; i<NX; ++i)
	for(int j=0; j<NZ; ++j)
	  {
	    double EPSR_IXm1_IZ = ( (LSiO2/ZWIDTH < host_dm->get_Z()->mesh(j)   && host_dm->get_Z()->mesh(j)   < (ZWIDTH-LSiO2)/ZWIDTH) ? host_phyco->__epsrSi : host_phyco->__epsrSiO2 );
	    double EPSR_IX_IZ   = ( (LSiO2/ZWIDTH < host_dm->get_Z()->mesh(j)   && host_dm->get_Z()->mesh(j)   < (ZWIDTH-LSiO2)/ZWIDTH) ? host_phyco->__epsrSi : host_phyco->__epsrSiO2 );
	    double EPSR_IX_IZm1 = ( (LSiO2/ZWIDTH < host_dm->get_Z()->mesh(j-1) && host_dm->get_Z()->mesh(j-1) < (ZWIDTH-LSiO2)/ZWIDTH) ? host_phyco->__epsrSi : host_phyco->__epsrSiO2 );
	    double EPSR_IXp1_IZ = ( (LSiO2/ZWIDTH < host_dm->get_Z()->mesh(j)   && host_dm->get_Z()->mesh(j)   < (ZWIDTH-LSiO2)/ZWIDTH) ? host_phyco->__epsrSi : host_phyco->__epsrSiO2 );
	    double EPSR_IX_IZp1 = ( (LSiO2/ZWIDTH < host_dm->get_Z()->mesh(j+1) && host_dm->get_Z()->mesh(j+1) < (ZWIDTH-LSiO2)/ZWIDTH) ? host_phyco->__epsrSi : host_phyco->__epsrSiO2 );
	    
	    if( 1<=i && i<=NX-2 && 1<=j && j<=NZ-2 )
	      {
		matrix_2dconst(EP,host_i_j(i,j),host_i_j(i-1,j)) += -host_adimpar->get_eta()*host_adimpar->get_eta()*DXM2*(.5*EPSR_IXm1_IZ+.5*EPSR_IX_IZ);
		matrix_2dconst(EP,host_i_j(i,j),host_i_j(i,j-1)) += -DZM2*(.5*EPSR_IX_IZm1+.5*EPSR_IX_IZ);
		matrix_2dconst(EP,host_i_j(i,j),host_i_j(i,j))   += host_adimpar->get_eta()*host_adimpar->get_eta()*DXM2*(.5*EPSR_IXm1_IZ+EPSR_IX_IZ+.5*EPSR_IXp1_IZ) + DZM2*(.5*EPSR_IX_IZm1+EPSR_IX_IZ+.5*EPSR_IX_IZp1);
		matrix_2dconst(EP,host_i_j(i,j),host_i_j(i,j+1)) += -DZM2*(.5*EPSR_IX_IZ+.5*EPSR_IX_IZp1);
		matrix_2dconst(EP,host_i_j(i,j),host_i_j(i+1,j)) += -host_adimpar->get_eta()*host_adimpar->get_eta()*DXM2*(.5*EPSR_IX_IZ+.5*EPSR_IXp1_IZ);
	      }
	    
	    // boundary conditions
	    if( 1<=i && i<=NX-2 && j==0 )
	      {
		switch( bc_confined(EP,i,j).condtype )
		  {
		  case DIRICHLET :
		    matrix_2dconst(EP,host_i_j(i,j),host_i_j(i,j)) = 1.;
		    matrix_2dconst(EP,host_i_j(i,j),host_i_j(i,j+1)) = 0;
		    break;
		    
		  case NEUMANN :
		    matrix_2dconst(EP,host_i_j(i,j),host_i_j(i,j)) = 1.;
		    matrix_2dconst(EP,host_i_j(i,j),host_i_j(i,j+1)) = -1.;
		    break;
		    
		  default : 
		    cerr << "For POISSON problems, only DIRICHLET, NEUMANN boundary conditions are allowed."
			 << "Please modify the boundary condition." << endl;
		    throw error_CONFIG_BOUNDCOND( i, j, bc_confined(EP,i,j), (EigenProblem)EP );
		    break;
		  }
	      }
	    
	    if( 0<=j && j<=NZ-1 && i==NX-1 )
	      {
		switch( bc_confined(EP,i,j).condtype )
		  {
		  case DIRICHLET :
		    matrix_2dconst(EP,host_i_j(i,j),host_i_j(i,j)) = 1.;
		    matrix_2dconst(EP,host_i_j(i,j),host_i_j(i-1,j)) = 0;
		    break;
		    
		  case NEUMANN :
		    matrix_2dconst(EP,host_i_j(i,j),host_i_j(i,j)) = 1.;
		    matrix_2dconst(EP,host_i_j(i,j),host_i_j(i-1,j)) = -1.;
		    break;
		    
		  default : 
		    cerr << "For POISSON problems, only DIRICHLET, NEUMANN boundary conditions are allowed."
			 << "Please modify the boundary condition." << endl;
		    throw error_CONFIG_BOUNDCOND( i, j, bc_confined(EP,i,j), (EigenProblem)EP );
		    break;
		  }
	      }
	    
	    if( j==NZ-1 && 1<=i && i<=NX-2 )
	      {
		switch( bc_confined(EP,i,j).condtype )
		  {
		  case DIRICHLET :
		    matrix_2dconst(EP,host_i_j(i,j),host_i_j(i,j)) = 1.;
		    matrix_2dconst(EP,host_i_j(i,j),host_i_j(i,j-1)) = 0;
		    break;
		    
		  case NEUMANN :
		    matrix_2dconst(EP,host_i_j(i,j),host_i_j(i,j)) = 1.;
		    matrix_2dconst(EP,host_i_j(i,j),host_i_j(i,j-1)) = -1.;
		    break;
		    
		  default : 
		    cerr << "For POISSON problems, only DIRICHLET, NEUMANN boundary conditions are allowed."
			 << "Please modify the boundary condition." << endl;
		    throw error_CONFIG_BOUNDCOND( i, j, bc_confined(EP,i,j), (EigenProblem)EP );
		    break;
		  }
	      }
	    
	    
	    if( i==0 && 0<=j && j<=NZ-1 )
	      {
		switch( bc_confined(EP,i,j).condtype )
		  {
		  case DIRICHLET :
		    matrix_2dconst(EP,host_i_j(i,j),host_i_j(i,j)) = 1.;
		    matrix_2dconst(EP,host_i_j(i,j),host_i_j(i+1,j)) = 0;
		    break;
		    
		  case NEUMANN :
		    matrix_2dconst(EP,host_i_j(i,j),host_i_j(i,j)) = 1.;
		    matrix_2dconst(EP,host_i_j(i,j),host_i_j(i+1,j)) = -1.;
		    break;
		    
		  default : 
		    cerr << "For POISSON problems, only DIRICHLET, NEUMANN boundary conditions are allowed."
			 << "Please modify the boundary condition." << endl;
		    throw error_CONFIG_BOUNDCOND( i, j, bc_confined(EP,i,j), (EigenProblem)EP );
		    break;
		  }
	      }
	    
	  }
    }

  // para CUDA
  for(int i=0; i<NX; ++i)
    for(int nu=0; nu<_NVALLEYS; ++nu)
      {
	const int start = (i*_NVALLEYS + nu)*_SCHROED_ROW_SIZE;

	double *D = &___A_const[start];
	double *S = &___A_const[start+_SCHROED_MATRIX_SIZE];
	
	// construction of the matrix representing the Schroedinger operator
      	for(int j=1; j<NZ-1; ++j)
	  D[j-1] = 0.5*host_adimpar->get_cs1()*DZM2*(.5/effmass(Zdim,nu,i,j-1)+1./effmass(Zdim,nu,i,j)+.5/effmass(Zdim,nu,i,j+1));

	for(int j=1; j<NZ-1; ++j)
	  D[j-1] -= (vconf(i,j));

	for(int j=1; j<NZ-2; ++j)
	  S[j-1]  = -0.5*host_adimpar->get_cs1()*DZM2*(.5/effmass(Zdim,nu,i,j) + .5/effmass(Zdim,nu,i,j+1));
      }

#ifdef __CUDACC__
  CUDA_SAFE_CALL( cudaMemcpy( _GPU___A_const,      ___A_const,      _NVALLEYS*NX*_SCHROED_ROW_SIZE*sizeof(double), cudaMemcpyHostToDevice ) );
  CUDA_SAFE_CALL( cudaMemcpy( _GPU_matrix_2dconst, _matrix_2dconst, 3*NX*NZ*(2*NZ+1)*sizeof(double),               cudaMemcpyHostToDevice ) );
#endif

  exitfunc;
  
  return;
}




/**
   PURPOSE:        This function sets all the constants intervening in the integration
                   of the scattering mechanisms.

   FILE:           cuda_config.cu

   NAME:           MOSFETProblem::config_scintegrate

   PARAMETERS:

   RETURN VALUE:   none

   CALLS TO:       MOSFETProblem::show_eta            (cuda_testing.h --- declared inline)

   CALLED FROM:    MOSFETProblem::cuda_config         (cuda_config.cu)

   DATA MODIFIED:  

   METHOD:         

   LAST UPDATE:    2022 March 28
*/
void MOSFETProblemCuda::config_scintegrate()
{
  enterfunc;

  const int NSBN  = host_dm -> get_NSBN();
  const int NX    = host_dm -> get_X()   -> get_N();
  const int NZ    = host_dm -> get_Z()   -> get_N();
  const int NW    = host_dm -> get_W()   -> get_N();
  const int NPHI  = host_dm -> get_PHI() -> get_N();
  const double XLENGTH = host_physdesc -> compute_XLENGTH();
  const double ZWIDTH  = host_physdesc -> get_ZWIDTH();
  const double DPHI = host_dm->get_PHI()->get_delta();

  for( int sc=0; sc<8; ++sc )
    {
      _omega[sc] = host_phyco->__kb*phtemp(sc)/host_phyco->__hbar;

      for(int nu=0; nu<_NVALLEYS; ++nu)
	for(int nup=0; nup<_NVALLEYS; ++nup)
	  occupations( sc, nu, nup ) = (sqrt(mass(Xdim,nu)*mass(Ydim,nu)/(mass(Xdim,nup)*mass(Ydim,nup))) * (1+2.*_kane[nu]*host_rescpar->get_epsstar())/(1+2.*_kane[nup]*host_rescpar->get_epsstar()) * exp(host_phyco->__hbar*_omega[sc]/host_rescpar->get_epsstar()) + 1)/( (exp(host_phyco->__hbar*_omega[sc]/host_rescpar->get_epsstar()) + 1)*(exp(host_phyco->__hbar*_omega[sc]/host_rescpar->get_epsstar()) - 1) );

      switch( scatttype(sc) )
	{
	case PHONON_ELASTIC :
	  {
	    const double deformation_potential = defpot(sc)*host_phyco->__eV;
	    _cscatt[ sc ] = host_phyco->__me*host_phyco->__kb*host_phyco->__latticetemp*pow( deformation_potential,2 );
	    _cscatt[ sc ] /= host_rescpar->get_scstar()*host_phyco->__hbar*host_phyco->__hbar*ZWIDTH*4.*M_PI*M_PI*host_phyco->__rhoSi*host_phyco->__hbar*host_phyco->__ul*host_phyco->__ul;
	    break;
	  }
	  
	case PHONON_GTYPE :
	  {
	    const double deformation_potential = defpot(sc)*1.e8*host_phyco->__eV*1.E2;
	    _cscatt[ sc ] = host_phyco->__me*pow( deformation_potential,2 );
	    _cscatt[ sc ] /= host_rescpar->get_scstar()*8.*M_PI*M_PI*host_phyco->__rhoSi*_omega[sc]*ZWIDTH*host_phyco->__hbar*host_phyco->__hbar;
	    break;
	  }

	case PHONON_FTYPE :
	  {
	    const double deformation_potential = defpot(sc)*1.e8*host_phyco->__eV*1.E2;
	    _cscatt[ sc ] = host_phyco->__me*pow( deformation_potential,2 );
	    _cscatt[ sc ] /= host_rescpar->get_scstar()*8.*M_PI*M_PI*host_phyco->__rhoSi*_omega[sc]*ZWIDTH*host_phyco->__hbar*host_phyco->__hbar;
	    break;
	  }

	case ROUGHNESS :
	  {
	    _c_SR = XLENGTH*pow(host_rescpar->get_kstar(),3)*pow(host_solvpar->get_LENGTH_SR(),2)/(4.*M_PI);
	    double * aux;
	    aux = new double[NPHI];
	    for(int nu=0; nu<_NVALLEYS; ++nu)
	      for(int l=0; l<NW; ++l)
		for(int m=0; m<NPHI; ++m)
		  {
		    for(int mp=0; mp<NPHI; ++mp)
		      aux[mp] = 1./pow( 1.+4.*pow(host_rescpar->get_kstar(),2)*pow(host_solvpar->get_LENGTH_SR(),2)*host_dm->get_W()->mesh(l)*(1+_kane[nu]*host_rescpar->get_epsstar()*host_dm->get_W()->mesh(l))*pow(sin(.5*(host_dm->get_PHI()->mesh(m)-host_dm->get_PHI()->mesh(mp))),2)*(mass(Xdim,nu)*pow(sin(.5*(host_dm->get_PHI()->mesh(m)+host_dm->get_PHI()->mesh(mp))),2)+mass(Ydim,nu)*pow(cos(.5*(host_dm->get_PHI()->mesh(m)+host_dm->get_PHI()->mesh(mp))),2)) ,1.5);

		    double res = 0;
		    for(int mp=0; mp<NPHI; ++mp)
		      res += aux[mp];
		    integrateddenom_SR(nu,l,m) = res * DPHI;
		  }
	    delete [] aux;

#ifdef __CUDACC__
	    checkCudaErrors( cudaMemcpy(_GPU_integrateddenom_SR, _integrateddenom_SR, _NVALLEYS*NW*NPHI*sizeof(double), cudaMemcpyHostToDevice) );
#endif

	    break;
	  }
	}
    }

  exitfunc;
  
  return;
}




/**
   PURPOSE:        

   FILE:           cuda_config.cu

   NAME:           MOSFETProblem::CPU_initialize_pdf

   PARAMETERS:

   RETURN VALUE:   none

   CALLS TO:       none

   CALLED FROM:    MOSFETProblem::cuda_config         (cuda_config.cu)

   DATA MODIFIED:  

   METHOD:         

   LAST UPDATE:    2022 February 17
*/
void MOSFETProblemCuda::CPU_initialize_pdf()
{
  enterfunc;

  const int NSBN  = host_dm -> get_NSBN();
  const int NX    = host_dm -> get_X()   -> get_N();
  const int NW    = host_dm -> get_W()   -> get_N();
  const int NPHI  = host_dm -> get_PHI() -> get_N();

#pragma omp parallel for
  for( int global_index=0; global_index < _NVALLEYS*NSBN*NX*NW*NPHI ; ++global_index )
    {
      int i, nu, p, l, m;
      GPU_map_1D_to_5D( global_index, &i, NX, &nu, _NVALLEYS, &p, NSBN, &l, NW, &m, NPHI );

      double wl = host_dm->get_W()->mesh(l);
      double s_chcoord = sqrt(mass(Xdim,nu)*mass(Ydim,nu))*(1.+2.*_kane[nu]*host_rescpar->get_epsstar()*wl);

      pdf( nu,p,i,l,m,0 ) = s_chcoord * surfdens_eq( nu,p,i ) * maxw(nu,l);
    }

  exitfunc;

  return;
}




/**
   PURPOSE:        This function sets the constant part for the right-hand side
                   of the Poisson equations involved in the ITER computational section.
		   More into details, when the initial condition has been computed,
		   the boundary conditions for the potential can be set, precisely
		   the Dirichlet condition at the source and the drain are a result
		   of solving two eigenproblems, the profile is not flat due to the
		   distortion due to the confinement.

   FILE:           cuda_config.cu

   NAME:           MOSFETProblem::config_constdata

   PARAMETERS:

   RETURN VALUE:   none

   CALLS TO:       none

   CALLED FROM:    MOSFETProblem::thermequi_on_cuda     (cuda_iter.cu)

   DATA MODIFIED:  

   METHOD:         

   LAST UPDATE:    2022 March 28
*/
void MOSFETProblemCuda::config_rhs()
{
  enterfunc;

  const int NSBN  = host_dm -> get_NSBN();
  const int NX    = host_dm -> get_X()   -> get_N();
  const int NZ    = host_dm -> get_Z()   -> get_N();
  const int NW    = host_dm -> get_W()   -> get_N();
  const int NPHI  = host_dm -> get_PHI() -> get_N();

  for(int i=0; i<NX; ++i)
    for(int j=0; j<NZ; ++j)
      {
  	int row_index = i*NZ+j;
	_rhs_const [ row_index ] = 0;

  	// boundary conditions
  	if( j==0 && 1<=i && i<=NX-2 )
  	  {
  	    switch( bc_confined(POTENTIAL,i,j).condtype )
  	      {
  	      case DIRICHLET :
  		_rhs_const[ row_index ] += bc_confined(POTENTIAL,i,j).point_value;
  		break;
	      
  	      case NEUMANN :
  		_rhs_const[ row_index ] += 0;
  		break;
	      
  	      default : 
  		cerr << "For POISSON problems, only DIRICHLET, NEUMANN boundary conditions are allowed."
  		     << "Please modify the boundary condition." << endl;
		throw error_CONFIG_BOUNDCOND( i, j, bc_confined(POTENTIAL,i,j), POTENTIAL );
  		break;
  	      }	
  	  }
      
  	if( i==NX-1 && 0<=j && j<=NZ-1 )
  	  {
  	    switch( bc_confined(POTENTIAL,i,j).condtype )
  	      {
  	      case DIRICHLET :
  		_rhs_const[ row_index ] += bc_confined(POTENTIAL,i,j).point_value;
  		break;
	      
  	      case NEUMANN :
  		_rhs_const[ row_index ] += 0;
  		break;
	      
  	      default : 
  		cerr << "For POISSON problems, only DIRICHLET, NEUMANN boundary conditions are allowed."
  		     << "Please modify the boundary condition." << endl;
		throw error_CONFIG_BOUNDCOND( i, j, bc_confined(POTENTIAL,i,j), POTENTIAL );
  		break;
  	      }
  	  }
      
  	if( j==NZ-1 && 1<=i && i<=NX-2 )
  	  {
  	    switch( bc_confined(POTENTIAL,i,j).condtype )
  	      {
  	      case DIRICHLET :
  		_rhs_const[ row_index ] += bc_confined(POTENTIAL,i,j).point_value;
  		break;
	      
  	      case NEUMANN :
  		_rhs_const[ row_index ] += 0;
  		break;
	      
  	      default : 
  		cerr << "For POISSON problems, only DIRICHLET, NEUMANN boundary conditions are allowed."
  		     << "Please modify the boundary condition." << endl;
		throw error_CONFIG_BOUNDCOND( i, j, bc_confined(POTENTIAL,i,j), POTENTIAL );
  		break;
  	      }
  	  }
      
  	if( i==0 && 0<=j && j<=NZ-1 )
  	  {
  	    switch( bc_confined(POTENTIAL,i,j).condtype )
  	      {
  	      case DIRICHLET :
  		_rhs_const[ row_index ] += bc_confined(POTENTIAL,i,j).point_value;
  		break;
	      
  	      case NEUMANN :
  		_rhs_const[ row_index ] += 0;
  		break;
	      
  	      default : 
  		cerr << "For POISSON problems, only DIRICHLET, NEUMANN boundary conditions are allowed."
  		     << "Please modify the boundary condition." << endl;
		throw error_CONFIG_BOUNDCOND( i, j, bc_confined(POTENTIAL,i,j), POTENTIAL );
  		break;
  	      }
  	  }
      
      }

#ifdef __CUDACC__
  CUDA_SAFE_CALL( cudaMemcpy( _GPU_rhs_const, _rhs_const, NX*NZ*sizeof(double), cudaMemcpyHostToDevice ) );
#endif

  exitfunc;

  return;
}





