#include "mosfetproblem.h"
#include "debug_flags.h"


#define    g(i) _g[i+3]
#define   df(i) _df[i+3]
#define gpos(i) _gpos[i+3]
#define gneg(i) _gneg[i+3]

/**********************************
 *      w-PARTIAL DERIVATIVE      *
 **********************************/
#ifdef __CUDACC__
/*
  CUDA kernel: 'cuda_WENO_W_20230524'
  
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  
             THE DESCRIPTION MUST BE UPDATED !!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  BRIEF DESCRIPTION
  =================
  This kernel computes the w-derivative of (a_3*pdf) by means of the order-5 WENO
  routine for flux reconstruction. Namely, it gives a value to the flux for (a_3*pdf)(nu,p,i,l,m) at points
  w_{l} for l = -1/2, 1/2, 3/2, ..., _NW-1/2
  so that by finite differences we have a high-order approximation for the derivative for
  (a_3*pdf)(nu,p,i,l,m) for l = 0, 1, 2, ..., _NW-1.
  
  As for the boundary conditions, specular reflection is used for the border w=0, in the sense
  that negative-energy points are physical, positive-energy points for the pi-shifted angle,
  while homogeneous Neumann boundary conditions are used for the border w=wmax, as, anyway,
  the distribution function should be numerically zero there, if parameter _BARN is properly chosen.

  In order to enforce mass conservation, an averaging is needed between the reconstructed fluxes
  at opposite angles. For this reason, the same thread takes care of two different angles.
  The specific parts of the code taking care of this are marked as MASSCONS_ENF.

  DATA FLOW
  =========
  The function takes as input:
       NAME       |                             DESCRIPTION                                       |                   SIZE                         |       ORDERING
  -------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  1)     _GPU_pdf | the probability distribution function                                         | _NVALLEYS*_NSBN*_NX*_NW*_NPHI*4*sizeof(double) | s > i > nu > p > l > m
  -------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  2) _GPU_deps_dx | the force field                                                               | _NVALLEYS*_NSBN*_NX*sizeof(double)             | i > nu > p
  -------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  3)        STAGE | the Runge-Kutta time-integration stage (0, 1 or 2, as 3-order TVD RK is used) | idT                                            |
  -------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  4)    _GPU_kane | the Kane non-parabolicity factors                                             | passed as a symbol, 3*sizeof(double)           | 
  -------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  5)    _GPU_mass | the Silicon effective masses                                                  | passed as a symbol, 9*sizeof(double)           |
  -------------------------------------------------------------------------------------------------------------------------------------------------------------------------

  The function returns as output:
       NAME       |                             DESCRIPTION                                       |                   SIZE                         |       ORDERING
  -------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  6) _GPU_rhs_pdf | the function MODIFIES this vector, representing the approximation             | _NVALLEYS*_NSBN*_NX*_NW*_NPHI*sizeof(double)   | i > nu > p > l > m
                  | of the right hand side in the transport equation, by adding the w-derivative. |                                                | 
  -------------------------------------------------------------------------------------------------------------------------------------------------------------------------

  KERNEL CALL: THREAD ORDERING
  ============================
  A total of _NVALLEYS*_NSBN*_NX*_NPHI/2 threads are constructed. As a reminder, _NPHI/2 is used
  and not _NPHI because every thread takes care of two opposite angles, for mass-conservation reasons.
  The force field _GPU_deps_dx does not depend on the angle, therefore _NPHI/2 threads are 
  using the same information from global memory.
  
  The thread ordering is i (_NX) > nu (_NVALLEYS) > p (_NSBN) > m (_NPHI/2).

  BRIEF DESCRIPTION (new)
  =======================
  This kernel comes as an improvement of the previous kernel cuda_WENO_W_20230223.
  There, some statically allocated arrays are used. The problem with that is that cuda does not allow that,
  as the vector size must be 'compile-time discoverable'. Therefore, a fixed maximum value is
  used, under the name of NWMAX, defined with a directive to the pre-compiler. This introduces rigidity in the
  code and a sub-optimal memory exploitation. 
  A new version avoid rigid static local vectors is needed.
 */
__global__ void cuda_WENO_W_20230524( const discrMeshes *dm, const rescalingParams *rescpar, const double *_GPU_pdf, double *_GPU_rhs_pdf, const double *_GPU_deps_dx, const idT STAGE )
{
  const idT global_index = blockIdx.x*blockDim.x + threadIdx.x; 
  const idT NSBN         = dm->get_NSBN();
  const idT NX           = dm->get_X()->get_N();
  const idT NW           = dm->get_W()->get_N();
  const idT NPHI         = dm->get_PHI()->get_N();
  const idT NPHI2        = NPHI/2;
  const double DW        = dm->get_W()->get_delta();
  const double DPHI      = dm->get_PHI()->get_delta();
  const double DWM1      = dm->get_W()->get_delta_m1();
                                                                 /*
								   Global index for the thread, 
								   and "external" values of the discretization meshes.
								 */

  __shared__ double _deps_dx;                                    /*
								   Theis variable contains the value of the eps derivative.
								 */
  extern __shared__ double sm[];
  double *flux = &sm[0];
  double *_g   = &sm[NW];
  double *_fh  = &sm[2*NW+6];
                                                                 /*
								   (i)   the values of flux flux[]              <--- NW values
								   (ii)  the values for flux _g                 <--- NW+6 values
								   (iii) the values for flux _fh reconstruction <--- NW+1 values
								 */

  __shared__ double _ff0m, _ff0mspec;                            /*
								   These two variables will contain the values needed for the
								   averaging for mass conservation.
								 */
  
  /*
    begin section (1) : split the global index into local indices
  */
  idT nu, p, i, l, m;
  GPU_map_1D_to_5D( global_index, &i, NX, &nu, _NVALLEYS, &p, NSBN, &m, NPHI2, &l, NW+1 );
  /*
    end section (1)
  */

  /*
    begin section (2) : load value of deps_Dx.
                        No check on indices should be needed if grid size and block size are correctly chosen.
  */
  if( threadIdx.x == 0 )
    {
      _deps_dx = GPU_deps_dx(nu,p,i);
    }
  /*
    end section (2)
  */
  
  /*
    begin section (3) : store values of flux into shared memory
  */
  if( l < NW )
    {
      const double wl   = dm->get_W()->mesh(l);
      const double aux  = rescpar->get_epsstar()*GPU_kane(nu)*wl;
      const double aux3 = sqrt(2.*wl*(1.+aux));
      flux[l] = -_deps_dx * aux3 * cos(dm->get_PHI()->mesh(m)) / (GPU_sqrtmass(Xdim,nu)*(1.+2.*aux));
    }
  __syncthreads();
  /*
    end section (3)
  */

  /*
    begin section (4) : compute the values of _fh[] for one m
  */

  /*
    section (4.1) : declare some local variables
  */
  double g_m2m3;
  double g_m1m2;
  double g_0m1;
  double g_p10;
  double g_p2p1;
  double t0;
  double t1;
  double t2;
  double tt0;
  double tt1;
  double tt2;
  double s0;
  double s1;
  double s2;
  double t00;
  double fromleft, fromright;

  /*
    section (4.2) : introduce g for internal points
  */
  if( l<NW )
    {
      g(l) = flux[l]*GPU_pdf( nu,p,i,l,m,STAGE );
    }
  
  __syncthreads();

  /*
    section (4.3) : set boundary conditions
  */
  if( l>=1 && l<=3 )
    {
      g(-l) = flux[l-1]*GPU_pdf( nu,p,i,l-1,(m+NPHI/2)%NPHI,STAGE );
      g( NW-1+l ) = g( NW-1 );
    }
  
  __syncthreads();
  
  /*
    section (4.4) : set auxiliary variables
  */
  // if(int l = 0; l <= NW; ++l)
  g_m2m3 = g( l-2 ) - g( l-3 );
  g_m1m2 = g( l-1 ) - g( l-2 );
  g_0m1  = g( l   ) - g( l-1 );
  g_p10  = g( l+1 ) - g( l   );
  g_p2p1 = g( l+2 ) - g( l+1 );
  
  /*
    section (4.5) : compute flux "from left"
  */
  // FROM LEFT
  t0 = g_m2m3 - g_m1m2;
  t1 = g_m1m2 - g_0m1;
  t2 = g_0m1 - g_p10;
  
  tt0 = 13. * t0*t0 + 3. * (    g_m2m3 - 3*g_m1m2 ) * (    g_m2m3 - 3*g_m1m2 );
  tt1 = 13. * t1*t1 + 3. * (    g_m1m2 +   g_0m1 ) * (    g_m1m2 +   g_0m1 );
  tt2 = 13. * t2*t2 + 3. * ( 3.*g_0m1 -   g_p10 ) * ( 3.*g_0m1 -   g_p10 );
  
  tt0 =  ( 1.e-6 + tt0 )*( 1.e-6 + tt0 );
  tt1 =  ( 1.e-6 + tt1 )*( 1.e-6 + tt1 );
  tt2 =  ( 1.e-6 + tt2 )*( 1.e-6 + tt2 );
  s0 =      tt1 * tt2;
  s1 = 6. * tt0 * tt2;
  s2 = 3. * tt0 * tt1;
  
  t00 = 1. / ( s0 + s1 + s2 );
  s0 = s0 * t00;
  s2 = s2 * t00;
  
  fromleft = ( -g(l-2) + 7.*(g(l-1)+g(l)) - g(l+1) ) / 12. + ( s0*(t1-t0) + (0.5*s2-0.25)*(t2-t1) ) /3.; 
  
  /*
    section (4.6) : compute flux "from right"
  */
  // FROM RIGHT 
  t0 = (-g_p2p1) - (-g_p10);
  t1 = (-g_p10) - (-g_0m1);
  t2 = (-g_0m1) - (-g_m1m2);
  
  tt0 = 13. * t0*t0 + 3. * (    (-g_p2p1) - 3.*(-g_p10) ) * (    (-g_p2p1) - 3.*(-g_p10) );
  tt1 = 13. * t1*t1 + 3. * (    (-g_p10) +    (-g_0m1) ) * (    (-g_p10) +    (-g_0m1) );
  tt2 = 13. * t2*t2 + 3. * ( 3.*(-g_0m1) -    (-g_m1m2) ) * ( 3.*(-g_0m1) -    (-g_m1m2) );
  
  tt0 =  ( 1.e-6 + tt0 )*( 1.e-6 + tt0 );
  tt1 =  ( 1.e-6 + tt1 )*( 1.e-6 + tt1 );
  tt2 =  ( 1.e-6 + tt2 )*( 1.e-6 + tt2 );
  s0 =      tt1 * tt2 ;
  s1 = 6. * tt0 * tt2 ;
  s2 = 3. * tt0 * tt1 ;
  t00 = 1. / ( s0 + s1 + s2 );
  s0 = s0 * t00 ;
  s2 = s2 * t00;
  
  fromright = ( -g(l-2) + 7.*(g(l-1)+g(l)) - g(l+1) ) / 12. + ( s0*(t1-t0) + (0.5*s2-0.25)*(t2-t1) ) /3.;
  
  /*
    section (4.7) : choose flux depending on upwinding
  */
  _fh[l] = ( flux[0] > 0 ? fromleft : fromright );

  __syncthreads();

  /*
    section (4.8) : store variables later used for averaging, and set border fluxes to zero
  */
  if( threadIdx.x == 0 )
    {
      _ff0m = _fh[0];
      _fh[0] = 0; _fh[NW] = 0;
    }

  __syncthreads();

  /*
    section (4.9) : compute partial derivatives
  */
  if( l<NW )
    {
      GPU_rhs_pdf( nu,p,i,l,m ) -= ( _fh[l+1]-_fh[l] )*DWM1;
    }

  __syncthreads(); // this might be pointless...
  /*
    end section (4)
  */

  /*
    begin section (5) : compute the values of _fh[] for the symmetric m
  */
  
  /*
    section (5.1) : take new angle
  */
  m = m+NPHI2;

  /*
    section (5.2) : set g[] for internal points
  */
  if( l<NW )
    {
      g(l) = -flux[l]*GPU_pdf( nu,p,i,l,m,STAGE );
    }
		  
  __syncthreads();

  /*
    section (5.3) : set boundary condtions
  */
  if( l>=1 && l<=3 )
    {
      g(-l) = -flux[l-1]*GPU_pdf( nu,p,i,l-1,(m+NPHI/2)%NPHI,STAGE );
      g( NW-1+l ) = g( NW-1 );
    }
	  
  __syncthreads();

  /*
    section (5.4) : set auxiliary variables
  */
  // for(int l = 0; l <= NW; ++l)
  g_m2m3 = g( l-2 ) - g( l-3 );
  g_m1m2 = g( l-1 ) - g( l-2 );
  g_0m1  = g( l   ) - g( l-1 );
  g_p10  = g( l+1 ) - g( l   );
  g_p2p1 = g( l+2 ) - g( l+1 );
  
  /*
    section (5.5) : compute flux "from left"
  */
  // FROM LEFT
  t0 = g_m2m3 - g_m1m2;
  t1 = g_m1m2 - g_0m1;
  t2 = g_0m1 - g_p10;
  
  tt0 = 13. * t0*t0 + 3. * (    g_m2m3 - 3*g_m1m2 ) * (    g_m2m3 - 3*g_m1m2 );
  tt1 = 13. * t1*t1 + 3. * (    g_m1m2 +   g_0m1 ) * (    g_m1m2 +   g_0m1 );
  tt2 = 13. * t2*t2 + 3. * ( 3.*g_0m1 -   g_p10 ) * ( 3.*g_0m1 -   g_p10 );
  
  tt0 =  ( 1.e-6 + tt0 )*( 1.e-6 + tt0 );
  tt1 =  ( 1.e-6 + tt1 )*( 1.e-6 + tt1 );
  tt2 =  ( 1.e-6 + tt2 )*( 1.e-6 + tt2 );
  s0 =      tt1 * tt2;
  s1 = 6. * tt0 * tt2;
  s2 = 3. * tt0 * tt1;
  
  t00 = 1. / ( s0 + s1 + s2 );
  s0 = s0 * t00;
  s2 = s2 * t00;
  
  fromleft = ( -g(l-2) + 7.*(g(l-1)+g(l)) - g(l+1) ) / 12. + ( s0*(t1-t0) + (0.5*s2-0.25)*(t2-t1) ) /3.; 
  
  /*
    section (5.6) : compute flux "from right"
  */
  // FROM RIGHT
  t0 = (-g_p2p1) - (-g_p10);
  t1 = (-g_p10) - (-g_0m1);
  t2 = (-g_0m1) - (-g_m1m2);
  
  tt0 = 13. * t0*t0 + 3. * (    (-g_p2p1) - 3.*(-g_p10) ) * (    (-g_p2p1) - 3.*(-g_p10) );
  tt1 = 13. * t1*t1 + 3. * (    (-g_p10) +    (-g_0m1) ) * (    (-g_p10) +    (-g_0m1) );
  tt2 = 13. * t2*t2 + 3. * ( 3.*(-g_0m1) -    (-g_m1m2) ) * ( 3.*(-g_0m1) -    (-g_m1m2) );
  
  tt0 =  ( 1.e-6 + tt0 )*( 1.e-6 + tt0 );
  tt1 =  ( 1.e-6 + tt1 )*( 1.e-6 + tt1 );
  tt2 =  ( 1.e-6 + tt2 )*( 1.e-6 + tt2 );
  s0 =      tt1 * tt2 ;
  s1 = 6. * tt0 * tt2 ;
  s2 = 3. * tt0 * tt1 ;
  t00 = 1. / ( s0 + s1 + s2 );
  s0 = s0 * t00 ;
  s2 = s2 * t00;
  
  fromright = ( -g(l-2) + 7.*(g(l-1)+g(l)) - g(l+1) ) / 12. + ( s0*(t1-t0) + (0.5*s2-0.25)*(t2-t1) ) /3.;

  /*
    section (5.7) : choose flux depending on upwinding
  */
  _fh[l] = ( -flux[0] > 0 ? fromleft : fromright );
	    
  __syncthreads();

  /*
    section (5.8) : store variables for averaging and set border fluxes to zero
  */
  if( threadIdx.x == 0 )
    {
      _ff0mspec = _fh[0];
      _fh[0] = 0; _fh[NW] = 0;
    }
	    
  __syncthreads();

  /*
    section (5.9) : compute partial derivatives
  */
  if( l<NW )
    {
      GPU_rhs_pdf( nu,p,i,l,m ) -= ( _fh[l+1]-_fh[l] )*DWM1;
    }

  __syncthreads();
  /*
    end section (5)
  */

  /*
    begin section (6) : averaging for mass conservation
  */
  if( threadIdx.x == 0 )
    {
      GPU_rhs_pdf( nu,p,i,0,m ) += .5*( _ff0mspec-_ff0m )*DWM1;
      m = m-NPHI2;
      GPU_rhs_pdf( nu,p,i,0,m ) += .5*( _ff0m-_ff0mspec )*DWM1;
    }
  /*
    end section (6)
  */
}


#endif

void MOSFETProblemCuda::CPU_WENO_W( const idT STAGE, double *test_fh )
{
  const idT NSBN  = host_dm -> get_NSBN();
  const idT NX    = host_dm -> get_X()   -> get_N();
  const idT NW    = host_dm -> get_W()   -> get_N();
  const idT NPHI  = host_dm -> get_PHI() -> get_N();

  const double DW = host_dm->get_W()->get_delta();
  const double DWM1 = host_dm->get_W()->get_delta_m1();
  const double DPHI = host_dm->get_PHI()->get_delta();

#pragma omp parallel for
  for(idT global_index=0; global_index < _NVALLEYS*NSBN*NX*NPHI/2; ++global_index)
    {
      const idT NPHI2 = NPHI/2;

      idT i, nu, p, m;
      GPU_map_1D_to_4D( global_index, &i, NX, &nu, _NVALLEYS, &p, NSBN, &m, NPHI2 );

      double _ff0m, _ff0mspec;
      const double dep  = deps_dx(nu,p,i);
      double flux[NW];

      double wl;
      double aux;
      double aux3;
      for( idT l=0; l<NW; ++l )
	{
	  wl   = (l+0.5)*DW;
	  aux  = host_rescpar->get_epsstar()*_kane[nu]*wl;
	  aux3 = sqrt(2.*wl*(1.+aux));
	  flux[l] = -dep*aux3*cos(m*DPHI)/(sqrtmass(Xdim,nu)*(1.+2.*aux));
	}

      double _g [NW+6];
      double _fh[NW+1];
      double g_m2m3;
      double g_m1m2;
      double g_0m1;
      double g_p10;
      double g_p2p1;
      double t0;
      double t1;
      double t2;
      double tt0;
      double tt1;
      double tt2;
      double s0;
      double s1;
      double s2;
      double t00;
      double fromleft, fromright;

      // for one m
      {
    	for( idT l=0; l<NW; ++l )
    	  g(l) = flux[l]*pdf( nu,p,i,l,m,STAGE );
		  
    	for( idT l=1; l<=3; ++l )
    	  {
    	    g(-l) = flux[l-1]*pdf( nu,p,i,l-1,(m+NPHI/2)%NPHI,STAGE );
    	    g( NW-1+l ) = g( NW-1 );
    	  }

      	for(idT l = 0; l <= NW; ++l)
      	  {
      	    g_m2m3 = g( l-2 ) - g( l-3 );
      	    g_m1m2 = g( l-1 ) - g( l-2 );
      	    g_0m1  = g( l   ) - g( l-1 );
      	    g_p10  = g( l+1 ) - g( l   );
      	    g_p2p1 = g( l+2 ) - g( l+1 );

      	    // FROM LEFT
      	    t0 = g_m2m3 - g_m1m2;
      	    t1 = g_m1m2 - g_0m1;
      	    t2 = g_0m1 - g_p10;

      	    tt0 = 13. * t0*t0 + 3. * (    g_m2m3 - 3*g_m1m2 ) * (    g_m2m3 - 3*g_m1m2 );
      	    tt1 = 13. * t1*t1 + 3. * (    g_m1m2 +   g_0m1 ) * (    g_m1m2 +   g_0m1 );
      	    tt2 = 13. * t2*t2 + 3. * ( 3.*g_0m1 -   g_p10 ) * ( 3.*g_0m1 -   g_p10 );
 
      	    tt0 =  ( 1.e-6 + tt0 )*( 1.e-6 + tt0 );
      	    tt1 =  ( 1.e-6 + tt1 )*( 1.e-6 + tt1 );
      	    tt2 =  ( 1.e-6 + tt2 )*( 1.e-6 + tt2 );
      	    s0 =      tt1 * tt2;
      	    s1 = 6. * tt0 * tt2;
      	    s2 = 3. * tt0 * tt1;
	    
      	    t00 = 1. / ( s0 + s1 + s2 );
      	    s0 = s0 * t00;
      	    s2 = s2 * t00;

      	    fromleft = ( -g(l-2) + 7.*(g(l-1)+g(l)) - g(l+1) ) / 12. + ( s0*(t1-t0) + (0.5*s2-0.25)*(t2-t1) ) /3.; 

      	    // FROM RIGHT 
      	    t0 = (-g_p2p1) - (-g_p10);
      	    t1 = (-g_p10) - (-g_0m1);
      	    t2 = (-g_0m1) - (-g_m1m2);
  
      	    tt0 = 13. * t0*t0 + 3. * (    (-g_p2p1) - 3.*(-g_p10) ) * (    (-g_p2p1) - 3.*(-g_p10) );
      	    tt1 = 13. * t1*t1 + 3. * (    (-g_p10) +    (-g_0m1) ) * (    (-g_p10) +    (-g_0m1) );
      	    tt2 = 13. * t2*t2 + 3. * ( 3.*(-g_0m1) -    (-g_m1m2) ) * ( 3.*(-g_0m1) -    (-g_m1m2) );
  
      	    tt0 =  ( 1.e-6 + tt0 )*( 1.e-6 + tt0 );
      	    tt1 =  ( 1.e-6 + tt1 )*( 1.e-6 + tt1 );
      	    tt2 =  ( 1.e-6 + tt2 )*( 1.e-6 + tt2 );
      	    s0 =      tt1 * tt2 ;
      	    s1 = 6. * tt0 * tt2 ;
      	    s2 = 3. * tt0 * tt1 ;
      	    t00 = 1. / ( s0 + s1 + s2 );
      	    s0 = s0 * t00 ;
      	    s2 = s2 * t00;

      	    fromright = ( -g(l-2) + 7.*(g(l-1)+g(l)) - g(l+1) ) / 12. + ( s0*(t1-t0) + (0.5*s2-0.25)*(t2-t1) ) /3.;

      	    _fh[l] = ( flux[0] > 0 ? fromleft : fromright );
      	  }

      	_ff0m = _fh[0];
      	_fh[0] = 0; _fh[NW] = 0;
	    
      	for( idT l=0; l<NW; ++l )
      	  rhs_pdf( nu,p,i,l,m ) -= ( _fh[l+1]-_fh[l] )*DWM1;
      }
      
      m = m+NPHI2;
      // for the other m
      {
      	for( idT l=0; l<NW; ++l )
      	  g(l) = -flux[l]*pdf( nu,p,i,l,m,STAGE );
		  
      	for( idT l=1; l<=3; ++l )
      	  {
      	    g(-l) = -flux[l-1]*pdf( nu,p,i,l-1,(m+NPHI/2)%NPHI,STAGE );
      	    g( NW-1+l ) = g( NW-1 );
      	  }
	  
      	for(idT l = 0; l <= NW; ++l)
      	  {
      	    g_m2m3 = g( l-2 ) - g( l-3 );
      	    g_m1m2 = g( l-1 ) - g( l-2 );
      	    g_0m1  = g( l   ) - g( l-1 );
      	    g_p10  = g( l+1 ) - g( l   );
      	    g_p2p1 = g( l+2 ) - g( l+1 );

      	    // FROM LEFT
      	    t0 = g_m2m3 - g_m1m2;
      	    t1 = g_m1m2 - g_0m1;
      	    t2 = g_0m1 - g_p10;

      	    tt0 = 13. * t0*t0 + 3. * (    g_m2m3 - 3*g_m1m2 ) * (    g_m2m3 - 3*g_m1m2 );
      	    tt1 = 13. * t1*t1 + 3. * (    g_m1m2 +   g_0m1 ) * (    g_m1m2 +   g_0m1 );
      	    tt2 = 13. * t2*t2 + 3. * ( 3.*g_0m1 -   g_p10 ) * ( 3.*g_0m1 -   g_p10 );
 
      	    tt0 =  ( 1.e-6 + tt0 )*( 1.e-6 + tt0 );
      	    tt1 =  ( 1.e-6 + tt1 )*( 1.e-6 + tt1 );
      	    tt2 =  ( 1.e-6 + tt2 )*( 1.e-6 + tt2 );
      	    s0 =      tt1 * tt2;
      	    s1 = 6. * tt0 * tt2;
      	    s2 = 3. * tt0 * tt1;
	    
      	    t00 = 1. / ( s0 + s1 + s2 );
      	    s0 = s0 * t00;
      	    s2 = s2 * t00;

      	    fromleft = ( -g(l-2) + 7.*(g(l-1)+g(l)) - g(l+1) ) / 12. + ( s0*(t1-t0) + (0.5*s2-0.25)*(t2-t1) ) /3.; 

      	    // FROM RIGHT
      	    t0 = (-g_p2p1) - (-g_p10);
      	    t1 = (-g_p10) - (-g_0m1);
      	    t2 = (-g_0m1) - (-g_m1m2);
  
      	    tt0 = 13. * t0*t0 + 3. * (    (-g_p2p1) - 3.*(-g_p10) ) * (    (-g_p2p1) - 3.*(-g_p10) );
      	    tt1 = 13. * t1*t1 + 3. * (    (-g_p10) +    (-g_0m1) ) * (    (-g_p10) +    (-g_0m1) );
      	    tt2 = 13. * t2*t2 + 3. * ( 3.*(-g_0m1) -    (-g_m1m2) ) * ( 3.*(-g_0m1) -    (-g_m1m2) );
  
      	    tt0 =  ( 1.e-6 + tt0 )*( 1.e-6 + tt0 );
      	    tt1 =  ( 1.e-6 + tt1 )*( 1.e-6 + tt1 );
      	    tt2 =  ( 1.e-6 + tt2 )*( 1.e-6 + tt2 );
      	    s0 =      tt1 * tt2 ;
      	    s1 = 6. * tt0 * tt2 ;
      	    s2 = 3. * tt0 * tt1 ;
      	    t00 = 1. / ( s0 + s1 + s2 );
      	    s0 = s0 * t00 ;
      	    s2 = s2 * t00;

      	    fromright = ( -g(l-2) + 7.*(g(l-1)+g(l)) - g(l+1) ) / 12. + ( s0*(t1-t0) + (0.5*s2-0.25)*(t2-t1) ) /3.;

      	    _fh[l] = ( -flux[0] > 0 ? fromleft : fromright );
      	  }
	    
      	_ff0mspec = _fh[0];
      	_fh[0] = 0; _fh[NW] = 0;
	    
      	for( idT l=0; l<NW; ++l )
      	  rhs_pdf( nu,p,i,l,m ) -= ( _fh[l+1]-_fh[l] )*DWM1;
      }

      rhs_pdf( nu,p,i,0,m ) += .5*( _ff0mspec-_ff0m )*DWM1;
      m = m-NPHI2;
      rhs_pdf( nu,p,i,0,m ) += .5*( _ff0m-_ff0mspec )*DWM1;
    }

  return;
}

void MOSFETProblemCuda::WENO_W( const idT STAGE )
{
#ifdef _SHOW_ETA
  ostringstream message;
  message << "called " << __func__;
  show_eta(message.str());
#endif

  start_time( _PHASE_BTE_WENOW );

#ifdef __CUDACC__
  const idT NSBN  = host_dm -> get_NSBN();
  const idT NX    = host_dm -> get_X()   -> get_N();
  const idT NW    = host_dm -> get_W()   -> get_N();
  const idT NPHI  = host_dm -> get_PHI() -> get_N();

  checkCudaErrors( cudaMemcpyToSymbol(_GPU_sqrtmass,  _sqrtmass, 9*sizeof(double), 0, cudaMemcpyHostToDevice) );

  const idT gridSize      = host_gridconfig -> cuda_WENO_W_20230524_config -> get_gridSize();
  const idT blockSize     = host_gridconfig -> cuda_WENO_W_20230524_config -> get_blockSize();
  const idT shmemSize     = host_gridconfig -> cuda_WENO_W_20230524_config -> get_shmemSize();
  const cudaFuncCache cfc = host_gridconfig -> cuda_WENO_W_20230524_config -> get_cfc();
  cudaDeviceSetCacheConfig( cfc );
  cuda_WENO_W_20230524 <<< gridSize, blockSize, shmemSize >>> ( device_dm, device_rescpar, _GPU_pdf, _GPU_rhs_pdf, _GPU_deps_dx, STAGE );

#else
  CPU_WENO_W ( STAGE );
#endif

  stop_time( _PHASE_BTE_WENOW );
  
#ifdef _SHOW_ETA
  cerr << "[ok] (function " << __func__ << ", line " << __LINE__ << endl;
#endif

  return;
}


/**********************************
 *      x-PARTIAL DERIVATIVE      *
 **********************************/
void MOSFETProblemCuda::WENO_X( const idT STAGE )
{
#ifdef _SHOW_ETA
  ostringstream message;
  message << "called " << __func__;
  show_eta(message.str());
#endif

  start_time( _PHASE_BTE_WENOX );
  
#ifdef __CUDACC__
  const idT NSBN  = host_dm -> get_NSBN();
  const idT NX    = host_dm -> get_X()   -> get_N();
  const idT NW    = host_dm -> get_W()   -> get_N();
  const idT NPHI  = host_dm -> get_PHI() -> get_N();

  checkCudaErrors( cudaMemcpyToSymbol(_GPU_mass,      _mass,         9*sizeof(double), 0, cudaMemcpyHostToDevice) );
  checkCudaErrors( cudaMemcpyToSymbol(_GPU_kane,      _kane,         3*sizeof(double), 0, cudaMemcpyHostToDevice) );

  const idT gridSize      = host_gridconfig -> cuda_WENO_X_20230525_config -> get_gridSize();
  const idT blockSize     = host_gridconfig -> cuda_WENO_X_20230525_config -> get_blockSize();
  const idT shmemSize     = host_gridconfig -> cuda_WENO_X_20230525_config -> get_shmemSize();
  const cudaFuncCache cfc = host_gridconfig -> cuda_WENO_X_20230525_config -> get_cfc();
  cudaDeviceSetCacheConfig( cfc );
  cuda_WENO_X_20230525 <<< gridSize, blockSize, shmemSize >>> ( device_dm, _GPU_pdf, _GPU_rhs_pdf, _GPU_surfdens, _GPU_surfdens_eq, STAGE, NULL, _GPU_a1, _GPU_vel );

#else
  CPU_WENO_X ( STAGE );
#endif

  stop_time( _PHASE_BTE_WENOX );

#ifdef _SHOW_ETA
  cerr << "[ok] (function " << __func__ << ", line " << __LINE__ << endl;
#endif

  return;
}


#ifdef __CUDACC__
__global__ void cuda_WENO_X_20230525( const discrMeshes *dm, const double * _GPU_pdf, double *_GPU_rhs_pdf, const double *_GPU_surfdens, const double *_GPU_surfdens_eq, const idT STAGE, double *test_fh, const double *_GPU_a1, const double *_GPU_vel )
{
  idT global_index   = blockIdx.x*blockDim.x + threadIdx.x;
  const idT NSBN     = dm -> get_NSBN();
  const idT NX       = dm -> get_X()   -> get_N();
  const idT NW       = dm -> get_W()   -> get_N();
  const idT NPHI     = dm -> get_PHI() -> get_N();
  const double DX    = dm -> get_X()   -> get_delta();
  const double DXM1  = dm -> get_X()   -> get_delta_m1();
                                                                 /*
								   Global index for the thread, 
								   and "external" values of the discretization meshes.
								 */

  extern __shared__ double sm[];
  double *_g   = &sm[0];
  double *_fh  = &sm[NX+6];
                                                                 /*
								   (i)  the values for flux _g                 <--- NX+6 values
								   (ii) the values for flux _fh reconstruction <--- NX+1 values
								 */

  /*
    begin section (1) : split the global index into local indices
  */
  idT nu, p, i, l, m;
  GPU_map_1D_to_5D( global_index, &nu, _NVALLEYS, &p, NSBN, &l, NW, &m, NPHI, &i, NX+1 );
  /*
    end section (1)
  */

  /*
    section (2.1) : introduce g[]
  */
  if( i<NX )
    {
      g(i) = GPU_a1(nu,l,m)*GPU_pdf( nu,p,i,l,m,STAGE );
    }
  __syncthreads();

  /*
    section (2.2) : introduce boundary conditions
  */
  if( i>=1 && i<=3 )
    {
      g(    -i) = ( GPU_vel(Xdim,nu,l,m) >= 0 ? GPU_a1(nu,l,m)*GPU_surfdens_eq( nu,p,    0 )/GPU_surfdens( nu,p,    0 )*GPU_pdf( nu,p,    0,l,m,STAGE ) : g(    0) );
      g(NX-1+i) = ( GPU_vel(Xdim,nu,l,m) <= 0 ? GPU_a1(nu,l,m)*GPU_surfdens_eq( nu,p, NX-1 )/GPU_surfdens( nu,p, NX-1 )*GPU_pdf( nu,p, NX-1,l,m,STAGE ) : g( NX-1) );
    }
  __syncthreads();

  /*
    section (2.3) : declare some auxiliary local variables
  */
  double g_m2m3;
  double g_m1m2;
  double g_0m1;
  double g_p10;
  double g_p2p1;
  double t0;
  double t1;
  double t2;
  double tt0;
  double tt1;
  double tt2;
  double t00;
  double s0;
  double s1;
  double s2;
  
  /*
    section (2.4) : WENO uses upwinding
  */
  if( GPU_a1( nu,l,m ) >= 0 )
    {
      g_m2m3 = g(i-2)-g(i-3);
      g_m1m2 = g(i-1)-g(i-2);
      g_0m1  = g(i  )-g(i-1);
      g_p10  = g(i+1)-g(i  );
      g_p2p1 = g(i+2)-g(i+1);
      
      // FROM LEFT
      t0 = g_m2m3 - g_m1m2;
      t1 = g_m1m2 - g_0m1;
      t2 = g_0m1  - g_p10;
      
      tt0 = 13. * t0*t0 + 3. * (    g_m2m3 - 3*g_m1m2 ) * (    g_m2m3 - 3*g_m1m2 );
      tt1 = 13. * t1*t1 + 3. * (    g_m1m2 +    g_0m1 ) * (    g_m1m2 +    g_0m1 );
      tt2 = 13. * t2*t2 + 3. * (  3.*g_0m1 -    g_p10 ) * (  3.*g_0m1 -    g_p10 );
      
      tt0 =  ( 1.e-6 + tt0 )*( 1.e-6 + tt0 );
      tt1 =  ( 1.e-6 + tt1 )*( 1.e-6 + tt1 );
      tt2 =  ( 1.e-6 + tt2 )*( 1.e-6 + tt2 );
      s0 =      tt1 * tt2;
      s1 = 6. * tt0 * tt2;
      s2 = 3. * tt0 * tt1;
      
      t00 = 1. / ( s0 + s1 + s2 );
      s0 = s0 * t00;
      s2 = s2 * t00;
      
      _fh[i] = ( -g(i-2) + 7.*(g(i-1)+g(i)) - g(i+1) ) / 12. + ( s0*(t1-t0) + (0.5*s2-0.25)*(t2-t1) ) /3.; 
    }
  else
    {
      g_m2m3 = g(i-2)-g(i-3);
      g_m1m2 = g(i-1)-g(i-2);
      g_0m1  = g(i  )-g(i-1);
      g_p10  = g(i+1)-g(i  );
      g_p2p1 = g(i+2)-g(i+1);
      
      // FROM RIGHT
      t0 = (-g_p2p1) - (-g_p10);
      t1 = (-g_p10) - (-g_0m1);
      t2 = (-g_0m1) - (-g_m1m2);
      
      tt0 = 13. * t0*t0 + 3. * (    (-g_p2p1) - 3.*(-g_p10) ) * (    (-g_p2p1) - 3.*(-g_p10) );
      tt1 = 13. * t1*t1 + 3. * (    (-g_p10) +    (-g_0m1) ) * (    (-g_p10) +    (-g_0m1) );
      tt2 = 13. * t2*t2 + 3. * ( 3.*(-g_0m1) -    (-g_m1m2) ) * ( 3.*(-g_0m1) -    (-g_m1m2) );
      
      tt0 =  ( 1.e-6 + tt0 )*( 1.e-6 + tt0 );
      tt1 =  ( 1.e-6 + tt1 )*( 1.e-6 + tt1 );
      tt2 =  ( 1.e-6 + tt2 )*( 1.e-6 + tt2 );
      s0 =      tt1 * tt2 ;
      s1 = 6. * tt0 * tt2 ;
      s2 = 3. * tt0 * tt1 ;
      t00 = 1. / ( s0 + s1 + s2 );
      s0 = s0 * t00 ;
      s2 = s2 * t00;
      
      _fh[i] = ( -g(i-2) + 7.*(g(i-1)+g(i)) - g(i+1) ) / 12. + ( s0*(t1-t0) + (0.5*s2-0.25)*(t2-t1) ) /3.;
    }

  __syncthreads();

  /*
    section (3) : compute partial derivatives
  */
  if( i<NX )
    {
      GPU_rhs_pdf( nu,p,i,l,m ) -= ( _fh[i+1]-_fh[i] )*DXM1;
    }
}










#endif
void MOSFETProblemCuda::CPU_WENO_X( const idT STAGE, double *test_fh )
{
  const idT NSBN  = host_dm -> get_NSBN();
  const idT NX    = host_dm -> get_X()   -> get_N();
  const idT NW    = host_dm -> get_W()   -> get_N();
  const idT NPHI  = host_dm -> get_PHI() -> get_N();

  // const double DXM1 = 1./_dx;
  const double DXM1 = host_dm->get_X()->get_delta_m1();

#pragma omp parallel for
  for(idT global_index=0; global_index < _NVALLEYS*NSBN*NW*NPHI; ++global_index)
    {
      idT nu, p, l, m;
      GPU_map_1D_to_4D( global_index, &nu, _NVALLEYS, &p, NSBN, &l, NW, &m, NPHI );

      const double DPHI = host_dm->get_PHI()->get_delta();

      double _g [NX+6];
      double _fh[NX+1];

      // const double wl = (l+.5)*_dw;
      const double wl = host_dm->get_W()->mesh(l);

      const double wind = sqrt(2.*wl*(1.+host_rescpar->get_epsstar()*_kane[nu]*wl))*cos(m*DPHI)/(sqrt(mass(Xdim,nu))*(1.+2.*host_rescpar->get_epsstar()*_kane[nu]*wl));

      for( idT i=0; i<NX; ++i )
	g(i) = wind*pdf( nu,p,i,l,m,STAGE );

      for( idT i=1; i<=3; ++i )
	{
	  g(-i) = ( wind >= 0 ? wind*surfdens_eq( nu,p,0 )/surfdens( nu,p,0 )*pdf( nu,p,0,l,m,STAGE ) : g(0) );
	  g(NX-1+i) = ( wind <= 0 ? wind*surfdens_eq( nu,p,NX-1 )/surfdens( nu,p,NX-1 )*pdf( nu,p,NX-1,l,m,STAGE ) : g(NX-1) );
	}

      double g_m2m3;
      double g_m1m2;
      double g_0m1;
      double g_p10;
      double g_p2p1;
      double t0;
      double t1;
      double t2;
      double tt0;
      double tt1;
      double tt2;
      double t00;
      double s0;
      double s1;
      double s2;
      double fromleft, fromright;

      // WENO
      for(idT i = 0; i <= NX; ++i)
      	{
      	  // FROM LEFT
      	  g_m2m3 = g(i-2)-g(i-3);
      	  g_m1m2 = g(i-1)-g(i-2);
      	  g_0m1  = g(i  )-g(i-1);
      	  g_p10  = g(i+1)-g(i  );
      	  g_p2p1 = g(i+2)-g(i+1);

      	  t0 = g_m2m3 - g_m1m2;
      	  t1 = g_m1m2 - g_0m1;
      	  t2 = g_0m1  - g_p10;

      	  tt0 = 13. * t0*t0 + 3. * (    g_m2m3 - 3*g_m1m2 ) * (    g_m2m3 - 3*g_m1m2 );
      	  tt1 = 13. * t1*t1 + 3. * (    g_m1m2 +    g_0m1 ) * (    g_m1m2 +    g_0m1 );
      	  tt2 = 13. * t2*t2 + 3. * (  3.*g_0m1 -    g_p10 ) * (  3.*g_0m1 -    g_p10 );
 
      	  tt0 =  ( 1.e-6 + tt0 )*( 1.e-6 + tt0 );
      	  tt1 =  ( 1.e-6 + tt1 )*( 1.e-6 + tt1 );
      	  tt2 =  ( 1.e-6 + tt2 )*( 1.e-6 + tt2 );
      	  s0 =      tt1 * tt2;
      	  s1 = 6. * tt0 * tt2;
      	  s2 = 3. * tt0 * tt1;
	    
      	  t00 = 1. / ( s0 + s1 + s2 );
      	  s0 = s0 * t00;
      	  s2 = s2 * t00;

      	  fromleft = ( -g(i-2) + 7.*(g(i-1)+g(i)) - g(i+1) ) / 12. + ( s0*(t1-t0) + (0.5*s2-0.25)*(t2-t1) ) /3.; 

      	  // FROM RIGHT
      	  t0 = (-g_p2p1) - (-g_p10);
      	  t1 = (-g_p10) - (-g_0m1);
      	  t2 = (-g_0m1) - (-g_m1m2);
  
      	  tt0 = 13. * t0*t0 + 3. * (    (-g_p2p1) - 3.*(-g_p10) ) * (    (-g_p2p1) - 3.*(-g_p10) );
      	  tt1 = 13. * t1*t1 + 3. * (    (-g_p10) +    (-g_0m1) ) * (    (-g_p10) +    (-g_0m1) );
      	  tt2 = 13. * t2*t2 + 3. * ( 3.*(-g_0m1) -    (-g_m1m2) ) * ( 3.*(-g_0m1) -    (-g_m1m2) );
  
      	  tt0 =  ( 1.e-6 + tt0 )*( 1.e-6 + tt0 );
      	  tt1 =  ( 1.e-6 + tt1 )*( 1.e-6 + tt1 );
      	  tt2 =  ( 1.e-6 + tt2 )*( 1.e-6 + tt2 );
      	  s0 =      tt1 * tt2 ;
      	  s1 = 6. * tt0 * tt2 ;
      	  s2 = 3. * tt0 * tt1 ;
      	  t00 = 1. / ( s0 + s1 + s2 );
      	  s0 = s0 * t00 ;
      	  s2 = s2 * t00;

      	  fromright = ( -g(i-2) + 7.*(g(i-1)+g(i)) - g(i+1) ) / 12. + ( s0*(t1-t0) + (0.5*s2-0.25)*(t2-t1) ) /3.;

      	  _fh[i] = ( wind >= 0 ? fromleft : fromright );

      	}

	for( idT i=0; i<NX; ++i )
	  rhs_pdf( nu,p,i,l,m ) -= ( _fh[i+1]-_fh[i] )*DXM1;
    }

  return;
}

/**********************************
 *     phi-PARTIAL DERIVATIVE     *
 **********************************/
#ifdef __CUDACC__
__global__ void cuda_WENO_PHI( const discrMeshes *dm, const double *_GPU_pdf, double *_GPU_rhs_pdf, const double *_GPU_deps_dx, const double MAX_A3, const idT STAGE, const double *_GPU_a3, double *test_fh )
{
  extern __shared__ double sg[];

  const idT NSBN     = dm->get_NSBN();
  const idT NX       = dm->get_X()->get_N();
  const idT NW       = dm->get_W()->get_N();
  const idT NPHI     = dm->get_PHI()->get_N();

  idT global_index = blockIdx.x*blockDim.x + threadIdx.x;

  if( global_index < _NVALLEYS*NSBN*NX*NW*NPHI )
    {
      const double DPHIM1 = dm->get_PHI()->get_delta_m1();

      /*******************************************
       *            BLOCK'S INFORMATION          *
       *******************************************/
      const idT block_first_idx = blockIdx.x*blockDim.x;
      const idT gap_sx          = block_first_idx%NPHI;
      const idT block_last_idx  = block_first_idx + blockDim.x - 1;
      const idT gap_dx          = block_last_idx%NPHI;
      const idT N_s             = block_last_idx/NPHI - block_first_idx/NPHI + 1;
      const idT sm_sz           = N_s*(NPHI+6);
      double *smpos             = (double*)&sg[0];
      double *smneg             = (double*)&sg[sm_sz];

      /*******************************************
       *           THREAD'S INFORMATION          *
       *******************************************/
      idT nu,p,i,l,m;
      GPU_map_1D_to_5D( global_index, &i, NX, &nu, _NVALLEYS, &p, NSBN, &l, NW, &m, NPHI );
      const idT sec_ind = (gap_sx + threadIdx.x)/NPHI;
      idT sm_idx        = gap_sx + 3+6*sec_ind + threadIdx.x;

      /*******************************************
       *     LOAD DATA INTO THE SHARED MEMORY    *
       *******************************************/
      const double flux = GPU_a3 ( nu,p,i,l,m       );
      const double pdf  = GPU_pdf( nu,p,i,l,m,STAGE );
      smpos[ sm_idx ] = 0.5*( flux+MAX_A3 )*pdf;
      smneg[ sm_idx ] = 0.5*( flux-MAX_A3 )*pdf;

      // boundary conditions
      idT idx = threadIdx.x;
      if( idx < N_s*3 )
      	{
      	  const idT s = idx/3;
      	  const idT M = idx-s*3 + 1;
      	  const idT m_start = ( s == 0     ? gap_sx : 0       );
      	  const idT m_stop  = ( s == N_s-1 ? gap_dx : NPHI-1 );

	  const idT glob_idx_s = block_first_idx + s*NPHI;
	  idT NU,P,I,L,_aux;
	  GPU_map_1D_to_5D( glob_idx_s, &I, NX, &NU, _NVALLEYS, &P, NSBN, &L, NW, &_aux, NPHI );

      	  //left
      	  idT loc_m = m_start-M;
      	  double loc_flux = GPU_a3( NU,P,I,L,(loc_m+NPHI)%NPHI );
      	  double loc_pdf = GPU_pdf( NU,P,I,L,(loc_m+NPHI)%NPHI,STAGE );
      	  smpos[ 3+s*(NPHI+6) + loc_m ] = 0.5*( loc_flux+MAX_A3 )*loc_pdf;
      	  smneg[ 3+s*(NPHI+6) + loc_m ] = 0.5*( loc_flux-MAX_A3 )*loc_pdf;

      	  //right
      	  loc_m = m_stop+M;
      	  loc_flux = GPU_a3( NU,P,I,L,loc_m%NPHI );
      	  loc_pdf = GPU_pdf( NU,P,I,L,loc_m%NPHI,STAGE );
      	  smpos[ 3+s*(NPHI+6) + loc_m ] = 0.5*( loc_flux+MAX_A3 )*loc_pdf;
      	  smneg[ 3+s*(NPHI+6) + loc_m ] = 0.5*( loc_flux-MAX_A3 )*loc_pdf;
      	}
      
      __syncthreads();

      /*******************************************
       *     NOW PROCEED WITH COMPUTATIONS       *
       *******************************************/
      double _fh[2];

      // POSITIVE FLUX
      // compute _fh[0]
      double g_m2m3 = smpos[ sm_idx-2 ] - smpos[ sm_idx-3 ];
      double g_m1m2 = smpos[ sm_idx-1 ] - smpos[ sm_idx-2 ];
      double g_0m1  = smpos[ sm_idx   ] - smpos[ sm_idx-1 ];
      double g_p10  = smpos[ sm_idx+1 ] - smpos[ sm_idx   ];
      double g_p2p1 = smpos[ sm_idx+2 ] - smpos[ sm_idx+1 ];
      double g_p3p2 = smpos[ sm_idx+3 ] - smpos[ sm_idx+2 ];
      {
      	double t0 = g_m2m3 - g_m1m2;
      	double t1 = g_m1m2 - g_0m1;
      	double t2 = g_0m1 - g_p10;
	
      	double tt0 = 13. * t0*t0 + 3. * (    g_m2m3 - 3*g_m1m2 ) * (    g_m2m3 - 3*g_m1m2 );
      	double tt1 = 13. * t1*t1 + 3. * (    g_m1m2 +   g_0m1 ) * (    g_m1m2 +   g_0m1 );
      	double tt2 = 13. * t2*t2 + 3. * ( 3.*g_0m1 -   g_p10 ) * ( 3.*g_0m1 -   g_p10 );
	
      	tt0 =  ( 1.e-6 + tt0 )*( 1.e-6 + tt0 );
      	tt1 =  ( 1.e-6 + tt1 )*( 1.e-6 + tt1 );
      	tt2 =  ( 1.e-6 + tt2 )*( 1.e-6 + tt2 );
      	double s0 =      tt1 * tt2;
      	double s1 = 6. * tt0 * tt2;
      	double s2 = 3. * tt0 * tt1;
	
      	double t00 = 1. / ( s0 + s1 + s2 );
      	s0 = s0 * t00;
      	s2 = s2 * t00;

      	_fh[0] = ( -smpos[sm_idx-2] + 7.*(smpos[sm_idx-1] + smpos[sm_idx]) - smpos[sm_idx+1] ) / 12. + ( s0*(t1-t0) + (0.5*s2-0.25)*(t2-t1) ) /3.; 
      }

      // compute _fh[1]
      {
      	++sm_idx;
      	double t0 = g_m1m2 - g_0m1;
      	double t1 = g_0m1 - g_p10;
      	double t2 = g_p10 - g_p2p1;
	
      	double tt0 = 13. * t0*t0 + 3. * (    g_m1m2 - 3*g_0m1 ) * (    g_m1m2 - 3*g_0m1 );
      	double tt1 = 13. * t1*t1 + 3. * (    g_0m1 +   g_p10 ) * (    g_0m1 +   g_p10 );
      	double tt2 = 13. * t2*t2 + 3. * ( 3.*g_p10 -   g_p2p1 ) * ( 3.*g_p10 -   g_p2p1 );
	
      	tt0 =  ( 1.e-6 + tt0 )*( 1.e-6 + tt0 );
      	tt1 =  ( 1.e-6 + tt1 )*( 1.e-6 + tt1 );
      	tt2 =  ( 1.e-6 + tt2 )*( 1.e-6 + tt2 );
      	double s0 =      tt1 * tt2;
      	double s1 = 6. * tt0 * tt2;
      	double s2 = 3. * tt0 * tt1;
	
      	double t00 = 1. / ( s0 + s1 + s2 );
      	s0 = s0 * t00;
      	s2 = s2 * t00;

      	_fh[1] = ( -smpos[sm_idx-2] + 7.*(smpos[sm_idx-1] + smpos[sm_idx]) - smpos[sm_idx+1] ) / 12. + ( s0*(t1-t0) + (0.5*s2-0.25)*(t2-t1) ) /3.; 
      }

      // NEGATIVE FLUX
      /*******************************************
       *     NOW PROCEED WITH COMPUTATIONS       *
       *******************************************/
      // compute _fh[0]
      --sm_idx;
      g_m2m3 = smneg[ sm_idx-2 ] - smneg[ sm_idx-3 ];
      g_m1m2 = smneg[ sm_idx-1 ] - smneg[ sm_idx-2 ];
      g_0m1  = smneg[ sm_idx   ] - smneg[ sm_idx-1 ];
      g_p10  = smneg[ sm_idx+1 ] - smneg[ sm_idx   ];
      g_p2p1 = smneg[ sm_idx+2 ] - smneg[ sm_idx+1 ];
      g_p3p2 = smneg[ sm_idx+3 ] - smneg[ sm_idx+2 ];
      {
	double t0 = (-g_p2p1) - (-g_p10);
	double t1 = (-g_p10) - (-g_0m1);
	double t2 = (-g_0m1) - (-g_m1m2);
	
	double tt0 = 13. * t0*t0 + 3. * (    (-g_p2p1) - 3.*(-g_p10) ) * (    (-g_p2p1) - 3.*(-g_p10) );
	double tt1 = 13. * t1*t1 + 3. * (    (-g_p10) +    (-g_0m1) ) * (    (-g_p10) +    (-g_0m1) );
	double tt2 = 13. * t2*t2 + 3. * ( 3.*(-g_0m1) -    (-g_m1m2) ) * ( 3.*(-g_0m1) -    (-g_m1m2) );
	
      	tt0 =  ( 1.e-6 + tt0 )*( 1.e-6 + tt0 );
      	tt1 =  ( 1.e-6 + tt1 )*( 1.e-6 + tt1 );
      	tt2 =  ( 1.e-6 + tt2 )*( 1.e-6 + tt2 );

      	double s0 =      tt1 * tt2;
      	double s1 = 6. * tt0 * tt2;
      	double s2 = 3. * tt0 * tt1;
	
      	double t00 = 1. / ( s0 + s1 + s2 );

      	s0 = s0 * t00;
      	s2 = s2 * t00;

      	_fh[0] += ( -smneg[sm_idx-2] + 7.*(smneg[sm_idx-1] + smneg[sm_idx]) - smneg[sm_idx+1] ) / 12. + ( s0*(t1-t0) + (0.5*s2-0.25)*(t2-t1) ) /3.; 
      }

      // compute _fh[1]
      {
      	++sm_idx;
      	double t0 = (-g_p3p2) - (-g_p2p1);
      	double t1 = (-g_p2p1) - (-g_p10);
      	double t2 = (-g_p10) - (-g_0m1);
	
      	double tt0 = 13. * t0*t0 + 3. * (    (-g_p3p2) - 3*(-g_p2p1) ) * (    (-g_p3p2) - 3*(-g_p2p1) );
      	double tt1 = 13. * t1*t1 + 3. * (    (-g_p2p1) +   (-g_p10) ) * (    (-g_p2p1) +   (-g_p10) );
      	double tt2 = 13. * t2*t2 + 3. * ( 3.*(-g_p10) -   (-g_0m1) ) * ( 3.*(-g_p10) -   (-g_0m1) );
	
      	tt0 =  ( 1.e-6 + tt0 )*( 1.e-6 + tt0 );
      	tt1 =  ( 1.e-6 + tt1 )*( 1.e-6 + tt1 );
      	tt2 =  ( 1.e-6 + tt2 )*( 1.e-6 + tt2 );
      	double s0 =      tt1 * tt2;
      	double s1 = 6. * tt0 * tt2;
      	double s2 = 3. * tt0 * tt1;
	
      	double t00 = 1. / ( s0 + s1 + s2 );
      	s0 = s0 * t00;
      	s2 = s2 * t00;

      	_fh[1] += ( -smneg[sm_idx-2] + 7.*(smneg[sm_idx-1] + smneg[sm_idx]) - smneg[sm_idx+1] ) / 12. + ( s0*(t1-t0) + (0.5*s2-0.25)*(t2-t1) ) /3.; 
      }

      // compute partial derivative
      GPU_rhs_pdf( nu,p,i,l,m ) -= ( _fh[1]-_fh[0] )*DPHIM1;
    }
}
#endif

void MOSFETProblemCuda::CPU_WENO_PHI( const idT STAGE, double *test_fh )
{
  const idT NSBN  = host_dm -> get_NSBN();
  const idT NX    = host_dm -> get_X()   -> get_N();
  const idT NW    = host_dm -> get_W()   -> get_N();
  const idT NPHI  = host_dm -> get_PHI() -> get_N();

  const double DPHIM1 = host_dm->get_PHI()->get_delta_m1();
  const double one_over_twelve = 1./12.;
  const double one_third = 1./3.;

#pragma omp parallel for
  for(idT global_index=0; global_index < _NVALLEYS*NSBN*NX*NW; ++global_index)
    {
      idT i, nu, p, l;
      GPU_map_1D_to_4D( global_index, &i, NX, &nu, _NVALLEYS, &p, NSBN, &l, NW );

      double _g [NPHI+6];
      double _fh[NPHI+1];

      // FROM LEFT
      for( idT m=0; m<NPHI; ++m )
	g(m) = .5*(a3(nu,p,i,l,m) - _max_a3)*pdf( nu,p,i,l,m,STAGE );

      for( idT m=1; m<=3; ++m )
	{
	  g(-m) = g(NPHI-m);
	  g(NPHI-1+m) = g(m-1);
	}

      for(idT m = 0; m <= NPHI; ++m)
      	{
	  double g_im2 = g(m-3);
	  double g_im1 = g(m-2);
	  double g_i = g(m-1);
	  double g_ip1 = g(m);
	  double g_ip2 = g(m+1);

	  double fromleft = ( -g_im1 + 7*(g_i+g_ip1) - g_ip2 ) *one_over_twelve + ( (1./(((1.e-6+13. * ( 2.*g_im1-g_im2 - g_i )*( 2.*g_im1-g_im2 - g_i ) + 3. * ( 4.*g_im1-g_im2 - 3.*g_i) * ( 4.*g_im1-g_im2 - 3.*g_i)) * (1.e-6+13. * ( 2.*g_im1-g_im2 - g_i )*( 2.*g_im1-g_im2 - g_i ) + 3. * ( 4.*g_im1-g_im2 - 3.*g_i) * ( 4.*g_im1-g_im2 - 3.*g_i)))*(1./((1.e-6+13. * ( 2.*g_im1-g_im2 - g_i )*( 2.*g_im1-g_im2 - g_i ) + 3. * ( 4.*g_im1-g_im2 - 3.*g_i) * ( 4.*g_im1-g_im2 - 3.*g_i)) * (1.e-6+13. * ( 2.*g_im1-g_im2 - g_i )*( 2.*g_im1-g_im2 - g_i ) + 3. * ( 4.*g_im1-g_im2 - 3.*g_i) * ( 4.*g_im1-g_im2 - 3.*g_i)))+6./((1.e-6+13. * ( 2.*g_i-g_im1 - g_ip1 )*( 2.*g_i-g_im1 - g_ip1 ) + 3. * ( -g_im1+g_ip1 ) * ( -g_im1+g_ip1 )) * (1.e-6+13. * ( 2.*g_i-g_im1 - g_ip1 )*( 2.*g_i-g_im1 - g_ip1 ) + 3. * ( -g_im1+g_ip1 ) * ( -g_im1+g_ip1 )))+3./((1.e-6+13. * ( 2.*g_ip1-g_i - g_ip2 )*( 2.*g_ip1-g_i - g_ip2 ) + 3. * ( 4.*g_ip1-3.*g_i-g_ip2 )*( 4.*g_ip1-3.*g_i-g_ip2 ))*(1.e-6+13. * ( 2.*g_ip1-g_i - g_ip2 )*( 2.*g_ip1-g_i - g_ip2 ) + 3. * ( 4.*g_ip1-3.*g_i-g_ip2 )*( 4.*g_ip1-3.*g_i-g_ip2 ))))))*(3.*g_i- g_ip1-3.*g_im1+g_im2) + (0.5*(3./(((1.e-6+13. * ( 2.*g_ip1-g_i - g_ip2 )*( 2.*g_ip1-g_i - g_ip2 ) + 3. * ( 4.*g_ip1-3.*g_i-g_ip2 )*( 4.*g_ip1-3.*g_i-g_ip2 ))*(1.e-6+13. * ( 2.*g_ip1-g_i - g_ip2 )*( 2.*g_ip1-g_i - g_ip2 ) + 3. * ( 4.*g_ip1-3.*g_i-g_ip2 )*( 4.*g_ip1-3.*g_i-g_ip2 )))*(1./((1.e-6+13. * ( 2.*g_im1-g_im2 - g_i )*( 2.*g_im1-g_im2 - g_i ) + 3. * ( 4.*g_im1-g_im2 - 3.*g_i) * ( 4.*g_im1-g_im2 - 3.*g_i)) * (1.e-6+13. * ( 2.*g_im1-g_im2 - g_i )*( 2.*g_im1-g_im2 - g_i ) + 3. * ( 4.*g_im1-g_im2 - 3.*g_i) * ( 4.*g_im1-g_im2 - 3.*g_i)))+6./((1.e-6+13. * ( 2.*g_i-g_im1 - g_ip1 )*( 2.*g_i-g_im1 - g_ip1 ) + 3. * ( -g_im1+g_ip1 ) * ( -g_im1+g_ip1 )) * (1.e-6+13. * ( 2.*g_i-g_im1 - g_ip1 )*( 2.*g_i-g_im1 - g_ip1 ) + 3. * ( -g_im1+g_ip1 ) * ( -g_im1+g_ip1 )))+3./((1.e-6+13. * ( 2.*g_ip1-g_i - g_ip2 )*( 2.*g_ip1-g_i - g_ip2 ) + 3. * ( 4.*g_ip1-3.*g_i-g_ip2 )*( 4.*g_ip1-3.*g_i-g_ip2 ))*(1.e-6+13. * ( 2.*g_ip1-g_i - g_ip2 )*( 2.*g_ip1-g_i - g_ip2 ) + 3. * ( 4.*g_ip1-3.*g_i-g_ip2 )*( 4.*g_ip1-3.*g_i-g_ip2 ))))))-0.25)*(3.*g_ip1- g_ip2-3.*g_i+g_im1) ) *one_third ;

	  _fh[m] = fromleft;
      	}

      for( idT m=0; m<NPHI; ++m )
	rhs_pdf( nu,p,i,l,m ) -= ( _fh[m+1]-_fh[m] )*DPHIM1;
      
	// FROM RIGHT
	for( idT m=0; m<NPHI; ++m )
	  g(m) = .5*(a3(nu,p,i,l,m) + _max_a3)*pdf( nu,p,i,l,m,STAGE );

	for( idT m=1; m<=3; ++m )
	  {
	    g(-m) = g(NPHI-m);
	    g(NPHI-1+m) = g(m-1);
	  }

	// WENO
	for(idT m = 0; m <= NPHI; ++m)
	  {
	    double g_im1 = g(m-2);
	    double g_i = g(m-1);
	    double g_ip1 = g(m);
	    double g_ip2 = g(m+1);
	    double g_ip3 = g(m+2);
	    
	    double fromright = ( -g_im1 + 7*(g_i+g_ip1) - g_ip2 ) *one_over_twelve + ( (1./((( 1.e-6 + 13. * (-g_ip3+2.*g_ip2-g_ip1)*(-g_ip3+2.*g_ip2-g_ip1) + 3. * (-g_ip3+4.*g_ip2-3.*g_ip1)*(-g_ip3+4.*g_ip2-3.*g_ip1))*( 1.e-6 + 13. * (-g_ip3+2.*g_ip2-g_ip1)*(-g_ip3+2.*g_ip2-g_ip1) + 3. * (-g_ip3+4.*g_ip2-3.*g_ip1)*(-g_ip3+4.*g_ip2-3.*g_ip1)))*(1./(( 1.e-6 + 13. * (-g_ip3+2.*g_ip2-g_ip1)*(-g_ip3+2.*g_ip2-g_ip1) + 3. * (-g_ip3+4.*g_ip2-3.*g_ip1)*(-g_ip3+4.*g_ip2-3.*g_ip1))*( 1.e-6 + 13. * (-g_ip3+2.*g_ip2-g_ip1)*(-g_ip3+2.*g_ip2-g_ip1) + 3. * (-g_ip3+4.*g_ip2-3.*g_ip1)*(-g_ip3+4.*g_ip2-3.*g_ip1)))+6./(( 1.e-6 + 13. * (-g_ip2+2.*g_ip1-g_i)*(-g_ip2+2.*g_ip1-g_i) + 3. * (-g_ip2+g_i)*(-g_ip2+g_i) )*( 1.e-6 + 13. * (-g_ip2+2.*g_ip1-g_i)*(-g_ip2+2.*g_ip1-g_i) + 3. * (-g_ip2+g_i)*(-g_ip2+g_i) ))+3./(( 1.e-6 + 13. * (-g_ip1+2.*g_i-g_im1)*(-g_ip1+2.*g_i-g_im1) + 3. * ( -3.*g_ip1+4.*g_i-g_im1)*( -3.*g_ip1+4.*g_i-g_im1) )*( 1.e-6 + 13. * (-g_ip1+2.*g_i-g_im1)*(-g_ip1+2.*g_i-g_im1) + 3. * ( -3.*g_ip1+4.*g_i-g_im1)*( -3.*g_ip1+4.*g_i-g_im1) )))))*(3.*g_ip1-g_i+g_ip3-3.*g_ip2) + (0.5*(3./((( 1.e-6 + 13. * (-g_ip1+2.*g_i-g_im1)*(-g_ip1+2.*g_i-g_im1) + 3. * ( -3.*g_ip1+4.*g_i-g_im1)*( -3.*g_ip1+4.*g_i-g_im1) )*( 1.e-6 + 13. * (-g_ip1+2.*g_i-g_im1)*(-g_ip1+2.*g_i-g_im1) + 3. * ( -3.*g_ip1+4.*g_i-g_im1)*( -3.*g_ip1+4.*g_i-g_im1) ))*(1./(( 1.e-6 + 13. * (-g_ip3+2.*g_ip2-g_ip1)*(-g_ip3+2.*g_ip2-g_ip1) + 3. * (-g_ip3+4.*g_ip2-3.*g_ip1)*(-g_ip3+4.*g_ip2-3.*g_ip1))*( 1.e-6 + 13. * (-g_ip3+2.*g_ip2-g_ip1)*(-g_ip3+2.*g_ip2-g_ip1) + 3. * (-g_ip3+4.*g_ip2-3.*g_ip1)*(-g_ip3+4.*g_ip2-3.*g_ip1)))+6./(( 1.e-6 + 13. * (-g_ip2+2.*g_ip1-g_i)*(-g_ip2+2.*g_ip1-g_i) + 3. * (-g_ip2+g_i)*(-g_ip2+g_i) )*( 1.e-6 + 13. * (-g_ip2+2.*g_ip1-g_i)*(-g_ip2+2.*g_ip1-g_i) + 3. * (-g_ip2+g_i)*(-g_ip2+g_i) ))+3./(( 1.e-6 + 13. * (-g_ip1+2.*g_i-g_im1)*(-g_ip1+2.*g_i-g_im1) + 3. * ( -3.*g_ip1+4.*g_i-g_im1)*( -3.*g_ip1+4.*g_i-g_im1) )*( 1.e-6 + 13. * (-g_ip1+2.*g_i-g_im1)*(-g_ip1+2.*g_i-g_im1) + 3. * ( -3.*g_ip1+4.*g_i-g_im1)*( -3.*g_ip1+4.*g_i-g_im1) )))))-0.25)*(3.*g_i-g_im1+g_ip2-3.*g_ip1) ) *one_third ;

	    _fh[m] = fromright;
	  }

	for( idT m=0; m<NPHI; ++m )
	  rhs_pdf( nu,p,i,l,m ) -= ( _fh[m+1]-_fh[m] )*DPHIM1;
	
    }

  return;
}


void MOSFETProblemCuda::WENO_PHI( const idT STAGE )
{
#ifdef _SHOW_ETA
  ostringstream message;
  message << "called " << __func__;
  show_eta(message.str());
#endif

  start_time( _PHASE_BTE_WENOPHI );

#ifdef __CUDACC__
  const idT NSBN  = host_dm -> get_NSBN();
  const idT NX    = host_dm -> get_X()   -> get_N();
  const idT NW    = host_dm -> get_W()   -> get_N();
  const idT NPHI  = host_dm -> get_PHI() -> get_N();

  // const idT partf_PHI_TPB = 96;
  // cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
  const idT gridSize      = host_gridconfig -> cuda_WENO_PHI_config -> get_gridSize();
  const idT blockSize     = host_gridconfig -> cuda_WENO_PHI_config -> get_blockSize();
  const idT shmemSize     = host_gridconfig -> cuda_WENO_PHI_config -> get_shmemSize();
  const cudaFuncCache cfc = host_gridconfig -> cuda_WENO_PHI_config -> get_cfc();
  cudaDeviceSetCacheConfig( cfc );
  cuda_WENO_PHI <<< gridSize, blockSize, shmemSize >>> ( device_dm, _GPU_pdf, _GPU_rhs_pdf, _GPU_deps_dx, _max_a3, STAGE, _GPU_a3 );
#else
  CPU_WENO_PHI ( STAGE );
#endif

  stop_time( _PHASE_BTE_WENOPHI );

#ifdef _SHOW_ETA
  cerr << "[ok] (function " << __func__ << ", line " << __LINE__ << endl;
#endif

  return;
}



/*************************************************
 *          OLD NON-OPTIMIZED ROUTINES           *
 *************************************************/
#define gweno(i)    _gweno[((i)-1)+3]
#define fhweno(i)   _fhweno[i]

void MOSFETProblemCuda::fdweno5_FROMLEFT(const double *_gweno, const idT NS, double *_fhweno)
{
  const double one_over_twelve = 1./12.;
  const double one_third = 1./3.;

  for(idT i = 0; i <= NS; ++i)
    {
      double g_im2 = gweno(i-2);
      double g_im1 = gweno(i-1);
      double g_i = gweno(i);
      double g_ip1 = gweno(i+1);
      double g_ip2 = gweno(i+2);

      fhweno(i) = ( -g_im1 + 7*(g_i+g_ip1) - g_ip2 ) *one_over_twelve + ( (1./(((1.e-6+13. * ( 2.*g_im1-g_im2 - g_i )*( 2.*g_im1-g_im2 - g_i ) + 3. * ( 4.*g_im1-g_im2 - 3.*g_i) * ( 4.*g_im1-g_im2 - 3.*g_i)) * (1.e-6+13. * ( 2.*g_im1-g_im2 - g_i )*( 2.*g_im1-g_im2 - g_i ) + 3. * ( 4.*g_im1-g_im2 - 3.*g_i) * ( 4.*g_im1-g_im2 - 3.*g_i)))*(1./((1.e-6+13. * ( 2.*g_im1-g_im2 - g_i )*( 2.*g_im1-g_im2 - g_i ) + 3. * ( 4.*g_im1-g_im2 - 3.*g_i) * ( 4.*g_im1-g_im2 - 3.*g_i)) * (1.e-6+13. * ( 2.*g_im1-g_im2 - g_i )*( 2.*g_im1-g_im2 - g_i ) + 3. * ( 4.*g_im1-g_im2 - 3.*g_i) * ( 4.*g_im1-g_im2 - 3.*g_i)))+6./((1.e-6+13. * ( 2.*g_i-g_im1 - g_ip1 )*( 2.*g_i-g_im1 - g_ip1 ) + 3. * ( -g_im1+g_ip1 ) * ( -g_im1+g_ip1 )) * (1.e-6+13. * ( 2.*g_i-g_im1 - g_ip1 )*( 2.*g_i-g_im1 - g_ip1 ) + 3. * ( -g_im1+g_ip1 ) * ( -g_im1+g_ip1 )))+3./((1.e-6+13. * ( 2.*g_ip1-g_i - g_ip2 )*( 2.*g_ip1-g_i - g_ip2 ) + 3. * ( 4.*g_ip1-3.*g_i-g_ip2 )*( 4.*g_ip1-3.*g_i-g_ip2 ))*(1.e-6+13. * ( 2.*g_ip1-g_i - g_ip2 )*( 2.*g_ip1-g_i - g_ip2 ) + 3. * ( 4.*g_ip1-3.*g_i-g_ip2 )*( 4.*g_ip1-3.*g_i-g_ip2 ))))))*(3.*g_i- g_ip1-3.*g_im1+g_im2) + (0.5*(3./(((1.e-6+13. * ( 2.*g_ip1-g_i - g_ip2 )*( 2.*g_ip1-g_i - g_ip2 ) + 3. * ( 4.*g_ip1-3.*g_i-g_ip2 )*( 4.*g_ip1-3.*g_i-g_ip2 ))*(1.e-6+13. * ( 2.*g_ip1-g_i - g_ip2 )*( 2.*g_ip1-g_i - g_ip2 ) + 3. * ( 4.*g_ip1-3.*g_i-g_ip2 )*( 4.*g_ip1-3.*g_i-g_ip2 )))*(1./((1.e-6+13. * ( 2.*g_im1-g_im2 - g_i )*( 2.*g_im1-g_im2 - g_i ) + 3. * ( 4.*g_im1-g_im2 - 3.*g_i) * ( 4.*g_im1-g_im2 - 3.*g_i)) * (1.e-6+13. * ( 2.*g_im1-g_im2 - g_i )*( 2.*g_im1-g_im2 - g_i ) + 3. * ( 4.*g_im1-g_im2 - 3.*g_i) * ( 4.*g_im1-g_im2 - 3.*g_i)))+6./((1.e-6+13. * ( 2.*g_i-g_im1 - g_ip1 )*( 2.*g_i-g_im1 - g_ip1 ) + 3. * ( -g_im1+g_ip1 ) * ( -g_im1+g_ip1 )) * (1.e-6+13. * ( 2.*g_i-g_im1 - g_ip1 )*( 2.*g_i-g_im1 - g_ip1 ) + 3. * ( -g_im1+g_ip1 ) * ( -g_im1+g_ip1 )))+3./((1.e-6+13. * ( 2.*g_ip1-g_i - g_ip2 )*( 2.*g_ip1-g_i - g_ip2 ) + 3. * ( 4.*g_ip1-3.*g_i-g_ip2 )*( 4.*g_ip1-3.*g_i-g_ip2 ))*(1.e-6+13. * ( 2.*g_ip1-g_i - g_ip2 )*( 2.*g_ip1-g_i - g_ip2 ) + 3. * ( 4.*g_ip1-3.*g_i-g_ip2 )*( 4.*g_ip1-3.*g_i-g_ip2 ))))))-0.25)*(3.*g_ip1- g_ip2-3.*g_i+g_im1) ) *one_third ;
    }

  return;
}


void MOSFETProblemCuda::fdweno5_FROMRIGHT(const double *_gweno, const idT NS, double *_fhweno)
{
  const double one_over_twelve = 1./12.;
  const double one_third = 1./3.;
  
  for(idT i = 0; i <= NS; ++i)
    {
      double g_im1 = gweno(i-1);
      double g_i = gweno(i);
      double g_ip1 = gweno(i+1);
      double g_ip2 = gweno(i+2);
      double g_ip3 = gweno(i+3);

      fhweno(i) = ( -g_im1 + 7*(g_i+g_ip1) - g_ip2 ) *one_over_twelve + ( (1./((( 1.e-6 + 13. * (-g_ip3+2.*g_ip2-g_ip1)*(-g_ip3+2.*g_ip2-g_ip1) + 3. * (-g_ip3+4.*g_ip2-3.*g_ip1)*(-g_ip3+4.*g_ip2-3.*g_ip1))*( 1.e-6 + 13. * (-g_ip3+2.*g_ip2-g_ip1)*(-g_ip3+2.*g_ip2-g_ip1) + 3. * (-g_ip3+4.*g_ip2-3.*g_ip1)*(-g_ip3+4.*g_ip2-3.*g_ip1)))*(1./(( 1.e-6 + 13. * (-g_ip3+2.*g_ip2-g_ip1)*(-g_ip3+2.*g_ip2-g_ip1) + 3. * (-g_ip3+4.*g_ip2-3.*g_ip1)*(-g_ip3+4.*g_ip2-3.*g_ip1))*( 1.e-6 + 13. * (-g_ip3+2.*g_ip2-g_ip1)*(-g_ip3+2.*g_ip2-g_ip1) + 3. * (-g_ip3+4.*g_ip2-3.*g_ip1)*(-g_ip3+4.*g_ip2-3.*g_ip1)))+6./(( 1.e-6 + 13. * (-g_ip2+2.*g_ip1-g_i)*(-g_ip2+2.*g_ip1-g_i) + 3. * (-g_ip2+g_i)*(-g_ip2+g_i) )*( 1.e-6 + 13. * (-g_ip2+2.*g_ip1-g_i)*(-g_ip2+2.*g_ip1-g_i) + 3. * (-g_ip2+g_i)*(-g_ip2+g_i) ))+3./(( 1.e-6 + 13. * (-g_ip1+2.*g_i-g_im1)*(-g_ip1+2.*g_i-g_im1) + 3. * ( -3.*g_ip1+4.*g_i-g_im1)*( -3.*g_ip1+4.*g_i-g_im1) )*( 1.e-6 + 13. * (-g_ip1+2.*g_i-g_im1)*(-g_ip1+2.*g_i-g_im1) + 3. * ( -3.*g_ip1+4.*g_i-g_im1)*( -3.*g_ip1+4.*g_i-g_im1) )))))*(3.*g_ip1-g_i+g_ip3-3.*g_ip2) + (0.5*(3./((( 1.e-6 + 13. * (-g_ip1+2.*g_i-g_im1)*(-g_ip1+2.*g_i-g_im1) + 3. * ( -3.*g_ip1+4.*g_i-g_im1)*( -3.*g_ip1+4.*g_i-g_im1) )*( 1.e-6 + 13. * (-g_ip1+2.*g_i-g_im1)*(-g_ip1+2.*g_i-g_im1) + 3. * ( -3.*g_ip1+4.*g_i-g_im1)*( -3.*g_ip1+4.*g_i-g_im1) ))*(1./(( 1.e-6 + 13. * (-g_ip3+2.*g_ip2-g_ip1)*(-g_ip3+2.*g_ip2-g_ip1) + 3. * (-g_ip3+4.*g_ip2-3.*g_ip1)*(-g_ip3+4.*g_ip2-3.*g_ip1))*( 1.e-6 + 13. * (-g_ip3+2.*g_ip2-g_ip1)*(-g_ip3+2.*g_ip2-g_ip1) + 3. * (-g_ip3+4.*g_ip2-3.*g_ip1)*(-g_ip3+4.*g_ip2-3.*g_ip1)))+6./(( 1.e-6 + 13. * (-g_ip2+2.*g_ip1-g_i)*(-g_ip2+2.*g_ip1-g_i) + 3. * (-g_ip2+g_i)*(-g_ip2+g_i) )*( 1.e-6 + 13. * (-g_ip2+2.*g_ip1-g_i)*(-g_ip2+2.*g_ip1-g_i) + 3. * (-g_ip2+g_i)*(-g_ip2+g_i) ))+3./(( 1.e-6 + 13. * (-g_ip1+2.*g_i-g_im1)*(-g_ip1+2.*g_i-g_im1) + 3. * ( -3.*g_ip1+4.*g_i-g_im1)*( -3.*g_ip1+4.*g_i-g_im1) )*( 1.e-6 + 13. * (-g_ip1+2.*g_i-g_im1)*(-g_ip1+2.*g_i-g_im1) + 3. * ( -3.*g_ip1+4.*g_i-g_im1)*( -3.*g_ip1+4.*g_i-g_im1) )))))-0.25)*(3.*g_i-g_im1+g_ip2-3.*g_ip1) ) *one_third ;
    }
  
  return;
}



cuda_bte_weno_kernels_config::cuda_bte_weno_kernels_config(const discrMeshes *dm)
{
  const idT NSBN  = dm -> get_NSBN();
  const idT NX    = dm -> get_X()   -> get_N();
  const idT NW    = dm -> get_W()   -> get_N();
  const idT NPHI  = dm -> get_PHI() -> get_N();

  /* WENO-X */
  const idT partf_X_TPB = NX + 1;                           /* 
							       One thread for each flux value _fh[] will be used. 
							    */
  const idT sm_size_X = (NX+6) + (NX+1);                    /* 
							       In shared memory, we shall store:
							       (i)  the values for flux _g                 <--- NX+6 values
							       (ii) the values for flux _fh reconstruction <--- NX+1 values
							    */
  cuda_WENO_X_20230525_config = new kernelConfig( nblocks(_NVALLEYS*NSBN*(NX+1)*NW*NPHI, partf_X_TPB),     partf_X_TPB,   sm_size_X*sizeof(double),         cudaFuncCachePreferL1,     "cuda_WENO_X_20230525" );

  /* WENO-W */
  const idT partf_W_TPB = NW + 1;                           /* 
							       One thread for each flux value _fh[] will be used. 
							    */
  const idT sm_size_W = NW + (NW+6) + (NW+1);               /* 
							       In shared memory, we shall store:
							       (i)   the values of flux flux[]              <--- NW values
							       (ii)  the values for flux _g                 <--- NW+6 values
							       (iii) the values for flux _fh reconstruction <--- NW+1 values
							    */
  cuda_WENO_W_20230524_config = new kernelConfig( nblocks( _NVALLEYS*NSBN*NX*(NW+1)*NPHI/2, partf_W_TPB ), partf_W_TPB,   sm_size_W*sizeof(double),         cudaFuncCachePreferL1,     "cuda_WENO_W_20230524" );

  /* WENO-PHI */
  const idT partf_PHI_TPB = 96;
  cuda_WENO_PHI_config        = new kernelConfig( nblocks( _NVALLEYS*NSBN*NX*NW*NPHI, partf_PHI_TPB ),     partf_PHI_TPB, 2*sm_size_N(partf_PHI_TPB, NPHI), cudaFuncCachePreferShared, "cuda_WENO_PHI"        );
}
