#include "mosfetproblem.h"
#include "debug_flags.h"


/**
   PURPOSE       :        

   FILE          : cuda_bte_scatterings_roughness.cu

   NAME          : MOSFETProblem::stretch_totvoldens

   PARAMETERS    :

   RETURN VALUE  : none

   CALLS TO      : MOSFETProblem::show_eta             (cuda_testing.h - declared inline)
 
   CALLED FROM   : MOSFETProblem::compute_initcond     (cuda_config.cu)

   DATA MODIFIED :  

   METHOD        :         

   AUTHOR        : Francesco VECIL

   LAST UPDATE   : 2023/06/01

   MODIFICATIONS : (i) 2023/06/01, 15H56' -- This config function is bugged. When called, it stalls.
                       Some messages are introduced in order to track where this happens exactly.
*/
// #define  _CONFIG_ROUGHNESS_DEBUG
void MOSFETProblemCuda::config_roughness()
{
#ifdef  _CONFIG_ROUGHNESS_DEBUG
  cerr << endl;
#endif

#ifdef  _CONFIG_ROUGHNESS_DEBUG
  cerr << " In function '" << __func__ << "', line " << __LINE__ << ", file '" << __FILE__ << "' --- reading constant data from host...";
#endif
  const int NX    = host_dm -> get_X()   -> get_N();
  const int NZ    = host_dm -> get_Z()   -> get_N();
  const int NZEXT = host_dm -> get_ZEXT()-> get_N();
  const int NW    = host_dm -> get_W()   -> get_N();
  const int NPHI  = host_dm -> get_PHI() -> get_N();

  enterfunc << "(may take some time, LAPACK has to invert a matrix of order " << NX*NZEXT << ")";

  const double LSOURCE    = host_physdesc -> get_LSOURCE();
  const double LCHANNEL   = host_physdesc -> get_LCHANNEL();
  const double LDRAIN     = host_physdesc -> get_LDRAIN();
  const double XLENGTH    = host_physdesc -> compute_XLENGTH();
  const double ZWIDTH     = host_physdesc -> get_ZWIDTH();
  const double eta        = ZWIDTH/XLENGTH;
  const double LSiO2      = host_physdesc -> get_LSiO2();
  const double NDHIGH     = host_physdesc -> get_NDHIGH();
  const double NDLOW      = host_physdesc -> get_NDLOW();
  const double VUPPERGATE = host_physdesc -> get_VUPPERGATE();
  const double VLOWERGATE = host_physdesc -> get_VLOWERGATE();
  const double VBIAS      = host_physdesc -> get_VBIAS();

  const double epsrSi     = host_phyco -> __epsrSi;
  const double epsrSiO2   = host_phyco -> __epsrSiO2;
#ifdef  _CONFIG_ROUGHNESS_DEBUG
  cerr << "[ok]" << endl;
#endif
  
#ifdef  _CONFIG_ROUGHNESS_DEBUG
  cerr << " In function '" << __func__ << "', line " << __LINE__ << ", file '" << __FILE__ << "' --- setting extended meshes...";
#endif
  // meshes
  // double adimensionalized_total_device_width = (ZWIDTH+host_solvpar->get_DELTA_SR())/ZWIDTH;
  // _dzext = adimensionalized_total_device_width/(NZEXT-1);
  // _dzextm1 = 1./_dzext;
  // _dzextm2 = 1./(_dzext*_dzext);
  const double DZEXTM1 = host_dm -> get_ZEXT() -> get_delta_m1();
  const double DZEXTM2 = host_dm -> get_ZEXT() -> get_delta_m2();

  // SILICON OXIDE in the extended domain for the surface roughness
  for( int i=0; i<NX; ++i )
    for( int j=0; j<NZEXT; ++j )
      {
	if( LSiO2/ZWIDTH < host_dm -> get_ZEXT() -> mesh(j) && host_dm -> get_ZEXT() -> mesh(j) < (ZWIDTH+host_solvpar->get_DELTA_SR()-LSiO2)/ZWIDTH )
	  isoxide_ext( i,j ) = false;
	else
	  isoxide_ext( i,j ) = true;
      }

  // SOURCE CONTACT in the extended domain for the surface roughness
  for( int i=0; i<NX; ++i )
    for( int j=0; j<NZEXT; ++j )
      {
	if( isoxide_ext( i,j ) == false )
	  {
	    // if( 0 <= x(i) && x(i) <= LSOURCE/XLENGTH )
	    const double xi = host_dm->get_X()->mesh(i);
	    if( 0 <= xi && xi <= LSOURCE/XLENGTH )
	      issource_ext( i,j ) = true;
	    else
	      issource_ext( i,j ) = false;
	  }
	else
	  issource_ext( i,j ) = false;
      }

  // DRAIN CONTACT in the extended domain for the surface roughness
  for( int i=0; i<NX; ++i )
    for( int j=0; j<NZEXT; ++j )
      {
	if( isoxide_ext( i,j ) == false )
	  {
	    const double xi = host_dm->get_X()->mesh(i);
	    // if( (LSOURCE+LCHANNEL)/XLENGTH-1.e-12 <= x(i) && x(i) <= (LSOURCE+LCHANNEL+LDRAIN)/XLENGTH )
	    if( (LSOURCE+LCHANNEL)/XLENGTH-1.e-12 <= xi && xi <= (LSOURCE+LCHANNEL+LDRAIN)/XLENGTH )
	      isdrain_ext( i,j ) = true;
	    else
	      isdrain_ext( i,j ) = false;
	  }
	else
	  isdrain_ext( i,j ) = false;
      }

  // in the extended domain for the surface roughness
  for(int j=0; j<NZEXT; ++j)
  {
      pot_b_ext(j) = interpolate_pot_b( resampling_extended_to_original(host_dm -> get_ZEXT() -> mesh(j)) );
  }

  // nd in the extended domain for the surface roughness
  for( int i=0; i<NX; ++i )
    for( int j=0; j<NZEXT; ++j )
      {
	nd_ext( i,j ) = 0.0;
	
	if( !isoxide_ext(i,j) )
	  {
	    if( issource_ext(i,j) || isdrain_ext(i,j) )
	      nd_ext(i,j) = NDHIGH/host_rescpar->get_nstar();
	    else
	      nd_ext(i,j) = NDLOW/host_rescpar->get_nstar();
	  }
      }

  // epsr in the extended domain for the surface roughness
  for( int i=0; i<NX; ++i )
    for( int j=0; j<NZEXT; ++j )
      {
	if( !isoxide_ext(i,j) )
	  epsr_ext (i,j) = epsrSi;
	else
	  epsr_ext (i,j) = epsrSiO2;
      }


  // fill with zeros
  for(int r=0; r<NX*NZEXT; ++r)
    for(int s=r-NZEXT; s<=r+NZEXT; ++s)
      matrix_2dconst_ext(r,s) = 0.0;
  
  // double DXM2 = 1./(_dx*_dx);
  double DXM2 = host_dm->get_X()->get_delta_m2();

  nnz_ext = 0;
  
  for(int j=0; j<NZEXT; ++j)
    {
      matrix_2dconst_ext(host_i_jext(0,j),host_i_jext(0,j)) = 1.; ++nnz_ext; // Dirichlet at source
    }

  for(int i=1; i<NX-1; ++i)
    {
      if(isgate(i,0))
	{
	  matrix_2dconst_ext(host_i_jext(i,0),host_i_jext(i,0)) = 1.; ++nnz_ext; // Dirichlet for oxide
	}
      else
	{
	  matrix_2dconst_ext(host_i_jext(i,0),host_i_jext(i,0)) = DZEXTM1; ++nnz_ext; // Neumann else
	  matrix_2dconst_ext(host_i_jext(i,0),host_i_jext(i,1)) = -DZEXTM1; ++nnz_ext;
	}

      // Laplacian
      for(int j=1; j<NZEXT-1; ++j)
	{
	  matrix_2dconst_ext(host_i_jext(i,j),host_i_jext(i-1,j)) += -eta*eta*DXM2*(.5*epsr_ext(i-1,j)+.5*epsr_ext(i,j)); ++nnz_ext;
	  matrix_2dconst_ext(host_i_jext(i,j),host_i_jext(i,j-1)) += -DZEXTM2*(.5*epsr_ext(i,j-1)+.5*epsr_ext(i,j)); ++nnz_ext;
	  matrix_2dconst_ext(host_i_jext(i,j),host_i_jext(i,j))   += eta*eta*DXM2*(.5*epsr_ext(i-1,j)+epsr_ext(i,j)+.5*epsr_ext(i+1,j)) 
	    + DZEXTM2*(.5*epsr_ext(i,j-1)+epsr_ext(i,j)+.5*epsr_ext(i,j+1)); ++nnz_ext;
	  matrix_2dconst_ext(host_i_jext(i,j),host_i_jext(i,j+1)) += -DZEXTM2*(.5*epsr_ext(i,j)+.5*epsr_ext(i,j+1)); ++nnz_ext;
	  matrix_2dconst_ext(host_i_jext(i,j),host_i_jext(i+1,j)) += -eta*eta*DXM2*(.5*epsr_ext(i,j)+.5*epsr_ext(i+1,j)); ++nnz_ext;
	}

      if(isgate(i,NZ-1))
	{
	  matrix_2dconst_ext(host_i_jext(i,NZEXT-1),host_i_jext(i,NZEXT-1)) = 1.; ++nnz_ext; // Dirichlet for oxide
	}
      else
	{
	  matrix_2dconst_ext(host_i_jext(i,NZEXT-1),host_i_jext(i,NZEXT-1)) = DZEXTM1; ++nnz_ext; // Neumann else
	  matrix_2dconst_ext(host_i_jext(i,NZEXT-1),host_i_jext(i,NZEXT-2)) = -DZEXTM1; ++nnz_ext;
	}
    }
  
  for(int j=0; j<NZEXT; ++j)
    {
      matrix_2dconst_ext(host_i_jext(NX-1,j),host_i_jext(NX-1,j)) = 1.; ++nnz_ext; // Dirichlet at drain
    }
#ifdef  _CONFIG_ROUGHNESS_DEBUG
  cerr << "[ok]" << endl;
#endif

#ifdef  _CONFIG_ROUGHNESS_DEBUG
  cerr << " In function '" << __func__ << "', line " << __LINE__ << ", file '" << __FILE__ << "' --- precomputing denom_SR...";
#endif
  // precompute denom(nu,l,m,mp)
  for(int nu=0; nu<_NVALLEYS; ++nu)
      for(int l=0; l<NW; ++l)
      {
	const double WL = host_dm->get_W()->mesh(l);
	double s_chcoord = sqrt(mass(Xdim,nu)*mass(Ydim,nu))*(1.+2.*_kane[nu]*host_rescpar->get_epsstar()*WL);
	double AUX_CST = 4.*host_rescpar->get_kstar()*host_rescpar->get_kstar()*host_solvpar->get_LENGTH_SR()*host_solvpar->get_LENGTH_SR()*WL*(1+_kane[nu]*host_rescpar->get_epsstar()*WL);
	for(int m=0; m<NPHI; ++m)
	  for(int mp=0; mp<NPHI; ++mp)
	    {
	      double SIN_MINUS   = sin( .5*(host_dm->get_PHI()->mesh(m)-host_dm->get_PHI()->mesh(mp)) );
	      double SIN_MINUS_2 = SIN_MINUS * SIN_MINUS;
	      double SIN_PLUS    = sin( .5*(host_dm->get_PHI()->mesh(m)+host_dm->get_PHI()->mesh(mp)) );
	      double SIN_PLUS_2  = SIN_PLUS * SIN_PLUS;
	      double COS_PLUS    = cos( .5*(host_dm->get_PHI()->mesh(m)+host_dm->get_PHI()->mesh(mp)) );
	      double COS_PLUS_2  = COS_PLUS * COS_PLUS;
	      
	      double denom = 1.+AUX_CST*SIN_MINUS_2*(mass(Xdim,nu)*SIN_PLUS_2+mass(Ydim,nu)*COS_PLUS_2);
	      denom_SR(nu,l,m,mp) = 1./pow( denom,1.5 );
	    }
      }
  
#ifdef __CUDACC__
  checkCudaErrors( cudaMemcpy(_GPU_denom_SR, _denom_SR, _NVALLEYS*NW*NPHI*NPHI*sizeof(double), cudaMemcpyHostToDevice) );
#endif

#ifdef  _CONFIG_ROUGHNESS_DEBUG
  cerr << "[ok]" << endl;
#endif

#ifdef  _CONFIG_ROUGHNESS_DEBUG
  cerr << " In function '" << __func__ << "', line " << __LINE__ << ", file '" << __FILE__ << "' --- precomputing sigma...";
#endif

  // compute sigma
  for(int j=0; j<NZEXT; ++j)
    {
      double Z = host_dm -> get_ZEXT() -> mesh(j);
      if( 0 <= Z && Z <= 0.5*(1.0-host_solvpar->get_DELTA_STRETCH_SR()) )
	_sigma[j] = Z;
      else if( 0.5*(1.0-host_solvpar->get_DELTA_STRETCH_SR()) < Z && Z <= 0.5*(1.0+host_solvpar->get_DELTA_STRETCH_SR())+host_solvpar->get_DELTA_SR()/ZWIDTH ) 
	_sigma[j] = 1./(1.+host_solvpar->get_DELTA_SR()/(host_solvpar->get_DELTA_STRETCH_SR()*ZWIDTH))*(Z-0.5*(1.0-host_solvpar->get_DELTA_STRETCH_SR()))+0.5*(1.0-host_solvpar->get_DELTA_STRETCH_SR());
      else if( 0.5*(1.0+host_solvpar->get_DELTA_STRETCH_SR())+host_solvpar->get_DELTA_SR()/ZWIDTH < Z && Z <= 1.+host_solvpar->get_DELTA_STRETCH_SR()/ZWIDTH )
	_sigma[j] = Z - host_solvpar->get_DELTA_SR()/ZWIDTH;
      else
	{
	  cerr << " no... no..." << endl;
	  exit(0);
	}
      _Sigma[j] = int(_sigma[j]*(NZ-1));
    }
#ifdef __CUDACC__
  checkCudaErrors( cudaMemcpy(_GPU_sigma, _sigma, NZEXT*sizeof(double), cudaMemcpyHostToDevice) );
  checkCudaErrors( cudaMemcpy(_GPU_Sigma, _Sigma, NZEXT*sizeof(double), cudaMemcpyHostToDevice) );
#endif

#ifdef  _CONFIG_ROUGHNESS_DEBUG
  cerr << "[ok]" << endl;
#endif

#ifdef  _CONFIG_ROUGHNESS_DEBUG
  cerr << " In function '" << __func__ << "', line " << __LINE__ << ", file '" << __FILE__ << "' --- precomputing constant parts of the linear system...";
#endif
  
#ifdef __CUDACC__
  // constant part of the rhs
  const double CP = host_adimpar->get_cp();
  double *_h_b = new double [ NX*NZEXT ];
  for( int line=0; line< NX*NZEXT; ++line )
    _h_b[line] = 0;
  for(int IX = 0; IX < NX; ++IX)
    for(int IZ = 0; IZ < NZEXT; ++IZ )
      {
  	int line = IX*NZEXT + IZ;

  	// inner points
  	if( 0 < IX && IX < NX-1 && 0 < IZ && IZ < NZEXT-1 )
  	  _h_b[line] += -CP*(-nd_ext(IX,IZ));

  	// boundary conditions
  	if( 0 < IX && IX < NX-1 && IZ == 0 )
  	  {
  	    if(isgate(IX,0))
  	      _h_b[line] += VLOWERGATE/host_rescpar->get_potstar();
  	    else
  	      _h_b[line] += 0;
  	  }

  	if( 0 < IX && IX < NX-1 && IZ == NZEXT-1 )
  	  {
  	    if(isgate(IX,NZ-1))
  	      _h_b[line] += VUPPERGATE/host_rescpar->get_potstar();
  	    else
  	      _h_b[line] += 0;
  	  }
      
  	if( 0 <= IZ && IZ <= NZEXT-1 && IX == 0 )
  	  _h_b[line] += pot_b_ext(IZ);

  	if( 0 <= IZ && IZ <= NZEXT-1 && IX == NX-1 )
  	  _h_b[line] += pot_b_ext(IZ) + VBIAS/host_rescpar->get_potstar();
      }

  checkCudaErrors( cudaMemcpy(_GPU_rhs_ext_const, _h_b, NX*NZEXT*sizeof(double), cudaMemcpyHostToDevice) );
  delete [] _h_b;
#endif

#ifdef  _CONFIG_ROUGHNESS_DEBUG
  cerr << "[ok]" << endl;
#endif

  if(host_solvpar->get_ROUGHNESS() == true)
    {
#ifdef  _CONFIG_ROUGHNESS_DEBUG
      cerr << " In function '" << __func__ << "', line " << __LINE__ << ", file '" << __FILE__ << "' --- preparing matrix_2dconst_ext_inv for inversion...";
#endif

      // Se trata de rellenar la matriz de la ecuacion de Poisson y de invertirla utilizando LAPACK
      // fill with zeros
      double *matrix_aux = new double [NX*NZEXT*NX*NZEXT];

      for(int r=0; r<NX*NZEXT; ++r)
	for(int s=0; s<NX*NZEXT; ++s)
	  matrix_2dconst_ext_inv(r,s) = 0.0;
  
      nnz_ext = 0;
  
      for(int j=0; j<NZEXT; ++j)
	{
	  matrix_2dconst_ext_inv(host_i_jext(0,j),host_i_jext(0,j)) = 1.; ++nnz_ext; // Dirichlet at source
	}

      for(int i=1; i<NX-1; ++i)
	{
	  if(isgate(i,0))
	    {
	      matrix_2dconst_ext_inv(host_i_jext(i,0),host_i_jext(i,0)) = 1.; ++nnz_ext; // Dirichlet for oxide
	    }
	  else
	    {
	      matrix_2dconst_ext_inv(host_i_jext(i,0),host_i_jext(i,0)) = DZEXTM1; ++nnz_ext; // Neumann else
	      matrix_2dconst_ext_inv(host_i_jext(i,0),host_i_jext(i,1)) = -DZEXTM1; ++nnz_ext;
	    }

	  // Laplacian
	  for(int j=1; j<NZEXT-1; ++j)
	    {
	      matrix_2dconst_ext_inv(host_i_jext(i,j),host_i_jext(i-1,j)) += -eta*eta*DXM2*(.5*epsr_ext(i-1,j)+.5*epsr_ext(i,j)); ++nnz_ext;
	      matrix_2dconst_ext_inv(host_i_jext(i,j),host_i_jext(i,j-1)) += -DZEXTM2*(.5*epsr_ext(i,j-1)+.5*epsr_ext(i,j)); ++nnz_ext;
	      matrix_2dconst_ext_inv(host_i_jext(i,j),host_i_jext(i,j))   += eta*eta*DXM2*(.5*epsr_ext(i-1,j)+epsr_ext(i,j)+.5*epsr_ext(i+1,j)) 
		+ DZEXTM2*(.5*epsr_ext(i,j-1)+epsr_ext(i,j)+.5*epsr_ext(i,j+1)); ++nnz_ext;
	      matrix_2dconst_ext_inv(host_i_jext(i,j),host_i_jext(i,j+1)) += -DZEXTM2*(.5*epsr_ext(i,j)+.5*epsr_ext(i,j+1)); ++nnz_ext;
	      matrix_2dconst_ext_inv(host_i_jext(i,j),host_i_jext(i+1,j)) += -eta*eta*DXM2*(.5*epsr_ext(i,j)+.5*epsr_ext(i+1,j)); ++nnz_ext;
	    }

	  if(isgate(i,NZ-1))
	    {
	      matrix_2dconst_ext_inv(host_i_jext(i,NZEXT-1),host_i_jext(i,NZEXT-1)) = 1.; ++nnz_ext; // Dirichlet for oxide
	    }
	  else
	    {
	      matrix_2dconst_ext_inv(host_i_jext(i,NZEXT-1),host_i_jext(i,NZEXT-1)) = DZEXTM1; ++nnz_ext; // Neumann else
	      matrix_2dconst_ext_inv(host_i_jext(i,NZEXT-1),host_i_jext(i,NZEXT-2)) = -DZEXTM1; ++nnz_ext;
	    }
	}
  
      for(int j=0; j<NZEXT; ++j)
	{
	  matrix_2dconst_ext_inv(host_i_jext(NX-1,j),host_i_jext(NX-1,j)) = 1.; ++nnz_ext; // Dirichlet at drain
	}

      memcpy( matrix_aux, _matrix_2dconst_ext_inv, NX*NZEXT*NX*NZEXT*sizeof(double) );
  
#ifdef  _CONFIG_ROUGHNESS_DEBUG
      cerr << "[ok]" << endl;
#endif

      // // y ahora hay que invertirla...
      int N = NX * NZEXT;
      int *IPIV = new int[N];
      int LWORK = N*N;
      double *WORK = new double[LWORK];
      int INFO;

#ifdef  _CONFIG_ROUGHNESS_DEBUG
      cerr << " In function '" << __func__ << "', line " << __LINE__ << ", file '" << __FILE__ << "' --- call to LAPACK DGETRF to invert matrix_2dconst_ext_inv...";
#endif

      dgetrf_(&N,&N,_matrix_2dconst_ext_inv,&N,IPIV,&INFO);

      if( INFO != 0 )
	{
	  cerr << " ERROR from function '" << __func__ << "', line " << __LINE__ << ", file '" << __FILE__ << "' --- DGETRF failed";
	  exit(-1);
	}
      
#ifdef  _CONFIG_ROUGHNESS_DEBUG
      cerr << "[ok]" << endl;
#endif

#ifdef  _CONFIG_ROUGHNESS_DEBUG
      cerr << " In function '" << __func__ << "', line " << __LINE__ << ", file '" << __FILE__ << "' --- call to LAPACK DGETRI to invert matrix_2dconst_ext_inv...";
#endif

      dgetri_(&N,_matrix_2dconst_ext_inv,&N,IPIV,WORK,&LWORK,&INFO);
  
      if( INFO != 0 )
	{
	  cerr << " ERROR from function '" << __func__ << "', line " << __LINE__ << ", file '" << __FILE__ << "' --- DGETRI failed";
	  exit(-1);
	}

      // delete from here
      cerr << " Counting the number of zeros...";
      int nzeros = 0;
      for(int u=0; u<N*N; ++u)
	if( fabs(_matrix_2dconst_ext_inv[u]) > 1.e-12 )
	  ++nzeros;
      cerr << " nzeros = " << nzeros << endl;
      cerr << "Spersiti pattern = " << (double)nzeros/(NX*NZEXT*NZ*NZEXT) << endl;
      // to here

      
      
#ifdef  _CONFIG_ROUGHNESS_DEBUG
      cerr << "[ok]" << endl;
#endif

      delete[] IPIV;
      delete[] WORK;

#ifdef __CUDACC__
      checkCudaErrors( cudaMemcpy(_GPU_matrix_2dconst_ext_inv, _matrix_2dconst_ext_inv, NX*NZEXT*NX*NZEXT*sizeof(double), cudaMemcpyHostToDevice) );
#endif

      delete [] matrix_aux;

    }

  exitfunc;
  
  return;
}



double MOSFETProblemCuda::resampling_extended_to_original( const double Z )
{
  const double ZWIDTH  = host_physdesc -> get_ZWIDTH();

  double res = NAN;

  if( 0 <= Z*ZWIDTH && Z*ZWIDTH <= .5*ZWIDTH-.5*host_solvpar->get_DELTA_STRETCH_SR()*ZWIDTH )
    res=Z;
  else if( .5*ZWIDTH-.5*host_solvpar->get_DELTA_STRETCH_SR()*ZWIDTH < Z*ZWIDTH && Z*ZWIDTH < .5*ZWIDTH+.5*host_solvpar->get_DELTA_STRETCH_SR()*ZWIDTH+host_solvpar->get_DELTA_SR() )
    res = ( .5*ZWIDTH-.5*host_solvpar->get_DELTA_STRETCH_SR()*ZWIDTH+(Z*ZWIDTH-(.5*ZWIDTH-.5*host_solvpar->get_DELTA_STRETCH_SR()*ZWIDTH))*(host_solvpar->get_DELTA_STRETCH_SR()*ZWIDTH)/(host_solvpar->get_DELTA_STRETCH_SR()*ZWIDTH+host_solvpar->get_DELTA_SR()) )/ZWIDTH;
  else if( .5*ZWIDTH+.5*host_solvpar->get_DELTA_STRETCH_SR()*ZWIDTH+host_solvpar->get_DELTA_SR() <= Z*ZWIDTH && Z*ZWIDTH <= ZWIDTH+host_solvpar->get_DELTA_SR() )
    res = (Z*ZWIDTH-host_solvpar->get_DELTA_SR())/ZWIDTH;

  return res;
}


double MOSFETProblemCuda::interpolate_pot_b( const double Z )
{
  const int NZ    = host_dm -> get_Z()   -> get_N();

  for( int j=0; j<NZ; ++j )
    if( fabs(host_dm->get_Z()->mesh(j)-Z) < 1.e-12 )
      return _pot_b[j];
  
  const double DZM1 = host_dm->get_Z()->get_delta_m1();
  
  const int jdown = (int)(Z*DZM1);
  const int jup = jdown+1;
  
  double res = ( _pot_b[jup]-_pot_b[jdown] )*DZM1*Z
    +( host_dm->get_Z()->mesh(jup)*_pot_b[jdown]-host_dm->get_Z()->mesh(jdown)*_pot_b[jup] )*DZM1;
  
  return res;
}



cuda_bte_scatterings_roughness_kernels_config::cuda_bte_scatterings_roughness_kernels_config( const discrMeshes *dm )
{
  const int NSBN  = dm -> get_NSBN();
  const int NX    = dm -> get_X()   -> get_N();
  const int NZ    = dm -> get_Z()   -> get_N();
  const int NZEXT = dm -> get_ZEXT()-> get_N();
  const int NW    = dm -> get_W()   -> get_N();
  const int NPHI  = dm -> get_PHI() -> get_N();
  
  int blocksize, gridsize, shmemsize;

  kernel_roughness_gain_7_config        = new kernelConfig( nblocks( _NVALLEYS*NSBN*NX*NW, NX*NSBN ), NX*NSBN,  NPHI*NPHI*sizeof(double), cudaFuncCachePreferNone,           "kernel_roughness_gain_7"   );

  blocksize = 64;
  shmemsize = ((blocksize+NPHI-2)/NPHI)*NPHI;
  kernel_roughness_gain_20230616_config = new kernelConfig( nblocks( _NVALLEYS*NSBN*NX*NW*NPHI, blocksize ), blocksize,  shmemsize*sizeof(double), cudaFuncCachePreferNone,           "kernel_roughness_gain_20230616"   );
  kernel_roughness_gain_20230616_2_config = new kernelConfig( nblocks( _NVALLEYS*NSBN*NX*NW*NPHI, blocksize ), blocksize,  shmemsize*sizeof(double), cudaFuncCachePreferNone,           "kernel_roughness_gain_20230616_2"   );

  kernel_roughness_loss_config          = new kernelConfig( nblocks( _NVALLEYS*NSBN*NX*NW*NPHI,32 ), 32,        NOSHMEM,                  cudaFuncCachePreferNone,           "kernel_roughness_loss"     );

  blocksize = 64; gridsize = ceil( (float)(NX*NZEXT)/ blocksize );
  kernel_stretch_totvoldens_config      = new kernelConfig( gridsize,                                blocksize, NOSHMEM,                  cudaFuncCachePreferNone,           "kernel_stretch_totvoldens" );

  blocksize = 64; gridsize = ceil( (float)(NX*NZEXT)/ blocksize );
  kernel_construct_rhs_ext_config       = new kernelConfig( gridsize,                                blocksize, NOSHMEM,                  cudaFuncCachePreferNone,           "kernel_construct_rhs_ext"  );

  blocksize = 64; gridsize = ceil( (float)(NX*NZ)/ blocksize );
  kernel_compute_Deltapot_config        = new kernelConfig( gridsize,                                blocksize, NOSHMEM,                  cudaFuncCachePreferNone,           "kernel_compute_Deltapot"   );

  blocksize = 1; gridsize = ceil( (float)(_NVALLEYS*NSBN*NX)/ blocksize );
  kernel_compute_overlap_SR_config      = new kernelConfig( gridsize,                                blocksize, NZ*sizeof(double),        cudaFuncCachePreferNone,           "kernel_compute_overlap_SR" );

  blocksize = 64; gridsize = ceil( (float)(NX*NZEXT)/ blocksize );
  multiply_config                       = new kernelConfig( gridsize,                                blocksize, NOSHMEM,                  cudaFuncCachePreferNone,           "multiply"                  );
}



