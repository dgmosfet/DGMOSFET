#include "mosfetproblem.h"
#include "debug_flags.h"

#ifdef __CUDACC__
__global__ void cuda_phonons_gain( const discrMeshes *dm, const rescalingParams *rescpar, const physConsts *pc, const double *_GPU_integrated_pdf_energy, const double *_GPU_Wm1, 
				   double *_GPU_rhs_pdf_gain, const double *_GPU_eps, const int STAGE, double *_GPU_test_gain )
{
  int global_index = blockIdx.x*blockDim.x + threadIdx.x;
    
  const int NSBN     = dm->get_NSBN();
  const int NX       = dm->get_X()->get_N();
  const int NW       = dm->get_W()->get_N();
  const int NPHI     = dm->get_PHI()->get_N();
  const double hbar  = pc->__hbar;

  if( global_index < _NVALLEYS*NSBN*NX*NW )
    {
      const double DW = dm->get_W()->get_delta();
      const double DWM1 = dm->get_W()->get_delta_m1();
      
      int i, nu, p, l;
      GPU_map_1D_to_4D( global_index, &i, NX, &nu, _NVALLEYS, &p, NSBN, &l, NW );

      const double EPSSTAR = rescpar->get_epsstar();
      
      double contrib = 0;

      const double eps_nu_p_i = GPU_eps( nu,p,i );

      double Gamma;
      int ldown;

      double eps_nu_pp_i;
      double Wm1_nu_pp_nu_p;

      for( int pp=0; pp<NSBN; ++pp )
	{
	  eps_nu_pp_i = GPU_eps( nu,pp,i );
	  Wm1_nu_pp_nu_p = GPU_Wm1( nu,pp,nu,p,i );

	  /*************************************
	   *               F-TYPE              *
	   *************************************/
	  for( int nup=0; nup<_NVALLEYS; ++nup )
	    if( nup != nu )
	      {
	  	const double Wm1_nup_pp_nu_p_i = GPU_Wm1( nup,pp,nu,p,i );
	  	const double eps_nup_pp_i = GPU_eps( nup,pp,i );
		
	  	for( int SC=4; SC<=6; ++SC )
	  	  {
	  	    // PLUS 
	  	    Gamma = ((l+.5)*DW)+eps_nu_p_i - eps_nup_pp_i + hbar*GPU_omega(SC)/EPSSTAR;
	  	    ldown = (int)(Gamma*DWM1-.5);
		    
	  	    contrib += (GPU_cscatt(SC)*(GPU_sqrtmassXY(nu)*(1.+2.*GPU_kane(nu)*EPSSTAR*((l+.5)*DW)))*2*(GPU_occupations(SC,nup,nu)+1)*Wm1_nup_pp_nu_p_i*(( (Gamma >= 0 && ldown<=NW-2) ? (GPU_integrated_pdf_energy(nup,pp,i,(ldown+1))-GPU_integrated_pdf_energy(nup,pp,i,ldown))*DWM1*Gamma+(((ldown+1)+0.5)*DW*GPU_integrated_pdf_energy(nup,pp,i,ldown)-((ldown+0.5)*DW)*GPU_integrated_pdf_energy(nup,pp,i,(ldown+1)))*DWM1 : 0 )));
		    
	  	    // MINUS 
	  	    Gamma = ((l+.5)*DW)+eps_nu_p_i - eps_nup_pp_i - hbar*GPU_omega(SC)/EPSSTAR;
	  	    ldown = (int)(Gamma*DWM1-.5);

	  	    contrib += (GPU_cscatt(SC)*(GPU_sqrtmassXY(nu)*(1.+2.*GPU_kane(nu)*EPSSTAR*((l+.5)*DW)))*2*GPU_occupations(SC,nup,nu)*Wm1_nup_pp_nu_p_i*(( (Gamma >= 0 && ldown<=NW-2) ? (GPU_integrated_pdf_energy(nup,pp,i,(ldown+1))-GPU_integrated_pdf_energy(nup,pp,i,ldown))*DWM1*Gamma+(((ldown+1)+0.5)*DW*GPU_integrated_pdf_energy(nup,pp,i,ldown)-((ldown+0.5)*DW)*GPU_integrated_pdf_energy(nup,pp,i,(ldown+1)))*DWM1 : 0 )));
	  	  }
	      }

	  /*************************************
	   *               ELASTIC             *
	   *************************************/
	  Gamma = (l+.5)*DW + eps_nu_p_i - eps_nu_pp_i;
	  ldown = (int)(Gamma*DWM1-.5);
	  
	  contrib += (GPU_cscatt(0)*Wm1_nu_pp_nu_p*(GPU_sqrtmassXY(nu)*(1.+2.*GPU_kane(nu)*EPSSTAR*((l+.5)*DW)))*(( (Gamma >= 0 && ldown<=NW-2) ? (GPU_integrated_pdf_energy(nu,pp,i,(ldown+1))-GPU_integrated_pdf_energy(nu,pp,i,ldown))*DWM1*Gamma+(((ldown+1)+0.5)*DW*GPU_integrated_pdf_energy(nu,pp,i,ldown)-((ldown+0.5)*DW)*GPU_integrated_pdf_energy(nu,pp,i,(ldown+1)))*DWM1 : 0 )));

	  /*************************************
	   *               G-TYPE              *
	   *************************************/
	  for( int SC=1; SC<=3; ++SC )
	    {
	      // PLUS 
	      Gamma = ((l+.5)*DW)+eps_nu_p_i - eps_nu_pp_i + hbar*GPU_omega(SC)/EPSSTAR;
	      ldown = (int)(Gamma*DWM1-.5);
		
	      contrib += (GPU_cscatt(SC)*(GPU_sqrtmassXY(nu)*(1.+2.*GPU_kane(nu)*EPSSTAR*((l+.5)*DW)))*(GPU_occupations(SC,nu,nu)+1)*Wm1_nu_pp_nu_p*(( (Gamma >= 0 && ldown<=NW-2) ? (GPU_integrated_pdf_energy(nu,pp,i,(ldown+1))-GPU_integrated_pdf_energy(nu,pp,i,ldown))*DWM1*Gamma+(((ldown+1)+0.5)*DW*GPU_integrated_pdf_energy(nu,pp,i,ldown)-((ldown+0.5)*DW)*GPU_integrated_pdf_energy(nu,pp,i,(ldown+1)))*DWM1 : 0 )));
		
	      // MINUS 
	      Gamma = ((l+.5)*DW)+eps_nu_p_i - eps_nu_pp_i - hbar*GPU_omega(SC)/EPSSTAR;
	      ldown = (int)(Gamma*DWM1-.5);

	      contrib += (GPU_cscatt(SC)*(GPU_sqrtmassXY(nu)*(1.+2.*GPU_kane(nu)*EPSSTAR*((l+.5)*DW)))*GPU_occupations(SC,nu,nu)*Wm1_nu_pp_nu_p*(( (Gamma >= 0 && ldown<=NW-2) ? (GPU_integrated_pdf_energy(nu,pp,i,(ldown+1))-GPU_integrated_pdf_energy(nu,pp,i,ldown))*DWM1*Gamma+(((ldown+1)+0.5)*DW*GPU_integrated_pdf_energy(nu,pp,i,ldown)-((ldown+0.5)*DW)*GPU_integrated_pdf_energy(nu,pp,i,(ldown+1)))*DWM1 : 0 )));
	    }
	}

      // for(int m=0; m<_NPHI; ++m)
      // 	GPU_test_gain(nu,p,i,l,m) = contrib;

      GPU_rhs_pdf_gain( nu,p,i,l ) = contrib;
    }
}


__global__ void cuda_phonons_loss( const discrMeshes *dm, const rescalingParams *rescpar, const physConsts *pc, const double* _GPU_pdf,
				   const double* _GPU_Wm1, 
				   double* _GPU_rhs_pdf, const double* _GPU_rhs_pdf_gain, const double* _GPU_eps, const int STAGE, double *_GPU_test_loss )
{
  extern __shared__ double shmem[];

  int global_index = blockIdx.x*blockDim.x + threadIdx.x;
    
  const int NSBN     = dm->get_NSBN();
  const int NX       = dm->get_X()->get_N();
  const int NW       = dm->get_W()->get_N();
  const int NPHI     = dm->get_PHI()->get_N();
  const double hbar  = pc->__hbar;

  if(global_index < _NVALLEYS*NSBN*NX*NW*NPHI)
    {
      const double DW = dm->get_W()->get_delta();

      int nu,p,i,l,m;
      GPU_map_1D_to_5D( global_index, &i, NX, &nu, _NVALLEYS, &p, NSBN, &l, NW, &m, NPHI );

      // USING SHARED MEMORY
      int i_0   = blockIdx.x*blockDim.x;
      i_0                = i_0/NPHI;
      i_0                = i_0/NW;
      i_0                = i_0/NSBN;
      i_0                = i_0/_NVALLEYS;
      const int i_0_N   = i_0*_NVALLEYS*NSBN;
      double *seps       = (double*)&shmem[0];

      if( threadIdx.x < 2*_NVALLEYS*NSBN && i_0_N + threadIdx.x < (_NVALLEYS*NSBN*NX) )
	seps[ threadIdx.x ] = _GPU_eps[ i_0_N + threadIdx.x ];

      __syncthreads();

      // --------------------------

      double contrib = 0;

      const double EPSSTAR = rescpar->get_epsstar();
      const double epsstarm1 = 1./EPSSTAR;
      
      double Gamma;

      for( int pp=0; pp<NSBN; ++pp )
    	{
    	  const double Wm1_nu_pp_nu_p = GPU_Wm1( nu,pp,nu,p,i );

    	  /*************************************
    	   *               F-TYPE              *
    	   *************************************/
    	  for( unsigned int nup=0; nup<_NVALLEYS; ++nup )
    	    if(nup != nu)
    	      {
    	  	const double aux3 = 2.*GPU_kane(nup)*EPSSTAR;

    	  	const double Wm1_nup_pp_nu_p_i = GPU_Wm1( nup,pp,nu,p,i );
		
	  	const double aux  = ((l+.5)*DW)+seps[ (i-i_0)*_NVALLEYS*NSBN + nu*NSBN + p ] - seps[ (i-i_0)*_NVALLEYS*NSBN + nup*NSBN + pp ];
    	  	for( int SC=4; SC<=6; ++SC )
    	  	  {
    	  	    const double aux2 = GPU_cscatt(SC)*Wm1_nup_pp_nu_p_i*GPU_sqrtmassXY(nup);
    	  	    const double aux4 = hbar*GPU_omega(SC)*epsstarm1;

    	  	    // PLUS 
    	  	    Gamma = aux + aux4;
    	  	    contrib -= aux2*(1.+aux3*Gamma)*2*GPU_occupations(SC,nu,nup)*(Gamma >= 0);
		    
    	  	    // MINUS 
    	  	    Gamma = aux - aux4;
    	  	    contrib -= aux2*(1.+aux3*Gamma)*2*(GPU_occupations(SC,nu,nup)+1)*(Gamma >= 0);
    	  	  }
    	      }

    	  /*************************************
    	   *               ELASTIC             *
    	   *************************************/
    	  Gamma = ((l+.5)*DW) + seps[ (i-i_0)*_NVALLEYS*NSBN + nu*NSBN + p ] - seps[ (i-i_0)*_NVALLEYS*NSBN + nu*NSBN + pp ];
    	  contrib -= (GPU_cscatt(0)*Wm1_nu_pp_nu_p*(GPU_sqrtmassXY(nu)*(1.+2.*GPU_kane(nu)*EPSSTAR*Gamma))*(Gamma >= 0));

    	  /*************************************
    	   *               G-TYPE              *
    	   *************************************/
	  const double aux = ((l+.5)*DW)+seps[ (i-i_0)*_NVALLEYS*NSBN + nu*NSBN + p ] - seps[ (i-i_0)*_NVALLEYS*NSBN + nu*NSBN + pp ];

    	  for( int SC=1; SC<=3; ++SC )
    	    {
    	      // PLUS 
    	      Gamma = aux + hbar*GPU_omega(SC)*epsstarm1;
    	      contrib -= (GPU_cscatt(SC)*Wm1_nu_pp_nu_p*(GPU_sqrtmassXY(nu)*(1.+2.*GPU_kane(nu)*EPSSTAR*Gamma))*GPU_occupations(SC,nu,nu)*(Gamma >= 0));
		
    	      // MINUS 
    	      Gamma = aux - hbar*GPU_omega(SC)*epsstarm1;
    	      contrib -= (GPU_cscatt(SC)*Wm1_nu_pp_nu_p*(GPU_sqrtmassXY(nu)*(1.+2.*GPU_kane(nu)*EPSSTAR*Gamma))*(GPU_occupations(SC,nu,nu)+1)*(Gamma >= 0));
    	    }

	}
      // GPU_test_loss(nu,p,i,l,m) = -GPU_pdf(nu,p,i,l,m,STAGE)*2.*M_PI*contrib;
      
      GPU_rhs_pdf( nu,p,i,l,m ) += GPU_pdf(nu,p,i,l,m,STAGE)*2.*M_PI*contrib + GPU_rhs_pdf_gain( nu,p,i,l );
    }

}


#else
void MOSFETProblemCuda::CPU_phonons_loss( const int STAGE )
{
  const int NSBN  = host_dm -> get_NSBN();
  const int NX    = host_dm -> get_X()   -> get_N();
  const int NW    = host_dm -> get_W()   -> get_N();
  const int NPHI  = host_dm -> get_PHI() -> get_N();
  const double DW = host_dm->get_W()->get_delta();
  const double DWM1 = host_dm->get_W()->get_delta_m1();

  // double _DW_DWM1[2];
  // _DW_DWM1[0] = _dw;
  // _DW_DWM1[1] = 1./_dw;

#pragma omp parallel for
  for( int global_index=0; global_index < _NVALLEYS*NSBN*NX*NW*NPHI; global_index++ )
    {
      int nu,p,i,l,m;
      GPU_map_1D_to_5D( global_index, &i, NX, &nu, _NVALLEYS, &p, NSBN, &l, NW, &m, NPHI );
      // int index = global_index;
      // i    = index;
      // index  = index/_NPHI;
      // m    = i-index*_NPHI;
      // i    = index;
      // index  = index/_NW;
      // l    = i-index*_NW;
      // i    = index;
      // index  = index/_NSBN;
      // p    = i-index*_NSBN;
      // i    = index;
      // index  = index/_NVALLEYS;
      // nu   = i-index*_NVALLEYS;
      // i    = index;

      double contrib = 0;

      double Gamma;

      double Wm1_nup_pp_nu_p_i;
      const double EPSSTAR = host_rescpar->get_epsstar();
      const double epsstarm1 = 1./EPSSTAR;

      for( int pp=0; pp<NSBN; ++pp )
    	{
    	  const double Wm1_nu_pp_nu_p = Wm1( nu,pp,nu,p,i );

    	  /*************************************
    	   *               F-TYPE              *
    	   *************************************/
    	  for( unsigned int nup=0; nup<_NVALLEYS; ++nup )
    	    if(nup != nu)
    	      {
    	  	const double aux3 = 2.*_kane[nup]*EPSSTAR;

    	  	Wm1_nup_pp_nu_p_i = Wm1( nup,pp,nu,p,i );
		
		const double aux  = ((l+.5)*DW)+eps(nu,p,i) - eps(nup,pp,i);
    	  	for( int SC=4; SC<=6; ++SC )
    	  	  {
    	  	    const double aux2 = _cscatt[SC]*Wm1_nup_pp_nu_p_i*sqrtmassXY(nup);
    	  	    const double aux4 = host_phyco->__hbar*_omega[SC]*epsstarm1;

    	  	    // PLUS 
    	  	    Gamma = aux + aux4;
    	  	    contrib -= aux2*(1.+aux3*Gamma)*2*occupations(SC,nu,nup)*(Gamma >= 0);
		    
    	  	    // MINUS 
    	  	    Gamma = aux - aux4;
    	  	    contrib -= aux2*(1.+aux3*Gamma)*2*(occupations(SC,nu,nup)+1)*(Gamma >= 0);
    	  	  }
    	      }

    	  /*************************************
    	   *               ELASTIC             *
    	   *************************************/
    	  Gamma = ((l+.5)*DW) + eps(nu,p,i) - eps(nu,pp,i);
    	  contrib -= (_cscatt[0]*Wm1_nu_pp_nu_p*(sqrtmassXY(nu)*(1.+2.*_kane[nu]*EPSSTAR*Gamma))*(Gamma >= 0));

    	  /*************************************
    	   *               G-TYPE              *
    	   *************************************/
	  const double aux = ((l+.5)*DW)+eps(nu,p,i) - eps(nu,pp,i);

    	  for( int SC=1; SC<=3; ++SC )
    	    {
    	      // PLUS 
    	      Gamma = aux + host_phyco->__hbar*_omega[SC]*epsstarm1;
    	      contrib -= (_cscatt[SC]*Wm1_nu_pp_nu_p*(sqrtmassXY(nu)*(1.+2.*_kane[nu]*EPSSTAR*Gamma))*occupations(SC,nu,nu)*(Gamma >= 0));
		
    	      // MINUS 
    	      Gamma = aux - host_phyco->__hbar*_omega[SC]*epsstarm1;
    	      contrib -= (_cscatt[SC]*Wm1_nu_pp_nu_p*(sqrtmassXY(nu)*(1.+2.*_kane[nu]*EPSSTAR*Gamma))*(occupations(SC,nu,nu)+1)*(Gamma >= 0));
    	    }
    	}
      
      rhs_pdf( nu,p,i,l,m ) += pdf(nu,p,i,l,m,STAGE)*2.*M_PI*contrib + rhs_pdf_gain( nu,p,i,l );
    }

  return;
}
#endif



/**
   PURPOSE:        

   FILE:           cuda_bte_scatterings_phonons.cu

   NAME:           MOSFETProblem::phonons_loss

   PARAMETERS:

   RETURN VALUE:   none

   CALLS TO:       MOSFETProblem::start_time           (cuda_comptime.h - declared inline)
		   MOSFETProblem::stop_time            (cuda_comptime.h - declared inline)
                   MOSFETProblem::show_eta             (cuda_testing.h - declared inline)
 
   CALLED FROM:    MOSFETProblem::scatterings          (cuda_time_integration.cu)

   DATA MODIFIED:  

   METHOD:         

   LAST UPDATE:    2022 February 16
*/
void MOSFETProblemCuda::phonons_loss ( const int STAGE )
{
#ifdef _SHOW_ETA
  ostringstream message;
  message << "called " << __func__;
  show_eta(message.str());
#endif

  start_time( _PHASE_BTE_SCATTERINGS_PHONONS_PHONONSLOSS );

#ifdef __CUDACC__
  const int NSBN  = host_dm -> get_NSBN();
  const int NX    = host_dm -> get_X()   -> get_N();
  const int NW    = host_dm -> get_W()   -> get_N();
  const int NPHI  = host_dm -> get_PHI() -> get_N();

  // delete from here ??
  checkCudaErrors( cudaMemcpyToSymbol(_GPU_sqrtmassXY,  _sqrtmassXY,  _NVALLEYS*sizeof(double),             0, cudaMemcpyHostToDevice) );
  checkCudaErrors( cudaMemcpyToSymbol(_GPU_mass,        _mass,        9*sizeof(double),                     0, cudaMemcpyHostToDevice) );
  checkCudaErrors( cudaMemcpyToSymbol(_GPU_kane,        _kane,        _NVALLEYS*sizeof(double),             0, cudaMemcpyHostToDevice) );
  checkCudaErrors( cudaMemcpyToSymbol(_GPU_occupations, _occupations, 8*_NVALLEYS*_NVALLEYS*sizeof(double), 0, cudaMemcpyHostToDevice) );
  checkCudaErrors( cudaMemcpyToSymbol(_GPU_cscatt,      _cscatt,      8*sizeof(double),                     0, cudaMemcpyHostToDevice) );
  checkCudaErrors( cudaMemcpyToSymbol(_GPU_omega,       _omega,       8*sizeof(double),                     0, cudaMemcpyHostToDevice) );
  // to here ??

  // int sm_size = _NVALLEYS*NSBN*3*sizeof(double);
  // cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
  const int gridSize      = host_gridconfig -> cuda_phonons_loss_config -> get_gridSize();
  const int blockSize     = host_gridconfig -> cuda_phonons_loss_config -> get_blockSize();
  const int shmemSize     = host_gridconfig -> cuda_phonons_loss_config -> get_shmemSize();
  const cudaFuncCache cfc = host_gridconfig -> cuda_phonons_loss_config -> get_cfc();
  cudaDeviceSetCacheConfig( cfc );
  cuda_phonons_loss <<< gridSize, blockSize, shmemSize >>>   ( device_dm, device_rescpar, device_phyco, _GPU_pdf, _GPU_Wm1, _GPU_rhs_pdf, _GPU_rhs_pdf_gain, _GPU_eps, STAGE, NULL );
#else
  CPU_phonons_loss ( STAGE );
#endif

  stop_time( _PHASE_BTE_SCATTERINGS_PHONONS_PHONONSLOSS );

#ifdef _SHOW_ETA
  cerr << "[ok] (function " << __func__ << ", line " << __LINE__ << ")" << endl;
#endif

  return;
}





#ifdef __CUDACC__
#else
void MOSFETProblemCuda::CPU_phonons_gain( const int STAGE )
{
  const double DW = host_dm->get_W()->get_delta();
  const double DWM1 = host_dm->get_W()->get_delta_m1();

  // double _DW_DWM1[2];
  // _DW_DWM1[0] = _dw;
  // _DW_DWM1[1] = 1./_dw;

  const int NSBN  = host_dm -> get_NSBN();
  const int NX    = host_dm -> get_X()   -> get_N();
  const int NW    = host_dm -> get_W()   -> get_N();
  const int NPHI  = host_dm -> get_PHI() -> get_N();

#pragma omp parallel for
  for(int global_index=0; global_index < _NVALLEYS*NSBN*NX*NW; ++global_index)
    {
      // int index = global_index;
      int i, nu, p, l;
      GPU_map_1D_to_4D( global_index, &i, NX, &nu, _NVALLEYS, &p, NSBN, &l, NW );
      // int i        = index;
      // index        = index/_NW;
      // int l        = i-index*_NW;
      // i            = index;
      // index        = index/_NSBN;
      // const int p  = i-index*_NSBN;
      // i            = index;
      // index        = index/_NVALLEYS;
      // const int nu = i-index*_NVALLEYS;
      // i            = index;

      double contrib = 0;

      const double eps_nu_p_i = eps( nu,p,i );

      const double EPSSTAR = host_rescpar->get_epsstar();
      
      double Gamma;
      int ldown;

      double Wm1_nup_pp_nu_p_i;
      double eps_nup_pp_i;
      double eps_nu_pp_i;
      double Wm1_nu_pp_nu_p;

      for( int pp=0; pp<NSBN; ++pp )
	{
	  /*************************************
	   *               F-TYPE              *
	   *************************************/
	  for( int nup=0; nup<_NVALLEYS; ++nup )
	    if( nup != nu )
	      {
		Wm1_nup_pp_nu_p_i = Wm1( nup,pp,nu,p,i );
		eps_nup_pp_i = eps( nup,pp,i );
		
		for( int SC=4; SC<=6; ++SC )
		  {
		    // PLUS 
		    Gamma = ((l+.5)*DW)+eps_nu_p_i - eps_nup_pp_i + host_phyco->__hbar*_omega[SC]/EPSSTAR;
		    ldown = (int)(Gamma*DWM1-.5);
		    
		    contrib += (_cscatt[SC]*(sqrtmassXY(nu)*(1.+2.*_kane[nu]*EPSSTAR*((l+.5)*DW)))*2*(occupations(SC,nup,nu)+1)*Wm1_nup_pp_nu_p_i*(( (Gamma >= 0 && ldown<=NW-2) ? (integrated_pdf_energy(nup,pp,i,(ldown+1))-integrated_pdf_energy(nup,pp,i,ldown))*DWM1*Gamma+(((ldown+1)+0.5)*DW*integrated_pdf_energy(nup,pp,i,ldown)-((ldown+0.5)*DW)*integrated_pdf_energy(nup,pp,i,(ldown+1)))*DWM1 : 0 )));
		    
		    // MINUS 
		    Gamma = ((l+.5)*DW)+eps_nu_p_i - eps_nup_pp_i - host_phyco->__hbar*_omega[SC]/EPSSTAR;
		    ldown = (int)(Gamma*DWM1-.5);

		    contrib += (_cscatt[SC]*(sqrtmassXY(nu)*(1.+2.*_kane[nu]*EPSSTAR*((l+.5)*DW)))*2*occupations(SC,nup,nu)*Wm1_nup_pp_nu_p_i*(( (Gamma >= 0 && ldown<=NW-2) ? (integrated_pdf_energy(nup,pp,i,(ldown+1))-integrated_pdf_energy(nup,pp,i,ldown))*DWM1*Gamma+(((ldown+1)+0.5)*DW*integrated_pdf_energy(nup,pp,i,ldown)-((ldown+0.5)*DW)*integrated_pdf_energy(nup,pp,i,(ldown+1)))*DWM1 : 0 )));
		  }
	      }

	  eps_nu_pp_i = eps( nu,pp,i );
	  Wm1_nu_pp_nu_p = Wm1( nu,pp,nu,p,i );

	  /*************************************
	   *               ELASTIC             *
	   *************************************/
	  Gamma = ((l+.5)*DW) + eps_nu_p_i - eps_nu_pp_i;
	  ldown = (int)(Gamma*DWM1-.5);
	  
	  contrib += (_cscatt[0]*Wm1_nu_pp_nu_p*(sqrtmassXY(nu)*(1.+2.*_kane[nu]*EPSSTAR*((l+.5)*DW)))*(( (Gamma >= 0 && ldown<=NW-2) ? (integrated_pdf_energy(nu,pp,i,(ldown+1))-integrated_pdf_energy(nu,pp,i,ldown))*DWM1*Gamma+(((ldown+1)+0.5)*DW*integrated_pdf_energy(nu,pp,i,ldown)-((ldown+0.5)*DW)*integrated_pdf_energy(nu,pp,i,(ldown+1)))*DWM1 : 0 )));

	  /*************************************
	   *               G-TYPE              *
	   *************************************/
	  for( int SC=1; SC<=3; ++SC )
	    {
	      // PLUS 
	      Gamma = ((l+.5)*DW)+eps_nu_p_i - eps_nu_pp_i + host_phyco->__hbar*_omega[SC]/EPSSTAR;
	      ldown = (int)(Gamma*DWM1-.5);
		
	      contrib += (_cscatt[SC]*(sqrtmassXY(nu)*(1.+2.*_kane[nu]*EPSSTAR*((l+.5)*DW)))*(occupations(SC,nu,nu)+1)*Wm1_nu_pp_nu_p*(( (Gamma >= 0 && ldown<=NW-2) ? (integrated_pdf_energy(nu,pp,i,(ldown+1))-integrated_pdf_energy(nu,pp,i,ldown))*DWM1*Gamma+(((ldown+1)+0.5)*DW*integrated_pdf_energy(nu,pp,i,ldown)-((ldown+0.5)*DW)*integrated_pdf_energy(nu,pp,i,(ldown+1)))*DWM1 : 0 )));
		
	      // MINUS 
	      Gamma = ((l+.5)*DW)+eps_nu_p_i - eps_nu_pp_i - host_phyco->__hbar*_omega[SC]/EPSSTAR;
	      ldown = (int)(Gamma*DWM1-.5);

	      contrib += (_cscatt[SC]*(sqrtmassXY(nu)*(1.+2.*_kane[nu]*EPSSTAR*((l+.5)*DW)))*occupations(SC,nu,nu)*Wm1_nu_pp_nu_p*(( (Gamma >= 0 && ldown<=NW-2) ? (integrated_pdf_energy(nu,pp,i,(ldown+1))-integrated_pdf_energy(nu,pp,i,ldown))*DWM1*Gamma+(((ldown+1)+0.5)*DW*integrated_pdf_energy(nu,pp,i,ldown)-((ldown+0.5)*DW)*integrated_pdf_energy(nu,pp,i,(ldown+1)))*DWM1 : 0 )));
	    }
	}
      
      rhs_pdf_gain( nu,p,i,l ) = contrib;
    }

  return;
}
#endif
/**
   PURPOSE:        

   FILE:           cuda_bte_scatterings_phonons.cu

   NAME:           MOSFETProblem::phonons_gain

   PARAMETERS:

   RETURN VALUE:   none

   CALLS TO:       MOSFETProblem::start_time           (cuda_comptime.h - declared inline)
		   MOSFETProblem::stop_time            (cuda_comptime.h - declared inline)
                   MOSFETProblem::show_eta             (cuda_testing.h - declared inline)
 
   CALLED FROM:    MOSFETProblem::scatterings          (cuda_time_integration.cu)

   DATA MODIFIED:  

   METHOD:         

   LAST UPDATE:    2022 February 16
*/
void MOSFETProblemCuda::phonons_gain( const int STAGE )
{
#ifdef _SHOW_ETA
  ostringstream message;
  message << "called " << __func__;
  show_eta(message.str());
#endif
  
  start_time( _PHASE_BTE_SCATTERINGS_PHONONS_PHONONSGAIN );

#ifdef __CUDACC__
  const int NSBN  = host_dm -> get_NSBN();
  const int NX    = host_dm -> get_X()   -> get_N();
  const int NW    = host_dm -> get_W()   -> get_N();
  const int NPHI  = host_dm -> get_PHI() -> get_N();

  // delete from here ??
  checkCudaErrors( cudaMemcpyToSymbol(_GPU_sqrtmassXY,  _sqrtmassXY,  _NVALLEYS*sizeof(double),             0, cudaMemcpyHostToDevice) );
  checkCudaErrors( cudaMemcpyToSymbol(_GPU_mass,        _mass,        9*sizeof(double),                     0, cudaMemcpyHostToDevice) );
  checkCudaErrors( cudaMemcpyToSymbol(_GPU_kane,        _kane,        _NVALLEYS*sizeof(double),             0, cudaMemcpyHostToDevice) );
  checkCudaErrors( cudaMemcpyToSymbol(_GPU_occupations, _occupations, 8*_NVALLEYS*_NVALLEYS*sizeof(double), 0, cudaMemcpyHostToDevice) );
  checkCudaErrors( cudaMemcpyToSymbol(_GPU_cscatt,      _cscatt,      8*sizeof(double),                     0, cudaMemcpyHostToDevice) );
  checkCudaErrors( cudaMemcpyToSymbol(_GPU_omega,       _omega,       8*sizeof(double),                     0, cudaMemcpyHostToDevice) );
  // to here ??

  const int gridSize      = host_gridconfig -> cuda_phonons_gain_config -> get_gridSize();
  const int blockSize     = host_gridconfig -> cuda_phonons_gain_config -> get_blockSize();
  const int shmemSize     = host_gridconfig -> cuda_phonons_gain_config -> get_shmemSize();
  const cudaFuncCache cfc = host_gridconfig -> cuda_phonons_gain_config -> get_cfc();
  cudaDeviceSetCacheConfig( cfc );
  cuda_phonons_gain <<< gridSize, blockSize, shmemSize >>> ( device_dm, device_rescpar, device_phyco, _GPU_integrated_pdf_energy, _GPU_Wm1, _GPU_rhs_pdf_gain, _GPU_eps, STAGE, NULL );
#else
  CPU_phonons_gain ( STAGE );
#endif

  stop_time( _PHASE_BTE_SCATTERINGS_PHONONS_PHONONSGAIN );

#ifdef _SHOW_ETA
  cerr << "[ok] (function " << __func__ << ", line " << __LINE__ << ")" << endl;
#endif
  return;
}



#ifdef __CUDACC__
__global__ void cuda_compute_Wm1( const discrMeshes *dm, double *_GPU_Wm1, const double *_GPU_chi )
{
  int global_index = blockIdx.x*blockDim.x + threadIdx.x;

  const int NSBN     = dm->get_NSBN();
  const int NX       = dm->get_X()->get_N();
  const int NZ       = dm->get_Z()->get_N();
  const int NW       = dm->get_W()->get_N();
  const int NPHI     = dm->get_PHI()->get_N();
  const double DZ    = dm->get_Z()->get_delta();
  
  if( global_index < _NVALLEYS*NSBN*_NVALLEYS*NSBN*NX )
    {
      int i, nu, p, nup, pp;
      GPU_map_1D_to_5D( global_index, &i, NX, &nu, _NVALLEYS, &p, NSBN, &nup, _NVALLEYS, &pp, NSBN );

      double sum = 0;
      for( int j=1; j<=NZ-2; ++j )
	{
	  const double c1 = GPU_chi(nu,p,i,j);
	  const double c2 = GPU_chi(nup,pp,i,j);
	  sum += c1*c1 * c2*c2;
	}
      GPU_Wm1(nu,p,nup,pp,i) = DZ*sum;
    }
}
#else
void MOSFETProblemCuda::CPU_compute_Wm1()
{
  const int NSBN  = host_dm->get_NSBN();
  const int NX    = host_dm->get_X()->get_N();
  const int NZ    = host_dm->get_Z()->get_N();
  const int NW    = host_dm->get_W()->get_N();
  const int NPHI  = host_dm->get_PHI()->get_N();
  const double DZ = host_dm->get_Z()->get_delta();

#pragma omp parallel for
  for( int global_index=0; global_index < _NVALLEYS*NSBN*_NVALLEYS*NSBN*NX; ++global_index )
    {
      int i, nu, p, nup, pp;
      GPU_map_1D_to_5D( global_index, &i, NX, &nu, _NVALLEYS, &p, NSBN, &nup, _NVALLEYS, &pp, NSBN );
      // int index = global_index;
      // i    = index;
      // index  = index/_NSBN;
      // pp    = i-index*_NSBN;
      // i    = index;
      // index  = index/_NVALLEYS;
      // nup   = i-index*_NVALLEYS;
      // i    = index;
      // index  = index/_NSBN;
      // p    = i-index*_NSBN;
      // i    = index;
      // index  = index/_NVALLEYS;
      // nu   = i-index*_NVALLEYS;
      // i    = index;

      double sum = 0;
      for( int j=1; j<=NZ-2; ++j )
	{
	  const double c1 = chi(nu,p,i,j);
	  const double c2 = chi(nup,pp,i,j);
	  sum += c1*c1 * c2*c2;
	}
      Wm1(nu,p,nup,pp,i) = DZ*sum;
    }
}
#endif
/**
   PURPOSE:        

   FILE:           cuda_bte_scatterings_phonons.cu

   NAME:           MOSFETProblem::compute_Wm1

   PARAMETERS:

   RETURN VALUE:   none

   CALLS TO:       MOSFETProblem::start_time           (cuda_comptime.h - declared inline)
		   MOSFETProblem::stop_time            (cuda_comptime.h - declared inline)
 
   CALLED FROM:    MOSFETProblem::scatterings          (cuda_time_integration.cu)

   DATA MODIFIED:  

   METHOD:         

   LAST UPDATE:    2022 February 16
*/
void MOSFETProblemCuda::compute_Wm1()
{
  start_time( _PHASE_BTE_SCATTERINGS_PHONONS_WM1 );

#ifdef __CUDACC__
  const int NSBN  = host_dm -> get_NSBN();
  const int NX    = host_dm -> get_X()   -> get_N();
  const int NW    = host_dm -> get_W()   -> get_N();
  const int NPHI  = host_dm -> get_PHI() -> get_N();

  // cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
  const int gridSize      = host_gridconfig -> cuda_compute_Wm1_config -> get_gridSize();
  const int blockSize     = host_gridconfig -> cuda_compute_Wm1_config -> get_blockSize();
  const int shmemSize     = host_gridconfig -> cuda_compute_Wm1_config -> get_shmemSize();
  const cudaFuncCache cfc = host_gridconfig -> cuda_compute_Wm1_config -> get_cfc();
  cudaDeviceSetCacheConfig( cfc );
  cuda_compute_Wm1 <<< gridSize, blockSize, shmemSize >>> ( device_dm, _GPU_Wm1, _GPU_chi );
#else
  CPU_compute_Wm1 ();
#endif

  stop_time( _PHASE_BTE_SCATTERINGS_PHONONS_WM1 );

  return;
}


cuda_bte_scatterings_phonons_kernels_config::cuda_bte_scatterings_phonons_kernels_config( const discrMeshes *dm )
{
  const int NSBN  = dm -> get_NSBN();
  const int NX    = dm -> get_X()   -> get_N();
  const int NW    = dm -> get_W()   -> get_N();
  const int NPHI  = dm -> get_PHI() -> get_N();

  cuda_phonons_loss_config = new kernelConfig( nblocks( _NVALLEYS*NSBN*NX*NW*NPHI, 32 ), 32, _NVALLEYS*NSBN*3*sizeof(double), cudaFuncCachePreferShared, "cuda_phonons_loss" );

  cuda_phonons_gain_config = new kernelConfig( nblocks( _NVALLEYS*NSBN*NX*NW*NPHI, 64 ), 64, NOSHMEM                        , cudaFuncCachePreferNone,   "cuda_phonons_gain" );

  cuda_compute_Wm1_config  = new kernelConfig( nblocks( _NVALLEYS*NSBN*NX*NW*NPHI, 64 ), 64, NOSHMEM                        , cudaFuncCachePreferL1,     "cuda_compute_Wm1"  );
}
