#include "mosfetproblem.h"
#include "debug_flags.h"

int MOSFETProblemCuda::generate_films_and_graphs()
{
  ofstream ostr;
  ostr.open("generate_films_and_graphs.sh", ios_base::out);

  // FILMS (TIME-EVOLUTION OF MULTI-DIMENSIONAL MAGNITUDES)
  ostr << "gnuplot scripts/film_currdens.gp" << endl;
  ostr << "gnuplot scripts/film_driftvel.gp" << endl;
  ostr << "gnuplot scripts/film_test.gp" << endl;
  ostr << "gnuplot scripts/film_eps.gp" << endl;
  ostr << "gnuplot scripts/film_potential.gp" << endl;
  ostr << "gnuplot scripts/film_surfdens.gp" << endl;
  ostr << "gnuplot scripts/film_totvoldens.gp" << endl;
  if( host_solvpar->get_ROUGHNESS() == true )
    {
      ostr << "gnuplot scripts/film_pot_ext.gp" << endl;
      ostr << "gnuplot scripts/film_totvoldens_ext.gp" << endl;
    }

  // GRAPHS (TIME-EVOLUTION OF SCALARS)
  ostr << "gnuplot scripts/graph_avgcurr.gp" << endl;
  ostr << "gnuplot scripts/graph_totmass.gp" << endl;
  ostr << "gnuplot scripts/graph_dt.gp" << endl;
  ostr << "gnuplot scripts/graph_maxpdfborder.gp" << endl;

  ostr.close();
  if(system("chmod +x generate_films_and_graphs.sh"));

  return 0;
}




void MOSFETProblemCuda::update_min_max(const int st)
{
  const int NSBN  = host_dm -> get_NSBN();
  const int NX    = host_dm -> get_X()   -> get_N();
  const int NZ    = host_dm -> get_Z()   -> get_N();
  const int NW    = host_dm -> get_W()   -> get_N();
  const int NPHI  = host_dm -> get_PHI() -> get_N();

  if(st == 0)
    {
      _min_currdens =  1.e36;
      _max_currdens = -1.e36;
    }
  for( int i=0; i<NX; ++i )
    {
      if(_totcurrdens[i] > _max_currdens) _max_currdens = _totcurrdens[i];
      if(_totcurrdens[i] < _min_currdens) _min_currdens = _totcurrdens[i];
    }

  // ____________________________________________________________

  if(st == 0)
    {
      _min_driftvel =  1.e36;
      _max_driftvel = -1.e36;
    }
  for( int i=0; i<NX; ++i )
    {
      if(_avgdriftvel[i] > _max_driftvel) _max_driftvel = _avgdriftvel[i];
      if(_avgdriftvel[i] < _min_driftvel) _min_driftvel = _avgdriftvel[i];
    }

  // ____________________________________________________________

  if(st == 0)
    {
      _min_test =  1.e36;
      _max_test = -1.e36;
    }
  for( int l=0; l<NW; ++l )
    {
      if(log(_integrated_pdf_nu_p_i_m[l]) > _max_test) _max_test = log(_integrated_pdf_nu_p_i_m[l]);
      if(log(_integrated_pdf_nu_p_i_m[l]) < _min_test) _min_test = log(_integrated_pdf_nu_p_i_m[l]);
    }

  // ____________________________________________________________

  if(st == 0)
    {
      _min_totvoldens =  1.e36;
      _max_totvoldens = -1.e36;
    }
  for(int i=0; i<NX; ++i)
    for(int j=0; j<NZ; ++j)
      {
	if(totvoldens(i,j) > _max_totvoldens) _max_totvoldens = totvoldens(i,j);
	if(totvoldens(i,j) < _min_totvoldens) _min_totvoldens = totvoldens(i,j);
      }

  // ____________________________________________________________

  if(st == 0)
    {
      _min_totsurfdens =  1.e36;
      _max_totsurfdens = -1.e36;
    }
  for( int i=0; i<NX; ++i )
    {
      if(_totsurfdens[i] > _max_totsurfdens) _max_totsurfdens = _totsurfdens[i];
      if(_totsurfdens[i] < _min_totsurfdens) _min_totsurfdens = _totsurfdens[i];
    }

  // ____________________________________________________________

  if(st == 0)
    {
      _min_eps =  1.e36;
      _max_eps = -1.e36;
    }
  for( int i=0; i<NX; ++i )
    for( int nu=0; nu<_NVALLEYS; ++nu )
      for (int p=0; p<NSBN; ++p )
	{
	  if(eps(i,nu,p) > _max_eps) _max_eps = eps(i,nu,p);
	  if(eps(i,nu,p) < _min_eps) _min_eps = eps(i,nu,p);
	}

  if(st == 0)
    {
      _min_deps_dx =  1.e36;
      _max_deps_dx = -1.e36;
    }
  for( int i=0; i<NX; ++i )
    for( int nu=0; nu<_NVALLEYS; ++nu )
      for (int p=0; p<NSBN; ++p )
	{
	  if(deps_dx(i,nu,p) > _max_deps_dx) _max_deps_dx = deps_dx(i,nu,p);
	  if(deps_dx(i,nu,p) < _min_deps_dx) _min_deps_dx = deps_dx(i,nu,p);
	}


  // ____________________________________________________________

  if(st == 0)
    {
      _min_potential =  1.e36;
      _max_potential = -1.e36;
    }
  for(int i=0; i<NX; ++i) 
    for(int j=0; j<NZ; ++j)
      if(pot(i,j) > _max_potential) _max_potential = pot(i,j);

  for(int i=0; i<NX; ++i) 
    for(int j=0; j<NZ; ++j)
      if(pot(i,j) < _min_potential) _min_potential = pot(i,j);

  return;
}



/**
   PURPOSE:        

   FILE:           cuda_filestorage.cu

   NAME:           MOSFETProblem::store_data

   PARAMETERS:

   RETURN VALUE:   none

   CALLS TO:       MOSFETProblem::macro          (cuda_reductions.cu)
		   MOSFETProblem::store_scalars  (cuda_filestorage.cu)

   CALLED FROM:    MOSFETProblem::solve_2        (cuda_solve.cu)

   DATA MODIFIED:  

   METHOD:         

   LAST UPDATE:    2022 February 16
*/
void MOSFETProblemCuda::store_data( const bool force )
{
  const int NX    = host_dm -> get_X()   -> get_N();
  const double ZWIDTH  = host_physdesc -> get_ZWIDTH();

  /***************************************************
   * then, store everything                          *
   ***************************************************/
  // store_scalars();
  ostringstream sstr;
  sstr<<"results/scalars.dat";
  string filename = sstr.str();
  ofstream str;
  str.open( filename.c_str(),ios_base::app );
  str << setprecision(12) 
      << setw(25) << _time*host_rescpar->get_tstar() // 1
      << setw(25) << _totmass // 2
      << setw(25) << _avgcurr // 3
      << setw(25) << _step // 4
      << setw(25) << _dt*host_rescpar->get_tstar() // 5
      << setw(25) << fabs(max_pdf_border(0)) // 6
      << setw(25) << _totsurfdens[     0 ]*host_rescpar->get_rhostar() - _intnd*host_rescpar->get_nstar()*ZWIDTH // 7
      << setw(25) << _totsurfdens[ NX-1 ]*host_rescpar->get_rhostar() - _intnd*host_rescpar->get_nstar()*ZWIDTH // 8
      << setw(25) << (_totsurfdens_eq[    0 ]-_totsurfdens[     0 ])*host_rescpar->get_rhostar() // 9
      << setw(25) << (_totsurfdens_eq[NX-1 ]-_totsurfdens[ NX-1 ])*host_rescpar->get_rhostar() // 10
      << setw(25) << iter_counter // 11
      << setw(25) << _avgdriftvel_scalar // 12
      << setw(25) << _avgdriftvel_max // 13
      << endl;
  str.close();
  
  if( _step % host_solvpar->get_FREQ() == 0 || force == true )
    {
      ostringstream message;
      message << "storing data into files...";
      show_eta(message.str());

      update_min_max(_step);
      store_totvoldens_dat();
      store_chi_dat();
      store_surfdens_dat();
      store_eps_dat();
      store_potential_dat();
      store_currdens_dat();
      store_driftvel_dat();
      store_test_dat();
	  
      write_gnuplotscripts_films();
      write_gnuplotscripts_scalars();
      generate_films_and_graphs();

      cerr << "[ok]" << endl;
    }

  return;
}


void MOSFETProblemCuda::store_currdens_dat()
{
  const int NSBN  = host_dm -> get_NSBN();
  const int NX    = host_dm -> get_X()   -> get_N();
  const double XLENGTH = host_physdesc -> compute_XLENGTH();
  
  ostringstream sstr;
  sstr << "results/currdens." << setw(DIGITS) << setfill('0') << _step << ".dat";
  ofstream str;
  str.open( sstr.str().c_str(),ios_base::out );
  str << scientific << setprecision(16);
  for( int i=0; i<NX; ++i )
    {
      const double xi = host_dm->get_X()->mesh(i);
      str << setw(25) << xi*XLENGTH
	  << setw(25) << _totcurrdens[ i ]*host_rescpar->get_jstar();

      for( int nu=0; nu<_NVALLEYS; ++nu )
	for( int p=0; p<NSBN; ++p )
	  str << setw(25) << currdens(nu,p,i)*host_rescpar->get_jstar();
      str << endl;
    }
  str.close();

  return;
}




void MOSFETProblemCuda::store_driftvel_dat()
{
  const int NX    = host_dm -> get_X()   -> get_N();
  const double XLENGTH = host_physdesc -> compute_XLENGTH();

  ostringstream sstr;
  sstr << "results/driftvel." << setw(DIGITS) << setfill('0') << _step << ".dat";
  ofstream str;
  str.open( sstr.str().c_str(),ios_base::out );
  str << scientific << setprecision(16);
  for( int i=0; i<NX; ++i )
    {
      const double xi = host_dm->get_X()->mesh(i);
      str << setw(25) << xi*XLENGTH
	  << setw(25) << host_rescpar->get_vstar()*_avgdriftvel[ i ]
	  << endl;
    }
  str.close();

  return;
}



void MOSFETProblemCuda::store_test_dat()
{
  const int NX    = host_dm -> get_X()   -> get_N();
  const int NW    = host_dm -> get_W()   -> get_N();

  ostringstream sstr;
  sstr << "results/test." << setw(DIGITS) << setfill('0') << _step << ".dat";
  ofstream str;
  str.open( sstr.str().c_str(),ios_base::out );
  str << scientific << setprecision(16);
  for( int l=0; l<NW; ++l )
    str << setw(25) << host_dm->get_W()->mesh(l)
	<< setw(25) << _integrated_pdf_nu_p_i_m[l]
	<< endl;
  str.close();

  return;
}



void MOSFETProblemCuda::store_totvoldens_dat()
{
  const int NSBN  = host_dm -> get_NSBN();
  const int NX    = host_dm -> get_X()   -> get_N();
  const int NZ    = host_dm -> get_Z()   -> get_N();
  const int NZEXT = host_dm -> get_ZEXT()-> get_N();
  const double XLENGTH = host_physdesc -> compute_XLENGTH();
  const double ZWIDTH  = host_physdesc -> get_ZWIDTH();

  ostringstream sstr;
  sstr << "results/voldens." << setw(DIGITS) << setfill('0') << _step << ".dat";
  ofstream str;
  str << scientific << setprecision(16);
  str.open( sstr.str().c_str(),ios_base::out );
  for(int i=0; i<NX; ++i)
    for(int j=0; j<NZ; ++j)
      {
	const double xi = host_dm->get_X()->mesh(i);
	const double zj = host_dm->get_Z()->mesh(j);
	str << setw(25) << xi*XLENGTH
	    << setw(25) << zj*ZWIDTH
	    << setw(25) << totvoldens( i,j )*host_rescpar->get_nstar();
	for( int nu=0; nu<_NVALLEYS; ++nu )
	  for (int p=0; p<NSBN; ++p )
	    str << setw(25) << voldens(nu,p,i,j)*host_rescpar->get_nstar();
	str << endl;

	if(j == NZ-1) str << endl;
      }
  str.close();

  if( host_solvpar->get_ROUGHNESS() == true )
    {
      sstr.str("");
      sstr<<"results/totvoldens_ext."<<setw(DIGITS)<<setfill('0')<<_step<<".dat";
      ofstream str;
      str.open( sstr.str().c_str(),ios_base::out );
      str << scientific << setprecision(16);
      for( int i=0; i<NX; ++i )
	{
	  const double xi = host_dm->get_X()->mesh(i);
	  for( int j=0; j<NZEXT; ++j )
	    str << setw(25) << xi*XLENGTH
		<< setw(25) << host_dm -> get_ZEXT() -> mesh(j)*ZWIDTH
		<< setw(25) << totvoldens_ext( i,j )*host_rescpar->get_nstar()
		<< setw(25) << nd_ext( i,j )*host_rescpar->get_nstar()
		<< setw(25) << epsr_ext( i,j )
		<< endl;
	  str<<endl;
	}
      str.close();
    }

  return;
}



void MOSFETProblemCuda::store_chi_dat()
{
  const int NSBN  = host_dm -> get_NSBN();
  const int NX    = host_dm -> get_X()   -> get_N();
  const int NZ    = host_dm -> get_Z()   -> get_N();

  ostringstream sstr;
  sstr << "results/chi." << setw(DIGITS) << setfill('0') << _step << ".dat";
  ofstream str;
  str << scientific << setprecision(16);
  str.open( sstr.str().c_str(),ios_base::out );
  for(int i=0; i<NX; ++i)
    for(int j=1; j<NZ-1; ++j)
      {
	for( int nu=0; nu<_NVALLEYS; ++nu )
	  for (int p=0; p<NSBN; ++p )
	    str << setw(25) << chi(nu,p,i,j);
	str << endl;

	if(j == NZ-2) str << endl;
      }
  str.close();

  return;
}



void MOSFETProblemCuda::store_surfdens_dat()
{
  const int NSBN  = host_dm -> get_NSBN();
  const int NX    = host_dm -> get_X()   -> get_N();
  const double XLENGTH = host_physdesc -> compute_XLENGTH();

  ostringstream sstr;
  
#ifdef __CUDACC__
  cudaMemcpy( _surfdens, _GPU_surfdens, _NVALLEYS*NSBN*NX*sizeof(double), cudaMemcpyDeviceToHost );
#endif
  
  sstr << "results/surfdens." << setw(DIGITS) << setfill('0') << _step << ".dat";
  ofstream str;
  str.open( sstr.str().c_str(),ios_base::out );
  str << scientific << setprecision(16);
  for( int i=0; i<NX; ++i )
    {
      const double xi = host_dm->get_X()->mesh(i);
      str << setw(25) << xi*XLENGTH
	  << setw(25) << _totsurfdens[ i ]*host_rescpar->get_rhostar();
      for( int nu=0; nu<_NVALLEYS; ++nu )
	for (int p=0; p<NSBN; ++p )
	  str << setw(25) << surfdens( nu,p,i )*host_rescpar->get_rhostar();
      str<<endl;
    }
  str.close();

  return;
}

void MOSFETProblemCuda::store_eps_dat()
{
  const int NSBN  = host_dm -> get_NSBN();
  const int NX    = host_dm -> get_X()   -> get_N();
  const double XLENGTH = host_physdesc -> compute_XLENGTH();

  ostringstream sstr;

#ifdef __CUDACC__
  cudaMemcpy( _eps, _GPU_eps, _NVALLEYS*NSBN*NX*sizeof(double), cudaMemcpyDeviceToHost );
#endif

  sstr << "results/eps." << setw(DIGITS) << setfill('0') << _step << ".dat";
  ofstream str;
  str.open( sstr.str().c_str(),ios_base::out );
  str << scientific << setprecision(16);
  for( int i=0; i<NX; ++i )
    {
      const double xi = host_dm->get_X()->mesh(i);
      str << setw(25) << xi*XLENGTH;
      for( int nu=0; nu<_NVALLEYS; ++nu )
  	for (int p=0; p<NSBN; ++p )
  	  str << setw(25) << eps( nu,p,i )*host_rescpar->get_epsstar()/host_phyco->__eV;
      str<<endl;
    }
  str.close();

  sstr.str("");

#ifdef __CUDACC__
  cudaMemcpy( _deps_dx, _GPU_deps_dx, _NVALLEYS*NSBN*NX*sizeof(double), cudaMemcpyDeviceToHost );
#endif

sstr << "results/deps_dx." << setw(DIGITS) << setfill('0') << _step << ".dat";
  str.open( sstr.str().c_str(),ios_base::out );
  str << scientific << setprecision(16);
  for( int i=0; i<NX; ++i )
    {
      const double xi = host_dm->get_X()->mesh(i);
      str << setw(25) << xi*XLENGTH;
      for( int nu=0; nu<_NVALLEYS; ++nu )
	for (int p=0; p<NSBN; ++p )
	  str << setw(25) << deps_dx( nu,p,i )*host_rescpar->get_epsstar()/(host_phyco->__eV*XLENGTH);
      str<<endl;
    }
  str.close();

  return;
}

void MOSFETProblemCuda::store_potential_dat()
{
  const int NX    = host_dm -> get_X()   -> get_N();
  const int NZ    = host_dm -> get_Z()   -> get_N();
  const int NZEXT = host_dm ->get_ZEXT() -> get_N();
  const double XLENGTH = host_physdesc -> compute_XLENGTH();
  const double ZWIDTH  = host_physdesc -> get_ZWIDTH();

#ifdef __CUDACC__
  checkCudaErrors( cudaMemcpy(_pot, _GPU_pot, NX*NZ*sizeof(double), cudaMemcpyDeviceToHost) );
  if( host_solvpar->get_ROUGHNESS() == true )
    {
      checkCudaErrors( cudaMemcpy(_pot_ext, _GPU_pot_ext, NX*NZEXT*sizeof(double), cudaMemcpyDeviceToHost) );
      checkCudaErrors( cudaMemcpy(_Deltapot_upper, _GPU_Deltapot_upper, NX*NZ*sizeof(double), cudaMemcpyDeviceToHost) );
      checkCudaErrors( cudaMemcpy(_Deltapot_lower, _GPU_Deltapot_lower, NX*NZ*sizeof(double), cudaMemcpyDeviceToHost) );
    }
#endif

  ostringstream sstr;
  sstr<<"results/potential."<<setw(DIGITS)<<setfill('0')<<_step<<".dat";
  ofstream str;
  str.open( sstr.str().c_str(),ios_base::out );
  str << scientific << setprecision(16);
  for( int i=0; i<NX; ++i )
    {
      const double xi = host_dm->get_X()->mesh(i);
      for( int j=0; j<NZ; ++j )
	{
	  const double zj = host_dm->get_Z()->mesh(j);
	  str << setw(25) << xi*XLENGTH
	      << setw(25) << zj*ZWIDTH
	      << setw(25) << pot( i,j )*host_rescpar->get_potstar()
	      << setw(25) << Deltapot_upper( i,j )*host_rescpar->get_potstar()
	      << setw(25) << Deltapot_lower( i,j )*host_rescpar->get_potstar()
	      << endl;
	}
      str<<endl;
    }
  str.close();

  if( host_solvpar->get_ROUGHNESS() == true )
    {
      sstr.str("");
      sstr<<"results/pot_ext."<<setw(DIGITS)<<setfill('0')<<_step<<".dat";
      ofstream str;
      str.open( sstr.str().c_str(),ios_base::out );
      str << scientific << setprecision(16);
      for( int i=0; i<NX; ++i )
	{
	  const double xi = host_dm->get_X()->mesh(i);
	  for( int j=0; j<NZEXT; ++j )
	    str << setw(25) << xi*XLENGTH
		<< setw(25) << host_dm -> get_ZEXT() -> mesh(j)*ZWIDTH
		<< setw(25) << pot_ext( i,j )*host_rescpar->get_potstar()
		<< endl;
	  str<<endl;
	}
      str.close();
    }

  return;
}




void MOSFETProblemCuda::write_gnuplotscripts_films()
{
  write_gnuplotscript_totvoldens();
  write_gnuplotscript_surfdens();
  write_gnuplotscript_eps();
  write_gnuplotscript_potential();
  write_gnuplotscript_currdens();
  write_gnuplotscript_driftvel();
  write_gnuplotscript_test();

  return;
}


void MOSFETProblemCuda::write_gnuplotscript_currdens()
{
  const int NSBN  = host_dm -> get_NSBN();
  const int NX    = host_dm -> get_X()   -> get_N();
  const double LSOURCE  = host_physdesc -> get_LSOURCE();
  const double LCHANNEL = host_physdesc -> get_LCHANNEL();
  const double LDRAIN   = host_physdesc -> get_LDRAIN();

  ofstream str;
  str.open( "scripts/film_currdens.gp",ios_base::out );
  for(int st=0; st<=_step; st+=host_solvpar->get_FREQ())
    {
      str << "!/bin/echo -en 'generating currdens.avi : step = " << st << "/" << _step << "                                      \\r' 1>&2" << endl;

      str<<"set terminal png enh"<<endl;
      ostringstream sstr;
      sstr  << "results/currdens."             << setw(DIGITS) << setfill('0') << st;

      const string filename_dat = sstr.str()+".dat";
      const string filename_png = sstr.str()+".png";
      str<<"set output '"<<filename_png<<"'"<<endl;
      str<<"set xlabel 'x [nm]'"<<endl;
      str<<"set xtics ( '0' 0,'"<<LSOURCE/1.E-9<<"' "<<LSOURCE<<",'"<<(LSOURCE+LCHANNEL)/1.E-9<<"' "<<LSOURCE+LCHANNEL<<",'"<<(LSOURCE+LCHANNEL+LDRAIN)/1.E-9<<"' "<<LSOURCE+LCHANNEL+LDRAIN<<" )"<<endl;
      str<<"set ylabel 'j [A m^{-1}]'"<<endl;

      str<<"set size 1,1"<<endl;
      str<<"set multiplot"<<endl;
      str<<"set size 1,.5"<<endl;
      str<<"set origin 0,.5"<<endl;
      str<<"unset key"<<endl;
      str<<"set title 'total current at approximate time "<<(_time/_step*st)*host_rescpar->get_tstar()/1.E-12<<" ps'"<<endl;
      str<<"set yrange [" << _min_currdens*host_rescpar->get_jstar() << ":" << _max_currdens*host_rescpar->get_jstar() << "]"<<endl;
      str<<"set ytics"<<endl;
      str<<"plot '" << filename_dat << "' using 1:2 with lines"<<endl;
      str<<"set size .33,.5"<<endl;
	  
      str<<"set origin 0,0"<<endl;
      str<<"unset key"<<endl;
      str<<"unset ylabel"<<endl;
      str<<"unset ytics"<<endl;
      str<<"set title '1^{st} valley'"<<endl;
      str<<"plot '" << filename_dat << "' using 1:3 with lines title 'p=0'";
      for( int p=1; p<NSBN; ++p )
	str<<",'" << filename_dat << "' using 1:"<<3+p<<" with lines title 'p="<<p<<"'";
      str<<endl;
	  
      str<<"set origin .33,0"<<endl;
      str<<"unset key"<<endl;
      str<<"set title '2^{nd} valley'"<<endl;
      str<<"plot '" << filename_dat << "' using 1:"<<3+NSBN<<" with lines title 'p=0'";
      for( int p=1; p<NSBN; ++p )
	str<<",'" << filename_dat << "' using 1:"<<3+p+NSBN<<" with lines title 'p="<<p<<"'";
      str<<endl;
	  
      str<<"set origin .66,0"<<endl;
      str<<"unset key"<<endl;
      str<<"set title '3^{rd} valley'"<<endl;
      str<<"plot '" << filename_dat << "' using 1:"<<3+2*NSBN<<" with lines title 'p=0'";
      for( int p=1; p<NSBN; ++p )
	str<<",'" << filename_dat << "' using 1:"<<3+p+2*NSBN<<" with lines title 'p="<<p<<"'";
      str<<endl;
	  
      str<<"unset multiplot"<<endl;

      str<<"set output"<<endl;

    }
  str << "!/bin/echo -en 'encoding...                                      \\r' 1>&2" << endl;
  str<<"!mencoder mf://results/currdens.*.png -mf fps=10 -o films_and_graphs/currdens.avi -ovc lavc -lavcopts vcodec=wmv2 1>/dev/null 2>/dev/null"<<endl;
  str<<"!rm results/*.png 1>/dev/null 2>/dev/null"<<endl;
  str << "!/bin/echo -en 'generating currdens.avi...[ok]\\n' 1>&2" << endl;
  str<<"quit"<<endl;


  str.close();

  return;
}



void MOSFETProblemCuda::write_gnuplotscript_driftvel()
{
  const double LSOURCE  = host_physdesc -> get_LSOURCE();
  const double LCHANNEL = host_physdesc -> get_LCHANNEL();
  const double LDRAIN   = host_physdesc -> get_LDRAIN();

  ofstream str;
  str.open( "scripts/film_driftvel.gp",ios_base::out );
  for(int st=0; st<=_step; st+=host_solvpar->get_FREQ())
    {
      str << "!/bin/echo -en 'generating driftvel.avi : step = " << st << "/" << _step << "                                      \\r' 1>&2" << endl;

      str<<"set terminal png enh"<<endl;
      ostringstream sstr;
      sstr  << "results/driftvel."             << setw(DIGITS) << setfill('0') << st;
      const string filename_dat = sstr.str()+".dat";
      const string filename_png = sstr.str()+".png";
      str<<"set output '"<<filename_png<<"'"<<endl;
      str<<"set xlabel 'unconfined dimension'"<<endl;
      str<<"set xtics ( '0 nm' 0,'"<<LSOURCE/1.E-9<<" nm' "<<LSOURCE<<",'"<<(LSOURCE+LCHANNEL)/1.E-9<<" nm' "<<LSOURCE+LCHANNEL<<",'"<<(LSOURCE+LCHANNEL+LDRAIN)/1.E-9<<" nm' "<<LSOURCE+LCHANNEL+LDRAIN<<" )"<<endl;
      str<<"set ylabel 'vel [ms^{-1}]'"<<endl;
      str<<"unset key"<<endl;
      str<<"set title 'average drift velocity at approximate time "<<(_time/_step*st)*host_rescpar->get_tstar()/1.E-12<<" ps'"<<endl;
      str<<"plot [][" << _min_driftvel*host_rescpar->get_vstar() << ":" << _max_driftvel*host_rescpar->get_vstar() << "] '" << filename_dat << "' using 1:2 with lines"<<endl;
      str<<"set output"<<endl;
    }
  str << "!/bin/echo -en 'encoding...                                      \\r' 1>&2" << endl;
  str<<"!mencoder mf://results/driftvel.*.png -mf fps=10 -o films_and_graphs/driftvel.avi -ovc lavc -lavcopts vcodec=wmv2 1>/dev/null 2>/dev/null"<<endl;
  str<<"!rm results/*.png 1>/dev/null 2>/dev/null"<<endl;
  str << "!/bin/echo -en 'generating driftvel.avi...[ok]\\n' 1>&2" << endl;
  str<<"quit"<<endl;


  str.close();

  return;
}



void MOSFETProblemCuda::write_gnuplotscript_test()
{
  ofstream str;
  str.open( "scripts/film_test.gp",ios_base::out );
  for(int st=0; st<=_step; st+=host_solvpar->get_FREQ())
    {
      str << "!/bin/echo -en 'generating test.avi : step = " << st << "/" << _step << "                                      \\r' 1>&2" << endl;

      str<<"set terminal png enh"<<endl;
      ostringstream sstr;
      sstr  << "results/test."             << setw(DIGITS) << setfill('0') << st;
      const string filename_dat = sstr.str()+".dat";
      const string filename_png = sstr.str()+".png";
      str<<"set output '"<<filename_png<<"'"<<endl;
      str<<"set xlabel 'w-dimension'"<<endl;
      str<<"unset xtics" << endl;
      str<<"unset key"<<endl;
      str<<"set title 'log of the (nu,p,i,m)-integrated pdf at time "<<(_time/_step*st)*host_rescpar->get_tstar()/1.E-12<<" ps'"<<endl;
      str<<"plot [][" << _min_test << ":" << _max_test << "] '" << filename_dat << "' using 1:(log($2)) with lines"<<endl;
      str<<"set output"<<endl;
    }
  str << "!/bin/echo -en 'encoding...                                      \\r' 1>&2" << endl;
  str<<"!mencoder mf://results/test.*.png -mf fps=10 -o films_and_graphs/test.avi -ovc lavc -lavcopts vcodec=wmv2 1>/dev/null 2>/dev/null"<<endl;
  str<<"!rm results/*.png 1>/dev/null 2>/dev/null"<<endl;
  str << "!/bin/echo -en 'generating test.avi...[ok]\\n' 1>&2" << endl;
  str<<"quit"<<endl;


  str.close();

  return;
}



void MOSFETProblemCuda::write_gnuplotscript_totvoldens()
{
  const double LSOURCE  = host_physdesc -> get_LSOURCE();
  const double LCHANNEL = host_physdesc -> get_LCHANNEL();
  const double LDRAIN   = host_physdesc -> get_LDRAIN();
  const double ZWIDTH  = host_physdesc -> get_ZWIDTH();
  const double LSiO2  = host_physdesc -> get_LSiO2();

  ofstream str;

  str.open( "scripts/film_totvoldens.gp",ios_base::out );
  for(int st=0; st<=_step; st+=host_solvpar->get_FREQ())
    {
      str << "!/bin/echo -en 'generating totvoldens.avi : step = " << st << "/" << _step << "                                      \\r' 1>&2" << endl;

      str<<"set terminal png enh"<<endl;
      ostringstream sstr;
      sstr  << "results/voldens."             << setw(DIGITS) << setfill('0') << st;
      const string filename_dat = sstr.str()+".dat";
      const string filename_png = sstr.str()+".png";
      str<<"set output '"<<filename_png<<"'"<<endl;
      str<<"set title 'total volume density at approximate time "<<(_time/_step*st)*host_rescpar->get_tstar()/1.E-12<<" ps,in [m^{-3}]'"<<endl;
      str<<"set xlabel 'unconfined dimension'"<<endl;
      str<<"set ylabel 'confined dimension'"<<endl;
      str<<"set xtics ( '0 nm' 0,'"<<LSOURCE/1.E-9<<" nm' "<<LSOURCE<<",'"<<(LSOURCE+LCHANNEL)/1.E-9<<" nm' "<<LSOURCE+LCHANNEL<<",'"<<(LSOURCE+LCHANNEL+LDRAIN)/1.E-9<<" nm' "<<LSOURCE+LCHANNEL+LDRAIN<<" )"<<endl;
      str<<"set ytics ( '0 nm' 0,'"<<LSiO2/1.E-9<<" nm' "<<LSiO2<<",'"<<(ZWIDTH-LSiO2)/1.E-9<<" nm' "<<ZWIDTH-LSiO2<<",'"<<ZWIDTH/1.E-9<<" nm' "<<ZWIDTH<<" )"<<endl;
      str<<"set pm3d map"<<endl;
      str<<"unset key"<<endl;
      str<<"set zrange [" << setprecision(12) << _min_totvoldens*host_rescpar->get_nstar() << ":" << _max_totvoldens*host_rescpar->get_nstar() << "]"<<endl;
      str<<"set cbrange [" << setprecision(12) << _min_totvoldens*host_rescpar->get_nstar() << ":" << _max_totvoldens*host_rescpar->get_nstar() << "]"<<endl;
      str<<"splot '" << filename_dat << "' using 1:2:3:3 with pm3d"<<endl;
      str<<"set output"<<endl;
    }
  str << "!/bin/echo -en 'encoding...                                      \\r' 1>&2" << endl;
  str<<"!mencoder mf://results/voldens.*.png -mf fps=10 -o films_and_graphs/totvoldens.avi -ovc lavc -lavcopts vcodec=wmv2 1>/dev/null 2>/dev/null"<<endl;
  str<<"!rm results/*.png 1>/dev/null 2>/dev/null"<<endl;
  str << "!/bin/echo -en 'generating totvoldens.avi...[ok]\\n' 1>&2" << endl;
  str<<"quit"<<endl;
  str.close();

  if( host_solvpar->get_ROUGHNESS() == true )
    {
      str.open( "scripts/film_totvoldens_ext.gp",ios_base::out );
      for(int st=0; st<=_step; st+=host_solvpar->get_FREQ())
	{
	  str << "!/bin/echo -en 'generating totvoldens_ext.avi : step = " << st << "/" << _step << "                                      \\r' 1>&2" << endl;

	  str<<"set terminal png enh"<<endl;
	  ostringstream sstr;
	  sstr  << "results/totvoldens_ext."             << setw(DIGITS) << setfill('0') << st;
	  const string filename_dat = sstr.str()+".dat";
	  const string filename_png = sstr.str()+".png";
	  str<<"set output '"<<filename_png<<"'"<<endl;
	  str<<"set title '(extended) total volume density at approximate time "<<(_time/_step*st)*host_rescpar->get_tstar()/1.E-12<<" ps,in [m^{-3}]'"<<endl;
	  str<<"set xlabel 'unconfined dimension'"<<endl;
	  str<<"set ylabel 'confined dimension'"<<endl;
	  str<<"set xtics ( '0 nm' 0,'"<<LSOURCE/1.E-9<<" nm' "<<LSOURCE<<",'"<<(LSOURCE+LCHANNEL)/1.E-9<<" nm' "<<LSOURCE+LCHANNEL<<",'"<<(LSOURCE+LCHANNEL+LDRAIN)/1.E-9<<" nm' "<<LSOURCE+LCHANNEL+LDRAIN<<" )"<<endl;
	  str<<"set ytics ( '0 nm' 0,'"<<LSiO2/1.E-9<<" nm' "<<LSiO2<<",'"<<(ZWIDTH+host_solvpar->get_DELTA_SR()-LSiO2)/1.E-9<<" nm' "<<ZWIDTH+host_solvpar->get_DELTA_SR()-LSiO2<<",'"<<(ZWIDTH+host_solvpar->get_DELTA_SR())/1.E-9<<" nm' "<<ZWIDTH+host_solvpar->get_DELTA_SR()<<" )"<<endl;
	  str<<"set pm3d map"<<endl;
	  str<<"unset key"<<endl;
	  str<<"set zrange [" << setprecision(12) << _min_totvoldens*host_rescpar->get_nstar() << ":" << _max_totvoldens*host_rescpar->get_nstar() << "]"<<endl;
	  str<<"set cbrange [" << setprecision(12) << _min_totvoldens*host_rescpar->get_nstar() << ":" << _max_totvoldens*host_rescpar->get_nstar() << "]"<<endl;
	  str<<"splot '" << filename_dat << "' using 1:2:3:3 with pm3d"<<endl;
	  str<<"set output"<<endl;
	}
      str << "!/bin/echo -en 'encoding...                                      \\r' 1>&2" << endl;
      str<<"!mencoder mf://results/totvoldens_ext.*.png -mf fps=10 -o films_and_graphs/totvoldens_ext.avi -ovc lavc -lavcopts vcodec=wmv2 1>/dev/null 2>/dev/null"<<endl;
      str<<"!rm results/*.png 1>/dev/null 2>/dev/null"<<endl;
      str << "!/bin/echo -en 'generating totvoldens_ext.avi...[ok]\\n' 1>&2" << endl;
      str<<"quit"<<endl;
      str.close();
    }

  return;
}

void MOSFETProblemCuda::write_gnuplotscript_surfdens()
{
  const int NSBN  = host_dm -> get_NSBN();
  const double LSOURCE  = host_physdesc -> get_LSOURCE();
  const double LCHANNEL = host_physdesc -> get_LCHANNEL();
  const double LDRAIN   = host_physdesc -> get_LDRAIN();

  ofstream str;

  str.open( "scripts/film_surfdens.gp",ios_base::out );
  for(int st=0; st<=_step; st+=host_solvpar->get_FREQ())
    {
      str << "!/bin/echo -en 'generating surfdens.avi : step = " << st << "/" << _step << "                                      \\r' 1>&2" << endl;

      str<<"set terminal png enh"<<endl;
      ostringstream sstr;
      sstr  << "results/surfdens."             << setw(DIGITS) << setfill('0') << st;
      const string filename_dat = sstr.str()+".dat";
      const string filename_png = sstr.str()+".png";
      str<<"set output '"<<filename_png<<"'"<<endl;
      str<<"set xlabel 'unconf. dimension'"<<endl;
      str<<"set xtics ( '0 nm' 0,'"<<LSOURCE/1.E-9<<" nm' "<<LSOURCE<<",'"<<(LSOURCE+LCHANNEL)/1.E-9<<" nm' "<<LSOURCE+LCHANNEL<<",'"<<(LSOURCE+LCHANNEL+LDRAIN)/1.E-9<<" nm' "<<LSOURCE+LCHANNEL+LDRAIN<<" )"<<endl;
      str<<"set ylabel 'rho [m^{-2}]'"<<endl;

      str<<"set multiplot"<<endl;
      str<<"set border"<<endl;
      str<<"set ytics"<<endl;
      str<<"unset key"<<endl;
      str<<"set xtics ( '"<<LSOURCE/1.E-9<<" nm' "<<LSOURCE<<",'"<<(LSOURCE+LCHANNEL)/1.E-9<<" nm' "<<LSOURCE+LCHANNEL<<")"<<endl;
	  
      str<<"set size 1,.5"<<endl;
      str<<"set origin 0,.5"<<endl;
      int nu = 0;
      str<<"set title 'total surface density at approximate time "<<(_time/_step*st)*host_rescpar->get_tstar()/1.E-12<<" ps'"<<endl;
      str<<"plot [][" << _min_totsurfdens*host_rescpar->get_rhostar() << ":"<<_max_totsurfdens*host_rescpar->get_rhostar()<<"] '" << filename_dat << "' using 1:2 with lines title 'total density'"<<endl;
	  
      str<<"unset ylabel"<<endl;
      str<<"set size .33,.5"<<endl;
	  
      str<<"set origin 0,0"<<endl;
      str<<"set title '1^{st} valley'"<<endl;
      str<<"unset ytics"<<endl;
      str<<"plot [][" << _min_totsurfdens*host_rescpar->get_rhostar() << ":"<<_max_totsurfdens*host_rescpar->get_rhostar()<<"] '" << filename_dat << "' using 1:($"<<3+nu*NSBN<<"*2) with lines title 'p=0'";
      for( int p=1; p<NSBN; ++p )
	str<<",'" << filename_dat << "' using 1:($"<<3+nu*NSBN+p<<"*2) with lines title 'p="<<p<<"'";
      str<<endl;
	  
      str<<"set size .33,.5"<<endl;
      str<<"set origin .33,0"<<endl;
      nu = 1;
      str<<"set title '2^{nd} valley'"<<endl;
      str<<"plot [][" << _min_totsurfdens*host_rescpar->get_rhostar() << ":"<<_max_totsurfdens*host_rescpar->get_rhostar()<<"] '" << filename_dat << "' using 1:($"<<3+nu*NSBN<<"*2) with lines title 'p=0'";
      for( int p=1; p<NSBN; ++p )
	str<<",'" << filename_dat << "' using 1:($"<<3+nu*NSBN+p<<"*2) with lines title 'p="<<p<<"'";
      str<<endl;
	  
      str<<"set size .33,.5"<<endl;
      str<<"set origin .66,0"<<endl;
      nu = 2;
      str<<"set title '3^{rd} valley'"<<endl;
      str<<"plot [][" << _min_totsurfdens*host_rescpar->get_rhostar() << ":"<<_max_totsurfdens*host_rescpar->get_rhostar()<<"] '" << filename_dat << "' using 1:($"<<3+nu*NSBN<<"*2) with lines title 'p=0'";
      for( int p=1; p<NSBN; ++p )
	str<<",'" << filename_dat << "' using 1:($"<<3+nu*NSBN+p<<"*2) with lines title 'p="<<p<<"'";
      str<<endl;
	  
      str<<"unset key"<<endl;
      str<<"unset xtics"<<endl;
      str<<"unset ytics"<<endl;
      str<<"unset ztics"<<endl;
      str<<"unset xlabel"<<endl;
      str<<"unset ylabel"<<endl;
      str<<"unset zlabel"<<endl;
      str<<"unset border"<<endl;
      str<<"unset title"<<endl;
      str<<"set size .33,.5"<<endl;
	  
      str<<"set origin .02,0"<<endl;
      str<<"set parametric"<<endl;
      str<<"splot [-3.5:3.5][-3.5:3.5][-3.5:3.5] u,0,0 lt 0 notitle,0,u,0 lt 0 notitle,0,0,u lt 0 notitle,"
	 <<"      "<<mass(Xdim,0)<<"*cos(u)*cos(v)+2,"<<mass(Ydim,0)<<"*sin(u)*cos(v)  ,"<<mass(Zdim,0)<<"*sin(v)   lt -1 notitle,"
	 <<"      "<<mass(Xdim,0)<<"*cos(u)*cos(v)-2,"<<mass(Ydim,0)<<"*sin(u)*cos(v)  ,"<<mass(Zdim,0)<<"*sin(v)   lt -1 title '1^{st} valley'"<<endl;
	  
      str<<"set origin .35,0"<<endl;
      str<<"set parametric"<<endl;
      str<<"splot [-3.5:3.5][-3.5:3.5][-3.5:3.5] u,0,0 lt 0 notitle,0,u,0 lt 0 notitle,0,0,u lt 0 notitle,"
	 <<"      "<<mass(Xdim,1)<<"*cos(u)*cos(v)  ,"<<mass(Ydim,1)<<"*sin(u)*cos(v)+2,"<<mass(Zdim,1)<<"*sin(v)   lt -1 notitle,"
	 <<"      "<<mass(Xdim,1)<<"*cos(u)*cos(v)  ,"<<mass(Ydim,1)<<"*sin(u)*cos(v)-2,"<<mass(Zdim,1)<<"*sin(v)   lt -1 title '2^{nd} valley'"<<endl;
	  
      str<<"set origin .68,0"<<endl;
      str<<"set parametric"<<endl;
      str<<"splot [-3.5:3.5][-3.5:3.5][-3.5:3.5] u,0,0 lt 0 notitle,0,u,0 lt 0 notitle,0,0,u lt 0 notitle,"
	 <<"      "<<mass(Xdim,2)<<"*cos(u)*cos(v)  ,"<<mass(Ydim,2)<<"*sin(u)*cos(v)  ,"<<mass(Zdim,2)<<"*sin(v)+2 lt -1 notitle,"
	 <<"      "<<mass(Xdim,2)<<"*cos(u)*cos(v)  ,"<<mass(Ydim,2)<<"*sin(u)*cos(v)  ,"<<mass(Zdim,2)<<"*sin(v)-2 lt -1 title '3^{rd} valley'"<<endl;
	  
      str<<"unset parametric"<<endl;
      str<<"unset multiplot"<<endl;

      str<<"set output"<<endl;
    }

  str << "!/bin/echo -en 'encoding...                                      \\r' 1>&2" << endl;
  str<<"!mencoder mf://results/surfdens.*.png -mf fps=10 -o films_and_graphs/surfdens.avi -ovc lavc -lavcopts vcodec=wmv2 1>/dev/null 2>/dev/null"<<endl;
  str<<"!rm results/*.png 1>/dev/null 2>/dev/null"<<endl;
  str << "!/bin/echo -en 'generating surfdens.avi...[ok]\\n' 1>&2" << endl;

  str<<"quit"<<endl;

  str.close();

  return;
}

void MOSFETProblemCuda::write_gnuplotscript_eps()
{
  const int NSBN  = host_dm -> get_NSBN();
  const double LSOURCE  = host_physdesc -> get_LSOURCE();
  const double LCHANNEL = host_physdesc -> get_LCHANNEL();
  const double LDRAIN   = host_physdesc -> get_LDRAIN();
  const double XLENGTH = host_physdesc -> compute_XLENGTH();

  ofstream str;

  str.open( "scripts/film_eps.gp",ios_base::out );
  for(int st=0; st<=_step; st+=host_solvpar->get_FREQ())
    {
      str << "!/bin/echo -en 'generating eps.avi : step = " << st << "/" << _step << "                                      \\r' 1>&2" << endl;

      str<<"set terminal png enh"<<endl;

      ostringstream sstr, sstr3;
      sstr  << "results/eps."             << setw(DIGITS) << setfill('0') << st;
      sstr3 << "results/deps_dx." << setw(DIGITS) << setfill('0') << st;
      const string filename_dat = sstr.str()+".dat";
      const string filename_dat_2 = sstr3.str()+".dat";
      const string filename_png = sstr.str()+".png";
      str<<"set output '"<<filename_png<<"'"<<endl;
      str<<"set xlabel 'unconfined dimension'"<<endl;
      str<<"set ylabel 'eps_{nu,p} [eV]'"<<endl;
      str<<"set yrange ["<<_min_eps*host_rescpar->get_epsstar()/host_phyco->__eV<<":"<<_max_eps*host_rescpar->get_epsstar()/host_phyco->__eV<<"]"<<endl;
      str<<"set y2range ["<<_min_deps_dx*host_rescpar->get_epsstar()/(host_phyco->__eV*XLENGTH)<<":"<<_max_deps_dx*host_rescpar->get_epsstar()/(host_phyco->__eV*XLENGTH)<<"]"<<endl;
      str<<"set y2label 'field = (eps_{nu,p})_{x} [eV m^{-1}]'"<<endl;
      str<<"set ytics nomirror"<<endl;
      str<<"set y2tics"<<endl;
      str<<"set xtics ( '0 nm' 0,'"<<LSOURCE/1.E-9<<" nm' "<<LSOURCE<<",'"<<(LSOURCE+LCHANNEL)/1.E-9<<" nm' "<<LSOURCE+LCHANNEL<<",'"<<(LSOURCE+LCHANNEL+LDRAIN)/1.E-9<<" nm' "<<LSOURCE+LCHANNEL+LDRAIN<<" )"<<endl;
      str<<"unset key"<<endl;

      str<<"set key box inside center top"<<endl;
      int nu = 0;
      str<<"set title 'sub-bands at approximate time "<<(_time/_step*st)*host_rescpar->get_tstar()/1.E-12<<" ps'"<<endl;
      str<<"plot '" << filename_dat   << "' using 1:"<<nu*NSBN+2<<"           with linespoints lc 1 pt 1 title '1^{st} v.'";
      str<<",    '" << filename_dat_2 << "' using 1:"<<nu*NSBN+2<<" axes x1y2 with linespoints lc 1 pt 1 notitle";
      for( int p=1; p<NSBN; ++p )
	{
	  str<<",'" << filename_dat   << "' using 1:"<<nu*NSBN+2+p<<"           with linespoints lc 1 pt 1 notitle";
	  str<<",'" << filename_dat_2 << "' using 1:"<<nu*NSBN+2+p<<" axes x1y2 with linespoints lc 1 pt 1 notitle";
	}
      nu = 1;
      str<<",'" << filename_dat   << "' using 1:"<<nu*NSBN+2<<"           with linespoints lc 2 pt 2 title '2^{nd} v.'";
      str<<",'" << filename_dat_2 << "' using 1:"<<nu*NSBN+2<<" axes x1y2 with linespoints lc 2 pt 2 notitle";
      for( int p=1; p<NSBN; ++p )
	{
	  str<<",'" << filename_dat   << "' using 1:"<<nu*NSBN+2+p<<"           with linespoints lc 2 pt 2 notitle";
	  str<<",'" << filename_dat_2 << "' using 1:"<<nu*NSBN+2+p<<" axes x1y2 with linespoints lc 2 pt 2 notitle";
	}
      nu = 2;
      str<<",'" << filename_dat   << "' using 1:"<<nu*NSBN+2<<"           with linespoints lc 3 pt 4 title '3^{rd} v.'";
      str<<",'" << filename_dat_2 << "' using 1:"<<nu*NSBN+2<<" axes x1y2 with linespoints lc 3 pt 4 notitle";
      for( int p=1; p<NSBN; ++p )
	{
	  str<<",'" << filename_dat   << "' using 1:"<<nu*NSBN+2+p<<"           with linespoints lc 3 pt 4 notitle";
	  str<<",'" << filename_dat_2 << "' using 1:"<<nu*NSBN+2+p<<" axes x1y2 with linespoints lc 3 pt 4 notitle";
	}
      str<<endl;
      
      str<<"set output"<<endl;
    }
  str << "!/bin/echo -en 'encoding...                                      \\r' 1>&2" << endl;
  str<<"!mencoder mf://results/eps.*.png -mf fps=10 -o films_and_graphs/eps.avi -ovc lavc -lavcopts vcodec=wmv2 1>/dev/null 2>/dev/null"<<endl;
  str<<"!rm results/*.png 1>/dev/null 2>/dev/null"<<endl;
  str << "!/bin/echo -en 'generating eps.avi...[ok]\\n' 1>&2" << endl;

  str<<"quit"<<endl;

  str.close();

  return;
}

void MOSFETProblemCuda::write_gnuplotscript_potential()
{
  const double LSOURCE  = host_physdesc -> get_LSOURCE();
  const double LCHANNEL = host_physdesc -> get_LCHANNEL();
  const double LDRAIN   = host_physdesc -> get_LDRAIN();
  const double LGATE    = host_physdesc -> get_LGATE();
  const double ZWIDTH  = host_physdesc -> get_ZWIDTH();
  const double LSiO2  = host_physdesc -> get_LSiO2();

  ofstream str;

  str.open( "scripts/film_potential.gp",ios_base::out );
  for(int st=0; st<=_step; st+=host_solvpar->get_FREQ())
    {
      str << "!/bin/echo -en 'generating potential.avi : step = " << st << "/" << _step << "                                      \\r' 1>&2" << endl;

      str<<"set terminal png enh"<<endl;
      ostringstream sstr;
      sstr<<"results/potential."<<setw(DIGITS)<<setfill('0')<<st;
      const string filename_dat = sstr.str()+".dat";
      const string filename_png = sstr.str()+".png";
      str<<"set output '"<<filename_png<<"'"<<endl;
      str<<"set title 'potential energy at approximate time "<<(_time/_step*st)*host_rescpar->get_tstar()/1.E-12<<" ps,in [eV]'"<<endl;
      str<<"set xlabel 'unconf. dim. [nm]'"<<endl;
      str<<"set ylabel 'conf. dim. [nm]'"<<endl;
      const double gatepos_start = (LSOURCE+LCHANNEL+LDRAIN-LGATE)/2;
      const double gatepos_end   = (LSOURCE+LCHANNEL+LDRAIN+LGATE)/2;
      str<<"set xtics ( '0' 0,'"<<gatepos_start/1.E-9<<"' "<<gatepos_start<<",'"<<LSOURCE/1.E-9<<"' "<<LSOURCE<<",'"<<(LSOURCE+LCHANNEL)/1.E-9<<"' "<<LSOURCE+LCHANNEL<<",'"<<gatepos_end/1.E-9<<"' "<<gatepos_end<<",'"<<(LSOURCE+LCHANNEL+LDRAIN)/1.E-9<<"' "<<LSOURCE+LCHANNEL+LDRAIN<<" )"<<endl;
      str<<"set ytics ( '0' 0,'"<<LSiO2/1.E-9<<"' "<<LSiO2<<",'"<<(ZWIDTH-LSiO2)/1.E-9<<"' "<<ZWIDTH-LSiO2<<",'"<<ZWIDTH/1.E-9<<"' "<<ZWIDTH<<" )"<<endl;
      str<<"set pm3d at s"<<endl;
      str<<"set cblabel 'U [eV]'"<<endl;
      str<<"unset key"<<endl;
      str << "set zrange [" << -_max_potential*host_rescpar->get_potstar() << ":" << -_min_potential*host_rescpar->get_potstar() << "]" << endl;
      str << "set cbrange [" << -_max_potential*host_rescpar->get_potstar() << ":" << -_min_potential*host_rescpar->get_potstar() << "]" << endl;
      str<<"splot '" << filename_dat << "' using 1:2:(-$3) with pm3d"<<endl;
      str<<"set output"<<endl;
    }
  str << "!/bin/echo -en 'encoding...                                      \\r' 1>&2" << endl;
  str<<"!mencoder mf://results/potential.*.png -mf fps=10 -o films_and_graphs/potential.avi -ovc lavc -lavcopts vcodec=wmv2 1>/dev/null 2>/dev/null"<<endl;
  str<<"!rm results/*.png 1>/dev/null 2>/dev/null"<<endl;
  str << "!/bin/echo -en 'generating potential.avi...[ok]\\n' 1>&2" << endl;
  str<<"quit"<<endl;
  str.close();

  if( host_solvpar->get_ROUGHNESS() == true )
    {
      str.open( "scripts/film_pot_ext.gp",ios_base::out );
      for(int st=0; st<=_step; st+=host_solvpar->get_FREQ())
	{
	  str << "!/bin/echo -en 'generating pot_ext.avi : step = " << st << "/" << _step << "                                      \\r' 1>&2" << endl;

	  str<<"set terminal png enh"<<endl;
	  ostringstream sstr;
	  sstr<<"results/pot_ext."<<setw(DIGITS)<<setfill('0')<<st;
	  const string filename_dat = sstr.str()+".dat";
	  const string filename_png = sstr.str()+".png";
	  str<<"set output '"<<filename_png<<"'"<<endl;
	  str<<"set title 'extended potential energy at approximate time "<<(_time/_step*st)*host_rescpar->get_tstar()/1.E-12<<" ps,in [eV]'"<<endl;
	  str<<"set xlabel 'unconf. dim. [nm]'"<<endl;
	  str<<"set ylabel 'conf. dim. [nm]'"<<endl;
	  const double gatepos_start = (LSOURCE+LCHANNEL+LDRAIN-LGATE)/2;
	  const double gatepos_end   = (LSOURCE+LCHANNEL+LDRAIN+LGATE)/2;
	  str<<"set xtics ( '0' 0,'"<<gatepos_start/1.E-9<<"' "<<gatepos_start<<",'"<<LSOURCE/1.E-9<<"' "<<LSOURCE<<",'"<<(LSOURCE+LCHANNEL)/1.E-9<<"' "<<LSOURCE+LCHANNEL<<",'"<<gatepos_end/1.E-9<<"' "<<gatepos_end<<",'"<<(LSOURCE+LCHANNEL+LDRAIN)/1.E-9<<"' "<<LSOURCE+LCHANNEL+LDRAIN<<" )"<<endl;
	  str<<"set ytics ( '0' 0,'"<<LSiO2/1.E-9<<"' "<<LSiO2<<",'"<<(ZWIDTH+host_solvpar->get_DELTA_SR()-LSiO2)/1.E-9<<"' "<<ZWIDTH+host_solvpar->get_DELTA_SR()-LSiO2<<",'"<<(ZWIDTH+host_solvpar->get_DELTA_SR())/1.E-9<<"' "<<ZWIDTH+host_solvpar->get_DELTA_SR()<<" )"<<endl;
	  str<<"set pm3d at s"<<endl;
	  str<<"set cblabel 'U [eV]'"<<endl;
	  str<<"unset key"<<endl;
	  str << "set zrange [" << -_max_potential*host_rescpar->get_potstar() << ":" << -_min_potential*host_rescpar->get_potstar() << "]" << endl;
	  str << "set cbrange [" << -_max_potential*host_rescpar->get_potstar() << ":" << -_min_potential*host_rescpar->get_potstar() << "]" << endl;
	  str<<"splot '" << filename_dat << "' using 1:2:(-$3) with pm3d"<<endl;
	  str<<"set output"<<endl;
	}
      str << "!/bin/echo -en 'encoding...                                      \\r' 1>&2" << endl;
      str<<"!mencoder mf://results/pot_ext.*.png -mf fps=10 -o films_and_graphs/pot_ext.avi -ovc lavc -lavcopts vcodec=wmv2 1>/dev/null 2>/dev/null"<<endl;
      str<<"!rm results/*.png 1>/dev/null 2>/dev/null"<<endl;
      str << "!/bin/echo -en 'generating pot_ext.avi...[ok]\\n' 1>&2" << endl;
      str<<"quit"<<endl;
      str.close();
    }

  return;
}



void MOSFETProblemCuda::write_gnuplotscripts_scalars()
{
  ofstream str;

  str.open( "scripts/graph_avgcurr.gp",ios_base::out );
  str << "!/bin/echo -en 'generating avgcurr.pdf...' 1>&2" << endl;
  str << "set terminal postscript eps enh color" << endl;
  str << "set output 'films_and_graphs/avgcurr.eps'" << endl;
  str << "set xlabel 'time [ps]'" << endl;
  str << "set title 'average current [which unit?]'" << endl;
  str << "unset key" << endl;
  str << "plot 'results/scalars.dat' using ($1*1.e12):3 with lines" << endl;
  str << "set output" << endl;
  str << "!epstopdf films_and_graphs/avgcurr.eps films_and_graphs/avgcurr.pdf" << endl;
  str << "!/bin/echo -en '[ok]\\n' 1>&2" << endl;
  str << "quit" << endl;
  str.close();

  str.open( "scripts/graph_totmass.gp",ios_base::out );
  str << "!/bin/echo -en 'generating totmass.pdf...' 1>&2" << endl;
  str << "set terminal postscript eps enh color" << endl;
  str << "set output 'films_and_graphs/totmass.eps'" << endl;
  str << "set xlabel 'time [ps]'" << endl;
  str << "set title 'total electron mass [which unit?]'" << endl;
  str << "unset key" << endl;
  str << "plot 'results/scalars.dat' using ($1*1.e12):2 with lines" << endl;
  str << "set output" << endl;
  str << "!epstopdf films_and_graphs/totmass.eps films_and_graphs/totmass.pdf" << endl;
  str << "!/bin/echo -en '[ok]\\n' 1>&2" << endl;
  str << "quit" << endl;
  str.close();

  str.open( "scripts/graph_dt.gp",ios_base::out );
  str << "!/bin/echo -en 'generating dt.pdf...' 1>&2" << endl;
  str << "set terminal postscript eps enh color" << endl;
  str << "set output 'films_and_graphs/dt.eps'" << endl;
  str << "set xlabel 'time [ps]'" << endl;
  str << "set title 'time stepping [which unit?]'" << endl;
  str << "unset key" << endl;
  str << "plot 'results/scalars.dat' using ($1*1.e12):5 with lines" << endl;
  str << "set output" << endl;
  str << "!epstopdf films_and_graphs/dt.eps films_and_graphs/dt.pdf" << endl;
  str << "!/bin/echo -en '[ok]\\n' 1>&2" << endl;
  str << "quit" << endl;
  str.close();

  str.open( "scripts/graph_maxpdfborder.gp",ios_base::out );
  str << "!/bin/echo -en 'generating maxpdfborder.pdf...' 1>&2" << endl;
  str << "set terminal postscript eps enh color" << endl;
  str << "set output 'films_and_graphs/maxpdfborder.eps'" << endl;
  str << "set xlabel 'time [ps]'" << endl;
  str << "set title 'max pdf at the border [which unit?]'" << endl;
  str << "unset key" << endl;
  str << "plot 'results/scalars.dat' using ($1*1.e12):6 with lines" << endl;
  str << "set output" << endl;
  str << "!epstopdf films_and_graphs/maxpdfborder.eps films_and_graphs/maxpdfborder.pdf" << endl;
  str << "!/bin/echo -en '[ok]\\n' 1>&2" << endl;
  str << "quit" << endl;
  str.close();

  return;
}



