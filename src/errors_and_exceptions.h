#include <exception>

/*******************************************************
 *              EXCEPTIONS AND EXIT STATUS             *
 *******************************************************/

#define     _ERROR_CONFIG_BOUNDCOND                       2
class error_CONFIG_BOUNDCOND : public exception
{
  const char* what() const throw()
  {
    return "error_CONFIG_BOUNDCOND";
  }
 public:
  int i, j;
  BoundCond bc;
  EigenProblem ep;
  error_CONFIG_BOUNDCOND(const int _i, const int _j, const BoundCond _bc, const EigenProblem _ep)
    {
      i = _i;
      j = _j;
      bc = _bc;
      ep = _ep;
    }
};

#define     _ERROR_CONFIG_NPHI_ODD                        3
class error_CONFIG_NPHI_ODD : public exception
{
  const char* what() const throw()
  {
    return "error_CONFIG_NPHI_ODD";
  }
};

#define     _ERROR_DENS                                   4
class error_DENS : public exception
{
  const char* what() const throw()
  {
    return "error_DENS";
  }
};

#define     _ERROR_DENS_PDFTILDE                          5
class error_DENS_PDFTILDE : public exception
{
  const char* what() const throw()
  {
    return "error_DENS_PDFTILDE";
  }
};

#define     _ERROR_DENS_SURFDENS                          6
class error_DENS_SURFDENS : public exception
{
  const char* what() const throw()
  {
    return "error_DENS_SURFDENS";
  }
};

#define     _ERROR_INCOMPLETE                             7
class error_INCOMPLETE : public exception
{
  const char* what() const throw()
  {
    return "error_INCOMPLETE";
  }
};

#define     _ERROR_ITER_CONSTRLINSYS_BOUNDCOND            8
class error_ITER_CONSTRLINSYS_BOUNDCOND : public exception
{
  const char* what() const throw()
  {
    return "error_ITER_CONSTRLINSYS_BOUNDCOND";
  }
 public:
  int i, j;
  BoundCond bc;
  EigenProblem ep;
  error_ITER_CONSTRLINSYS_BOUNDCOND(const int _i, const int _j, const BoundCond _bc, const EigenProblem _ep)
    {
      i = _i;
      j = _j;
      bc = _bc;
      ep = _ep;
    }
};

#define     _ERROR_ITER_EIGENSTATES_MULTISEC              9
class error_ITER_EIGENSTATES_MULTISEC : public exception
{
  const char* what() const throw()
  {
    return "error_ITER_EIGENSTATES_MULTISEC";
  }
 public:
  int nmulti;
  error_ITER_EIGENSTATES_MULTISEC(const int _nmulti)
    {
      nmulti = _nmulti;
    }
};

#define     _ERROR_ITER_NR_NOTCONV                        10
class error_ITER_NR_NOTCONV : public exception
{
  const char* what() const throw()
  {
    return "error_ITER_NR_NOTCONV";
  }
 public:
  int itermax;
  double *vect_L2V;
  double *vect_L2N;
  double *vect_LinfV;
  double *vect_LinfN;
  error_ITER_NR_NOTCONV(const discrMeshes *dm, const physDevice *pd, const int _itermax, const double *_vect_L2V, const double *_vect_L2N, const double *_vect_LinfV, const double *_vect_LinfN, const double *_init_surfdens, const double *_init_pot, const double *_init_voldens, const double *_init_eps, const double *_init_chi)
    {
      const int NSBN = dm -> get_NSBN();
      const int NX   = dm -> get_X() -> get_N();
      const int NZ   = dm -> get_Z() -> get_N();
      // const double XLENGTH = pd -> compute_XLENGTH();
      // const double ZWIDTH  = pd -> get_ZWIDTH();
      
      itermax = _itermax;
      vect_L2V = new double[itermax];
      vect_L2N = new double[itermax];
      vect_LinfV = new double[itermax];
      vect_LinfN = new double[itermax];
      for(int i=0; i<itermax; ++i)
	{
	  vect_L2V[i] = _vect_L2V[i];
	  vect_L2N[i] = _vect_L2N[i];
	  vect_LinfV[i] = _vect_LinfV[i];
	  vect_LinfN[i] = _vect_LinfN[i];
	}

      ofstream str;
      // surface densities
      cerr << " Writing file error_ITER_NR_NOTCONV_init_surfdens.dat'...";
      double *_init_totsurfdens = new double[ NX ];
      for( int i=0; i<NX; ++i )
	{
	  _init_totsurfdens[i] = 0;
	  for( int nu=0; nu<_NVALLEYS; ++nu )
	    for( int p=0; p<NSBN; ++p )
	      {
		_init_totsurfdens[i] += _init_surfdens[i*_NVALLEYS*NSBN + nu*NSBN + p];
	      }
	  _init_totsurfdens[i] *= 2.;
	}
      str.open("error_ITER_NR_NOTCONV_init_surfdens.dat", ios_base::out);
      str << scientific << setprecision(16);
      for( int i=0; i<NX; ++i )
	{
	  str << setw(25) << dm -> get_X() -> mesh(i)
	      << setw(25) << _init_totsurfdens[ i ];
	  for( int nu=0; nu<_NVALLEYS; ++nu )
	    for (int p=0; p<NSBN; ++p )
	      str << setw(25) << _init_surfdens[i*_NVALLEYS*NSBN + nu*NSBN + p];
	  str<<endl;
	}
      str.close();
      delete [] _init_totsurfdens;
      cerr << "[ok]" << endl;
      // initial potential
      cerr << " Writing file error_ITER_NR_NOTCONV_init_pot.dat'...";
      str.open("error_ITER_NR_NOTCONV_init_pot.dat", ios_base::out);
      str << scientific << setprecision(16);
      for( int i=0; i<NX; ++i )
	{
	  for( int j=0; j<NZ; ++j )
	    str << setw(25) << dm -> get_X() -> mesh(i)
		<< setw(25) << dm -> get_Z() -> mesh(j)
		<< setw(25) << _init_pot[i*NZ + j]
		<< endl;
	  str<<endl;
	}
      str.close();
      cerr << "[ok]" << endl;
      // volume densities
      cerr << " Writing file error_ITER_NR_NOTCONV_init_voldens.dat'...";
      double *_init_totvoldens = new double[ NX * NZ ];
      for( int i=0; i<NX; ++i )
	for( int j=0; j<NZ; ++j )
	  {
	    _init_totvoldens[i*NZ + j] = 0;
	    for( int nu=0; nu<_NVALLEYS; ++nu )
	      for( int p=0; p<NSBN; ++p )
		{
		  _init_totvoldens[i*NZ + j] += _init_voldens[(i*NZ+j)*_NVALLEYS*NSBN + nu*NSBN + p];
		}
	    _init_totvoldens[i*NZ + j] *= 2.;
	  }
      str.open("error_ITER_NR_NOTCONV_init_voldens.dat", ios_base::out);
      str << scientific << setprecision(16);
      for(int i=0; i<NX; ++i)
	for(int j=0; j<NZ; ++j)
	  {
	    str << setw(25) << dm -> get_X() -> mesh(i)
		<< setw(25) << dm -> get_Z() -> mesh(j)
		<< setw(25) << _init_totvoldens[i*NZ + j];
	    for( int nu=0; nu<_NVALLEYS; ++nu )
	      for (int p=0; p<NSBN; ++p )
		str << setw(25) << _init_voldens[(i*NZ+j)*_NVALLEYS*NSBN + nu*NSBN + p];
	    str << endl;

	    if(j == NZ-1) str << endl;
	  }
      str.close();
      delete [] _init_totvoldens;
      cerr << "[ok]" << endl;
      // eigenvalues
      cerr << " Writing file error_ITER_NR_NOTCONV_init_eps.dat'...";
      str.open("error_ITER_NR_NOTCONV_init_eps.dat", ios_base::out);
      str << scientific << setprecision(16);
      for( int i=0; i<NX; ++i )
	{
	  str << setw(25) << dm -> get_X() -> mesh(i);
	  for( int nu=0; nu<_NVALLEYS; ++nu )
	    for (int p=0; p<NSBN; ++p )
	      str << setw(25) << _init_eps[i*_NVALLEYS*NSBN+nu*NSBN+p];
	  str<<endl;
	}
      str.close();
      cerr << "[ok]" << endl;
      // eigenvectors
      cerr << " Writing file error_ITER_NR_NOTCONV_init_chi.dat'...";
      str.open("error_ITER_NR_NOTCONV_init_chi.dat", ios_base::out);
      str << "#" << setw(15) << "i" << setw(16) << "j";
      ostringstream osstr;
      for(int nu=0; nu<_NVALLEYS; ++nu)
	for(int p=0; p<NSBN; ++p)
	  {
	    osstr.str("");
	    osstr << "(nu,p)=(" << nu << "," << p << ")";
	    str << setw(32) << osstr.str();
	  }
      str << endl;
      
      for(int i=0; i<NX; ++i)
	{
	  for(int j=1; j<NZ-1; ++j)
	    {
	      str << setw(16) << i
		  << setw(16) << j;
	      for(int nu=0; nu<_NVALLEYS; ++nu)
		for(int p=0; p<NSBN; ++p)
		  str << setw(32) << _init_chi[ i*_NVALLEYS*NSBN*_SCHROED_MATRIX_SIZE_PAD + nu*NSBN*_SCHROED_MATRIX_SIZE_PAD + p*_SCHROED_MATRIX_SIZE_PAD + (j-1) ];
	      str << endl;
	    }
	  str << endl;
	}
      str.close();
      cerr << "[ok]" << endl;
    }
};

#define     _ERROR_ITER_NR_INITCOND                       11
class error_ITER_NR_NOTCONV_INITCOND : public exception
{
  const char* what() const throw()
  {
    return "error_ITER_NR_NOTCONV_INITCOND";
  }
};

#define     _ERROR_ITER_NR_THERMEQUIL                     12
class error_ITER_NR_NOTCONV_THERMEQUIL : public exception
{
  const char* what() const throw()
  {
    return "error_ITER_NR_NOTCONV_THERMEQUIL";
  }
};

#define     _ERROR_ITER_SPSTEP                            13
class error_ITER_SPSTEP : public exception
{
  const char* what() const throw()
  {
    return "error_ITER_SPSTEP";
  }
};

#define     _ERROR_ITER_SPSTEP_FRECHET_MATERIAL            14
class error_ITER_SPSTEP_FRECHET_MATERIAL : public exception
{
  const char* what() const throw()
  {
    return "error_ITER_SPSTEP_FRECHET_MATERIAL";
  }
};

#define     _ERROR_ITER_SPSTEP_SOLVELINSYS_SRJ_BSIZE      15
class error_ITER_SPSTEP_SOLVELINSYS_SRJ_BSIZE : public exception
{
  const char* what() const throw()
  {
    return "error_ITER_SPSTEP_SOLVELINSYS_SRJ_BSIZE";
  }
};

#define     _ERROR_ITER_SPSTEP_SOLVELINSYS_DGESV          16
class error_ITER_SPSTEP_SOLVELINSYS_DGESV : public exception
{
  const char* what() const throw()
  {
    return "error_ITER_SPSTEP_SOLVELINSYS_DGESV";
  }
};

#define     _ERROR_MOSFETPROBLEM_MEMORY                   17
class error_MOSFETPROBLEM_MEMORY : public exception
{
  const char* what() const throw()
  {
    return "error_MOSFETPROBLEM_MEMORY";
  }
};

#define     _ERROR_REDUCTION_MATERIAL                     18
class error_REDUCTION_MATERIAL : public exception
{
  const char* what() const throw()
  {
    return "error_REDUCTION_MATERIAL";
  }
};

#define     _ERROR_REDUCTION_WRONG_EIGENPROBLEM           19
class error_REDUCTION_WRONG_EIGENPROBLEM : public exception
{
  const char* what() const throw()
  {
    return "error_REDUCTION_WRONG_EIGENPROBLEM";
  }
};

#define     _ERROR_REDUCTION_WRONG_SPPHASE                20
class error_REDUCTION_WRONG_SPPHASE : public exception
{
  const char* what() const throw()
  {
    return "error_REDUCTION_WRONG_SPPHASE";
  }
};

#define     _ERROR_TESTING_COMPHOSTDEV                    21
class error_TESTING_COMPHOSTDEV : public exception
{
  const char* what() const throw()
  {
    return "error_TESTING_COMPHOSTDEV";
  }
};

#define     _ERROR_TESTING_EIGENSTATES                    22
class error_TESTING_EIGENSTATES : public exception
{
  const char* what() const throw()
  {
    return "error_TESTING_EIGENSTATES";
  }
};

#define     _ERROR_TESTING_EIGENVALUES                    23
class error_TESTING_EIGENVALUES : public exception
{
  const char* what() const throw()
  {
    return "error_TESTING_EIGENVALUES";
  }
};

#define     _ERROR_TESTING_NANORINF                       24
class error_TESTING_NANORINF : public exception
{
  const char* what() const throw()
  {
    return "error_CONFIG_NPHI_ODD";
  }
};

#define     _ERROR_TESTING_PDF                            25
class error_TESTING_PDF : public exception
{
  const char* what() const throw()
  {
    return "error_TESTING_PDF";
  }
};

#define     _ERROR_TESTING_RHS                            26
class error_TESTING_RHS : public exception
{
  const char* what() const throw()
  {
    return "error_TESTING_RHS";
  }
};

#define     _ERROR_WARMUP                                 27
class error_WARMUP : public exception
{
  const char* what() const throw()
  {
    return "error_WARMUP";
  }
};

#define     _ERROR_RK                                     28
class error_RK : public exception
{
  const char* what() const throw()
  {
    return "error_RK";
  }
};

