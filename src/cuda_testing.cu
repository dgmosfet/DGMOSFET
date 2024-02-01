#include "mosfetproblem.h"

#ifdef __CUDACC__
#define CUDA_SAFE_CALL( call ) {                                         \
 cudaError_t err = call;                                                 \
 if( cudaSuccess != err ) {                                              \
   fprintf(stderr,"CUDA: error occurred in cuda routine. Exiting...\n"); \
   exit(err);                                                            \
 } }



bool MOSFETProblemCuda::are_there_nan_inf(double *data, const int N, const string message)
{
  // cerr << " Called '" << __func__ << "'...";

  double *d_data = new double[N];
  CUDA_SAFE_CALL( cudaMemcpy(d_data, data, N*sizeof(double), cudaMemcpyDeviceToHost) );
  for(int i=0; i<N; ++i)
    {
      if( isnan( d_data[i] ) || isinf( d_data[i] ) )
	{
	  cerr << " Message from " << __func__ << ", line " << __LINE__ << " : message =" << message << endl;
	  cerr << "d_data[" << i << "] = " << d_data[i] << endl;
	  return true;		 
	}
    }
      
  delete [] d_data;
      
  return false;
}



int MOSFETProblemCuda::save_GPU_matrix_2d(const string filename)
{
  const int NX    = host_dm -> get_X()   -> get_N();
  const int NZ    = host_dm -> get_Z()   -> get_N();

  cerr << " Seving file " << filename << "...";

  const int N = NX*NZ*(2*NZ+1);
  double *_array = new double [ N ];
  CUDA_SAFE_CALL( cudaMemcpy(_array, _GPU_matrix_2d, N*sizeof(double), cudaMemcpyDeviceToHost) );

  ofstream ostr;
  ostr.open(filename.c_str(), ios_base::out);
  for(int line=0; line<NX*NZ; ++line)
    for(int column=max(0,line-NZ); column<=min(NX*NZ-1, line+NZ); ++column)
    {
      ostr << setw(20) << line
	   << setw(20) << column
	   << setw(20) << _array[ (line)*(2*NZ+1) + ((column)+NZ-(line)) ]
	   << endl;
    }
  ostr.close();
  
  delete [] _array;
  
  cerr << "[ok] (function " << __func__ << ", line " << __LINE__ << ")" << endl;
  
  return 0;
}


int MOSFETProblemCuda::save_GPU_pot(const string filename)
{
  const int NX    = host_dm -> get_X()   -> get_N();
  const int NZ    = host_dm -> get_Z()   -> get_N();

  cerr << " Seving file " << filename << "...";

  double *_array = new double [ NX*NZ ];
  CUDA_SAFE_CALL( cudaMemcpy(_array, _GPU_pot, NX*NZ*sizeof(double), cudaMemcpyDeviceToHost) );

  ofstream ostr;
  ostr.open(filename.c_str(), ios_base::out);
  for(int i=0; i<NX; ++i)
    for(int j=0; j<NZ; ++j)
      {
	ostr << setw(20) << i
	     << setw(20) << j
	     << setw(20) << _array[ i*NZ + j ]
	     << endl;
      }
  ostr.close();

  delete [] _array;

  cerr << "[ok] (function " << __func__ << ", line " << __LINE__ << ")" << endl;
  
  return 0;
}


int MOSFETProblemCuda::save_device_array(double *pt, const int N, const string filename)
{
  cerr << " Seving file " << filename << "...";

  double *_array = new double [ N ];
  CUDA_SAFE_CALL( cudaMemcpy(_array, pt, N*sizeof(double), cudaMemcpyDeviceToHost) );

  ofstream ostr;
  ostr.open(filename.c_str(), ios_base::out);
  for(int line=0; line<N; ++line)
    {
      ostr << setw(20) << line
	   << setw(20) << _array[line]
	   << endl;
    }
  ostr.close();

  delete [] _array;

  cerr << "[ok] (function " << __func__ << ", line " << __LINE__ << ")" << endl;
  
  return 0;
}

int MOSFETProblemCuda::save_host_array(double *pt, const int N, const string filename)
{
  cerr << " Seving file " << filename << "...";

  double *_array = pt;
  
  ofstream ostr;
  ostr.open(filename.c_str(), ios_base::out);
  for(int line=0; line<N; ++line)
    {
      ostr << setw(20) << line
	   << setw(20) << _array[line]
	   << endl;
    }
  ostr.close();

  delete [] _array;

  cerr << "[ok] (function " << __func__ << ", line " << __LINE__ << ")" << endl;
  
  return 0;
}

template <class T> void MOSFETProblemCuda::test( T *dmem, const int N, const string funcname )
{
  cerr << " testing '" << funcname << "'...";
  T *_test;
  _test = new T[ N ];
  checkCudaErrors( cudaMemcpy(_test, dmem, N*sizeof(T), cudaMemcpyDeviceToHost) );
  for(int i=0; i<N; ++i)
    {
      T val = _test[i];
      if( isinf(val) || isnan(val) )
	{
	  // cerr << " dmem[" << i << "] = " << val << endl;
	  throw error_TESTING_NANORINF();
	}
    }
  delete [] _test;
  cerr << "[ok] (function " << __func__ << ", line " << __LINE__ << ")" << endl;
  
  return;
}

void MOSFETProblemCuda::compare_iter()
{
  const int NSBN  = host_dm -> get_NSBN();
  const int NX    = host_dm -> get_X()   -> get_N();
  const int NZ    = host_dm -> get_Z()   -> get_N();

  cerr << " Called '" << __func__ << "'...";

  if(compare_host_device(_surfdens,       _GPU_surfdens,        _NVALLEYS*NSBN*NX)) { cerr << " surfdens!" << endl; }
  if(compare_host_device(_pot,            _GPU_pot,             NX*NZ            )) { cerr << " pot!" << endl; }
  if(compare_host_device(_pot_OLD,        _GPU_pot_OLD,         NX*NZ            )) { cerr << " pot_OLD!" << endl; }
  if(compare_host_device(_totvoldens,     _GPU_totvoldens,      NX*NZ            )) { cerr << " totvoldens!" << endl; }
  if(compare_host_device(_totvoldens_OLD, _GPU_totvoldens_OLD,  NX*NZ            )) { cerr << " totvoldens_OLD!" << endl; }
  if(compare_host_device(_eps,            _GPU_eps,             _NVALLEYS*NSBN*NX)) { cerr << " eps!" << endl; }
  
  return;
}

int MOSFETProblemCuda::compare_linsys_host_device(const string filename)
{
  cerr << " From '" << __func__ << "' : testing " << filename << "...";

  const int NX    = host_dm -> get_X()   -> get_N();
  const int NZ    = host_dm -> get_Z()   -> get_N();

  const double tol = 1.e-12;

  lis_output_matrix( _matrix_2d_LIS, LIS_FMT_MM, (char*)"linsys_CPU_MMF.txt" );
  string *scol1 = new string [NX*NZ*NX*NZ];
  string *scol2 = new string [NX*NZ*NX*NZ];
  string *scol3 = new string [NX*NZ*NX*NZ];
  int *col1 = new int [NX*NZ*NX*NZ];
  int *col2 = new int [NX*NZ*NX*NZ];
  double *col3 = new double [NX*NZ*NX*NZ];
  ifstream istr;
  ofstream ostr;
  istr.open("linsys_CPU_MMF.txt", ios_base::in);
  string spalabra1, spalabra2, spalabra3, spalabra4, spalabra5, spalabra6, spalabra7, snumlineas;
  istr >> spalabra1 >> spalabra2 >> spalabra3 >> spalabra4 >> spalabra5 >> spalabra6 >> spalabra7 >> snumlineas;
  int numlineas = atoi(snumlineas.c_str());
  int nle = 0;
  for(int linea=0; linea<numlineas; ++linea)
    {
      istr >> scol1[linea] >> scol2[linea] >> scol3[linea];
      if(atoi(scol1[linea].c_str()) != 0)
	{
	  col1[linea] = atoi(scol1[linea].c_str())-1;
	  col2[linea] = atoi(scol2[linea].c_str())-1;
	  col3[linea] = atof(scol3[linea].c_str());
	  ++nle;
	}
    }
  istr.close();

  // from GPU
  double *_matrix_2d = new double [NX*NZ*(2*NZ+1)];
  CUDA_SAFE_CALL( cudaMemcpy(_matrix_2d, _GPU_matrix_2d, NX*NZ*(2*NZ+1)*sizeof(double),cudaMemcpyDeviceToHost) );

  double linfdiff=0;
  int line1=0;
  for(int line=0; line<NX*NZ*NX*NZ; ++line)
    {
      int r=line/(NX*NZ);
      int s=line-r*(NX*NZ);
      if( r == col1[line1] && s == col2[line1] )
	linfdiff = max( linfdiff, fabs(col3[line1] - matrix_2d(r,s)) );
      ++line1;
    }
  if( linfdiff > tol )
    cerr << "WARNING! ";
  cerr << "linfdiff = " << linfdiff << "...";

  delete [] scol1;
  delete [] scol2;
  delete [] scol3;
  delete [] col1;
  delete [] col2;
  delete [] col3;
  delete [] _matrix_2d;

  cerr << "[ok] (function " << __func__ << ", line " << __LINE__ << ")" << endl;

  return 0;
}
  

int MOSFETProblemCuda::compare_rhs_host_device(const string filename)
{
  cerr << " From '" << __func__ << "' : testing " << filename << "...";

  const int NX    = host_dm -> get_X()   -> get_N();
  const int NZ    = host_dm -> get_Z()   -> get_N();

  const double tol = 1.e-12;

  ifstream istr;
  ofstream ostr;
  string spalabra1, spalabra2, spalabra3, spalabra4, spalabra5, spalabra6, spalabra7, snumlineas;
  lis_output_vector( _rhs_LIS, LIS_FMT_MM, (char*)"rhs_CPU_MMF.txt" );
  string *scol1 = new string [NX*NZ];
  string *scol2 = new string [NX*NZ];
  int *col1 = new int [NX*NZ];
  double *col3 = new double [NX*NZ];
  istr.open("rhs_CPU_MMF.txt", ios_base::in);
  istr >> spalabra1 >> spalabra2 >> spalabra3 >> spalabra4 >> spalabra5 >> snumlineas;
  int numlineas = atoi(snumlineas.c_str());
  for(int linea=0; linea<numlineas; ++linea)
    {
      istr >> scol1[linea] >> scol2[linea];
      col1[linea] = atoi(scol1[linea].c_str())-1;
      col3[linea] = atof(scol2[linea].c_str());

      if(isinf(col3[linea]) || isnan(col3[linea]))
	{
	  // cerr << " col3[" << linea << "] = " << col3[linea] << endl;
	  throw error_TESTING_RHS();
	}
    }
  istr.close();
  double *_rhs_2d = new double [NX*NZ ];
  CUDA_SAFE_CALL( cudaMemcpy(_rhs_2d, _GPU_rhs, NX*NZ*sizeof(double),cudaMemcpyDeviceToHost) );

  // delete from here
  ostr.open("rhs.txt", ios_base::out);
  for(int line=0; line<NX*NZ; ++line)
    {
      ostr << setw(20) << line
	   << setw(20) << col3[line]
	   << setw(20) << _rhs_2d[line]
	   << setw(20) << fabs( col3[line] - _rhs_2d[line] )
	   << endl;
    }
  ostr.close();
  // to here

  double linfdiff = 0, linfhost = 0, linfdevice = 0;
  for(int line=0; line<NX*NZ; ++line)
    {
      linfdiff = max( linfdiff, fabs( col3[line] - _rhs_2d[line] ) );
      linfhost = max( linfhost, fabs(col3[line]) );
      linfdevice = max( linfdevice, fabs(_rhs_2d[line]) );
    }
  if( linfdiff > tol )
    cerr << "WARNING! ";
  cerr << "linfdiff = " << linfdiff << "..." << "linfhost = " << linfhost << "..." << "linfdevice = " << linfdevice << "...";

  delete [] scol1;
  delete [] scol2;
  delete [] col1;
  delete [] col3;
  delete [] _rhs_2d;

  cerr << "[ok] (function " << __func__ << ", line " << __LINE__ << ")" << endl;

  return 0;
}
  

int MOSFETProblemCuda::compare_chi_host_device(const double* HOST, const double * device, const int N, const string filename)
{
  cerr << " From '" << __func__ << "' : testing " << filename << "...";
  const double tol = 1.e-12;

  double *host = new double [N];
  memcpy(host, HOST, N*sizeof(double));
  double *copy = new double [N];
  checkCudaErrors( cudaMemcpy(copy, device, N*sizeof(double), cudaMemcpyDeviceToHost) );

  for(int i=0; i<N; ++i)
    {
      host[i] = fabs(host[i]);
      copy[i] = fabs(copy[i]);
    }
  
  double linfdiff=0;
  for(int i=0; i<N; ++i)
    linfdiff = max (linfdiff, fabs(host[i]-copy[i]));
  if( linfdiff > tol )
    cerr << "WARNING! ";
  cerr << "linfdiff = " << linfdiff << "...";

  ofstream ostr;
  ostr.open("compare_host.txt", ios_base::out);
  for(int i=0; i<N; ++i)
    ostr << setw(25) << i
	 << setw(25) << host[i]
	 // << setw(25) << copy[i]
	 <<endl;
  ostr.close();
  ostr.open("compare_device.txt", ios_base::out);
  for(int i=0; i<N; ++i)
    ostr << setw(25) << i
	 // << setw(25) << host[i]
	 << setw(25) << copy[i]
	 <<endl;
  ostr.close();
  ostr.open(filename.c_str(), ios_base::out);
  for(int i=0; i<N; ++i)
    ostr << setw(25) << i
	 << setw(25) << host[i]
	 << setw(25) << copy[i]
	 << setw(25) << fabs( host[i] - copy[i] )
	 <<endl;
  ostr.close();
  delete [] host;
  delete [] copy;
  
  cerr << "[ok] (function " << __func__ << ", line " << __LINE__ << ")" << endl;

  return ( system("diff compare_host.txt compare_device.txt 1>comp_stdout.txt 2>comp_stderr.txt") );
}

int MOSFETProblemCuda::compare_host_device(const double* host, const double * device, const int N, const string filename)
{
  cerr << " From '" << __func__ << "' : testing " << filename << "...";
  const double tol = 1.E-10;

  double *copy = new double [N];
  checkCudaErrors( cudaMemcpy(copy, device, N*sizeof(double), cudaMemcpyDeviceToHost) );

  double linfdiff=0;
  double linfhost=0;
  double linfdevice=0;
  for(int i=0; i<N; ++i)
    {
      linfdiff = max (linfdiff, fabs(host[i]-copy[i]));
      linfhost = max (linfhost, fabs(host[i]));
      linfdevice = max (linfdevice, fabs(copy[i]));
    }
  if( linfdiff > tol )
    {
      cerr << " From '" << __func__ << "' : testing " << filename << "...";
      cerr << "ERROR! ";
      cerr << "linfdiff = " << linfdiff << "..." << "linfhost = " << linfhost << "..." << "linfdevice = " << linfdevice << "...";
      throw error_TESTING_COMPHOSTDEV();
    }

  cerr << " linfhost = " << linfhost << " --- linfdevice = " << linfdevice << " --- linfdiff = " << linfdiff << endl;
  
  // ofstream ostr;
  // ostr.open("compare_host.txt", ios_base::out);
  // for(int i=0; i<N; ++i)
  //   ostr << setw(25) << i
  // 	 << setw(25) << host[i]
  // 	 // << setw(25) << copy[i]
  // 	 <<endl;
  // ostr.close();
  // ostr.open("compare_device.txt", ios_base::out);
  // for(int i=0; i<N; ++i)
  //   ostr << setw(25) << i
  // 	 // << setw(25) << host[i]
  // 	 << setw(25) << copy[i]
  // 	 <<endl;
  // ostr.close();
  // ostr.open(filename.c_str(), ios_base::out);
  // for(int i=0; i<N; ++i)
  //   ostr << setw(25) << i
  // 	 << setw(25) << host[i]
  // 	 << setw(25) << copy[i]
  // 	 << setw(25) << fabs( host[i] - copy[i] )
  // 	 <<endl;
  // ostr.close();
  // delete [] copy;
  
  // return ( system("diff compare_host.txt compare_device.txt 1>comp_stdout.txt 2>comp_stderr.txt") );

  return 0;
}
  

int MOSFETProblemCuda::compare_device_device(const double* device1, const double * device2, const int N, const string filename)
{
  cerr << " From '" << __func__ << "' : testing " << filename << "...";

  const double tol = 1.e-12;

  double *copy1 = new double [N];
  checkCudaErrors( cudaMemcpy(copy1, device1, N*sizeof(double), cudaMemcpyDeviceToHost) );
  double *copy2 = new double [N];
  checkCudaErrors( cudaMemcpy(copy2, device2, N*sizeof(double), cudaMemcpyDeviceToHost) );

  double linfdiff=0;
  double linf_copy1=0;
  double linf_copy2=0;
  for(int i=0; i<N; ++i)
    {
      linfdiff = max (linfdiff, fabs(copy1[i]-copy2[i]));
      linf_copy1 = max (linf_copy1, fabs(copy1[i]));
      linf_copy2 = max (linf_copy2, fabs(copy2[i]));
    }
  if( linfdiff > tol )
    cerr << "WARNING! ";
  // cerr << "linfdiff = " << linfdiff << "..." << "linf_copy1 = " << linf_copy1 << "..." << "linf_copy2 = " << linf_copy2 << "...";

  cerr << "linfdiff = " << linfdiff;

  // for(int i=0; i<N; ++i)
  //   if( fabs(copy1[i]-copy2[i]) > tol )
  //     {
  // 	cerr << "WARNING! device[" << i << "] = " << copy2[i] << " != copy1[" << i << "] = " << copy1[i] << endl;
  // 	return 1;
  //     }

  // ofstream ostr;
  // ostr.open("compare_device1.txt", ios_base::out);
  // for(int i=0; i<N; ++i)
  //   ostr << setw(25) << i
  // 	 << setw(25) << copy1[i]
  // 	 // << setw(25) << copy2[i]
  // 	 <<endl;
  // ostr.close();
  // ostr.open("compare_device2.txt", ios_base::out);
  // for(int i=0; i<N; ++i)
  //   ostr << setw(25) << i
  // 	 // << setw(25) << copy1[i]
  // 	 << setw(25) << copy2[i]
  // 	 <<endl;
  // ostr.close();
  delete [] copy1;
  delete [] copy2;

  return ( system("diff compare_device1.txt compare_device2.txt 1>/dev/null 2>/dev/null") );
}




void MOSFETProblemCuda::test_pdf( const int STAGE, const string funcname )
{
  cerr << " testing pdf inside function '" << funcname << "'...";

  const int NSBN  = host_dm -> get_NSBN();
  const int NX    = host_dm -> get_X()   -> get_N();
  const int NW    = host_dm -> get_W()   -> get_N();
  const int NPHI  = host_dm -> get_PHI() -> get_N();

  double *_test;
  _test = new double[ _NVALLEYS*NSBN*NX*NW*NPHI*4 ];
  checkCudaErrors( cudaMemcpy(_test, _GPU_pdf, _NVALLEYS*NSBN*NX*NW*NPHI*4*sizeof(double), cudaMemcpyDeviceToHost) );
  for(int nu=0; nu<_NVALLEYS; ++nu)
    for(int p=0; p<NSBN; ++p)
      for(int i=0; i<NX; ++i)
	for(int l=0; l<NW; ++l)
	  for(int m=0; m<NPHI; ++m)
	    {
	      double val = _test[(STAGE)*NX*_NVALLEYS*NSBN*NW*NPHI + (i)*_NVALLEYS*NSBN*NW*NPHI + (nu)*NSBN*NW*NPHI + (p)*NW*NPHI + (l)*NPHI + (m)];
	      if( isinf(val) || isnan(val) )
		{
		  // cerr << " pdf(" << nu << "," << p << "," << i << "," << l << "," << m << "," << STAGE << ") = " << val << endl;
		  throw error_TESTING_PDF();
		}
	    }
  delete [] _test;
  cerr << "[ok] (function " << __func__ << ", line " << __LINE__ << ")" << endl;
  
  return;
}

void MOSFETProblemCuda::test_rhs( const string funcname )
{
  cerr << " testing rhs inside function '" << funcname << "'...";

  const int NSBN  = host_dm -> get_NSBN();
  const int NX    = host_dm -> get_X()   -> get_N();
  const int NW    = host_dm -> get_W()   -> get_N();
  const int NPHI  = host_dm -> get_PHI() -> get_N();

  double *_test;
  _test = new double[ _NVALLEYS*NSBN*NX*NW*NPHI ];
  checkCudaErrors( cudaMemcpy(_test, _GPU_rhs_pdf, _NVALLEYS*NSBN*NX*NW*NPHI*sizeof(double), cudaMemcpyDeviceToHost) );
  for(int nu=0; nu<_NVALLEYS; ++nu)
    for(int p=0; p<NSBN; ++p)
      for(int i=0; i<NX; ++i)
	for(int l=0; l<NW; ++l)
	  for(int m=0; m<NPHI; ++m)
	    {
	      double val = _test[(i)*_NVALLEYS*NSBN*NW*NPHI + (nu)*NSBN*NW*NPHI + (p)*NW*NPHI + (l)*NPHI + (m)];
	      if( isinf(val) || isnan(val) )
		{
		  // cerr << " rhs(" << nu << "," << p << "," << i << "," << l << "," << m << ") = " << val << endl;
		  throw error_TESTING_RHS();
		}
	    }
  delete [] _test;
  cerr << "[ok] (function " << __func__ << ", line " << __LINE__ << ")" << endl;
  
  return;
}

void MOSFETProblemCuda::test_eigenstates_GPU()
{
  cerr << " Testing eigenstates on GPU...";

  const int NSBN  = host_dm -> get_NSBN();
  const int NX    = host_dm -> get_X()   -> get_N();
  const int NZ    = host_dm -> get_Z()   -> get_N();

  cerr << "copying eps...";
  CUDA_SAFE_CALL( cudaMemcpy( _eps, _GPU_eps, _NVALLEYS*NSBN*NX*sizeof(double), cudaMemcpyDeviceToHost ) );
  cerr << "copying chi...";
  CUDA_SAFE_CALL( cudaMemcpy( _chi, _GPU_chi, _NVALLEYS*NSBN*NX*_SCHROED_MATRIX_SIZE_PAD*sizeof(double), cudaMemcpyDeviceToHost ) );
  double *h_A = new double[ _NVALLEYS*NX*_SCHROED_ROW_SIZE ];
  cerr << "copying d_A...";
  CUDA_SAFE_CALL( cudaMemcpy( h_A, d_A, _NVALLEYS*NX*_SCHROED_ROW_SIZE*sizeof(double), cudaMemcpyDeviceToHost ) );
  double val;
  const int N = NZ-2;
  double *test = new double[ N ];

  cerr << "testing...";
 
  for(int nu=0; nu<_NVALLEYS; ++nu)
    for(int i=0; i<NX; ++i)
      {
	int line = i*_NVALLEYS + nu;
	double *D = &h_A[ line * _SCHROED_ROW_SIZE ];
	double *S = D + _SCHROED_MATRIX_SIZE;
	for(int p=0; p<NSBN; ++p)
	  {
	    double eigval = eps(nu,p,i);
	    double *eigvec = &chi(nu,p,i,1);

	    val = D[0] * eigvec[0] + S[0] * eigvec[1] - eigval * eigvec[0];
	    test[0] = val;
	    if( fabs(val) > 1.e-6 )
	      cerr << "D[0] * eigvec[0] + S[0] * eigvec[1] - eigval * eigvec[0] = (" << D[0] << ")*(" << eigvec[0] << ") + (" << S[0] << ")*(" << eigvec[1] << ") - (" << eigval << ")*(" << eigvec[0] << ") = " << D[0] * eigvec[0] + S[0] * eigvec[1] - eigval * eigvec[0] << endl;
	    for(int j=1; j<N-1; ++j)
	      {
		val = S[j-1] * eigvec[j-1] + D[j] * eigvec[j] + S[j] * eigvec[j+1] - eigval * eigvec[j];
		test[j] = val;
		if( fabs(val) > 1.e-6 )
		  cerr << "S[" << j-1 << "] * eigvec[" << j-1 << "] + D[" << j << "] * eigvec[" << j << "] + S[" << j <<"] * eigvec[" << j+1 << "] - eigval * eigvec[" << j <<"] = (" << S[j-1] << ")*(" << eigvec[j-1]  << ") + (" << D[j] << ")*(" << eigvec[j] << ") + (" << S[j] << ")*(" << eigvec[j+1] << ") - (" << eigval << ")*(" << eigvec[j] << ") = " << S[j-1] * eigvec[j-1] + D[j] * eigvec[j] + S[j] * eigvec[j+1] - eigval * eigvec[j] << endl;
	      }
	    val = S[N-2] * eigvec[N-2] + D[N-1] * eigvec[N-1] - eigval * eigvec[N-1];
	    test[N-1] = val;
	    if( fabs(val) > 1.e-6 )
	      cerr << "S[N-2] * eigvec[N-2] + D[N-1] * eigvec[N-1] - eigval * eigvec[N-1] = (" << S[N-2] << ")*(" <<  eigvec[N-2] << ") + (" << D[N-1] << ")*(" << eigvec[N-1] << ") - (" << eigval << ")*(" << eigvec[N-1] << ") = " << S[N-2] * eigvec[N-2] + D[N-1] * eigvec[N-1] - eigval * eigvec[N-1] << endl;
	    
	    double nrm = 0;
	    for(int j=0; j<N; ++j)
	      nrm += test[j] * test[j];
	    nrm = sqrt(nrm);

	    if( nrm > 1.e-6 )
	    // if( nrm > 1.e-12 )
	      {
		// cerr << " ERROR from function " << __func__ << " : there must REALLY be a problem. The eigenstates are not properly computed." << endl;
		// cerr << " L2_error = " << nrm << endl;
		// for(int j=1; j<_NZ-1; ++j)
		//   cerr << " eps("     << nu << "," << p << "," << i << ") = "             << eps(nu,p,i)
		//        << " --- chi(" << nu << "," << p << "," << i << "," << j << ") * chi(" << nu << "," << p << "," << i << "," << j << ") = " << chi(nu,p,i,j) * chi(nu,p,i,j)
		//        << " --- eigvec[" << j-1 << "] * eigvec[" << j-1 << "] = " << eigvec[j-1] * eigvec[j-1]
		//        << endl;
		// for(int j=0; j<N; ++j)
		//   cerr << " test[" << j << "] = " << test[j]
		//        << " --- D[" << j << "] = " << D[j]
		//        << " --- S[" << j << "] = " << S[j] << endl;
		throw error_TESTING_EIGENSTATES();
	      }

	  }
      }
  delete [] h_A;
  delete [] test;

  cerr << "[ok] (function " << __func__ << ", line " << __LINE__ << ")" << endl;
}

void MOSFETProblemCuda::test_cuda_gershgorin( double *_GPU_Y, double *_GPU_Z )
{
  cerr << " Called '" << __func__ << "'...";

  const int NX    = host_dm -> get_X()   -> get_N();

  double *_Y = new double[_NVALLEYS*NX];
  double *_Z = new double[_NVALLEYS*NX];
  CUDA_SAFE_CALL( cudaMemcpy( _Y, _GPU_Y, _NVALLEYS*NX*sizeof(double), cudaMemcpyDeviceToHost ) );
  CUDA_SAFE_CALL( cudaMemcpy( _Z, _GPU_Z, _NVALLEYS*NX*sizeof(double), cudaMemcpyDeviceToHost ) );

  for(int line=0; line<_NVALLEYS*NX; ++line)
    if( isnan(_Y[line]) || isinf(_Y[line]) || isnan(_Z[line]) || isinf(_Z[line]) )
      {
	cerr << " _Y[" << line << "] = " << _Y[line] << endl
	     << " _Z[" << line << "] = " << _Z[line] << endl;
      }
  
  delete [] _Y;
  delete [] _Z;
}


void MOSFETProblemCuda::test_vec( double *d_vec, int taille, char* name )
{
  cerr << " testing '" << name << "'...";

  cerr << " copying vector to host...";
  double *h_vec = new double[ taille ];
  CUDA_SAFE_CALL( cudaMemcpy( h_vec, d_vec, taille*sizeof(double), cudaMemcpyDeviceToHost ) );

  cerr << " testing for inf or nan...";
  for(int i=0; i<taille; ++i)
    {
      double val = h_vec[ i ];
      if( isnan(val) || isinf(val) )
	{
	  // cerr << " " << name << "[ " << i << "] = " << h_vec[ i ] << endl;
	  throw error_TESTING_NANORINF();
	}
    }
	
  cerr << " deleting host copy...";
  delete [] h_vec;

  cerr << "[ok] (function " << __func__ << ", line " << __LINE__ << ")" << endl;
}


template <class T> void MOSFETProblemCuda::print_to_file( T *dmem, const int N, const string filename )
{
  cerr << " Called '" << __func__ << "'...";
  double *hmem = new double[N];
  CUDA_SAFE_CALL( cudaMemcpy(hmem, dmem, N*sizeof(T), cudaMemcpyDeviceToHost) );
  
  ofstream ostr;
  ostr.open( filename.c_str(), ios_base::out );

  for(int i=0; i<N; ++i)
    ostr << setw(20) << i
	 << setw(20) << hmem[i]
	 << endl;
  
  ostr.close();

  delete [] hmem;
}



void MOSFETProblemCuda::test_eigenvalues_device()
{
  cerr << " Testing eigenvalues...";
  
  const int NSBN  = host_dm -> get_NSBN();
  const int NX    = host_dm -> get_X()   -> get_N();
  const int NZ    = host_dm -> get_Z()   -> get_N();

  double *h_A = new double[ _NVALLEYS*NX*_SCHROED_ROW_SIZE ];
  CUDA_SAFE_CALL( cudaMemcpy(h_A, d_A, _NVALLEYS*NX*_SCHROED_ROW_SIZE*sizeof(double), cudaMemcpyDeviceToHost) );
  CUDA_SAFE_CALL( cudaMemcpy(_eps, _GPU_eps, _NVALLEYS*NSBN*NX*sizeof(double), cudaMemcpyDeviceToHost) );

  const int n = NZ-2;

  int cuantos_mal = 0;
  int cuantos_bien = 0;
  
  for(int nu=0; nu<_NVALLEYS; ++nu)
    for(int i=0; i<NX; ++i)
      {
	const int line = i*_NVALLEYS + nu;
	const double *D = &h_A[ line*_SCHROED_ROW_SIZE ]; 
	const double *S = D + _SCHROED_MATRIX_SIZE;

	// double _D[_NZ];
	// double _E[_NZ];

	// for(int j=1; j<_NZ-1; ++j)
	//   _D[j-1]  = 0.5*_cs1*DZM2*(.5/effmass(Z,nu,i,j-1)+1./effmass(Z,nu,i,j)+.5/effmass(Z,nu,i,j+1));

	// for(int j=1; j<_NZ-1; ++j)
	//   _D[j-1] -= (pot(i,j)+vconf(i,j));

	// for(int j=1; j<_N; ++j)
	//   _E[j-1]  = -0.5*_cs1*DZM2*(.5/effmass(Z,nu,i,j) + .5/effmass(Z,nu,i,j+1));

	for(int p=0; p<NSBN; ++p)
	  {
	    double P[n+1];
	    double X = eps(nu,p,i);
	    P[0] = 1.0;
	    // P[1] = _D[0] - X;
	    P[1] = D[0] - X;
	    for(int j=2; j<=n; ++j)
	      // P[j] = (_D[j-1] - X) * P[j-1] - _E[j-2]*_E[j-2] * P[j-2];
	      P[j] = (D[j-1] - X) * P[j-1] - S[j-2]*S[j-2] * P[j-2];

	    const double val = P[n];
	    if(fabs(val) > 1.e-6)
	      {
		++cuantos_mal;
		// cerr << " From " << __func__ << " : ERROR numero " << setw(6) << cuantos_mal << "! Hay un problema con el valor propio (nu,p,i) = (" << nu << "," << p << "," << i << ") --- val = " << val << endl;
		// for(int j=0; j<=n; ++j)
		//   {
		//     cerr << " eps(" << nu << "," << p << "," << i << ") = " << eps(nu,p,i) << " --- ";
		//     cerr << " D[" << j << "] = " << D[j] << " --- ";
		//     cerr << " S[" << j << "] = " << S[j] << " --- ";
		//     cerr << " P[" << j << "] = " << P[j] << endl;
		//   }
		throw error_TESTING_EIGENVALUES();
	      }
	    else
	      {
		++cuantos_bien;
		cerr << " From " << __func__ << " : Caso bueno numero " << setw(6) << cuantos_bien << ". El valor propio (nu,p,i) = (" << nu << "," << p << "," << i << ") esta bien --- val = " << val << endl;
	      }
	  }
      }
  
  cerr << " cuantos_mal = " << cuantos_mal << " --- cuantos_bien = " << cuantos_bien << endl;

  if (cuantos_mal != 0)
    throw error_TESTING_EIGENVALUES();

  delete [] h_A;

  cerr << "[ok] (function " << __func__ << ", line " << __LINE__ << ")" << endl;
    
}



void MOSFETProblemCuda::test_eigenvalues_host()
{
  cerr << " Testing eigenvalues on HOST...";

  const int NSBN  = host_dm -> get_NSBN();
  const int NX    = host_dm -> get_X()   -> get_N();
  const int NZ    = host_dm -> get_Z()   -> get_N();
  
  double *h_A = new double[ _NVALLEYS*NX*_SCHROED_ROW_SIZE ];
  CUDA_SAFE_CALL( cudaMemcpy(h_A, d_A, _NVALLEYS*NX*_SCHROED_ROW_SIZE*sizeof(double), cudaMemcpyDeviceToHost) );

  double DZM2 = host_dm->get_Z()->get_delta_m2();
  int _N      = NZ-2; 

  int cuantos_mal = 0;
  int cuantos_bien = 0;
  
  for(int nu=0; nu<_NVALLEYS; ++nu)
    for(int i=0; i<NX; ++i)
      {
	// const int line = i*_NVALLEYS + nu;
	// const double *D = &h_A[ line*_SCHROED_ROW_SIZE ]; 
	// const double *S = D + _SCHROED_MATRIX_SIZE;

	double D[NZ];
	double S[NZ];

	for(int j=1; j<NZ-1; ++j)
	  D[j-1]  = 0.5*host_adimpar->get_cs1()*DZM2*(.5/effmass(Zdim,nu,i,j-1)+1./effmass(Zdim,nu,i,j)+.5/effmass(Zdim,nu,i,j+1));

	for(int j=1; j<NZ-1; ++j)
	  D[j-1] -= (pot(i,j)+vconf(i,j));

	for(int j=1; j<_N; ++j)
	  S[j-1]  = -0.5*host_adimpar->get_cs1()*DZM2*(.5/effmass(Zdim,nu,i,j) + .5/effmass(Zdim,nu,i,j+1));

	for(int p=0; p<NSBN; ++p)
	  {
	    double X = eps(nu,p,i);
	    
	    double P = 1;
	    double Q = D[0] - X;
	    double R;
	  
	    for( int j = 1; j < _SCHROED_MATRIX_SIZE; j++ )
	      {
		R = ( D[j] - X ) * Q - S[j-1]*S[j-1] * P;
		P = Q;
		Q = R;
	      }

	    const double val = R;
	    if(fabs(val) > 1.e-6)
	      {
		++cuantos_mal;
		cerr << " From " << __func__ << " : ERROR numero " << setw(6) << cuantos_mal << "! Hay un problema con el valor propio (nu,p,i) = (" << nu << "," << p << "," << i << ") --- val = " << val << endl;
	      }
	    else
	      {
		++cuantos_bien;
		cerr << " From " << __func__ << " : Caso bueno numero " << setw(6) << cuantos_bien << ". El valor propio (nu,p,i) = (" << nu << "," << p << "," << i << ") esta bien --- val = " << val << endl;
	      }
	  }
      }
  
  cerr << " cuantos_mal = " << cuantos_mal << " --- cuantos_bien = " << cuantos_bien << endl;

  if (cuantos_mal != 0)
    throw error_TESTING_EIGENVALUES();

  delete [] h_A;

  cerr << "[ok] (function " << __func__ << ", line " << __LINE__ << ")" << endl;
    
}

void MOSFETProblemCuda::test_eigenvalues_matrices()
{
  cerr << " Testing eigenvalues matrices...";
  
  const int NSBN  = host_dm -> get_NSBN();
  const int NX    = host_dm -> get_X()   -> get_N();
  const int NZ    = host_dm -> get_Z()   -> get_N();
  const int NW    = host_dm -> get_W()   -> get_N();
  const int NPHI  = host_dm -> get_PHI() -> get_N();

  double *h_A = new double[ _NVALLEYS*NX*_SCHROED_ROW_SIZE ];
  CUDA_SAFE_CALL( cudaMemcpy(h_A, d_A, _NVALLEYS*NX*_SCHROED_ROW_SIZE*sizeof(double), cudaMemcpyDeviceToHost) );

  // save_device_array(d_A, _NVALLEYS*_NX*_SCHROED_ROW_SIZE, "d_A.dat");

  // save_device_array(d_A, _NVALLEYS*_NX*_SCHROED_ROW_SIZE, "d_A_2.dat");
  // save_host_array(h_A, _NVALLEYS*_NX*_SCHROED_ROW_SIZE, "h_A.dat");

  CUDA_SAFE_CALL( cudaMemcpy(_pot, _GPU_pot, NX*NZ*sizeof(double), cudaMemcpyDeviceToHost) );

  double DZM2 = host_dm->get_Z()->get_delta_m2();
  int _N      = NZ-2; 

  int cuantos_mal = 0;
  int cuantos_bien = 0;
  
  for(int nu=0; nu<_NVALLEYS; ++nu)
    for(int i=0; i<NX; ++i)
      {
	const int line = i*_NVALLEYS + nu;
	const double *D = &h_A[ line*_SCHROED_ROW_SIZE ]; 
	const double *S = D + _SCHROED_MATRIX_SIZE;

	double _D[NZ];
	double _E[NZ];

	for(int j=1; j<NZ-1; ++j)
	  _D[j-1]  = 0.5*host_adimpar->get_cs1()*DZM2*(.5/effmass(Zdim,nu,i,j-1)+1./effmass(Zdim,nu,i,j)+.5/effmass(Zdim,nu,i,j+1));

	for(int j=1; j<NZ-1; ++j)
	  _D[j-1] -= (pot(i,j)+vconf(i,j));

	for(int j=1; j<_N; ++j)
	  _E[j-1]  = -0.5*host_adimpar->get_cs1()*DZM2*(.5/effmass(Zdim,nu,i,j) + .5/effmass(Zdim,nu,i,j+1));

	for(int j=0; j<NZ-2; ++j)
	  if(fabs(D[j] - _D[j]) > 1.e-6)
	    {
	      ++cuantos_mal;
	      cerr << " (nu,i) = (" << nu << "," << i << ") --- D[" << j << "] = " << D[j] << " --- _D[" << j << "] = " << _D[j] << endl;
	    }
	  else
	    ++cuantos_bien;

	for(int j=0; j<NZ-1; ++j)
	  if(fabs(S[j] - _E[j]) > 1.e-6)
	    {
	      ++cuantos_mal;
	      cerr << " (nu,i) = (" << nu << "," << i << ") --- S[" << j << "] = " << S[j] << " --- _E[" << j << "] = " << _E[j] << endl;
	    }
	  else
	    ++cuantos_bien;
      }
  
  cerr << " cuantos_mal = " << cuantos_mal << " --- cuantos_bien = " << cuantos_bien << endl;

  if (cuantos_mal != 0)
    throw error_TESTING_EIGENVALUES();
  
  delete [] h_A;

  cerr << "[ok] (function " << __func__ << ", line " << __LINE__ << ")" << endl;
    
}




#endif



double MOSFETProblemCuda::max_pdf_border(const int s)
{
  // cerr << " Called '" << __func__ << "'...";

  const int NSBN  = host_dm -> get_NSBN();
  const int NX    = host_dm -> get_X()   -> get_N();
  const int NW    = host_dm -> get_W()   -> get_N();
  const int NPHI  = host_dm -> get_PHI() -> get_N();

  double maxpdfe = 0;
  double *_ipe = new double[ _NVALLEYS*NSBN*NX*NW ];
#ifdef __CUDACC__
  checkCudaErrors( cudaMemcpy(_ipe, _GPU_integrated_pdf_energy, _NVALLEYS*NSBN*NX*NW*sizeof(double), cudaMemcpyDeviceToHost) );
#else
  memcpy( _ipe, _integrated_pdf_energy, _NVALLEYS*NSBN*NX*NW*sizeof(double) );
#endif

  for(int i=0; i<NX; ++i)
    for(int nu=0; nu<_NVALLEYS; ++nu)
      for(int p=0; p<NSBN; ++p)
	{
	  if( fabs(_ipe[(i)*_NVALLEYS*NSBN*NW + (nu)*NSBN*NW + (p)*NW + (NW-1)]) > maxpdfe )
	    maxpdfe = fabs(_ipe[(i)*_NVALLEYS*NSBN*NW + (nu)*NSBN*NW + (p)*NW + (NW-1)]);
	  if( _ipe[(i)*_NVALLEYS*NSBN*NW + (nu)*NSBN*NW + (p)*NW + (NW-1)] + 1.e-6 < 0 )
	    cerr << " No me hallo muy bien." << endl;
	}

  for(int l=0; l<NW; ++l)
    {
      _integrated_pdf_nu_p_i_m[l] = 0;
      for(int i=0; i<NX; ++i)
	for(int nu=0; nu<_NVALLEYS; ++nu)
	  for(int p=0; p<NSBN; ++p)
	    _integrated_pdf_nu_p_i_m[l] += _ipe[(i)*_NVALLEYS*NSBN*NW + (nu)*NSBN*NW + (p)*NW + (l)];
    }

  delete [] _ipe;    

  return maxpdfe;
}


// int MOSFETProblemCuda::compute_eta()
// {
//   double time_now = gettime();
//   double TIME = _time*_tstar;

//   double slope = ( time_now-_time_init )/TIME;
//   _ETA = slope*_TMAX;

//   seconds = (int)_ETA % 60;
//   int totminutes = ((int)_ETA - seconds)/60;
//   minutes = totminutes % 60;
//   hours = (totminutes-minutes)/60;

//   return 0;
// }



// void MOSFETProblemCuda::test_some_norms( const int s )
// {
//   double L1_pdf = norm_L1_device( &GPU_pdf(0,0,0,0,0,s), _NVALLEYS*_NSBN*_NX*_NW*_NPHI );
//   double L2_pdf = norm_L2_device( &GPU_pdf(0,0,0,0,0,s), _NVALLEYS*_NSBN*_NX*_NW*_NPHI );
//   double Linf_pdf = norm_Linf_device( &GPU_pdf(0,0,0,0,0,s), _NVALLEYS*_NSBN*_NX*_NW*_NPHI );
//   double L1_rhs = norm_L1_device( &GPU_rhs_pdf(0,0,0,0,0), _NVALLEYS*_NSBN*_NX*_NW*_NPHI );
//   double L2_rhs = norm_L2_device( &GPU_rhs_pdf(0,0,0,0,0), _NVALLEYS*_NSBN*_NX*_NW*_NPHI );
//   double Linf_rhs = norm_Linf_device( &GPU_rhs_pdf(0,0,0,0,0), _NVALLEYS*_NSBN*_NX*_NW*_NPHI );
//   cerr << setprecision(12)
//        << " ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++" << endl
//        << "s = " << s << " --- "
//        << "pdf_sum = " << sum_device( &GPU_pdf(0,0,0,0,0,s), _NVALLEYS*_NSBN*_NX*_NW*_NPHI ) << " --- "
//        << "L1_pdf = " << L1_pdf << " --- "
//        << "L2_pdf = " << L2_pdf << " --- "
//        << "Linf_pdf = " << Linf_pdf << " --- "
//        << "rhs_sum = " << sum_device( &GPU_rhs_pdf(0,0,0,0,0), _NVALLEYS*_NSBN*_NX*_NW*_NPHI ) << " --- "
//        << "L1_rhs = " << L1_rhs << " --- "
//        << "L2_rhs = " << L2_rhs << " --- "
//        << "Linf_rhs = " << Linf_rhs << " --- "
//        << "_max_deps_dx = " << _max_deps_dx << endl
//        << " ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++" << endl;

//   return;
// }



#ifdef __CUDACC__
__global__ void cuda_test_data( const solverParams *sp, const double *start_address, const int N, const char *msg )
{
  const int idx = blockIdx.x*blockDim.x + threadIdx.x;
  if(idx < N)
    {
      if( isinf( start_address[idx] ) || isnan( start_address[idx] ) )
	{
	  printf("Function 'test_data' returned an error: start_address[%i] = %e\n", idx, start_address[idx]);
	  assert(0);
	}
    }
}
#endif




#ifdef __CUDACC__
__global__ void cuda_test_pdf( const solverParams *sp, const double *_GPU_pdf, const double *_GPU_rhs_pdf, const int stage )
{
  const int global_index = blockIdx.x*blockDim.x + threadIdx.x;

  const int NSBN     = sp->get_NSBN();
  const int NX       = sp->get_NX();
  const int NW       = sp->get_NW();
  const int NPHI     = sp->get_NPHI();

  if(global_index < _NVALLEYS * NSBN * NX * NW * NPHI)
    {
      int nu,p,i,l,m;
      GPU_map_1D_to_5D( global_index, &i, NX, &nu, _NVALLEYS, &p, NSBN, &l, NW, &m, NPHI );
      
      if( isinf(GPU_pdf(nu,p,i,l,m,stage)) || isnan(GPU_pdf(nu,p,i,l,m,stage)) || isinf(GPU_rhs_pdf(nu,p,i,l,m)) || isnan(GPU_rhs_pdf(nu,p,i,l,m)) )
	{
	  printf("Function 'test_pdf' returned an error: GPU_pdf(%i, %i, %i, %i, %i, %i) = %e --- GPU_rhs_pdf(%i, %i, %i, %i, %i) = %e\n", nu, p, i, l, m, stage, GPU_pdf(nu,p,i,l,m,stage), nu, p, i, l, m, GPU_rhs_pdf(nu,p,i,l,m));
	  assert(0);
	}
    }
}
#endif



#ifdef __CUDACC__
__global__ void cuda_test_pdf( const solverParams *sp, const double *_GPU_rhs_pdf )
{
  const int global_index = blockIdx.x*blockDim.x + threadIdx.x;

  const int NSBN     = sp->get_NSBN();
  const int NX       = sp->get_NX();
  const int NW       = sp->get_NW();
  const int NPHI     = sp->get_NPHI();

  if(global_index < _NVALLEYS * NSBN * NX * NW * NPHI)
    {
      int nu,p,i,l,m;
      GPU_map_1D_to_5D( global_index, &i, NX, &nu, _NVALLEYS, &p, NSBN, &l, NW, &m, NPHI );
      
      if( isinf( GPU_rhs_pdf(nu,p,i,l,m) ) || isnan( GPU_rhs_pdf(nu,p,i,l,m) ) )
	{
	  printf("Function 'test_rhs' returned an error: GPU_pdf(%i, %i, %i, %i, %i) = %e\n", nu, p, i, l, m, GPU_rhs_pdf(nu,p,i,l,m));
	  assert(0);
	}
    }
}
#endif



#ifdef __CUDACC__
__global__ void cuda_compare( const solverParams *sp, const double *_vec1, const double *_vec2, const int N )
{
  const int global_index = blockIdx.x*blockDim.x + threadIdx.x;
  const int NSBN     = sp->get_NSBN();
  const int NX       = sp->get_NX();
  const int NW       = sp->get_NW();
  const int NPHI     = sp->get_NPHI();
  if(global_index < N)
    {
      int nu,p,i,l,m;
      GPU_map_1D_to_5D( global_index, &i, NX, &nu, _NVALLEYS, &p, NSBN, &l, NW, &m, NPHI );
      
      if( _vec1[global_index] != _vec2[global_index] )
	{
	  printf("Function 'cuda_compare' returned an error: vec1(%i, %i, %i, %i, %i) = %e --- vec2(%i, %i, %i, %i, %i) = %e\n", nu, p, i, l, m, _vec1[global_index], nu, p, i, l, m, _vec2[global_index]);
	  assert(0);
	}
    }
}
#endif



cuda_testing_kernels_config::cuda_testing_kernels_config(const discrMeshes *dm)
{
  cuda_test_data_config = new kernelConfig;
  cuda_test_pdf_config = new kernelConfig;
  cuda_test_rhs_config = new kernelConfig;
  cuda_compare_config = new kernelConfig;
}
