#include "srjparams.h"

srjParams::srjParams(const int N, const int P, const string filename)
{
  __P = P;
  ifstream istr;
  istr.open(filename.c_str(), ios_base::in);
  if( !istr )
    {
      cerr << " ERROR! Could not open file '" << filename << "'" << endl;
      exit(-1);
    }

  string token;

  // 1 - first element is M
  getline( istr, token);
  __M = stoi( token );
  // cerr << " __M = " << __M << endl;
  __relaxpar = new double[ __M ];

  // 2 - relaxation parameters
  for( int line=0; line < __M; ++line )
    {
      getline( istr, token);
      __relaxpar[line] = stof( token );
      // cerr << "__relaxpar[" << line << "] = "<< __relaxpar[line] << endl;
    }

  // int line=0;
  // while( !istr.eof() )
  //   {
  //     getline( istr, token);
  //     __relaxpar[line] = stof( token );
  //     cerr << "__relaxpar[" << line << "] = "<< __relaxpar[line] << endl;
  //     ++line;
  //   }

  // if( line != __M )
  //   {
  //     cerr << " ERROR! Something went wrong while reading file '" << filename << "'" << endl;
  //     exit(-1);
  //   }
  
  istr.close();
}

srjParams::srjParams(const int N, const int P)
{
  if( P > 0 )
    {
      __P     = P;
      __omega = new double[P];
      __beta  = new double[P];
      __q     = new int[P];
    }
  else
    {
      cerr << " ERROR! Parameter P=" << P << " for the SRJ method is not valid." << endl;
      exit(-1);
    }
  
  // if( N == 32 && P == 7 )
  //   {
  //     __omega[0] = 370.035;   __omega[1] =   167.331; __omega[2] =   51.1952; __omega[3] =   13.9321; __omega[4] =  3.80777; __omega[5] =  1.18727; __omega[6] = 0.556551;
  //     __beta [0] = 0.0107542; __beta [1] = 0.0171537; __beta [2] = 0.0336988; __beta [3] = 0.0699421; __beta [4] = 0.144888; __beta [5] = 0.282064; __beta [6] = 0.441499;
  //     compute_M();
  //     compute_q();
  //     compute_relaxparams();
  //   }
  if( N == 32 && P == 7 )
    {
      __M = 93;
      __relaxpar = new double[ __M ];
      
      __relaxpar[0] = 370.035;
      __relaxpar[1] = 0.556551;
      __relaxpar[2] = 167.331;
      __relaxpar[3] = 0.556551;
      __relaxpar[4] = 167.331;
      __relaxpar[5] = 0.556551;
      __relaxpar[6] = 51.1952;
      __relaxpar[7] = 0.556551;
      __relaxpar[8] = 51.1952;
      __relaxpar[9] = 0.556551;
      __relaxpar[10] = 51.1952;
      __relaxpar[11] = 0.556551;
      __relaxpar[12] = 13.9321;
      __relaxpar[13] = 0.556551;
      __relaxpar[14] = 13.9321;
      __relaxpar[15] = 0.556551;
      __relaxpar[16] = 13.9321;
      __relaxpar[17] = 0.556551;
      __relaxpar[18] = 13.9321;
      __relaxpar[19] = 0.556551;
      __relaxpar[20] = 13.9321;
      __relaxpar[21] = 0.556551;
      __relaxpar[22] = 13.9321;
      __relaxpar[23] = 0.556551;
      __relaxpar[24] = 13.9321;
      __relaxpar[25] = 0.556551;
      __relaxpar[26] = 3.80777;
      __relaxpar[27] = 0.556551;
      __relaxpar[28] = 3.80777;
      __relaxpar[29] = 0.556551;
      __relaxpar[30] = 3.80777;
      __relaxpar[31] = 0.556551;
      __relaxpar[32] = 3.80777;
      __relaxpar[33] = 0.556551;
      __relaxpar[34] = 3.80777;
      __relaxpar[35] = 0.556551;
      __relaxpar[36] = 3.80777;
      __relaxpar[37] = 0.556551;
      __relaxpar[38] = 3.80777;
      __relaxpar[39] = 0.556551;
      __relaxpar[40] = 3.80777;
      __relaxpar[41] = 0.556551;
      __relaxpar[42] = 3.80777;
      __relaxpar[43] = 0.556551;
      __relaxpar[44] = 3.80777;
      __relaxpar[45] = 0.556551;
      __relaxpar[46] = 3.80777;
      __relaxpar[47] = 0.556551;
      __relaxpar[48] = 3.80777;
      __relaxpar[49] = 0.556551;
      __relaxpar[50] = 3.80777;
      __relaxpar[51] = 0.556551;
      __relaxpar[52] = 0.556551;
      __relaxpar[53] = 0.556551;
      __relaxpar[54] = 0.556551;
      __relaxpar[55] = 0.556551;
      __relaxpar[56] = 0.556551;
      __relaxpar[57] = 0.556551;
      __relaxpar[58] = 0.556551;
      __relaxpar[59] = 0.556551;
      __relaxpar[60] = 0.556551;
      __relaxpar[61] = 0.556551;
      __relaxpar[62] = 0.556551;
      __relaxpar[63] = 0.556551;
      __relaxpar[64] = 0.556551;
      __relaxpar[65] = 0.556551;
      __relaxpar[66] = 0.556551;
      __relaxpar[67] = 1.18727;
      __relaxpar[68] = 1.18727;
      __relaxpar[69] = 1.18727;
      __relaxpar[70] = 1.18727;
      __relaxpar[71] = 1.18727;
      __relaxpar[72] = 1.18727;
      __relaxpar[73] = 1.18727;
      __relaxpar[74] = 1.18727;
      __relaxpar[75] = 1.18727;
      __relaxpar[76] = 1.18727;
      __relaxpar[77] = 1.18727;
      __relaxpar[78] = 1.18727;
      __relaxpar[79] = 1.18727;
      __relaxpar[80] = 1.18727;
      __relaxpar[81] = 1.18727;
      __relaxpar[82] = 1.18727;
      __relaxpar[83] = 1.18727;
      __relaxpar[84] = 1.18727;
      __relaxpar[85] = 1.18727;
      __relaxpar[86] = 1.18727;
      __relaxpar[87] = 1.18727;
      __relaxpar[88] = 1.18727;
      __relaxpar[89] = 1.18727;
      __relaxpar[90] = 1.18727;
      __relaxpar[91] = 1.18727;
      __relaxpar[92] = 1.18727;
    }
  else
    {
      cerr << " ERROR! Parameters (N,P)=(" << N << "," << P << ") for the SRJ method not implemented." << endl;
      exit(-1);
    }

}



void srjParams::compute_M()
{
  __M = int(1./__beta[0] + .5);
  
  return;
}

void srjParams::compute_q()
{
  const double beta0_aux = 1./__M;
  for(int i=0; i<__P; ++i)
    {
      const double aux = __beta[i] / beta0_aux;
      __q[i] = int(aux + .5);
    }

  return;
}

void srjParams::compute_relaxparams()
{
  cerr << "computing relaxation parameters...";
  
  // construct relaxation parameters list
  __relaxpar = new double[ __M ];

  int index = 0;
  for( int i=0; i<__P; ++i )
    {
      int offset = (int)( (double)__M / (double)__q[i] );

      for( int j=index++; j<__M && __q[i] > 0; j+= offset )
	{
	  int pos = j;
	  while( __relaxpar[pos] != 0 )
	    {
	      ++pos;
	      pos %= __M;
	    }
	  if( 0 <= pos && pos < __M )
	    {
	      __relaxpar[pos] = __omega[i];
	      --__q[i];
	    }
	}
    }

  cerr << "[ok]" << endl;

  return;
}



void srjParams::printDataFields()
{
  cerr << " ********** class srjParams ********** " << endl;
  cerr << " __P = " << __P << endl;
  // for(int i=0; i<__P; ++i)
  //   {
  //     cerr << " __omega[" << i << "] = " << __omega[i] << endl;
  //     cerr << "  __beta[" << i << "] = " << __beta[i]  << endl;
  //     cerr << "     __q[" << i << "] = " << __q[i]     << endl;
  //   }
  cerr << " __M = " << __M << endl;
  for(int i=0; i<__M; ++i)
    {
      cerr << " __relaxpar[" << i << "] = " << __relaxpar[i] << endl;
    }
}
