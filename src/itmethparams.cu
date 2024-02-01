#include "itmethparams.h"


/*
 begin section : 'ItMethPar02'
 written       : 20230526
 last modified : 20230526
 author        : Francesco Vecil

 Description
 ===========
    This is the constructor of class itmethParams, which is described in the header file.
*/

itmethParams::itmethParams()
{
  __TOL_NR_INITCOND      = 1.E-12;
  __TOL_NR_THERMEQUIL    = 1.E-12;
  __TOL_NR_POTENTIAL     = 1.E-6;
  __ITERMAX_NR_POTENTIAL = 50000;
  __TOL_EIGVALS          = 1.E-12;
  __TOL_EIGVECS_IPIM     = 1.E-8;
  __TOL_LINSYS_SRJ       = 1.E-8;
  __ITERMAX_LINSYS_SRJ   = 50000;
}

/*
 end section: 'ItMethPar02'
*/

/*
 begin section : 'ItMethPar03'
 written       : 20230526
 last modified : 20230526
 author        : Francesco Vecil

 Description
 ===========
    This function will read a configuration file, and modify accordingly the
    data contained in the class.
*/

void itmethParams::readConfFile(const string filename)
{
  /*
    Try and open configuration file.
    If the operation fails, exit with a certain code.
   */
  ifstream istr;
  istr.open(filename.c_str(), ios_base::in);

  if( !istr )
    {
      cerr << " From file '" << __FILE__ << "', from function '" << __func__ << "', from line '" << __LINE__ << "': ERROR! Could not open configuration file '" << filename << "'" << endl;
      exit(_ERR_ITMETHPAR_NOTOPEN);
    }

  string token;
  while( !istr.eof() )
    {
      // 1 - get token
      getline( istr, token, ';');
      
      // 2 - delete LF characters from token
      while( token.find('\n') != string::npos )
	{
	  token.erase( token.find('\n'), 1 );
	}

      // 3 - delete space characters from token
      while( token.find(' ') != string::npos || token.find('\t') != string::npos )
	{
	  if( token.find(' ') != string::npos )
	    {
	      token.erase( token.find(' '), 1 );
	    }
	  else
	    {
	      token.erase( token.find('\t'), 1 );
	    }
	}

      // 4 - treat token only if it not empty
      if( !token.empty() )
	{
	  // 4.1 - treat the token only if it does not start with '#', used for comments
	  if(token[0] != '#')
	    {
	      // 4.1.0 - print token
	      // cerr << " From line " << __LINE__ << " : token = '" << token << "'" << endl;

	      // 4.1.1 - get the '=' character; if not found, return an error message
	      size_t seppos = token.find('=');
	      if( seppos == string::npos )
		{
		  cerr << " From file '" << __FILE__ << "', from function '" << __func__ << "', from line '" << __LINE__ << "': ERROR! In file '" << filename << "' no separator for token '" << token << "'" << endl;
		  exit(_ERR_ITMETHPAR_TOKNOSEP);
		}

	      // 4.1.2 - get the identifier (left of '=' character)
	      string identifier = token.substr( 0, seppos );
	      // cerr << " identifier = " << identifier << endl;

	      // 4.1.3 - get the value (right of '=' character): if empty return an error message
	      string value = token.substr( seppos+1 );
	      if( value.empty() )
		{
		  cerr << " From file '" << __FILE__ << "', from function '" << __func__ << "', from line '" << __LINE__ << "': ERROR! In file '" << filename << "' empyu value field for token '" << token << "'" << endl;
		  exit(_ERR_ITMETHPAR_EMPTYVAL);
		}

	      // 4.1.4 - now switch on the identifier and assign the corresponding value after conversion
	      if( identifier == "TOL_NR_INITCOND" )
		{
		  double dval = stof( value );
		  __TOL_NR_INITCOND = dval;
		}
	      else if( identifier == "TOL_NR_THERMEQUIL" )
		{
		  double dval = stof( value );
		  __TOL_NR_THERMEQUIL = dval;
		}
	      else if( identifier == "TOL_NR_POTENTIAL" )
		{
		  double dval = stof( value );
		  __TOL_NR_POTENTIAL = dval;
		}
	      else if( identifier == "ITERMAX_NR_POTENTIAL" )
		{
		  int ival = stoi( value );
		  __ITERMAX_NR_POTENTIAL = ival;
		}
	      else if( identifier == "TOL_EIGVALS" )
		{
		  double dval = stof( value );
		  __TOL_EIGVALS = dval;
		}
	      else if( identifier == "TOL_EIGVECS_IPIM" )
		{
		  double dval = stof( value );
		  __TOL_EIGVECS_IPIM = dval;
		}
	      else if( identifier == "TOL_LINSYS_SRJ" )
		{
		  double dval = stof( value );
		  __TOL_LINSYS_SRJ = dval;
		}
	      else if( identifier == "ITERMAX_LINSYS_SRJ" )
		{
		  int ival = stoi( value );
		  __ITERMAX_LINSYS_SRJ = ival;
		}
	      else
		{
		  // cerr << " From file '" << __FILE__ << "', from function '" << __func__ << "', from line '" << __LINE__ << "': WARNING! In file '" << filename
		  //      << "', in token '" << token << "', identifier '" << identifier << "' does not exist and will not be taken into account." << endl;
		}

	    }
	  
	}
    }
  
  istr.close();
}

/*
  end section: 'ItMethPar03'
*/


/*
 begin section : 'ItMethPar05'
 written       : 20230526
 last modified : 20230526
 author        : Francesco Vecil

 Description
 ===========
*/

void itmethParams::readInline(int _argc, char **_argv)
{
  string token;
  for( int it = 1; it < _argc; ++it ) /*
					read the whole list of parameters,
					excluding the name of the executable
				      */
    {
      // 1 - get token
      token = _argv[it];
     
      // 2 - treat token only if it starts with '--'
      if( token[0] == '-' && token[1] == '-' && token.find('=') )
	{
	  // 2.1 - get the '=' character; if not found, do not treat the token
	  size_t seppos = token.find('=');
	  if( seppos == string::npos )
	    {
	      continue;
	    }
	  
	  // 2.2 - get the identifier (left of '=' character)
	  string identifier = token.substr( 2, seppos-2 );
	  // cerr << " identifier = " << identifier << endl;
	  
	  // 2.3 - get the value (right of '=' character): if empty return an error message
	  string value = token.substr( seppos+1 );
	  if( value.empty() )
	    {
	      cerr << " From file '" << __FILE__ << "', from function '" << __func__ << "', from line '" << __LINE__ << "': ERROR! Empty value field for token '" << token << "'" << endl;
	      exit(_ERR_ITMETHPAR_EMPTYVAL);
	    }
	  
	  // 2.3.1 - if requested, read configuration file
	  if( identifier == "configfile" )
	    {
	      readConfFile(value);
	    }

	  // 2.4 - now switch on the identifier and assign the corresponding value after conversion
	  else if( identifier == "TOL_NR_INITCOND" )
	    {
	      double dval = stof( value );
	      __TOL_NR_INITCOND = dval;
	    }
	  else if( identifier == "TOL_NR_THERMEQUIL" )
	    {
	      double dval = stof( value );
	      __TOL_NR_THERMEQUIL = dval;
	    }
	  else if( identifier == "TOL_NR_POTENTIAL" )
	    {
	      double dval = stof( value );
	      __TOL_NR_POTENTIAL = dval;
	    }
	  else if( identifier == "ITERMAX_NR_POTENTIAL" )
	    {
	      int ival = stoi( value );
	      __ITERMAX_NR_POTENTIAL = ival;
	    }
	  else if( identifier == "TOL_EIGVALS" )
	    {
	      double dval = stof( value );
	      __TOL_EIGVALS = dval;
	    }
	  else if( identifier == "TOL_EIGVECS_IPIM" )
	    {
	      double dval = stof( value );
	      __TOL_EIGVECS_IPIM = dval;
	    }
	  else if( identifier == "TOL_LINSYS_SRJ" )
	    {
	      double dval = stof( value );
	      __TOL_LINSYS_SRJ = dval;
	    }
	  else if( identifier == "ITERMAX_LINSYS_SRJ" )
	    {
	      int ival = stoi( value );
	      __ITERMAX_LINSYS_SRJ = ival;
	    }
	  else
	    {
	      // cerr << " From file '" << __FILE__ << "', from function '" << __func__ << "', from line '" << __LINE__ << "': WARNING! In file '" << filename
	      //      << "', in token '" << token << "', identifier '" << identifier << "' does not exist and will not be taken into account." << endl;
	    }

	}
	  
    }
}

/*
  end section: 'ItMethPar05'
*/


/*
 begin section : 'ItMethPar04'
 written       : 20230526
 last modified : 20230526
 author        : Francesco Vecil

 Description
 ===========
    This method prints the data contained in the class.
*/

void itmethParams::printDataFields()
{
  cerr                                                         << endl;
  cerr << " *** data from class 'itmethParams' ***     "       << endl;
  cerr << "      TOL_NR_INITCOND = " << __TOL_NR_INITCOND      << endl;
  cerr << "    TOL_NR_THERMEQUIL = " << __TOL_NR_THERMEQUIL    << endl;
  cerr << "     TOL_NR_POTENTIAL = " << __TOL_NR_POTENTIAL     << endl;
  cerr << " ITERMAX_NR_POTENTIAL = " << __ITERMAX_NR_POTENTIAL << endl;
  cerr << "          TOL_EIGVALS = " << __TOL_EIGVALS          << endl;
  cerr << "     TOL_EIGVECS_IPIM = " << __TOL_EIGVECS_IPIM     << endl;
  cerr << "       TOL_LINSYS_SRJ = " << __TOL_LINSYS_SRJ       << endl;
  cerr << "   ITERMAX_LINSYS_SRJ = " << __ITERMAX_LINSYS_SRJ   << endl;
  cerr                                                         << endl;

  return;
}

/*
  end section: 'ItMethPar04'
*/


