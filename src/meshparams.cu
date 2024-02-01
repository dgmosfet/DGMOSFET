#include "meshparams.h"

/*
 begin section : 'meshPar02'
 written       : 20230526
 last modified : 20230526
 author        : Francesco Vecil

 Description
 ===========
    This is the default constructor of class meshParams, which is described in the header file.
*/

meshParams::meshParams()
{
  __NSBN             = 6;
  __NX               = 65;
  __NZ               = 65;
  __NW               = 300;
  __BARN             = 15;
  __NPHI             = 48;
  __NZEXT            = 84;
  __CFL              = .6;
  __STEPMAX          = 99999999;
  __TMAX             = 5.E-12;
  __FREQ             = 500;
}

/*
 end section: 'meshPar02'
*/

/*
 begin section : 'meshPar03'
 written       : 20230526
 last modified : 20230526
 author        : Francesco Vecil

 Description
 ===========
    This function will read a configuration file, and modify accordingly the
    data contained in the class.
*/

void meshParams::readConfFile(const string filename)
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
      exit(_ERR_MESHPAR_NOTOPEN);
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
		  exit(_ERR_MESHPAR_TOKNOSEP);
		}

	      // 4.1.2 - get the identifier (left of '=' character)
	      string identifier = token.substr( 0, seppos );
	      // cerr << " identifier = " << identifier << endl;

	      // 4.1.3 - get the value (right of '=' character): if empty return an error message
	      string value = token.substr( seppos+1 );
	      if( value.empty() )
		{
		  cerr << " From file '" << __FILE__ << "', from function '" << __func__ << "', from line '" << __LINE__ << "': ERROR! In file '" << filename << "' empyu value field for token '" << token << "'" << endl;
		  exit(_ERR_MESHPAR_EMPTYVAL);
		}

	      // 4.1.4 - now switch on the identifier and assign the corresponding value after conversion
	      if( identifier == "NSBN" )
		{
		  int ival = stoi( value );
		  __NSBN = ival;
		}
	      else if( identifier == "NX" )
		{
		  int ival = stoi( value );
		  __NX = ival;
		}
	      else if( identifier == "NZ" )
		{
		  int ival = stoi( value );
		  __NZ = ival;
		}
	      else if( identifier == "NW" )
		{
		  int ival = stoi( value );
		  __NW = ival;
		}
	      else if( identifier == "BARN" )
		{
		  int ival = stoi( value );
		  __BARN = ival;
		}
	      else if( identifier == "NPHI" )
		{
		  int ival = stoi( value );
		  __NPHI = ival;
		}
	      else if( identifier == "CFL" )
		{
		  double dval = stod( value );
		  __CFL = dval;
		}
	      else if( identifier == "STEPMAX" )
		{
		  int ival = stoi( value );
		  __STEPMAX = ival;
		}
	      else if( identifier == "TMAX" )
		{
		  double dval = stod( value );
		  __TMAX = dval;
		}
	      else if( identifier == "FREQ" )
		{
		  int ival = stoi( value );
		  __FREQ = ival;
		}
	      else if( identifier == "NZEXT" )
		{
		  int ival = stoi( value );
		  __NZEXT = ival;
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
  end section: 'meshPar03'
*/


/*
 begin section : 'meshPar05'
 written       : 20230526
 last modified : 20230526
 author        : Francesco Vecil

 Description
 ===========
*/

void meshParams::readInline(int _argc, char **_argv)
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
	      exit(_ERR_MESHPAR_EMPTYVAL);
	    }
	  
	  // 2.3.1 - if requested, read configuration file
	  if( identifier == "configfile" )
	    {
	      readConfFile(value);
	    }

	  // 2.4 - now switch on the identifier and assign the corresponding value after conversion
	  else if( identifier == "NSBN" )
	    {
	      int ival = stoi( value );
	      __NSBN = ival;
	    }
	  else if( identifier == "NX" )
	    {
	      int ival = stoi( value );
	      __NX = ival;
	    }
	  else if( identifier == "NZ" )
	    {
	      int ival = stoi( value );
	      __NZ = ival;
	    }
	  else if( identifier == "NW" )
	    {
	      int ival = stoi( value );
	      __NW = ival;
	    }
	  else if( identifier == "BARN" )
	    {
	      int ival = stoi( value );
	      __BARN = ival;
	    }
	  else if( identifier == "NPHI" )
	    {
	      int ival = stoi( value );
	      __NPHI = ival;
	    }
	  else if( identifier == "CFL" )
	    {
	      double dval = stod( value );
	      __CFL = dval;
	    }
	  else if( identifier == "STEPMAX" )
	    {
	      int ival = stoi( value );
	      __STEPMAX = ival;
	    }
	  else if( identifier == "TMAX" )
	    {
	      double dval = stod( value );
	      __TMAX = dval;
	    }
	  else if( identifier == "FREQ" )
	    {
	      int ival = stoi( value );
	      __FREQ = ival;
	    }
	  else if( identifier == "NZEXT" )
	    {
	      int ival = stoi( value );
	      __NZEXT = ival;
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
  end section: 'meshPar05'
*/


/*
 begin section : 'meshPar04'
 written       : 20230526
 last modified : 20230526
 author        : Francesco Vecil

 Description
 ===========
    This method prints the data contained in the class.
*/

void meshParams::printDataFields()
{
  cerr                                                  << endl;
  cerr << " *** data from class 'meshParams' ***      " << endl;
  cerr << "             NSBN = " << __NSBN              << endl;
  cerr << "               NX = " << __NX                << endl;
  cerr << "               NZ = " << __NZ                << endl;
  cerr << "               NW = " << __NW                << endl;
  cerr << "             BARN = " << __BARN              << endl;
  cerr << "             NPHI = " << __NPHI              << endl;
  cerr << "              CFL = " << __CFL               << endl;
  cerr << "          STEPMAX = " << __STEPMAX           << endl;
  cerr << "             TMAX = " << __TMAX              << endl;
  cerr << "             FREQ = " << __FREQ              << endl;
  cerr << "            NZEXT = " << __NZEXT             << endl;
  cerr                                                  << endl;

  return;
}

/*
  end section: 'meshPar04'
*/
