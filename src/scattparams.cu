#include "scattparams.h"

/*
 begin section : 'scattPar02'
 written       : 20230526
 last modified : 20230526
 author        : Francesco Vecil

 Description
 ===========
    This is the constructor of class scattParams, which is described in the header file.
*/

scattParams::scattParams()
{
  __PHONONS          = true;
  __ROUGHNESS        = false;
  __LENGTH_SR        = 1.5E-9;
  __DELTA_STRETCH_SR = .2;
  __DELTA_SR         = .3E-9;
}

/*
 end section: 'scattPar02'
*/

/*
 begin section : 'scattPar03'
 written       : 20230526
 last modified : 20230526
 author        : Francesco Vecil

 Description
 ===========
    This function will read a configuration file, and modify accordingly the
    data contained in the class.
*/

void scattParams::readConfFile(const string filename)
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
      exit(_ERR_SCATTPAR_NOTOPEN);
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
		  exit(_ERR_SCATTPAR_TOKNOSEP);
		}

	      // 4.1.2 - get the identifier (left of '=' character)
	      string identifier = token.substr( 0, seppos );
	      // cerr << " identifier = " << identifier << endl;

	      // 4.1.3 - get the value (right of '=' character): if empty return an error message
	      string value = token.substr( seppos+1 );
	      if( value.empty() )
		{
		  cerr << " From file '" << __FILE__ << "', from function '" << __func__ << "', from line '" << __LINE__ << "': ERROR! In file '" << filename << "' empyu value field for token '" << token << "'" << endl;
		  exit(_ERR_SCATTPAR_EMPTYVAL);
		}

	      // 4.1.4 - now switch on the identifier and assign the corresponding value after conversion
	      if( identifier == "PHONONS" )
		{
		  if( value == "false" || value == "0" )
		    __PHONONS = (bool)0;
		  else if( value == "true" || value == "1" )
		    __PHONONS = (bool)1;
		  else
		    {
		      cerr << " From file '" << __FILE__ << "', from function '" << __func__ << "', from line '" << __LINE__ << "': ERROR! In file '" << filename
			   << "', syntax error in token '" << token << "'!" << endl;
		    }
		}
	      else if( identifier == "ROUGHNESS" )
		{
		  if( value == "false" || value == "0" )
		    __ROUGHNESS = (bool)0;
		  else if( value == "true" || value == "1" )
		    __ROUGHNESS = (bool)1;
		  else
		    {
		      cerr << " From file '" << __FILE__ << "', from function '" << __func__ << "', from line '" << __LINE__ << "': ERROR! In file '" << filename
			   << "', syntax error in token '" << token << "'!" << endl;
		    }
		}
	      else if( identifier == "LENGTH_SR" )
		{
		  double dval = stod( value );
		  // cerr << " dval = " << dval << endl;
		  __LENGTH_SR = dval;
		}
	      else if( identifier == "DELTA_STRETCH_SR" )
		{
		  double dval = stod( value );
		  // cerr << " dval = " << dval << endl;
		  __DELTA_STRETCH_SR = dval;
		}
	      else if( identifier == "DELTA_SR" )
		{
		  double dval = stod( value );
		  // cerr << " dval = " << dval << endl;
		  __DELTA_SR = dval;
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
  end section: 'scattPar03'
*/


/*
 begin section : 'scattPar05'
 written       : 20230526
 last modified : 20230526
 author        : Francesco Vecil

 Description
 ===========
*/

void scattParams::readInline(int _argc, char **_argv)
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
	      exit(_ERR_SCATTPAR_EMPTYVAL);
	    }
	  
	  // 2.3.1 - if requested, read configuration file
	  if( identifier == "configfile" )
	    {
	      readConfFile(value);
	    }

	  // 2.4 - now switch on the identifier and assign the corresponding value after conversion
	  else if( identifier == "PHONONS" )
	    {
	      if( value == "false" || value == "0" )
		__PHONONS = (bool)0;
	      else if( value == "true" || value == "1" )
		__PHONONS = (bool)1;
	      else
		{
		  cerr << " From file '" << __FILE__ << "', from function '" << __func__ << "', from line '" << __LINE__ << "': ERROR! Syntax error in token '" << token << "'!" << endl;
		}
	    }
	  else if( identifier == "ROUGHNESS" )
	    {
	      if( value == "false" || value == "0" )
		__ROUGHNESS = (bool)0;
	      else if( value == "true" || value == "1" )
		__ROUGHNESS = (bool)1;
	      else
		{
		  cerr << " From file '" << __FILE__ << "', from function '" << __func__ << "', from line '" << __LINE__ << "': ERROR! Syntax error in token '" << token << "'!" << endl;
		}
	    }
	  else if( identifier == "LENGTH_SR" )
	    {
	      double dval = stod( value );
	      __LENGTH_SR = dval;
	    }
	  else if( identifier == "DELTA_STRETCH_SR" )
	    {
	      double dval = stod( value );
	      __DELTA_STRETCH_SR = dval;
	    }
	  else if( identifier == "DELTA_SR" )
	    {
	      double dval = stod( value );
	      __DELTA_SR = dval;
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
  end section: 'scattPar05'
*/


/*
 begin section : 'scattPar04'
 written       : 20230526
 last modified : 20230526
 author        : Francesco Vecil

 Description
 ===========
    This method prints the data contained in the class.
*/

void scattParams::printDataFields()
{
  cerr                                                  << endl;
  cerr << " *** data from class 'scattParams' ***     " << endl;
  cerr << "          PHONONS = " << __PHONONS           << endl;
  cerr << "        ROUGHNESS = " << __ROUGHNESS         << endl;
  cerr << "        LENGTH_SR = " << __LENGTH_SR         << endl;
  cerr << " DELTA_STRETCH_SR = " << __DELTA_STRETCH_SR  << endl;
  cerr << "         DELTA_SR = " << __DELTA_SR          << endl;
  cerr                                                  << endl;

  return;
}

/*
  end section: 'scattPar04'
*/

