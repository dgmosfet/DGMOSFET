#include "physdevice.h"

/*
 begin section: 'PhysDev02'
 written: 202303209
 last modified: 20230424
 author: Francesco Vecil

 Description
 ===========
    This is the constructor of class physDevice, which is described in the header file.

    The constructor introduces the values into the data structures by one of the three means,
    in increasing order of priority:

    1) by default values

    2) by means of a configuration file, whose syntax will be on the model of
       LSOURCE = 10.E-9;
       Units of the International System will be used.

    3) by inline values given when calling the executable, like
       ./detmosfet --NX 129
       which will overwrite the default value, and the value into the configuration file, if any.

 Modifications
 =============
    1) The class constructor takes as parameters _argc and _argv.
*/

physDevice::physDevice()
{
  __LSOURCE         = 5.E-9;
  __LCHANNEL        = 10.E-9;
  __LDRAIN          = 5.E-9;
  __LGATE           = 12.E-9;
  __ZWIDTH          = 8.E-9;
  __LSiO2           = 1.E-9;
  __NDHIGH          = 1.E26;
  __NDLOW           = 1.E18;
  __VCONF_WELL      = -3.15;
  __CONTACTMATERIAL = ALLUMINIUM;
  __VUPPERGATE      = 0.5;
  __VLOWERGATE      = 0.5;
  __VBIAS           = 0.1;
}

/*
 end section: 'PhysDev02'
*/

/*
 begin section: 'PhysDev03'
 written: 202303209
 last modified: 20230403
 author: Francesco Vecil

 Description
 ===========
    This function will read a configuration file, and modify accordingly the
    data contained in the class.
*/

void physDevice::readConfFile(const string filename)
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
      exit(_ERR_PHYSDEV_NOTOPEN);
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
		  exit(_ERR_PHYSDEV_TOKNOSEP);
		}

	      // 4.1.2 - get the identifier (left of '=' character)
	      string identifier = token.substr( 0, seppos );
	      // cerr << " identifier = " << identifier << endl;

	      // 4.1.3 - get the value (right of '=' character): if empty return an error message
	      string value = token.substr( seppos+1 );
	      if( value.empty() )
		{
		  cerr << " From file '" << __FILE__ << "', from function '" << __func__ << "', from line '" << __LINE__ << "': ERROR! In file '" << filename << "' empyu value field for token '" << token << "'" << endl;
		  exit(_ERR_PHYSDEV_EMPTYVAL);
		}

	      // 4.1.4 - now switch on the identifier and assign the corresponding value after conversion
	      if( identifier == "LSOURCE" )
		{
		  double dval = stod( value );
		  // cerr << " dval = " << dval << endl;
		  __LSOURCE = dval;
		}
	      else if( identifier == "LCHANNEL" )
		{
		  double dval = stod( value );
		  // cerr << " dval = " << dval << endl;
		  __LCHANNEL = dval;
		}
	      else if( identifier == "LDRAIN" )
		{
		  double dval = stod( value );
		  // cerr << " dval = " << dval << endl;
		  __LDRAIN = dval;
		}
	      else if( identifier == "LGATE" )
		{
		  double dval = stod( value );
		  // cerr << " dval = " << dval << endl;
		  __LGATE = dval;
		}
	      else if( identifier == "ZWIDTH" )
		{
		  double dval = stod( value );
		  // cerr << " dval = " << dval << endl;
		  __ZWIDTH = dval;
		}
	      else if( identifier == "LSiO2" )
		{
		  double dval = stod( value );
		  // cerr << " dval = " << dval << endl;
		  __LSiO2 = dval;
		}
	      else if( identifier == "NDHIGH" )
		{
		  double dval = stod( value );
		  // cerr << " dval = " << dval << endl;
		  __NDHIGH = dval;
		}
	      else if( identifier == "NDLOW" )
		{
		  double dval = stod( value );
		  // cerr << " dval = " << dval << endl;
		  __NDLOW = dval;
		}
	      else if( identifier == "VCONF_WELL" )
		{
		  double dval = stod( value );
		  // cerr << " dval = " << dval << endl;
		  __VCONF_WELL = dval;
		}
	      else if( identifier == "CONTACTMATERIAL" )
		{
		  if( value == "ALLUMINIUM" )
		    {
		      __CONTACTMATERIAL = ALLUMINIUM;
		    }
		  else if( value == "TUNGSTEN" )
		    {
		      __CONTACTMATERIAL = TUNGSTEN;
		    }
		  else
		    {
		      cerr << " From file '" << __FILE__ << "', from function '" << __func__ << "', from line '" << __LINE__ << "': WARNING! In file '" << filename
			   << "', in token '" << token << "', unidentified contact material. Using ALLUMINIUM instead." << endl;
		      __CONTACTMATERIAL = ALLUMINIUM;
		    }
		}
	      else if( identifier == "VUPPERGATE" )
		{
		  double dval = stod( value );
		  // cerr << " dval = " << dval << endl;
		  __VUPPERGATE = dval;
		}
	      else if( identifier == "VLOWERGATE" )
		{
		  double dval = stod( value );
		  // cerr << " dval = " << dval << endl;
		  __VLOWERGATE = dval;
		}
	      else if( identifier == "VBIAS" )
		{
		  double dval = stod( value );
		  // cerr << " dval = " << dval << endl;
		  __VBIAS = dval;
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
  end section: 'PhysDev03'
*/

/*
 begin section: 'PhysDev05'
 written: 20230424
 last modified: 20230424
 author: Francesco Vecil

 Description
 ===========
    This function will read inline information, and modify accordingly the
    data contained in the class.
*/

void physDevice::readInline(int _argc, char **_argv)
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
	      exit(_ERR_PHYSDEV_EMPTYVAL);
	    }
	  
	  // 2.3.1 - if requested, read configuration file
	  if( identifier == "configfile" )
	    {
	      readConfFile(value);
	    }

	  // 2.4 - now switch on the identifier and assign the corresponding value after conversion
	  else if( identifier == "LSOURCE" )
	    {
	      double dval = stod( value );
	      // cerr << " dval = " << dval << endl;
	      __LSOURCE = dval;
	    }
	  else if( identifier == "LCHANNEL" )
	    {
	      double dval = stod( value );
	      // cerr << " dval = " << dval << endl;
	      __LCHANNEL = dval;
	    }
	  else if( identifier == "LDRAIN" )
	    {
	      double dval = stod( value );
	      // cerr << " dval = " << dval << endl;
	      __LDRAIN = dval;
	    }
	  else if( identifier == "LGATE" )
	    {
	      double dval = stod( value );
	      // cerr << " dval = " << dval << endl;
	      __LGATE = dval;
	    }
	  else if( identifier == "ZWIDTH" )
	    {
	      double dval = stod( value );
	      // cerr << " dval = " << dval << endl;
	      __ZWIDTH = dval;
	    }
	  else if( identifier == "LSiO2" )
	    {
	      double dval = stod( value );
	      // cerr << " dval = " << dval << endl;
	      __LSiO2 = dval;
	    }
	  else if( identifier == "NDHIGH" )
	    {
	      double dval = stod( value );
	      // cerr << " dval = " << dval << endl;
	      __NDHIGH = dval;
	    }
	  else if( identifier == "NDLOW" )
	    {
	      double dval = stod( value );
	      // cerr << " dval = " << dval << endl;
	      __NDLOW = dval;
	    }
	  else if( identifier == "VCONF_WELL" )
	    {
	      double dval = stod( value );
	      // cerr << " dval = " << dval << endl;
	      __VCONF_WELL = dval;
	    }
	  else if( identifier == "CONTACTMATERIAL" )
	    {
	      if( value == "ALLUMINIUM" )
		{
		  __CONTACTMATERIAL = ALLUMINIUM;
		}
	      else if( value == "TUNGSTEN" )
		{
		  __CONTACTMATERIAL = TUNGSTEN;
		}
	      else
		{
		  cerr << " From file '" << __FILE__ << "', from function '" << __func__ << "', from line '" << __LINE__ << "': WARNING! In token '" << token << "', unidentified contact material. Using ALLUMINIUM instead." << endl;
		  __CONTACTMATERIAL = ALLUMINIUM;
		}
	    }
	  else if( identifier == "VUPPERGATE" )
	    {
	      double dval = stod( value );
	      // cerr << " dval = " << dval << endl;
	      __VUPPERGATE = dval;
	    }
	  else if( identifier == "VLOWERGATE" )
	    {
	      double dval = stod( value );
	      // cerr << " dval = " << dval << endl;
	      __VLOWERGATE = dval;
	    }
	  else if( identifier == "VBIAS" )
	    {
	      double dval = stod( value );
	      // cerr << " dval = " << dval << endl;
	      __VBIAS = dval;
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
  end section: 'PhysDev05'
*/




/*
 begin section: 'PhysDev04'
 written: 20230404
 last modified: 20230404
 author: Francesco Vecil

 Description
 ===========
    This method prints the data contained in the class.
*/

void physDevice::printDataFields()
{
  cerr                                              << endl;
  cerr << " *** data of class 'physDevice' ***"     << endl;
  cerr << "        LSOURCE = " << __LSOURCE         << endl;
  cerr << "       LCHANNEL = " << __LCHANNEL        << endl;
  cerr << "         LDRAIN = " << __LDRAIN          << endl;
  cerr << "          LGATE = " << __LGATE           << endl;
  cerr << "         ZWIDTH = " << __ZWIDTH          << endl;
  cerr << "          LSiO2 = " << __LSiO2           << endl;
  cerr << "         NDHIGH = " << __NDHIGH          << endl;
  cerr << "          NDLOW = " << __NDLOW           << endl;
  cerr << "     VCONF_WELL = " << __VCONF_WELL      << endl;
  cerr << "CONTACTMATERIAL = " << __CONTACTMATERIAL << endl;
  cerr << "     VUPPERGATE = " << __VUPPERGATE      << endl;
  cerr << "     VLOWERGATE = " << __VLOWERGATE      << endl;
  cerr << "          VBIAS = " << __VBIAS           << endl;
  cerr                                              << endl;

  return;
}

/*
  end section: 'PhysDev04'
*/


