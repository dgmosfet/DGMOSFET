#include "mosfetproblem.h"
#include "debug_flags.h"



void MOSFETProblemCuda::config_comptime()
{
  fieldname[ _PHASE_STEP ] = "step";                                                            // 0
  
  fieldname[ _PHASE_BTE ] = "BTE";                                                              // 1
  fieldname[ _PHASE_BTE_SETFLUXES ] = "fluxes";                                                 // 2
  fieldname[ _PHASE_BTE_SCATTERINGS ] = "scatterings";                                          // 3
  fieldname[ _PHASE_BTE_SCATTERINGS_PHONONS_WM1 ] = "phonons.overlap";                          // 4
  fieldname[ _PHASE_BTE_SCATTERINGS_PHONONS ] = "phonons";                                      // 
  fieldname[ _PHASE_BTE_SCATTERINGS_ROUGHNESS ] = "roughness";                                  // 
  fieldname[ _PHASE_BTE_SCATTERINGS_PHONONS_PHONONSGAIN ] = "phonons.gain";                     // 5
  fieldname[ _PHASE_BTE_SCATTERINGS_PHONONS_PHONONSLOSS ] = "phonons.loss";                     // 6
  fieldname[ _PHASE_BTE_SCATTERINGS_ROUGHNESS_ISR ] = "roughness.ISR";                          // 7
  fieldname[ _PHASE_BTE_SCATTERINGS_ROUGHNESS_ISR_STRETCHTOTVOLDENS ] = "roughness.stretch";    // 8
  fieldname[ _PHASE_BTE_SCATTERINGS_ROUGHNESS_ISR_SOLVEPOISSEXT ] = "roughness.Poisson";        // 9
  fieldname[ _PHASE_BTE_SCATTERINGS_ROUGHNESS_ISR_DELTAPOT ] = "roughness.Delta";               // 10
  fieldname[ _PHASE_BTE_SCATTERINGS_ROUGHNESS_ISR_OVERLAPSR ] = "roughness.overlap";            // 11
  fieldname[ _PHASE_BTE_SCATTERINGS_ROUGHNESS_ROUGHNESSGAIN ] = "roughness.gain";               // 12
  fieldname[ _PHASE_BTE_SCATTERINGS_ROUGHNESS_ROUGHNESSLOSS ] = "roughness.loss";               // 13
  fieldname[ _PHASE_BTE_WENOX ] = "BTE.WENOX";                                                  // 14
  fieldname[ _PHASE_BTE_WENOW ] = "BTE.WENOW";                                                  // 15
  fieldname[ _PHASE_BTE_WENOPHI ] = "BTE.WENOPHI";                                              // 16
  fieldname[ _PHASE_BTE_RK ] = "BTE.RK";                                                        // 17

  fieldname[ _PHASE_DENS ] = "dens";                                                            // 18
 
  fieldname[ _PHASE_ITER ] = "iter";                                                            // 19
  fieldname[ _PHASE_ITER_EIGENSTATES ] = "iter.eigen";                                          // 20
  fieldname[ _PHASE_ITER_EIGENSTATES_PREPARE ] = "iter.eigen.prepare";                          // 21
  fieldname[ _PHASE_ITER_EIGENSTATES_EIGENVALUES ] = "iter.eigen.vals";                         // 22
  fieldname[ _PHASE_ITER_EIGENSTATES_EIGENVECTORS ] = "iter.eigen.vecs";                        // 23
  fieldname[ _PHASE_ITER_SPSTEP ] = "iter.spstep";                                              // 24
  fieldname[ _PHASE_ITER_SPSTEP_FRECHET ] = "iter.Frechet";                                     // 25
  fieldname[ _PHASE_ITER_SPSTEP_CONSTRLINSYS ] = "iter.constrlinsys";                           // 26
  fieldname[ _PHASE_ITER_SPSTEP_SOLVELINSYS ] = "iter.solvelinsys";                             // 27
}


double MOSFETProblemCuda::gettime()
{
  timeval _timeval_now;
  gettimeofday(&_timeval_now,NULL);
  return _timeval_now.tv_sec+_timeval_now.tv_usec/1000000.;
}




/**
   PURPOSE:        

   FILE:           cuda_comptime.cu

   NAME:           MOSFETProblem::analysis_comptime

   PARAMETERS:

   RETURN VALUE:   none

   CALLS TO:       MOSFETProblem::save_times             (cuda_comptime.cu)
                   MOSFETProblem::save_piecharts         (cuda_comptime.cu)
                   MOSFETProblem::save_histograms        (cuda_comptime.cu)
                   MOSFETProblem::save_speedups          (cuda_comptime.cu)

   CALLED FROM:    MOSFETProblem::solve_2                (cuda_solve.cu)

   DATA MODIFIED:  

   METHOD:         

   LAST UPDATE:    2022 February 16
*/
void MOSFETProblemCuda::analysis_comptime()
{
  if(system("rm generate_comptime_analysis.sh 1>/dev/null 2>/dev/null"));

  save_times();
  // save_piecharts();
  save_histograms();
  save_speedups();

  if(system("chmod +x generate_comptimes_analysis.sh"));

  return;
}

// string ReplaceAll(string str, const string& from, const string& to)
// {
//   size_t start_pos = 0;
//   while((start_pos = str.find(from, start_pos)) != string::npos)
//     {
//       str.replace(start_pos, from.length(), to);
//       start_pos += to.length(); // Handles case where 'to' is a substring of 'from'
//     }
//   return str;
// }


void MOSFETProblemCuda::save_times()
{
  const int NSBN  = host_dm -> get_NSBN();
  const int NX    = host_dm -> get_X()   -> get_N();
  const int NZ    = host_dm -> get_Z()   -> get_N();
  const int NW    = host_dm -> get_W()   -> get_N();
  const int NPHI  = host_dm -> get_PHI() -> get_N();

  const int spacing = 64;
  
  ostringstream osstr;

  ifstream test;
  test.open("times.DAT", ios_base::in);
  if(!test)
    {
      test.close();
      ofstream tempostr;
      tempostr.open("times.DAT", ios_base::out);

      tempostr << setw(spacing) << "#1_comment"
	       << setw(spacing) << "#2_CPU_GPU"
	       << setw(spacing) << "#3_NUMTHREADS";
      int i;
      for(i=0; i<_NUM_CT_FIELDS; ++i)
	{
	  osstr << "#" << i+4 << "_" << fieldname[i];
	  tempostr << setw(spacing) << osstr.str();
	  osstr.str("");
	}
      osstr << "#" << i+4 << "_parameters";
      tempostr << setw(spacing) << osstr.str();
      osstr.str("");

      tempostr << endl;

      tempostr.close();
    }
  test.close();

  osstr << "\"" << comment << "\"" ;

  ofstream ostr;
  ostr.open("times.DAT", ios_base::app);
  ostr << setw(spacing) << osstr.str();
  osstr.str("");
#if defined(__CUDACC__)
  osstr.str("");
  osstr << "\"" << props.name << "\"";
  string s = ReplaceAll(osstr.str(), string(" "), string("_"));
  ostr << setw(spacing) << s;
  osstr.str("");
#endif
#if !defined(__CUDACC__)
  osstr.str("");
  osstr << "\"" << CPU_INFO << "\"";
  ostr << setw(spacing) << osstr.str();
  osstr.str("");
#endif
  ostr << setw(spacing) << omp_get_max_threads();
  for(int i=0; i<_NUM_CT_FIELDS; ++i)
    ostr << setw(spacing) << _ct[i]/_step;
  osstr  << "\"" << NSBN << "x" << NX << "x" << NZ << "x" << NW << "x" << NPHI << "\"";
  ostr << setw(spacing) << osstr.str();
  osstr.str("");
  ostr << endl;
  ostr.close();

return;
}


void MOSFETProblemCuda::save_piecharts()
{
  ofstream ostr;

  int nvals = 2;
  int field[100];
  field[0] = _PHASE_BTE;
  field[1] = _PHASE_ITER;
  // field[2] = _PHASE_DENS;
  save_piecharts_2( field, nvals, "phases" );
  ostr.open("generate_comptimes_analysis.sh", ios_base::app);
  ostr << "gnuplot scripts/piechart_phases.gp" << endl;
  ostr.close();

  nvals = 5;
  field[0] = _PHASE_ITER_EIGENSTATES_EIGENVALUES;
  field[1] = _PHASE_ITER_EIGENSTATES_EIGENVECTORS;
  field[2] = _PHASE_ITER_SPSTEP_FRECHET;
  field[3] = _PHASE_ITER_SPSTEP_CONSTRLINSYS;
  field[4] = _PHASE_ITER_SPSTEP_SOLVELINSYS;
  save_piecharts_2( field, nvals, "iter", _PHASE_ITER );
  ostr.open("generate_comptimes_analysis.sh", ios_base::app);
  ostr << "gnuplot scripts/piechart_iter.gp" << endl;
  ostr.close();
}


void MOSFETProblemCuda::save_histograms()
{
  ofstream ostr;

  int nvals = 2;
  int field[100];
  field[0] = _PHASE_BTE;
  // field[2] = _PHASE_DENS;
  field[1] = _PHASE_ITER;
  save_histograms_2( field, nvals, "phases", _PHASE_STEP );
  ostr.open("generate_comptimes_analysis.sh", ios_base::app);
  ostr << "gnuplot scripts/histogram_phases.gp" << endl;
  ostr.close();

  nvals = 4;
  field[0] = _PHASE_ITER_EIGENSTATES;
  field[1] = _PHASE_ITER_SPSTEP_FRECHET;
  field[2] = _PHASE_ITER_SPSTEP_CONSTRLINSYS;
  field[3] = _PHASE_ITER_SPSTEP_SOLVELINSYS;
  // nvals = 5;
  // field[0] = _PHASE_ITER_EIGENSTATES_EIGENVALUES;
  // field[1] = _PHASE_ITER_EIGENSTATES_EIGENVECTORS;
  // field[2] = _PHASE_ITER_SPSTEP_FRECHET;
  // field[3] = _PHASE_ITER_SPSTEP_CONSTRLINSYS;
  // field[4] = _PHASE_ITER_SPSTEP_SOLVELINSYS;
  save_histograms_2( field, nvals, "iter", _PHASE_ITER );
  ostr.open("generate_comptimes_analysis.sh", ios_base::app);
  ostr << "gnuplot scripts/histogram_iter.gp" << endl;
  ostr.close();
}


void MOSFETProblemCuda::save_speedups()
{
  ofstream ostr;

  // save_speedups_2( "phases", _PHASE_STEP );
  // ostr.open("generate_comptimes_analysis.sh", ios_base::app);
  // ostr << "gnuplot scripts/speedups_phases.gp" << endl;
  // ostr.close();
  
  // save_speedups_2( "iter", _PHASE_ITER );
  // ostr.open("generate_comptimes_analysis.sh", ios_base::app);
  // ostr << "gnuplot scripts/speedups_iter.gp" << endl;
  // ostr.close();
  
  // save_speedups_2( "eigenstates", _PHASE_ITER_EIGENSTATES );
  // ostr.open("generate_comptimes_analysis.sh", ios_base::app);
  // ostr << "gnuplot scripts/speedups_eigenstates.gp" << endl;
  // ostr.close();
  
  // save_speedups_2( "frechet", _PHASE_ITER_SPSTEP_FRECHET );
  // ostr.open("generate_comptimes_analysis.sh", ios_base::app);
  // ostr << "gnuplot scripts/speedups_frechet.gp" << endl;
  // ostr.close();
  
  // save_speedups_2( "constrlinsys", _PHASE_ITER_SPSTEP_CONSTRLINSYS );
  // ostr.open("generate_comptimes_analysis.sh", ios_base::app);
  // ostr << "gnuplot scripts/speedups_constrlinsys.gp" << endl;
  // ostr.close();
  
  // save_speedups_2( "solvelinsys", _PHASE_ITER_SPSTEP_SOLVELINSYS );
  // ostr.open("generate_comptimes_analysis.sh", ios_base::app);
  // ostr << "gnuplot scripts/speedups_solvelinsys.gp" << endl;
  // ostr.close();

  // all speedups in one graphic
  save_speedups_3();
  ostr.open("generate_comptimes_analysis.sh", ios_base::app);
  ostr << "gnuplot scripts/speedups.gp" << endl;
  ostr.close();
}

void MOSFETProblemCuda::save_piecharts_2( const int *field, int nvals, const string filename, const int complem )
{
  ostringstream ofilename;
  ofilename << "scripts/piechart_" << filename << ".gp";

  ofstream ostr;
  ostr.open(ofilename.str().c_str(), ios_base::out);

  string name[nvals+1];
  double val[nvals+1], normval[nvals+1];
  ostringstream oname; oname << std::fixed << setprecision(2);
  double SUM=0;
  for(int i=0; i<nvals; ++i)
    {
      val[i] = _ct[field[i]];
      SUM += val[i];
      oname << fieldname[field[i]] << " (" << 100.*val[i]/_ct[complem] << " %)";
      name[i] = oname.str().c_str();
      oname.str("");
    }
  val[nvals] = _ct[ complem ] - SUM;
  oname << "other" << " (" << 100.*val[nvals]/_ct[complem] << " %)";
  name[nvals] = oname.str().c_str();
  oname.str("");
  ++nvals;
  
  ostr << "!/bin/echo -en 'generating " << ofilename.str() << "...' 1>&2" << endl;
  ostr << "set term postscript eps enh color" << endl;
  ostr << "set out 'comptimes_analysis/" << filename << ".eps'" << endl;
  ostr << "set parametric" << endl;
  ostr << "unset border; unset tics; unset key; unset colorbox" << endl;
  ostr << "set xrange [-1.1:1.1]; set yrange [-1.1:1.1]" << endl;

  for(int i=0; i<nvals; ++i)
    ostr << "val" << i << " = " << val[i] << endl;
  ostr << "sum = (val0)";
  for(int i=1; i<nvals; ++i) 
    ostr << " + (val" << i << ")";
  ostr << endl;
  for(int i=0; i<nvals; ++i)
    ostr << "normval" << i << " = val" << i << "/sum" << endl;

  ostr << "START = 0" << endl;
  ostr << "STOP  = 1" << endl;

  ostr << "start0 = START" << endl;
  for(int i=1; i<nvals; ++i)
    ostr << "start" << i << " = start" << i-1 << "+normval" << i-1 << "*(STOP-START)" << endl;
  for(int i=0; i<nvals; ++i)
    ostr << "stop" << i << " = start" << i << "+normval" << i << "*(STOP-START)" << endl;

  ostr << "position0 = start0 + .5*normval0*(STOP-START)" << endl;
  for(int i=1; i<nvals; ++i)
    ostr << "position" << i << " = start" << i << " + .5*normval" << i << "*(STOP-START)" << endl;

  for(int i=0; i<nvals; ++i)
    {
      double position = 0;
      for(int j=0; j<i; ++j) 
	position += normval[j];
      position += .5*normval[i];
      ostr << "set label " << i+1 << " \"" << name[i] << "\" at .5*cos(position" << i << "*2*pi), .5*sin(position" << i << "*2*pi) centre font \",16\" rotate by 360*position" << i << endl;
    }

  ostr << "R = 0.8" << endl;
  ostr << "set multiplot; set size 1,1" << endl;

  for(int i=0; i<nvals; ++i)
    ostr << "plot [t=-start" << i << "*2*pi+pi/2:-stop" << i << "*2*pi+pi/2] R*sin(t),R*cos(t) with filledcurve xy=0,0 lc " << i+1 << endl;
  for(int i=0; i<nvals; ++i)
    {
      ostr << "plot [t=0:1] t*R*sin(-start" << i << "*2*pi+pi/2),t*R*cos(-start" << i << "*2*pi+pi/2) lt -1 lw 2" << endl;
      ostr << "plot [t=0:1] t*R*sin(-stop" << i << "*2*pi+pi/2),t*R*cos(-stop" << i << "*2*pi+pi/2) lt -1 lw 2" << endl;
    }
  ostr << "plot [t=0:2*pi] R*sin(t),R*cos(t) lt -1 lw 2" << endl;

  ostr << "unset multiplot; set out" << endl;
  ostr << "!epstopdf comptimes_analysis/" << filename << ".eps" << endl;
  ostr << "reset" << endl;
  ostr << "!/bin/echo -en '[ok]\\n' 1>&2" << endl;
  ostr << "quit" << endl;

  return;
}


void MOSFETProblemCuda::save_histograms_2( const int *field, int nvals, const string filename, const int complem )
{
  ostringstream ofilename;
  ofilename << "scripts/histogram_" << filename << ".gp";

  ofstream ostr;
  ostr.open(ofilename.str().c_str(), ios_base::out);

  ostr << "!/bin/echo -en 'generating " << ofilename.str() << "...' 1>&2" << endl;
  ostr << "set term postscript eps enh color" << endl;
  ostr << "set out 'comptimes_analysis/histogram_" << filename << ".eps'" << endl;

  ostr << "set boxwidth 0.45 absolute" << endl;
  ostr << "set style fill solid 1.00 border -1" << endl;
  ostr << "set key box inside top right" << endl;
  ostr << "set style histogram clustered gap 2 title offset character 0, 0, 0" << endl;
  ostr << "set datafile missing '-'" << endl;
  ostr << "set style data histograms" << endl;
  ostr << "set xtics border in scale 1,0.5 nomirror rotate by -45  offset character 0, 0, 0" << endl;
  ostr << "set ytics nomirror" << endl;
  ostr << "set ylabel 'absolute weight (in seconds)'" << endl;
  ostr << "set noborder" << endl;
  // --------- LABELS WITH THE SPEEDUP -----------
  ostr << "row = 0" << endl;
  ostr << "col = " << complem+4 << endl;
  ostr << "stats 'times.DAT' every ::row::row using col nooutput" << endl;
  ostr << "refval = STATS_min" << endl;
  ostr << "stats 'times.DAT' using col nooutput" << endl;
  ostr << "nrows = STATS_records" << endl;
  ostr << "do for [i=2:nrows] {" << endl;
  ostr << "    linea = i-1; columna = " << complem+4 << "; stats 'times.DAT' every ::linea::linea using columna nooutput; valor = STATS_min" << endl;
  ostr << "    spdup = refval/valor" << endl;
  ostr << "    set label i sprintf(\"speedup\\n%.1f\", spdup) at graph 1.*i/(nrows+1), graph .5" << endl;
  ostr << "}" << endl;
  // ---------------------------------------------
  ostr << "plot 'times.DAT' using " << field[0]+4 << ":xticlabels(1) title '" << fieldname[field[0]] << "'";
  if(nvals > 1)
    for(int i=1; i<nvals; ++i)
      ostr << ", 'times.DAT' using ($" << field[i]+4 << "):xticlabels(1) title '" << fieldname[field[i]] << "'";
  ostr << endl;
  // --------- LABELS WITH THE SPEEDUP -----------
  ostr << "do for [i=2:nrows] {" << endl;
  ostr << "    unset label i" << endl;
  ostr << "}" << endl;
  // ---------------------------------------------

  ostr << "set output" << endl;
  ostr << "!epstopdf comptimes_analysis/histogram_" << filename << ".eps" << endl;

  ostr << "!/bin/echo -en '[ok]\\n' 1>&2" << endl;
  ostr << "quit" << endl;
  
  return;
}


void MOSFETProblemCuda::save_histograms_3( const int *field, int nvals, const string filename, const int complem )
{
  ostringstream ofilename;
  ofilename << "scripts/histogram_" << filename << ".gp";

  ofstream ostr;
  ostr.open(ofilename.str().c_str(), ios_base::out);

  ostr << "!/bin/echo -en 'generating " << ofilename.str() << "...' 1>&2" << endl;
  ostr << "set term postscript eps enh color" << endl;
  ostr << "set out 'comptimes_analysis/histogram_" << filename << ".eps'" << endl;
  ostr << "set size 1,2" << endl;
  ostr << "set multiplot" << endl;
  ostr << "set size 1,1" << endl;

  ostr << "set origin 0,1" << endl;
  ostr << "set boxwidth 0.45 absolute" << endl;
  ostr << "set style fill solid 1.00 border -1" << endl;
  ostr << "set key box inside top right" << endl;
  ostr << "set style histogram clustered gap 2 title offset character 0, 0, 0" << endl;
  ostr << "set datafile missing '-'" << endl;
  ostr << "set style data histograms" << endl;
  ostr << "set xtics border in scale 1,0.5 nomirror rotate by -45  offset character 0, 0, 0" << endl;
  ostr << "set ytics nomirror" << endl;
  ostr << "set ylabel 'absolute weight (in seconds)'" << endl;
  ostr << "set noborder" << endl;
  // --------- LABELS WITH THE SPEEDUP -----------
  ostr << "row = 0" << endl;
  ostr << "col = " << complem+4 << endl;
  ostr << "stats 'times.DAT' every ::row::row using col nooutput" << endl;
  ostr << "refval = STATS_min" << endl;
  ostr << "stats 'times.DAT' using col nooutput" << endl;
  ostr << "nrows = STATS_records" << endl;
  ostr << "do for [i=2:nrows] {" << endl;
  ostr << "    linea = i-1; columna = " << complem+4 << "; stats 'times.DAT' every ::linea::linea using columna nooutput; valor = STATS_min" << endl;
  ostr << "    spdup = refval/valor" << endl;
  ostr << "    set label i sprintf(\"speedup\\n%.1f\", spdup) at graph 1.*i/(nrows+1), graph .5" << endl;
  ostr << "}" << endl;
  // ---------------------------------------------
  ostr << "plot 'times.DAT' using " << field[0]+4 << ":xticlabels(1) title '" << fieldname[field[0]] << "'";
  if(nvals > 1)
    for(int i=1; i<nvals; ++i)
      ostr << ", 'times.DAT' using ($" << field[i]+4 << "):xticlabels(1) title '" << fieldname[field[i]] << "'";
  // ostr << ", 'times.DAT' using ($" << complem+4;
  // for(int i=0; i<nvals; ++i)
  //   ostr << "-$" << field[i]+4;
  // ostr << "):xticlabels(1) title 'other'";
  ostr << endl;
  // --------- LABELS WITH THE SPEEDUP -----------
  ostr << "do for [i=2:nrows] {" << endl;
  ostr << "    unset label i" << endl;
  ostr << "}" << endl;
  // ---------------------------------------------

  ostr << "set origin 0,0" << endl;
  ostr << "set boxwidth 0.45 absolute" << endl;
  ostr << "set style fill solid 1.00 border -1" << endl;
  ostr << "set key box inside top right" << endl;
  ostr << "set style histogram clustered gap 2 title offset character 0, 0, 0" << endl;
  ostr << "set datafile missing '-'" << endl;
  ostr << "set style data histograms" << endl;
  ostr << "set xtics border in scale 1,0.5 nomirror rotate by -45  offset character 0, 0, 0" << endl;
  ostr << "set ytics nomirror" << endl;
  ostr << "set ylabel 'relative weight over one step (percentage)'" << endl;
  ostr << "set noborder" << endl;
  ostr << "plot 'times.DAT' using (($" << field[0]+4 << ")/$" << complem+4 << "*100):xticlabels(1) title '" << fieldname[field[0]] << "'";
  if(nvals > 1)
    for(int i=1; i<nvals; ++i)
      ostr << ", 'times.DAT' using (($" << field[i]+4 << ")/$" << complem+4 << "*100):xticlabels(1) title '" << fieldname[field[i]] << "'";
  // ostr << ", 'times.DAT' using (($" << complem+4;
  // for(int i=0; i<nvals; ++i)
  //   ostr << "-$" << field[i]+4;
  // ostr << ")/$" << complem+4 << "*100):xticlabels(1) title 'other'";
  ostr << endl;

  ostr << "unset multiplot" << endl;
  ostr << "set output" << endl;
  ostr << "!epstopdf comptimes_analysis/histogram_" << filename << ".eps" << endl;

  ostr << "!/bin/echo -en '[ok]\\n' 1>&2" << endl;
  ostr << "quit" << endl;
  
  return;
}


void MOSFETProblemCuda::save_speedups_2( const string filename, const int complem )
{
  ostringstream ofilename;
  ofilename << "scripts/speedups_" << filename << ".gp";

  ofstream ostr;
  ostr.open(ofilename.str().c_str(), ios_base::out);

  ostr << "!/bin/echo -en 'generating " << ofilename.str() << "...' 1>&2" << endl;
  ostr << "set term postscript eps enh color" << endl;
  ostr << "set out 'comptimes_analysis/speedups_" << filename << ".eps'" << endl;
  ostr << "set key box inside top right" << endl;
  ostr << "set ytics nomirror" << endl;
  ostr << "set ylabel 'speedup'" << endl;
  ostr << "unset key" << endl;
  ostr << "unset xtics" << endl;
  ostr << "unset border" << endl;

  ostr << "row = 0" << endl;
  ostr << "col = " << complem+4 << endl;
  ostr << "stats 'times.DAT' every ::row::row using col nooutput" << endl;
  ostr << "refval = STATS_min" << endl;
  ostr << "stats 'times.DAT' using col nooutput" << endl;
  ostr << "nrows = STATS_records" << endl;
  ostr << "plot [-1:nrows] 'times.DAT' using ($0):(refval/$" << complem+4 << ") with lines lt 0, 'times.DAT' using ($0):(refval/$" << complem+4 << "):1 with labels" << endl;
  ostr << "set output" << endl;
  ostr << "!epstopdf comptimes_analysis/speedups_" << filename << ".eps" << endl;

  ostr << "!/bin/echo -en '[ok]\\n' 1>&2" << endl;
  ostr << "quit" << endl;
  
  return;
}


void MOSFETProblemCuda::save_speedups_3()
{
  ostringstream ofilename;
  ofilename << "scripts/speedups.gp";

  ofstream ostr;
  ostr.open(ofilename.str().c_str(), ios_base::out);

  ostr << "!/bin/echo -en 'generating " << ofilename.str() << "...' 1>&2" << endl;
  ostr << "set term postscript eps enh color" << endl;
  ostr << "set out 'comptimes_analysis/speedups.eps'" << endl;
  ostr << "set key box inside top right" << endl;
  ostr << "set ytics nomirror" << endl;
  ostr << "set ylabel 'speedup'" << endl;
  ostr << "set key inside top left box" << endl;
  ostr << "unset xtics" << endl;
  ostr << "unset border" << endl;
  ostr << "set ytics 0, 10" << endl;

  ostr << "row = 0" << endl;
  ostr << "stats 'times.DAT' using " << _PHASE_ITER_EIGENSTATES + 4 << " nooutput" << endl;
  ostr << "nrows = STATS_records" << endl;

  // eigenstates
  ostr << "col = " << _PHASE_ITER_EIGENSTATES + 4 << endl;
  ostr << "stats 'times.DAT' every ::row::row using col nooutput" << endl;
  ostr << "refval1 = STATS_min" << endl;

  // Frechet
  ostr << "col = " << _PHASE_ITER_SPSTEP_FRECHET + 4 << endl;
  ostr << "stats 'times.DAT' every ::row::row using col nooutput" << endl;
  ostr << "refval2 = STATS_min" << endl;

  // constrlinsys
  ostr << "col = " << _PHASE_ITER_SPSTEP_CONSTRLINSYS + 4 << endl;
  ostr << "stats 'times.DAT' every ::row::row using col nooutput" << endl;
  ostr << "refval3 = STATS_min" << endl;
  
  // solvelinsys
  ostr << "col = " << _PHASE_ITER_SPSTEP_SOLVELINSYS + 4 << endl;
  ostr << "stats 'times.DAT' every ::row::row using col nooutput" << endl;
  ostr << "refval4 = STATS_min" << endl;

  // plot
  ostr << "plot [-1:nrows] 'times.DAT' using ($0):(refval1/$ " << _PHASE_ITER_EIGENSTATES + 4         << ") title 'eigenstates' w lp lt 1 lc 1 lw 4, 'times.DAT' using ($0):(-1):1 with labels notitle\\" << endl;
  ostr << "               ,'times.DAT' using ($0):(refval2/$ " << _PHASE_ITER_SPSTEP_FRECHET + 4      << ") title 'Frechet' w lp lt 2 lc 2 lw 4 \\"        << endl;
  ostr << "               ,'times.DAT' using ($0):(refval3/$ " << _PHASE_ITER_SPSTEP_CONSTRLINSYS + 4 << ") title 'constr. linsys' w lp lt 3 lc 3 lw 4 \\" << endl;
  ostr << "               ,'times.DAT' using ($0):(refval4/$ " << _PHASE_ITER_SPSTEP_SOLVELINSYS + 4  << ") title 'solve linsys' w lp lt 4 lc 4 lw 4 "     << endl;
  // ostr << "               ,'times.DAT' using ($0):(refval2/$ title 'Frechet' "        << _PHASE_ITER_SPSTEP_FRECHET + 4      << ") w lp lt 1 lc 1, 'times.DAT' using ($0):(refval2/$" << _PHASE_ITER_SPSTEP_FRECHET + 4      << "):1 with labels \\" << endl;
  // ostr << "               ,'times.DAT' using ($0):(refval3/$ title 'constr. linsys' " << _PHASE_ITER_SPSTEP_CONSTRLINSYS + 4 << ") w lp lt 2 lc 3, 'times.DAT' using ($0):(refval3/$" << _PHASE_ITER_SPSTEP_CONSTRLINSYS + 4 << "):1 with labels \\" << endl;
  // ostr << "               ,'times.DAT' using ($0):(refval4/$ title 'solve linsys' "   << _PHASE_ITER_SPSTEP_SOLVELINSYS + 4  << ") w lp lt 3 lc 4, 'times.DAT' using ($0):(refval4/$" << _PHASE_ITER_SPSTEP_SOLVELINSYS + 4  << "):1 with labels   " << endl;

  ostr << "set output" << endl;
  ostr << "!epstopdf comptimes_analysis/speedups.eps" << endl;
  ostr << "!/bin/echo -en '[ok]\\n' 1>&2" << endl;
  ostr << "quit" << endl;
  
  return;
}
