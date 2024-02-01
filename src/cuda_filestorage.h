/***********************************
 *           FILE STORAGE          *
 ***********************************/
int generate_films_and_graphs();

// DATA STORAGE
static const int DIGITS = 8;
void store_data( const bool=false );
void update_min_max(const int);
void store_totvoldens_dat();
void store_chi_dat();
void store_surfdens_dat();
void store_eps_dat();
void store_potential_dat();
void store_currdens_dat();
void store_driftvel_dat();
void store_test_dat();
/* void store_scalars(); */

// GNUPLOT SCRIPTS: FILMS
void write_gnuplotscripts_films();
void write_gnuplotscript_totvoldens();
void write_gnuplotscript_surfdens();
void write_gnuplotscript_eps();
void write_gnuplotscript_potential();
void write_gnuplotscript_currdens();
void write_gnuplotscript_driftvel();
void write_gnuplotscript_test();

// GNUPLOT SCRIPTS: SCALARS
void write_gnuplotscripts_scalars();
