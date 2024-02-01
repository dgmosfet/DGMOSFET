void CPU_WENO_X( const int STAGE, double *test_CPU=NULL );
void CPU_WENO_W( const int STAGE, double *test_CPU=NULL );
void CPU_WENO_PHI( const int STAGE, double *test_CPU=NULL );
void CPU_WENO_PHI_2( const int STAGE, double *test_CPU=NULL );
void WENO_X( const int STAGE );
void WENO_W( const int STAGE );
void WENO_PHI( const int STAGE );

// FOR TESTING PURPOSE
void fdweno5_FROMLEFT(const double *_g, const int NS, double *_fh);
void fdweno5_FROMRIGHT(const double *_g, const int NS, double *_fh);
