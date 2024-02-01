void CPU_compute_frechet();
void GPU_compute_frechet();
void compute_frechet();

/*
  name          : 'check_compute_frechet_input', 'check_compute_frechet_output'
  last modified : 2023/06/06
  author        : Francesco VECIL

  description   : This functions check the input and the output of the 'compute_frechet' method.
                  As input  : (i) eps, (ii) chi
		  As output : (iii) frechet
 */
#define  _ERR_CHECK_COMPUTE_FRECHET_INPUT_OUTPUT   161
void check_compute_frechet_input();
void check_compute_frechet_output();
