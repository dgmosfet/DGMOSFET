void update_cfl();

/*
  begin section : timint_01
  author        : Francesco VECIL
  date          : 2023/06/12

  description   : The time integrator tries using Newton-Raphson for the computation of the eigenstates.
                  If Newton-Raphson fails, it switches to Gummel. If Gummel fails too, it exits the execution 
		  with an error message.
 */
void perform_step_3();
/*
  end section : timint_01
 */



/*
  begin section : timint_02
  author        : Francesco VECIL
  date          : 2023/06/12

  description   : This time integrator version is used only for debugging purposes. It is meant to validate
                  the results given by Gummel for the eigenstates. Therefore, it executes Newton-Raphson,
		  stores the results, restores the initial conditions, then executes Gummel.
		  Finally it compares the results given by the two methods and warns about discrepancies.
 */
void perform_step_4();
/*
  end section : timint_02
 */
