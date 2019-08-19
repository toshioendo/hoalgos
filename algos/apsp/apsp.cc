// hierarchy oblivious all pairs shortest algorithm
// main function
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/time.h>
#include <assert.h>

#include <homm.h>
#include "apsp.h"

extern double logtime;
extern double basetime;

int init_mat(long n, REAL **Ap)
{
  // A: N*N matrix
  size_t Aelems = n*n;
  REAL *A = (REAL *)homm_galloc(sizeof(REAL)*Aelems);

#if VERBOSE >= 10
  printf("[init] allocated %ldMiB\n",
	 (Aelems)*sizeof(REAL) >> 20L);
#endif

  *Ap = A;

  return 0;
}

int rand_mat(long n, REAL *A)
{
  /* initial values */
  double st = Wtime(), et;

  long j;
  const REAL infval = 1.0e+8;
  const int percent = 10;
  long concount = 0;
#ifdef USE_OMP
#pragma omp parallel for
#endif
  for (j = 0; j < n; j++) {
    long i;
    unsigned int seed = j;
    srand(j);
    for (i = 0; i < n; i++) {
      REAL v;
      if (rand_r(&seed) % 100 < percent) {
	v = 1.0;
	concount++;
      }
      else v = infval;
      A[i+j*n] = v;
    }
  }

  et = Wtime();
#if VERBOSE >= 10
  printf("[rand_mat] matrix of size %ld, address [%p,%p) initialized\n",
	 n, A, A+n*n);
  printf("[rand_mat] making a random mat took %.3lfsec\n", et-st);
  printf("[rand_mat] %ld connections out of %ld made\n", concount, n*n);
#endif
  return 0;
}


int main(int argc, char *argv[])
{
  long n;
  REAL *A;
  /* default setting option */
  n = 1024; 
  
  if (argc >= 2) {
    n = atol(argv[1]);
  }

  printf("type=[%s], size=(%ld)\n",
	 TYPENAME, n);

  homm_init();
  init_mat(n, &A);
  init_algo();

  // main computation
  int i;
  for (i = 0; i < 5; i++) {
    double st, et;
    double nops = (double)n*n*n*2.0;
    rand_mat(n, A);

    st = Wtime();

    algo(n, A, n);

    et = Wtime();

#if 1
    printf("A[0] = %lf, A[%ld] = %lf\n", 
	   A[0], n*n-1, A[n*n-1]);
#endif

    printf("%.3lf sec -> %lf MFlops\n\n",
	   (et-st), nops/(et-st)/1000000.0);

  }

  homm_finish();
}
