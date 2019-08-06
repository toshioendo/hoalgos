// hierarchy oblivious matrix multiply
//   single address space version
// main function
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/time.h>
#include <assert.h>

#include <homm.h>
#include "matmul.h"

extern double logtime;
extern double basetime;

int init_mats(long m, long n, long k, REAL **Ap, REAL **Bp, REAL **Cp)
{
  // A: M*K matrix
  size_t Aelems = m*k;
  REAL *A = (REAL *)homm_galloc(sizeof(REAL)*Aelems);
  // B: K*N matrix
  size_t Belems = k*n;
  REAL *B = (REAL *)homm_galloc(sizeof(REAL)*Belems);
  // C: M*N matrix
  size_t Celems = m*n;
  REAL *C = (REAL *)homm_galloc(sizeof(REAL)*Celems);

#if VERBOSE >= 10
  printf("[init] allocated %ldMiB\n",
	 (Aelems+Belems+Celems)*sizeof(REAL) >> 20L);
#endif

  /* initial values */
  double st = Wtime(), et;

  long i;
#pragma omp parallel for
  for (i = 0; i < Aelems; i++) {
    A[i] = 1.0;
  }

#pragma omp parallel for
  for (i = 0; i < Belems; i++) {
    B[i] = 2.0;
  }

#pragma omp parallel for
  for (i = 0; i < Celems; i++) {
    C[i] = 0.0;
  }

  et = Wtime();
#if VERBOSE >= 10
  printf("[init_mats] array initialization took %.3lfsec\n", et-st);
#endif

  *Ap = A;
  *Bp = B;
  *Cp = C;

  return 0;
}


int main(int argc, char *argv[])
{
  long m, n, k;
  double *A, *B, *C;
  /* default setting option */
  m = 1024;
  n = 1024; 
  k = 1024;
  
  if (argc >= 4) {
    m = atol(argv[1]);
    n = atol(argv[2]);
    k = atol(argv[3]);
  }
  else if (argc >= 2) {
    m = atol(argv[1]);
    n = m;
    k = m;
  }



  printf("type=[%s], size=(%ld, %ld, %ld)\n",
	 TYPENAME, m, n, k);

  homm_init();
  init_mats(m, n, k, &A, &B, &C);
  init_algo();

  // main computation
  int i;
  for (i = 0; i < 5; i++) {
    double st, et;
    double nops = (double)m*n*k*2.0;
    st = Wtime();

    algo(m, n, k, A, m, B, k, C, m);

    et = Wtime();

    printf("%.3lf sec -> %lf MFlops\n",
	   (et-st), nops/(et-st)/1000000.0);

#if 1
    printf("C[0] = %lf, C[%ld] = %lf\n", 
	   C[0], m*n-1, C[m*n-1]);
#endif

  }

  homm_finish();
}
