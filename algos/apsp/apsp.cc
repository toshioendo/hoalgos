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
  const int percent = 2;
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
      if (i == j) {
	v = 0.0;
      }
      else if (rand_r(&seed) % 100 < percent) {
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

int summary_mat(long n, REAL *A)
{
  long freq[12]; // 0...9 and others and infinity
  long i;
  for (i = 0; i < sizeof(freq)/sizeof(long); i++) {
    freq[i] = 0;
  }
  for (i = 0; i < n*n; i++) {
    long val = (long)(A[i]+0.5); // to long
    if (val >= 0 && val < 10) {
      freq[val]++;
    }
    else if (val > 1.0e+7) {
      freq[11]++;
    }
    else {
      freq[10]++;
    }
  }

  printf("Summarize A of size %ld\n", n);
  for (i = 0; i < 10; i++) {
    if (freq[i] > 0) {
      printf("dist %ld: %ld elements\n", i, freq[i]);
    }
  }
  printf("dist others: %ld elements\n", freq[10]);
  printf("dist infinity: %ld elements\n", freq[11]);
  printf("\n");
  return 0;
}

int main(int argc, char *argv[])
{
  long n;
  REAL *A;
  /* default setting option */
  n = 1024; 

  homm_init();
  init_algo(&argc, &argv);
  
  if (argc >= 2) {
    n = atol(argv[1]);
  }

  printf("type=[%s], size=(%ld)\n",
	 TYPENAME, n);

  init_mat(n, &A);

  // main computation
  int i;
  int niter = 5;
  for (i = 0; i < niter; i++) {
    double st, et;
    double nops = (double)n*n*n*2.0;
    rand_mat(n, A);

    st = Wtime();

    algo(n, A, n);

    et = Wtime();

    printf("\nsize=%ld (%d/%d) %.3lf sec -> %lf MFlops\n\n",
	   n, i, niter, (et-st), nops/(et-st)/1000000.0);

    summary_mat(n, A);
  }

  homm_finish();
}
