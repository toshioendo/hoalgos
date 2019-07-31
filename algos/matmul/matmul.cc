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

// global variables
struct global g;

double logtime = 0.0;

int init_mats()
{
  // A: M*K matrix
  size_t Aelems = g.size.x*g.size.z;
  g.A = (REAL *)homm_galloc(sizeof(REAL)*Aelems);
  // B: K*N matrix
  size_t Belems = g.size.z*g.size.y;
  g.B = (REAL *)homm_galloc(sizeof(REAL)*Belems);
  // C: M*N matrix
  size_t Celems = g.size.x*g.size.y;
  g.C = (REAL *)homm_galloc(sizeof(REAL)*Celems);

#if VERBOSE >= 10
  printf("[init] allocated %ldMiB\n",
	 (Aelems+Belems+Celems)*sizeof(REAL) >> 20L);
#endif

  /* initial values */
  double st = Wtime(), et;

  long i;
#pragma omp parallel for
  for (i = 0; i < Aelems; i++) {
    g.A[i] = 1.0;
  }

#pragma omp parallel for
  for (i = 0; i < Belems; i++) {
    g.B[i] = 2.0;
  }

#pragma omp parallel for
  for (i = 0; i < Celems; i++) {
    g.C[i] = 0.0;
  }

  et = Wtime();
#if VERBOSE >= 10
  printf("[init_mats] array initialization took %.3lfsec\n", et-st);
#endif

  return 0;
}


int base(vec3 v0, vec3 v1)
{

  double st, et;
  // base case
#if VERBOSE >= 20
  printf("[base] START [(%ld,%ld,%ld), (%ld,%ld,%ld))\n", 
	 v0.x, v0.y, v0.z, v1.x, v1.y, v1.z);
#endif

  st = Wtime();

#ifdef USE_CPUBLAS
  // computation
  char TA = 'N';
  char TB = 'N';
  REAL alpha = 1.0;
  REAL beta = 1.0;
  BLASINT lda = (BLASINT)g.size.x; // m
  BLASINT ldb = (BLASINT)g.size.z; // k
  BLASINT ldc = (BLASINT)g.size.x; // m

  BLASINT m = (BLASINT)(v1.x-v0.x);
  BLASINT n = (BLASINT)(v1.y-v0.y);
  BLASINT k = (BLASINT)(v1.z-v0.z);

  REAL *A = &g.A[v0.x + v0.z * lda];
  REAL *B = &g.B[v0.z + v0.y * ldb];
  REAL *C = &g.A[v0.x + v0.y * lda];

  GEMM(&TA, &TB, &m, &n, &k,
       &alpha, A, &lda, B, &ldb,
       &beta, C, &ldc);
#elif defined USE_CPULOOP
  long lda = g.size.x; // m
  long ldb = g.size.z; // k
  long ldc = g.size.x; // m

  long m = (long)(v1.x-v0.x);
  long n = (long)(v1.y-v0.y);
  long k = (long)(v1.z-v0.z);

  REAL *A = &g.A[v0.x + v0.z * lda];
  REAL *B = &g.B[v0.z + v0.y * ldb];
  REAL *C = &g.C[v0.x + v0.y * lda];

#ifdef USE_OMP
#pragma omp parallel for
#endif
  for (long j = 0; j < n; j++) {
    for (long l = 0; l < k; l++) {
      REAL blj = B[l+j*ldb];
      REAL *Ap = &A[0+l*lda];
      REAL *Cp = &C[0+j*ldc];
#pragma unroll
      for (long i = 0; i < m; i++) {
	REAL ail = *Ap;
	*Cp += ail*blj;
	Ap++;
	Cp++;
      }
    }
  }
#else
#error
#endif

  // print periodically
  et = Wtime();
#if VERBOSE >= 10
#if VERBOSE >= 30
  if (1)
#else
  if (et > logtime+1.0)
#endif
    {
      printf("[base] END [(%ld,%ld,%ld), (%ld,%ld,%ld)) -> %.3lfsec\n", 
	     v0.x, v0.y, v0.z, v1.x, v1.y, v1.z, et-st);
      logtime = et;
    }
#endif

  return 0;
}


int recalgo(vec3 v0, vec3 v1)
{
#if VERBOSE >= 30
  printf("[recalgo] [(%d,%d,%d), (%d,%d,%d))\n",
	 v0.x, v0.y, v0.z, v1.x, v1.y, v1.z);
#endif

  if (v0.x >= v1.x || v0.y >= v1.y || v0.z >= v1.z) {
    printf("[recalgo] (do nothing)\n");
    return 0;
  }

  long dx = v1.x-v0.x;
  long dy = v1.y-v0.y;
  long dz = v1.z-v0.z;

  size_t stopsize = g.stopsizemb * 1024L*1024L;
  size_t accsize = (dx*dz+dz*dy+dx*dy) * sizeof(REAL);

  if (accsize <= stopsize) {
    // base case
    base(v0, v1);
  }
  else {
    // divide long dimension
    char dim;
    if (dx >= dy*16 && dx >= dz*16) dim = 'X';
    else if (dy >= dz) dim = 'Y';
    else dim = 'Z';

    long mid = (v0.get(dim) + v1.get(dim))/2;

    // first task
    recalgo(v0, vec3mod(v1, dim, mid));
    // second task
    recalgo(vec3mod(v0, dim, mid), v1);
  }

  return 0;
}

int algo(vec3 v0, vec3 v1)
{
#ifdef USE_OMPTASK
#pragma omp parallel
#pragma omp single
#endif // USE_OMPTASK
  recalgo(v0, v1);

  return 0;
}

int main(int argc, char *argv[])
{

  /* default setting option */
  g.size = vec3(1024, 1024, 1024);
  g.stopsizemb = 16; //128;
  
  while (argc >= 2 && argv[1][0] == '-') {
    if (strcmp(argv[1], "-ss") == 0) {
      g.stopsizemb = atol(argv[2]);
      argc -= 2;
      argv += 2;
    }
    else break;
  }
  
  if (argc >= 4) {
    // x=M, y=N, z=K
    g.size = vec3(atol(argv[1]), atol(argv[2]), atol(argv[3]));
  }

  printf("Hierarchy Oblivious Matrix Multiply sample (Single Address Space version)\n");
  char use_cuda = 'N';
#ifdef USE_CUDA
  use_cuda = 'Y';
#endif
  char use_omp = 'N';
#ifdef USE_OMP
  use_omp = 'Y';
#endif
  char use_omptask = 'N';
#ifdef USE_OMPTASK
  use_omptask = 'Y';
#endif

  printf("  Compile time options: USE_CUDA %c, USE_OMP %c, USE_OMPTASK %c\n",
	 use_cuda, use_omp, use_omptask);

  printf("type=[%s], m=%ld, n=%ld, k=%ld, stopsize=%ldMiB\n",
	 TYPENAME, g.size.x, g.size.y, g.size.z, g.stopsizemb);

  homm_init();

  init_mats();

  // main computation
  int i;
  for (i = 0; i < 3; i++) {
    double st, et;
    double nops = (double)g.size.vol()*2.0;
    st = Wtime();

    algo(vec3(0,0,0), g.size);

    et = Wtime();

    printf("%.3lf sec -> %lf MFlops\n",
	   (et-st), nops/(et-st)/1000000.0);
  }

  homm_finish();
}
