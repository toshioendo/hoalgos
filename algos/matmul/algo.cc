// hierarchy oblivious matrix multiply
// recusvie algorithm

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/time.h>
#include <assert.h>

#include <homm.h>
#include "matmul.h"

int base(vec3 v0, vec3 v1)
{
  char TA = 'N';
  char TB = 'N';
  REAL alpha = -1.0;
  REAL beta = 1.0;
  BLASINT lda = (BLASINT)g.size.x; // m
  BLASINT ldb = (BLASINT)g.size.z; // k
  BLASINT ldc = (BLASINT)g.size.x; // m

  double st, et;
  // base case
#if VERBOSE >= 20
  printf("[base] START [(%ld,%ld,%ld), (%ld,%ld,%ld))\n", 
	 v0.x, v0.y, v0.z, v1.x, v1.y, v1.z);
#endif

  st = Wtime();

  // computation
  BLASINT m = (BLASINT)(v1.x-v0.x);
  BLASINT n = (BLASINT)(v1.y-v0.y);
  BLASINT k = (BLASINT)(v1.z-v0.z);

  REAL *A = &g.A[v0.x + v0.z * lda];
  REAL *B = &g.B[v0.z + v0.y * ldb];
  REAL *C = &g.A[v0.x + v0.y * lda];

  GEMM(&TA, &TB, &m, &n, &k,
       &alpha, A, &lda, B, &ldb,
       &beta, C, &ldc);

  // print periodically
  et = Wtime();
  double wt = et;
#if VERBOSE >= 10
#if VERBOSE >= 30
  if (1)
#else
  if (wt > logtime+1.0)
#endif
    {
      printf("[base] END [(%ld,%ld,%ld), (%ld,%ld,%ld)) -> %.3lfsec\n", 
	     v0.x, v0.y, v0.z, v1.x, v1.y, v1.z, et-st);
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
    // divide longest dimension
    char dim;
    if (dx >= dy && dx >= dz) dim = 'X';
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
