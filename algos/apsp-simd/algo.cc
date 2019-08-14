// hierarchy oblivious all pairs shortest path algorithm
// algorithm
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/time.h>
//#include <immintrin.h>
#include <assert.h>

#include <homm.h>
#include "apsp.h"

// global variables
struct global g;


double logtime = 0.0;
double kernel1time = 0.0;
double kernel2time = 0.0;
long kernel1count = 0;
long kernel2count = 0;
long ncopy = 0;
long copysize = 0;
double copytime = 0.0;

int init_algo()
{
  printf("Hierarchy Oblivious All Pairs Shortest Path sample\n");
  char use_avx2 = 'N';
#ifdef USE_AVX2
  use_avx2 = 'Y';
#endif
  char use_avx512 = 'N';
#ifdef USE_AVX512
  use_avx512 = 'Y';
#endif
  char use_omp = 'N';
#ifdef USE_OMP
  use_omp = 'Y';
#endif
  char use_omptask = 'N';
#ifdef USE_OMPTASK
  use_omptask = 'Y';
#endif

  g.ndiv = 8;

  //g.basesize = basesize_float_simd();
  g.basesize = vec3(16, 16, 16);
  printf("[matmul:init_algo]  Compile time options: USE_AVX2 %c, USE_AVX512 %c, USE_OMP %c, USE_OMPTASK %c\n",
	 use_avx2, use_avx512, use_omp, use_omptask);
  printf("[matmul:init_algo] type=[%s] basesize=(%ld,%ld,%ld)\n",
	 TYPENAME, g.basesize.x, g.basesize.y, g.basesize.z);

  // for internal copy buffer
  g.bufsize = 128*1024*1024;
  g.buf = (REAL*)homm_galloc(sizeof(REAL)*g.bufsize);

  printf("[matmul] bufsize=%ld*%ld=%ld Bytes\n", g.bufsize, sizeof(REAL), sizeof(REAL)*g.bufsize);

  return 0;
}

int base_pivot_cpuloop(vec3 v0, vec3 v1, REAL *Am, long lda)
{
#if VERBOSE >= 20
  printf("[base_pivot]\n");
#endif
  long m = (long)(v1.x-v0.x);
  long n = (long)(v1.y-v0.y);
  long k = (long)(v1.z-v0.z);

  REAL *A = &Am[v0.x + v0.z * lda];
  REAL *B = &Am[v0.z + v0.y * lda];
  REAL *C = &Am[v0.x + v0.y * lda];

  // in pivot cases, two or all of A, B, C are overlapped.
  // L loop must be outermost
  for (long l = 0; l < k; l++) {
#ifdef USE_OMP
#pragma omp parallel for
#endif
    for (long j = 0; j < n; j++) {
      REAL Blj = B[l+j*lda];
#pragma unroll
      for (long i = 0; i < m; i++) {
	REAL Ail = A[i+l*lda];
	REAL Cij = C[i+j*lda];
	if (Ail + Blj < Cij) {
	  Cij = Ail + Blj;
	}
	C[i+j*lda] = Cij;
      }
    }
  }

  return 0;
}

int base_nonpivot_cpuloop(vec3 v0, vec3 v1, REAL *Am, long lda)
{
#if VERBOSE >= 20
  printf("[base_nonpivot]\n");
#endif
  long m = (long)(v1.x-v0.x);
  long n = (long)(v1.y-v0.y);
  long k = (long)(v1.z-v0.z);

  REAL *A = &Am[v0.x + v0.z * lda];
  REAL *B = &Am[v0.z + v0.y * lda];
  REAL *C = &Am[v0.x + v0.y * lda];

#ifdef USE_OMP
#pragma omp parallel for
#endif
  for (long j = 0; j < n; j++) {
    for (long l = 0; l < k; l++) {
      REAL Blj = B[l+j*lda];
#pragma unroll
      for (long i = 0; i < m; i++) {
	REAL Ail = A[i+l*lda];
	REAL Cij = C[i+j*lda];
	if (Ail + Blj < Cij) {
	  Cij = Ail + Blj;
	}
	C[i+j*lda] = Cij;
      }
    }
  }

  return 0;
}

int base(vec3 v0, vec3 v1, REAL *Am, long lda)
{

  double st, et;
  // base case
#if VERBOSE >= 20
  printf("[base] START [(%ld,%ld,%ld), (%ld,%ld,%ld))\n", 
	 v0.x, v0.y, v0.z, v1.x, v1.y, v1.z);
#endif

  st = Wtime();

  if (v0.x == v0.z || v0.y == v0.z) {
    base_pivot_cpuloop(v0, v1, Am, lda);
    et = Wtime();
    kernel1time += (et-st);
    kernel1count++;
  }
  else {
    base_nonpivot_cpuloop(v0, v1, Am, lda);
    et = Wtime();
    kernel2time += (et-st);
    kernel2count++;
  }


  // print periodically
#if VERBOSE >= 10
#if VERBOSE >= 30
  if (1)
#else
  if (et > logtime+1.0)
#endif
    {
      printf("[base] END [(%ld,%ld,%ld), (%ld,%ld,%ld)) -> %.6lfsec\n", 
	     v0.x, v0.y, v0.z, v1.x, v1.y, v1.z, et-st);
      logtime = et;
    }
#endif

  return 0;
}

int recMM(bool inbuf, vec3 v0, vec3 v1, REAL *Am, long lda);

int recAPSP(bool inbuf, vec3 v0, vec3 v1, REAL *Am, long lda)
{
#if VERBOSE >= 30
  printf("[recAPSP] [(%d,%d,%d), (%d,%d,%d))\n",
	 v0.x, v0.y, v0.z, v1.x, v1.y, v1.z);
#endif

  if (v0.x >= v1.x || v0.y >= v1.y || v0.z >= v1.z) {
    printf("[recAPSP] (do nothing)\n");
    printf("[recAPSP] [(%d,%d,%d), (%d,%d,%d))\n",
	   v0.x, v0.y, v0.z, v1.x, v1.y, v1.z);
    return 0;
  }

  assert(v0.x == v0.y && v0.x == v0.z);
  assert(v1.x == v1.y && v1.x == v1.z);

  long cdim = v1.x-v0.x;
  if (cdim <= g.basesize.x) {
    // base case
    base(v0, v1, Am, lda);
  }
  else {
    // divide the task into 8
    long n0 = v0.x;
    long n1 = v1.x;
    long nm = (n0+n1)/2;
    long align = g.basesize.x; //
    nm = ((nm+align-1)/align)*align;

    // 1
    recAPSP(inbuf, vec3(n0, n0, n0), vec3(nm, nm, nm),
	    Am, lda);
    // 2
    recMM(inbuf, vec3(nm, n0, n0), vec3(n1, nm, nm),
	  Am, lda);
    // 3
    recMM(inbuf, vec3(n0, nm, n0), vec3(nm, n1, nm),
	  Am, lda);
    // 4
    recMM(inbuf, vec3(nm, nm, n0), vec3(n1, n1, nm),
	  Am, lda);
    // 5
    recAPSP(inbuf, vec3(nm, nm, nm), vec3(n1, n1, n1),
	    Am, lda);
    // 6
    recMM(inbuf, vec3(n0, nm, nm), vec3(nm, n1, n1),
	  Am, lda);
    // 7
    recMM(inbuf, vec3(nm, n0, nm), vec3(n1, nm, n1),
	  Am, lda);
    // 8
    recMM(inbuf, vec3(n0, n0, nm), vec3(nm, nm, n1),
	  Am, lda);

  }

  return 0;
}

int recMM(bool inbuf, vec3 v0, vec3 v1, REAL *Am, long lda)
{
#if VERBOSE >= 30
  printf("[recMM] [(%d,%d,%d), (%d,%d,%d))\n",
	 v0.x, v0.y, v0.z, v1.x, v1.y, v1.z);
#endif

  if (v0.x >= v1.x || v0.y >= v1.y || v0.z >= v1.z) {
    printf("[recMM] (do nothing)\n");
    printf("[recMM] [(%d,%d,%d), (%d,%d,%d))\n",
	   v0.x, v0.y, v0.z, v1.x, v1.y, v1.z);
    return 0;
  }

  vec3 csize = vec3sub(v1, v0);
  long cx = csize.x;
  long cy = csize.y;
  long cz = csize.z;
  if (cx <= g.basesize.x && cy <= g.basesize.y && cz <= g.basesize.z) {
    // base case
    base(v0, v1, Am, lda);
  }
  else {
    // divide long dimension
    char dim;
    if (cx >= cy*4 && cx >= cz*4 && csize.x > g.basesize.x) dim = 'X';
    else if (cy >= cz && csize.y > g.basesize.y) dim = 'Y';
    else if (csize.z > g.basesize.z) dim = 'Z';
    else if (csize.y > g.basesize.y) dim = 'Y';
    else dim = 'X';

    long mid = (v0.get(dim) + v1.get(dim))/2;
    long align = g.basesize.get(dim); //
    mid = ((mid+align-1)/align)*align;

    // first task
    recMM(inbuf, v0, vec3mod(v1, dim, mid),
	    Am, lda);
    // second task
    recMM(inbuf, vec3mod(v0, dim, mid), v1,
	    Am, lda);
  }

  return 0;
}


int algo(long n, REAL *Am, long lda)
{
  printf("[APSP:algo] type=[%s] size=%ld\n",
	 TYPENAME, n);

  if (n % g.basesize.x != 0) {
    printf("[APSP:algo] ERROR: currently, size (%ld) must be a multiple of %ld\n",
	   n, g.basesize.x);
    exit(1);
  }

  // statistics
  ncopy = 0;
  copysize = 0;
  copytime = 0.0;
  kernel1time = 0.0;
  kernel2time = 0.0;
  kernel1count = 0;
  kernel2count = 0;

#if 0 /////////////
  // Recursive algorithm

  recAPSP(false, vec3(0, 0, 0), vec3(n, n, n), Am, lda);

#elif 0 ///////////
#warning base slow algorithm for debug

  base(vec3(0, 0, 0), vec3(n, n, n), Am, lda);
  printf("[APSP:algo] BASE SLOW ALGORITHM is used\n");

#else ///////////
#warning Non-recursive. loop-based algorithm
  long i, j, l;
  long ms = g.basesize.x;
  long ns = g.basesize.y;
  long ks = g.basesize.z;
  for (l = 0; l < n; l += ks) {
    for (j = 0; j < n; j += ns) {
      for (i = 0; i < n; i += ms) {
	base(vec3(i, j, l), vec3(i+ms, j+ns, l+ks),
	     Am, lda);
      }
    }
  }

  printf("[APSP:algo] NON-RECURSIVE ALGORITHM is used\n");

#endif

  printf("[matmul:algo] pivot kernel: %.3lf sec, %ld times\n",
	 kernel1time, kernel1count);
  printf("[matmul:algo] nonpivot kernel: %.3lf sec, %ld times\n",
	 kernel2time, kernel2count);

  return 0;
}

