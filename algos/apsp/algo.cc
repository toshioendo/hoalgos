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

#ifdef USE_OMP
#include <omp.h>
#endif

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

  g.basesize = basesize_float_simd();

  g.task_thre = 512;
  char *envstr;
  envstr = getenv("TASK_THRE");
  if (envstr != NULL) {
    g.task_thre = atol(envstr);
    if (g.task_thre < 16) {
      printf("TASK_THRE(%ld) must be >=16 (such as 512)\n", g.task_thre);
      exit(1);
    }
  }
  

  printf("[apsp:init_algo]  Compile time options: USE_AVX2 %c, USE_AVX512 %c, USE_OMP %c\n",
	 use_avx2, use_avx512, use_omp);
  printf("[apsp:init_algo] type=[%s] basesize=(%ld,%ld,%ld)\n",
	 TYPENAME, g.basesize.x, g.basesize.y, g.basesize.z);
  printf("[apsp:init_algo] TASK_THRE=%ld\n", g.task_thre);

  // for internal copy buffer
  g.bufsize = 128*1024*1024;
  g.buf = (REAL*)homm_galloc(sizeof(REAL)*g.bufsize);

  printf("[apsp:init_algo] bufsize=%ld*%ld=%ld Bytes\n", g.bufsize, sizeof(REAL), sizeof(REAL)*g.bufsize);

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

inline int base(vec3 v0, vec3 v1, REAL *Am, long lda)
{
  // base case
#if VERBOSE >= 20
  printf("[base] START [(%ld,%ld,%ld), (%ld,%ld,%ld))\n", 
	 v0.x, v0.y, v0.z, v1.x, v1.y, v1.z);
#endif

  bool meas_kernel = true;
#ifdef USE_OMP
  if (omp_get_thread_num() != 0) meas_kernel = false;
#endif

  double st = 0.0;
  if (meas_kernel) st = Wtime();

  if (v0.x == v0.z || v0.y == v0.z) {
#if 1
    base_pivot_float_simd(v0, v1, Am, lda);
#else
    base_pivot_cpuloop(v0, v1, Am, lda);
#endif
  }
  else {
#if 1
    base_nonpivot_float_simd(v0, v1, Am, lda);
#else
    base_nonpivot_cpuloop(v0, v1, Am, lda);
#endif
  }

  double et = 0.0;
  if (meas_kernel) {
    et = Wtime();
    kernel1time += (et-st);
    kernel1count++;
  }

  // print periodically
#if VERBOSE >= 10
#if VERBOSE >= 30
  if (meas_kernel)
#else
  if (meas_kernel && et > logtime+1.0)
#endif
    {
      double t = et-st;
      logtime = et;
      printf("[base] END [(%ld,%ld,%ld), (%ld,%ld,%ld)) -> %.6lfsec\n", 
	     v0.x, v0.y, v0.z, v1.x, v1.y, v1.z, t);
    }
#endif


  return 0;
}

int recalgo(bool inbuf, vec3 v0, vec3 v1, REAL *Am, long lda)
{
#if VERBOSE >= 30
  printf("[recAPSP] [(%d,%d,%d), (%d,%d,%d))\n",
	 v0.x, v0.y, v0.z, v1.x, v1.y, v1.z);
#endif

  if (v0.x >= v1.x || v0.y >= v1.y || v0.z >= v1.z) {
    printf("[recalgo] (do nothing)\n");
    printf("[recalgo] [(%d,%d,%d), (%d,%d,%d))\n",
	   v0.x, v0.y, v0.z, v1.x, v1.y, v1.z);
    return 0;
  }

  vec3 csize = vec3sub(v1, v0);
  long cx = v1.x-v0.x;
  long cy = v1.y-v0.y;
  long cz = v1.z-v0.z;
  if (cx <= g.basesize.x && cy <= g.basesize.y && cz <= g.basesize.z) {
    // base case
    base(v0, v1, Am, lda);
  }
  else {
    // general case
    bool onpivot = (v0.x == v0.z || v0.y == v0.z);

    long len = cx;
    if (cy > len) len = cy;
    if (cz > len) len = cz;
    long chunklen = g.basesize.x;
    while (chunklen*2 < len) {
      chunklen *= 2;
    }
    assert(chunklen > 0);

    // divide the task into 8
    long x0 = v0.x;
    long x1 = v1.x;
    long y0 = v0.y;
    long y1 = v1.y;
    long z0 = v0.z;
    long z1 = v1.z;

    long xm = x0+chunklen;
    if (xm > x1) xm = x1;
    long ym = y0+chunklen;
    if (ym > y1) ym = y1;
    long zm = z0+chunklen;
    if (zm > z1) zm = z1;

    bool small = (chunklen < g.task_thre);

    // 1
#pragma omp task if(!small && !onpivot)
    recalgo(inbuf, vec3(x0, y0, z0), vec3(xm, ym, zm),
	    Am, lda);
    // 2
    if (xm < x1) 
#pragma omp task if(!small)
      recalgo(inbuf, vec3(xm, y0, z0), vec3(x1, ym, zm), Am, lda);
    // 3
    if (ym < y1)
#pragma omp task if(!small)
      recalgo(inbuf, vec3(x0, ym, z0), vec3(xm, y1, zm), Am, lda);

    if (!small && onpivot) {
#pragma omp taskwait
    }

    // 4
    if (xm < x1 && ym < y1) 
#pragma omp task if(!small && !onpivot)
      recalgo(inbuf, vec3(xm, ym, z0), vec3(x1, y1, zm),
	    Am, lda);

    if (!small) {
#pragma omp taskwait
    }

    if (zm < z1) {
      // 5
      if (xm < x1 && ym < y1) 
#pragma omp task if(!small && !onpivot)
	recalgo(inbuf, vec3(xm, ym, zm), vec3(x1, y1, z1),
		Am, lda);
      // 6
      if (ym < y1)
#pragma omp task if(!small)
	recalgo(inbuf, vec3(x0, ym, zm), vec3(xm, y1, z1),
		Am, lda);
      // 7
      if (xm < x1) 
#pragma omp task if(!small)
	recalgo(inbuf, vec3(xm, y0, zm), vec3(x1, ym, z1),
		Am, lda);

      if (!small && onpivot) {
#pragma omp taskwait
      }

      // 8
#pragma omp task if(!small && !onpivot)
      recalgo(inbuf, vec3(x0, y0, zm), vec3(xm, ym, z1),
	      Am, lda);

      if (!small) {
#pragma omp taskwait
      }
    }
  }

  return 0;
}


#ifdef USE_PACK_MAT
int pack_mats(long n,  REAL *Am, long lda)
{
  double st = Wtime();
  long nup = roundup(n, g.basesize.x);
  if (nup*nup >= g.bufsize) {
    printf("(%d) is too large. to be fixed\n", n);
  }

  g.nb = nup/g.basesize.x;

  long i, j;
  long s;
  REAL *p = g.buf;
  g.Abuf = p;
  for (j = 0; j < nup; j += g.basesize.y) {
    for (i = 0; i < nup; i += g.basesize.x) {
      s = base_float_packA(&Am[i+j*lda], lda, p);
      p += s;
    }
  }

  double et = Wtime();
  copytime += (et-st);

  return 0;
}

int unpack_mats(long n, REAL *Am, long lda)
{
  double st = Wtime();
  long nup = roundup(n, g.basesize.x);

  long i, j;
  long s;
  REAL *p = g.Abuf;
  for (j = 0; j < nup; j += g.basesize.y) {
    for (i = 0; i < nup; i += g.basesize.x) {
      s = base_float_unpackA(&Am[i+j*lda], lda, p);
      p += s;
    }
  }

  double et = Wtime();
  copytime += (et-st);
  return 0;
}
#endif

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


#if 1 /////////////
  // Recursive algorithm

#ifdef USE_PACK_MAT
  pack_mats(n, Am, lda);
#endif

#ifdef USE_OMP
#pragma omp parallel
#pragma omp single
#endif // USE_OMP
  recalgo(false, vec3(0, 0, 0), vec3(n, n, n), Am, lda);

#ifdef USE_PACK_MAT
  unpack_mats(n, Am, lda);
#endif

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

  printf("[APSP:algo] kernel: %.3lf sec, %ld times\n",
	 kernel1time, kernel1count);
  printf("[APSP:algo] copy: %.3lf sec\n",
	 copytime);

  return 0;
}

