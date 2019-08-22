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

  g.task_thre = 32; //64; //128; //256; //512;
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
#ifdef USE_OMP
  printf("[apsp:init_algo] #threads=%d\n", omp_get_max_threads());
#endif

#if 1
  g.bufsize = 0L;
  g.buf = NULL;
#else
  // for internal copy buffer
  g.bufsize = 4L*1024*1024*1024;
  g.buf = (REAL*)homm_galloc(sizeof(REAL)*g.bufsize);
#endif

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
  if (meas_kernel && et > logtime+2.0)
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
#if 0
    printf("[recalgo] (do nothing)\n");
    printf("[recalgo] [(%d,%d,%d), (%d,%d,%d))\n",
	   v0.x, v0.y, v0.z, v1.x, v1.y, v1.z);
#endif
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

    bool small = (chunklen <= g.task_thre);

    vec3 v0s[8], v1s[8];
    v0s[0] = vec3(x0, y0, z0); v1s[0] = vec3(xm, ym, zm);
    v0s[1] = vec3(xm, y0, z0); v1s[1] = vec3(x1, ym, zm);
    v0s[2] = vec3(x0, ym, z0); v1s[2] = vec3(xm, y1, zm);
    v0s[3] = vec3(xm, ym, z0); v1s[3] = vec3(x1, y1, zm);
    v0s[4] = vec3(xm, ym, zm); v1s[4] = vec3(x1, y1, z1);
    v0s[5] = vec3(x0, ym, zm); v1s[5] = vec3(xm, y1, z1);
    v0s[6] = vec3(xm, y0, zm); v1s[6] = vec3(x1, ym, z1);
    v0s[7] = vec3(x0, y0, zm); v1s[7] = vec3(xm, ym, z1);

    if (small) {
      int it;
      for (it = 0; it < 4; it++) {
	recalgo(inbuf, v0s[it], v1s[it], Am, lda);
      }
      if (zm < z1) {
	for (it = 4; it < 8; it++) {
	  if (v0s[it].x < v1s[it].x && v0s[it].y < v1s[it].y) { 
	    recalgo(inbuf, v0s[it], v1s[it], Am, lda);
	  }
	}
      }
    }
    else if (onpivot) {
      // 0
      recalgo(inbuf, v0s[0], v1s[0], Am, lda);
      // 1
#pragma omp task
      recalgo(inbuf, v0s[1], v1s[1], Am, lda);
      // 2
#pragma omp task
      recalgo(inbuf, v0s[2], v1s[2], Am, lda);

#pragma omp taskwait

      // 3
      recalgo(inbuf, v0s[3], v1s[3], Am, lda);

      if (zm < z1) {
	// 4
	recalgo(inbuf, v0s[4], v1s[4], Am, lda);
	// 5
#pragma omp task
	recalgo(inbuf, v0s[5], v1s[5], Am, lda);
	// 6
#pragma omp task
	recalgo(inbuf, v0s[6], v1s[6], Am, lda);
	
#pragma omp taskwait
	// 7
	recalgo(inbuf, v0s[7], v1s[7], Am, lda);
      }
    }
    else {
      // nonpivot
      int it;
      for (it = 0; it < 4; it++) {
#pragma omp task
	recalgo(inbuf, v0s[it], v1s[it], Am, lda);
      }

#pragma omp taskwait

      if (zm < z1) {
	for (it = 4; it < 8; it++) {
	  if (v0s[it].x < v1s[it].x && v0s[it].y < v1s[it].y) { 
#pragma omp task
	    recalgo(inbuf, v0s[it], v1s[it], Am, lda);
	  }
	}
      }
#pragma omp taskwait
    }
  }

  return 0;
}


#ifdef USE_PACK_MAT
int pack_mats(long n,  REAL *Am, long lda)
{
  double st = Wtime();
  long nup = roundup(n, g.basesize.x);
  if (nup*nup > g.bufsize) {
    printf("[pack_mats] n=%d is too large to buffer. to be fixed\n", n);
    exit(1);
  }

  g.nb = nup/g.basesize.x;

  long blocklen = g.basesize.x*g.basesize.y; // in words
  REAL *p = g.buf;
  g.Abuf = p;

  long jb;
#if 1 && defined USE_OMP
#pragma omp parallel for
#endif
  for (jb = 0; jb < g.nb; jb ++) {
    long ib;
    long j = jb*g.basesize.y;
    for (ib = 0; ib < g.nb; ib ++) {
      long i = ib*g.basesize.x;

      base_float_packA(&Am[i+j*lda], lda, &g.Abuf[(ib+jb*g.nb)*blocklen]);
    }
  }
  p += g.nb*g.nb*blocklen;

  double et = Wtime();
  copytime += (et-st);

  return 0;
}

int unpack_mats(long n, REAL *Am, long lda)
{
  double st = Wtime();
  long nup = roundup(n, g.basesize.x);

  long blocklen = g.basesize.x*g.basesize.y; // in words
  REAL *p = g.Abuf;
  long jb;
#if 1 && defined USE_OMP
#pragma omp parallel for
#endif
  for (jb = 0; jb < g.nb; jb ++) {
    long ib;
    long j = jb*g.basesize.y;
    for (ib = 0; ib < g.nb; ib ++) {
      long i = ib*g.basesize.x;

      base_float_unpackA(&Am[i+j*lda], lda, &g.Abuf[(ib+jb*g.nb)*blocklen]);
    }
  }
  p += g.nb*g.nb*blocklen;

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

#ifdef USE_PACK_MAT

  if (n*n > g.bufsize) {
    // allocate internal copy buffer eagerly
    g.bufsize = n*n;
    g.buf = (REAL*)homm_galloc(sizeof(REAL)*g.bufsize);
  }

  pack_mats(n, Am, lda);
#endif


#if 1 /////////////
  // Recursive algorithm


#ifdef USE_OMP
#pragma omp parallel
#pragma omp single
#endif // USE_OMP
  recalgo(false, vec3(0, 0, 0), vec3(n, n, n), Am, lda);

#elif 1
#warning Non-recursive. loop-based algorithm

  printf("[APSP:algo] NON-RECURSIVE ALGORITHM is used\n");

  long l;
  long ms = g.basesize.x;
  long ns = g.basesize.y;
  long ks = g.basesize.z;
  for (l = 0; l < n; l += ks) {
    long i, j;
    // pivot tile
    base(vec3(l, l, l), vec3(l+ms, l+ns, l+ks),
	 Am, lda);

    // pivot col
    {
      long i;
#pragma omp parallel for
      for (i = 0; i < n; i += ms) {
	if (i != l) {
	  base(vec3(i, l, l), vec3(i+ms, l+ns, l+ks),
	       Am, lda);
	}
      }
    }

    // pivot row
#pragma omp parallel for
    for (j = 0; j < n; j += ns) {
      if (j != l) {
	base(vec3(l, j, l), vec3(l+ms, j+ns, l+ks),
	     Am, lda);
      }
    }

    // other tiles
#pragma omp parallel for private (i)
    for (j = 0; j < n; j += ns) {
      if (j != l) {
	for (i = 0; i < n; i += ms) {
	  if (i != l) {
	    base(vec3(i, j, l), vec3(i+ms, j+ns, l+ks),
		 Am, lda);
	  }
	}
      }
    }
  }


#else ///////////
#warning base slow algorithm for debug

  printf("[APSP:algo] BASE SLOW ALGORITHM is used\n");
  base(vec3(0, 0, 0), vec3(n, n, n), Am, lda);

#endif

#ifdef USE_PACK_MAT
  unpack_mats(n, Am, lda);
#endif

  printf("[APSP:algo] kernel: %.3lf sec, %ld times\n",
	 kernel1time, kernel1count);
  printf("[APSP:algo] copy: %.3lf sec\n",
	 copytime);

  return 0;
}

