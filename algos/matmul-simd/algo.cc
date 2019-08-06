// hierarchy oblivious matrix multiply
//   single address space version
// algorithm
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/time.h>
//#include <immintrin.h>
#include <assert.h>

#include <homm.h>
#include "matmul.h"

// global variables
struct global g;


double logtime = 0.0;
double kernelfasttime = 0.0;
double kernelslowtime = 0.0;
long ncopy = 0;
long copysize = 0;
double copytime = 0.0;

int init_algo()
{
  printf("Hierarchy Oblivious Matrix Multiply sample\n");
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

  g.basesize = basesize_double_simd();
  printf("[matmul:init_algo]  Compile time options: USE_AVX2 %c, USE_AVX512 %c, USE_OMP %c, USE_OMPTASK %c\n",
	 use_avx2, use_avx512, use_omp, use_omptask);
  printf("[matmul:init_algo] type=[%s] basesize=(%ld,%ld,%ld)\n",
	 TYPENAME, g.basesize.x, g.basesize.y, g.basesize.z);

  // for internal copy buffer
#if 1
  g.bufsize = 128*1024*1024;
#else
  g.bufsize = (g.basesize.x*g.basesize.z + g.basesize.z*g.basesize.y 
	       + g.basesize.x*g.basesize.y) * (g.ndiv * g.ndiv);
#endif
  g.buf = (REAL*)homm_galloc(sizeof(REAL)*g.bufsize);

  printf("[matmul] bufsize=%ld*%ld=%ld Bytes\n", g.bufsize, sizeof(REAL), sizeof(REAL)*g.bufsize);

  return 0;
}

int base_cpuloop(vec3 v0, vec3 v1, REAL *Am, long lda, REAL *Bm, long ldb, REAL *Cm, long ldc)
{
  long m = (long)(v1.x-v0.x);
  long n = (long)(v1.y-v0.y);
  long k = (long)(v1.z-v0.z);

  REAL *A = &Am[v0.x + v0.z * lda];
  REAL *B = &Bm[v0.z + v0.y * ldb];
  REAL *C = &Cm[v0.x + v0.y * lda];

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

  return 0;
}

int base(vec3 v0, vec3 v1, REAL *Am, long lda, REAL *Bm, long ldb, REAL *Cm, long ldc)
{

  double st, et;
  // base case
#if VERBOSE >= 20
  printf("[base] START [(%ld,%ld,%ld), (%ld,%ld,%ld))\n", 
	 v0.x, v0.y, v0.z, v1.x, v1.y, v1.z);
#endif

  st = Wtime();

  if (v1.x-v0.x == g.basesize.x &&
      v1.y-v0.y == g.basesize.y &&
      v1.z-v0.z == g.basesize.z) {
    base_double_simd(v0, v1, Am, lda, Bm, ldb, Cm, ldc);
    et = Wtime();
    kernelfasttime += (et-st);
  }
  else {
    base_cpuloop(v0, v1, Am, lda, Bm, ldb, Cm, ldc);
    et = Wtime();
    kernelslowtime += (et-st);
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

int recalgo(bool inbuf, vec3 v0, vec3 v1, REAL *Am, long lda, REAL *Bm, long ldb, REAL *Cm, long ldc);

int copyandrec(vec3 v0, vec3 v1, REAL *Am, long lda, REAL *Bm, long ldb, REAL *Cm, long ldc)
{
#if 01
  recalgo(true, v0, v1, Am, lda, Bm, ldb, Cm, ldc);
#else
  // DO COPY
  long m = (long)(v1.x-v0.x);
  long n = (long)(v1.y-v0.y);
  long k = (long)(v1.z-v0.z);

  double st, et;
  // copyin
  printf("[copyandrec] %ld x %ld x %d\n", m, n, k);
  st = Wtime();
  REAL *p = g.buf;
  REAL *A = &Am[v0.x + v0.z * lda];
  REAL *B = &Bm[v0.z + v0.y * ldb];
  REAL *C = &Cm[v0.x + v0.y * lda];
  long align;
  long j;
  // copy A
  REAL *Abuf = p;
  align = g.basesize.get('X');
  long ldabuf = roundup(m, align);
  for (j = 0; j < k; j++) {
    memcpy(p, A, sizeof(REAL)*m);
    A += lda;
    p += ldabuf;
  }

  copysize += sizeof(REAL)*m*k;

  // copy B
  REAL *Bbuf = p;
  align = g.basesize.get('Z');
  long ldbbuf = roundup(k, align);
  for (j = 0; j < n; j++) {
    memcpy(p, B, sizeof(REAL)*k);
    B += ldb;
    p += ldbbuf;
  }

  copysize += sizeof(REAL)*n*k;

  // copy C
  REAL *Cbuf = p;
  align = g.basesize.get('X');
  long ldcbuf = roundup(m, align);
  for (j = 0; j < n; j++) {
    memcpy(p, C, sizeof(REAL)*m);
    C += ldc;
    p += ldcbuf;
  }

  copysize += sizeof(REAL)*m*n;
  et = Wtime();
  copytime += (et-st);

  recalgo(true, vec3(0,0,0), vec3(m,n,k), Abuf, ldabuf, Bbuf, ldbbuf, Cbuf, ldcbuf);

  st = Wtime();
  // copyback C
  C = &Cm[v0.x + v0.y * lda];
  align = g.basesize.get('X');
  p = Cbuf;
  for (j = 0; j < n; j++) {
    memcpy(C, p, sizeof(REAL)*m);
    C += ldc;
    p += ldcbuf;
  }

  copysize += sizeof(REAL)*m*n;
  ncopy++;
  et = Wtime();
  copytime += (et-st);
#endif
  return 0;
}

int recalgo(bool inbuf, vec3 v0, vec3 v1, REAL *Am, long lda, REAL *Bm, long ldb, REAL *Cm, long ldc)
{
#if VERBOSE >= 30
  printf("[recalgo] [(%d,%d,%d), (%d,%d,%d))\n",
	 v0.x, v0.y, v0.z, v1.x, v1.y, v1.z);
#endif

  if (v0.x >= v1.x || v0.y >= v1.y || v0.z >= v1.z) {
    printf("[recalgo] (do nothing)\n");
    printf("[recalgo] [(%d,%d,%d), (%d,%d,%d))\n",
	   v0.x, v0.y, v0.z, v1.x, v1.y, v1.z);
    return 0;
  }


  vec3 csize = vec3sub(v1, v0);
  long cx = csize.x;
  long cy = csize.y;
  long cz = csize.z;
  if (inbuf && cx <= g.basesize.x && cy <= g.basesize.y && cz <= g.basesize.z) {
    // base case
    base(v0, v1, Am, lda, Bm, ldb, Cm, ldc);
  }
  else if (!inbuf && (cx*cz+cz*cy+cx*cy <= g.bufsize)) {
    copyandrec(v0, v1, Am, lda, Bm, ldb, Cm, ldc);
  }
  else {
    // divide long dimension
    char dim;
    if (cx >= cy*4 && cx >= cz*4 && csize.x > g.basesize.x) dim = 'X';
    else if (cy >= cz && csize.y > g.basesize.y) dim = 'Y';
    else if (csize.z > g.basesize.z) dim = 'Z';
    else if (csize.y > g.basesize.y) dim = 'Y';
    else dim = 'X';

#if 1
    long len = csize.get(dim);
    long align = g.basesize.get(dim);
    assert(len > align);
    // try to find chunk size which is alignsize*ndiv^i
    long chunklen = align;
    while (chunklen*g.ndiv < len) {
      chunklen *= g.ndiv;
    }
    assert(chunklen > 0);

    long idx0 = v0.get(dim);
    long idx1 = v1.get(dim);
    //fprintf(stderr, "start div[%c], len=%ld, chunklen=%ld\n", dim, len, chunklen);
    // regular part (same chunklens)
    vec3 chunksize = vec3mod(csize, dim, chunklen);
    vec3 diffvec = vec3mod(vec3(0,0,0), dim, chunklen);

    if (vec3eq(chunksize, g.basesize)) {
      // childrens are base cases.
      // kernel is called directly for optimzation
      double st = Wtime();
      long s;
      //printf("calling fast kernels. dim=%c\n", dim);
      for (s = idx0; s+chunklen <= idx1; s += chunklen) {
	long ns = s+chunklen;
	base_double_simd(vec3mod(v0, dim, s), vec3mod(v1, dim, ns),
			 Am, lda, Bm, ldb, Cm, ldc);
      }

      double et = Wtime();
      kernelfasttime += (et-st);

      // rest part
      if (s < idx1) {
	long ns = idx1;
	base(vec3mod(v0, dim, s), vec3mod(v1, dim, ns),
	     Am, lda, Bm, ldb, Cm, ldc);
      }

    }
    else {
      // general case 
      long s;
      for (s = idx0; s < idx1; s += chunklen) {
	long ns = s+chunklen;
	if (ns > idx1) ns = idx1;
	recalgo(inbuf, vec3mod(v0, dim, s), vec3mod(v1, dim, ns),
		Am, lda, Bm, ldb, Cm, ldc);
      }
    }

#else
    long mid = (v0.get(dim) + v1.get(dim))/2;
    long align = g.basesize.get(dim);
    mid = ((mid+align-1)/align)*align;

    // first task
    recalgo(inbuf, v0, vec3mod(v1, dim, mid),
	    Am, lda, Bm, ldb, Cm, ldc);
    // second task
    recalgo(inbuf, vec3mod(v0, dim, mid), v1,
	    Am, lda, Bm, ldb, Cm, ldc);
#endif
  }

  return 0;
}

#ifdef USE_PACK_MAT
int pack_mats(long m, long n, long k, REAL *Am, long lda, REAL *Bm, long ldb, REAL *Cm, long ldc)
{
  double st = Wtime();
  long mup = roundup(m, g.basesize.x);
  long nup = roundup(n, g.basesize.y);
  long kup = roundup(k, g.basesize.z);
  if (mup*nup + nup*kup + mup*kup >= g.bufsize) {
    printf("(%d,%d,%d) is too large. to be fixed\n", m, n, k);
  }

  g.mb = mup/g.basesize.x;
  g.nb = nup/g.basesize.y;
  g.kb = kup/g.basesize.z;

  long i, j, l;
  long s;
  REAL *p = g.buf;
  g.Abuf = p;
  for (l = 0; l < kup; l += g.basesize.z) {
    for (i = 0; i < mup; i += g.basesize.x) {
      s = base_double_packA(&Am[i+l*lda], lda, p);
      p += s;
    }
  }
  g.Bbuf = p;
  for (j = 0; j < nup; j += g.basesize.y) {
    for (l = 0; l < kup; l += g.basesize.z) {
      s = base_double_packB(&Bm[l+j*lda], ldb, p);
      p += s;
    }
  }
  g.Cbuf = p;
  for (j = 0; j < nup; j += g.basesize.y) {
    for (i = 0; i < mup; i += g.basesize.x) {
      s = base_double_packC(&Cm[i+j*lda], ldc, p);
      p += s;
    }
  }

  double et = Wtime();
  copytime += (et-st);

  return 0;
}

int unpack_mats(long m, long n, long k, REAL *Am, long lda, REAL *Bm, long ldb, REAL *Cm, long ldc)
{
  double st = Wtime();
  long mup = roundup(m, g.basesize.x);
  long nup = roundup(n, g.basesize.y);
  long kup = roundup(k, g.basesize.z);

  long i, j;
  long s;
  REAL *p = g.Cbuf;
  for (j = 0; j < nup; j += g.basesize.y) {
    for (i = 0; i < mup; i += g.basesize.x) {
      s = base_double_unpackC(&Cm[i+j*lda], ldc, p);
      p += s;
    }
  }

  double et = Wtime();
  copytime += (et-st);
  return 0;
}
#endif

int algo(long m, long n, long k, REAL *Am, long lda, REAL *Bm, long ldb, REAL *Cm, long ldc)
{
  printf("[matmul:algo] type=[%s] size=(%ld,%ld,%ld) basesize=(%ld,%ld,%ld)\n",
	 TYPENAME, m, n, k, g.basesize.x, g.basesize.y, g.basesize.z);
#if 01
  // Recursive algorithm
  ncopy = 0;
  copysize = 0;
  kernelfasttime = 0.0;
  kernelslowtime = 0.0;
  copytime = 0.0;

#ifdef USE_PACK_MAT
  pack_mats(m, n, k, Am, lda, Bm, ldb, Cm, ldc);
#endif

#ifdef USE_OMPTASK
#pragma omp parallel
#pragma omp single
#endif // USE_OMPTASK
  recalgo(false, vec3(0, 0, 0), vec3(m, n, k), Am, lda, Bm, ldb, Cm, ldc);

#ifdef USE_PACK_MAT
  unpack_mats(m, n, k, Am, lda, Bm, ldb, Cm, ldc);
#endif

#else
#warning Non-recursive. loop-based algorithm
  long i, j, l;
  long ms = g.basesize.x;
  long ns = g.basesize.y;
  long ks = g.basesize.z;
  for (j = 0; j < n; j += ns) {
    for (l = 0; l < k; l += ks) {
      for (i = 0; i < m; i += ms) {
	base_double_simd(vec3(i, j, l), vec3(i+ms, j+ns, l+ks),
			 Am, lda, Bm, ldb, Cm, ldc);
      }
    }
  }

  if (m % ms != 0 || n % ns != 0 || k % ks != 0) {
    printf("TODO: I must consider indivible cases!\n");
  }
#endif

  printf("[matmul:algo] %.3lf sec in fast kernel, %.3lf sec in slow kernel\n", kernelfasttime, kernelslowtime);
  printf("[matmul:algo] buf-copied %ld time, total %ld bytes, Copy took %.3lf sec\n", ncopy, copysize, copytime);


  return 0;
}

