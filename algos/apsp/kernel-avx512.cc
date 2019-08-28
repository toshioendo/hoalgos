#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/time.h>
#include <immintrin.h>
#include <assert.h>

#include <homm.h>
#include "apsp.h"


vec3 basesize_float_simd()
{
  return vec3(16*KERNEL_MAG, 16*KERNEL_MAG, 16*KERNEL_MAG);
}


long base_float_pack_gen(long m, long n, REAL *A, long lda, REAL *buf)
{
  assert(m % 16 == 0);
  REAL *Ap = A;
  for (long j = 0; j < n; j++) {
    for (long i = 0; i < m; i+=16) {
      _mm512_storeu_ps(&buf[i+j*m], _mm512_loadu_ps(&A[i+j*lda]));
    }
  }

  return m*n;
}

long base_float_packA(REAL *A, long lda, REAL *buf)
{
  return base_float_pack_gen(g.basesize.x, g.basesize.y, A, lda, buf);
}


long base_float_unpack_gen(long m, long n, REAL *A, long lda, REAL *buf)
{
  assert(m % 16 == 0);
  REAL *Ap = A;
  for (long j = 0; j < n; j++) {
    for (long i = 0; i < m; i+= 16) {
      _mm512_storeu_ps(&A[i+j*lda], _mm512_loadu_ps(&buf[i+j*m]));
    }
  }

  return m*n;
}

long base_float_unpackA(REAL *A, long lda, REAL *buf)
{
  return base_float_unpack_gen(g.basesize.x, g.basesize.y, A, lda, buf);
}

#define USE_RUCCI_KERNEL1  // only in pivot kernel
//#define USE_ZMM_ARRAY
//#define USE_ALWAYS_KERNEL1

// definitions used both for pivot/nonpivot kernels
#define ONE_COL(L0, J0, CNAME) \
  {							\
    __m512 vb;						\
    __m512 vsum;					\
    __mmask16 mask;					\
    vb = _mm512_set1_ps(B[(L0)+(J0)*lda]);		\
    vsum = _mm512_add_ps(va0, vb);			\
    mask = _mm512_cmp_ps_mask(vsum, CNAME, _CMP_LT_OQ);	\
    CNAME = _mm512_mask_blend_ps(mask, CNAME, vsum);	\
  }	


#ifndef USE_ZMM_ARRAY

// this is defined only if !USE_ZMM_ARRAY  
#define ONE_STEP(L0)					\
  {							\
    __m512 va0 = _mm512_loadu_ps(&A[0+(L0)*lda]);		\
    ONE_COL(L0, 0, vc00);				\
    ONE_COL(L0, 1, vc01);				\
    ONE_COL(L0, 2, vc02);				\
    ONE_COL(L0, 3, vc03);				\
    ONE_COL(L0, 4, vc04);				\
    ONE_COL(L0, 5, vc05);				\
    ONE_COL(L0, 6, vc06);				\
    ONE_COL(L0, 7, vc07);				\
    ONE_COL(L0, 8, vc08);				\
    ONE_COL(L0, 9, vc09);				\
    ONE_COL(L0, 10, vc0a);				\
    ONE_COL(L0, 11, vc0b);				\
    ONE_COL(L0, 12, vc0c);				\
    ONE_COL(L0, 13, vc0d);				\
    ONE_COL(L0, 14, vc0e);				\
    ONE_COL(L0, 15, vc0f);				\
  }

#endif // !USE_ZMM_ARRAY


// This kernel is for pivot computation 
// for size [16,16,k]
// for every iteration, results must be updated on array
int kernel_pivot_float_simd(long k, REAL *A, REAL *B, REAL *C, long lda)
{
#define CREAD(I0, J0) (_mm512_loadu_ps(&C[(I0)+(J0)*lda]))
#define CWRITE(I0, J0, CNAME)				\
  _mm512_storeu_ps(&C[(I0)+(J0)*lda], CNAME)

  const REAL infval = 1.0e+8;

#ifdef USE_RUCCI_KERNEL1
  // no accumlator
#elif USE_ZMM_ARRAY
  register __m512 vcs[16];
#pragma unroll
  for (long ic = 0; ic < 16; ic++)
    vcs[ic] = CREAD(0, ic);

#else
  __m512 vc00, vc01, vc02, vc03, vc04, vc05, vc06, vc07;
  __m512 vc08, vc09, vc0a, vc0b, vc0c, vc0d, vc0e, vc0f;

  vc00 = CREAD(0, 0);
  vc01 = CREAD(0, 1);
  vc02 = CREAD(0, 2);
  vc03 = CREAD(0, 3);
  vc04 = CREAD(0, 4);
  vc05 = CREAD(0, 5);
  vc06 = CREAD(0, 6);
  vc07 = CREAD(0, 7);
  vc08 = CREAD(0, 8);
  vc09 = CREAD(0, 9);
  vc0a = CREAD(0, 10);
  vc0b = CREAD(0, 11);
  vc0c = CREAD(0, 12);
  vc0d = CREAD(0, 13);
  vc0e = CREAD(0, 14);
  vc0f = CREAD(0, 15);

#endif // !USE_ZMM_ARRAY

  long l;
  for (l = 0; l < k; l ++) {
#ifdef USE_RUCCI_KERNEL1
    __m512 va = _mm512_loadu_ps(&A[0+l*lda]);
#pragma unroll
    for (long j = 0; j < 16; j++) {
      __m512 vc = _mm512_loadu_ps(&C[0+j*lda]);
      __m512 vb = _mm512_set1_ps(B[l+j*lda]);
      __m512 vsum = _mm512_add_ps(va, vb);
      __mmask16 mask = _mm512_cmp_ps_mask(vsum, vc, _CMP_LT_OQ);
      _mm512_mask_storeu_ps(&C[0+j*lda], mask, vsum);
    }

#elif defined USE_ZMM_ARRAY
    __m512 va0 = _mm512_loadu_ps(&A[0+(L0)*lda]);
#pragma unroll
    for (long j = 0; j < 16; j++) {
      ONE_COL(l, j, vcs[j]);			
      CWRITE(0, j, vcs[j]);
    }

#else
    ONE_STEP(l+0);
    CWRITE(0, 0, vc00);
    CWRITE(0, 1, vc01);
    CWRITE(0, 2, vc02);
    CWRITE(0, 3, vc03);
    CWRITE(0, 4, vc04);
    CWRITE(0, 5, vc05);
    CWRITE(0, 6, vc06);
    CWRITE(0, 7, vc07);
    CWRITE(0, 8, vc08);
    CWRITE(0, 9, vc09);
    CWRITE(0, 10, vc0a);
    CWRITE(0, 11, vc0b);
    CWRITE(0, 12, vc0c);
    CWRITE(0, 13, vc0d);
    CWRITE(0, 14, vc0e);
    CWRITE(0, 15, vc0f);
#endif // !USE_ZMM_ARRAY
  }


#undef CREAD
#undef CWRITE

  //printf("after kernel (%ld,%ld,%ld): last of C is %lf\n", v0.x, v0.y, v0.z, C[15+15*lda]);
  
  return 0;
}

#ifdef USE_ALWAYS_KERNEL1

#define kernel_nonpivot_float_simd kernel_pivot_float_simd

#else
// This kernel is for nonpivot computation 
// for size [16,16,k]
int kernel_nonpivot_float_simd(long k, REAL *A, REAL *B, REAL *C, long lda)
{
  // designed for 16x16x(mul of 4)
#ifdef USE_ZMM_ARRAY
  register __m512 vcs[16];
#else
  __m512 vc00, vc01, vc02, vc03, vc04, vc05, vc06, vc07;
  __m512 vc08, vc09, vc0a, vc0b, vc0c, vc0d, vc0e, vc0f;
#endif

  const REAL infval = 1.0e+8;
#define CCLEAR() (_mm512_set1_ps(infval))

#ifdef USE_ZMM_ARRAY
#pragma unroll
  for (long ic = 0; ic < 16; ic++)
    vcs[ic] = CCLEAR();
  
#else
  vc00 = CCLEAR();
  vc01 = CCLEAR();
  vc02 = CCLEAR();
  vc03 = CCLEAR();
  vc04 = CCLEAR();
  vc05 = CCLEAR();
  vc06 = CCLEAR();
  vc07 = CCLEAR();
  vc08 = CCLEAR();
  vc09 = CCLEAR();
  vc0a = CCLEAR();
  vc0b = CCLEAR();
  vc0c = CCLEAR();
  vc0d = CCLEAR();
  vc0e = CCLEAR();
  vc0f = CCLEAR();
#endif // !USE_ZMM_ARRAY

  long l;
#ifdef USE_ZMM_ARRAY
  for (l = 0; l < k; l += 1) {
    __m512 va0 = _mm512_loadu_ps(&A[0+l*lda]);
#pragma unroll
    for (long j = 0; j < 16; j++) {
      ONE_COL(l, j, vcs[j]);
    }
  }
#else
  for (l = 0; l < k; l += 2) {
    ONE_STEP(l+0);
    ONE_STEP(l+1);
  }
#endif

#define CUPDATE(I0, J0, CNAME)				\
  {							\
    __m512 v;						\
    __mmask16 mask;					\
    v = _mm512_loadu_ps(&C[(I0)+(J0)*lda]);		\
    mask = _mm512_cmp_ps_mask(CNAME, v, _CMP_LT_OQ);	\
    v = _mm512_mask_blend_ps(mask, v, CNAME);		\
    _mm512_storeu_ps(&C[(I0)+(J0)*lda], v);		\
  } 

#ifdef USE_ZMM_ARRAY
  for (long ic = 0; ic < 16; ic++)
    CUPDATE(0, ic, vcs[ic]);

#else
  CUPDATE(0, 0, vc00);
  CUPDATE(0, 1, vc01);
  CUPDATE(0, 2, vc02);
  CUPDATE(0, 3, vc03);
  CUPDATE(0, 4, vc04);
  CUPDATE(0, 5, vc05);
  CUPDATE(0, 6, vc06);
  CUPDATE(0, 7, vc07);
  CUPDATE(0, 8, vc08);
  CUPDATE(0, 9, vc09);
  CUPDATE(0, 10, vc0a);
  CUPDATE(0, 11, vc0b);
  CUPDATE(0, 12, vc0c);
  CUPDATE(0, 13, vc0d);
  CUPDATE(0, 14, vc0e);
  CUPDATE(0, 15, vc0f);
#endif

#undef CUPDATE

  //printf("after kernel (%ld,%ld,%ld): last of C is %lf\n", v0.x, v0.y, v0.z, C[15+15*ldc]);
  
  return 0;
}

#endif // !USE_ALWAYS_KERNEL1



#undef ONE_COL
#undef ONE_STEP


//////////////////////////////////////////
int base_float_simd(bool onpivot, vec3 v0, vec3 v1)
{
  assert(vec3eq(g.basesize, vec3sub(v1, v0)));

  REAL *A, *B, *C;
  size_t offs00, offs01, offs10, offs11;
  long lda;

  const long sbs = 16; // small block
  const long bs = sbs*KERNEL_MAG; // large block
  
  if (g.use_pack_mat) {
    // overwrite lda, ldb, ldc
    lda = bs;
    
    // block idx
    long xb = v0.x/lda;
    long yb = v0.y/lda;
    long zb = v0.z/lda;
  
    const long blocklen = bs*bs;

    A = &g.Abuf[(xb+zb*g.nb)*blocklen];
    B = &g.Abuf[(zb+yb*g.nb)*blocklen];
    C = &g.Abuf[(xb+yb*g.nb)*blocklen];
  }
  else {
    lda = g.lda;

    REAL *Am = g.Amat;
    A = &Am[v0.x + v0.z * lda];
    B = &Am[v0.z + v0.y * lda];
    C = &Am[v0.x + v0.y * lda];
  }

  offs00 = 0;
  offs01 = 0 + sbs*lda;
  offs10 = sbs + 0*lda;
  offs11 = sbs + sbs*lda;

#if KERNEL_MAG == 1
  if (onpivot) {
    kernel_pivot_float_simd(sbs, A, B, C, lda);
  }
  else {
    kernel_nonpivot_float_simd(sbs, A, B, C, lda);
  }
#elif KERNEL_MAG == 2
  if (onpivot) {
    // divide task into 8
    // 0
    kernel_pivot_float_simd(sbs, A+offs00, B+offs00, C+offs00, lda);
    // 1
    kernel_pivot_float_simd(sbs, A+offs10, B+offs00, C+offs10, lda);
    // 2
    kernel_pivot_float_simd(sbs, A+offs00, B+offs01, C+offs01, lda);
    // 3
    kernel_nonpivot_float_simd(sbs, A+offs10, B+offs01, C+offs11, lda);
    // 4
    kernel_pivot_float_simd(sbs, A+offs11, B+offs11, C+offs11, lda);
    // 5
    kernel_pivot_float_simd(sbs, A+offs01, B+offs11, C+offs01, lda);
    // 6
    kernel_pivot_float_simd(sbs, A+offs11, B+offs10, C+offs10, lda);
    // 7
    kernel_nonpivot_float_simd(sbs, A+offs01, B+offs10, C+offs00, lda);
  }
  else {
    // divide task into 4 (as large as possible)
    // 0
    kernel_nonpivot_float_simd(bs, A+offs00, B+offs00, C+offs00, lda);
    // 1
    kernel_nonpivot_float_simd(bs, A+offs10, B+offs00, C+offs10, lda);
    // 2
    kernel_nonpivot_float_simd(bs, A+offs00, B+offs01, C+offs01, lda);
    // 3
    kernel_nonpivot_float_simd(bs, A+offs10, B+offs01, C+offs11, lda);
  }
#else 
#error
  if (onpivot) {
    for (long kb = 0; kb < KERNEL_MAG; kb++) {
      // pivot tile [kb,kb]
    }
  }
  else {
  }

#endif

  return 0;
}

