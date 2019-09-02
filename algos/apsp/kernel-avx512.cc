#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/time.h>
#include <immintrin.h>
#include <assert.h>

#include <homm.h>
#include "apsp.h"

//#define USE_ZMM_ARRAY

// SIMD width of float type in AVX512
#define DWIDTH 16

vec3 basesize_float_simd()
{
  return vec3(DWIDTH*4, DWIDTH*4, DWIDTH*4);
}


long base_float_pack_gen(long m, long n, REAL *A, long lda, REAL *buf)
{
  assert(m % DWIDTH == 0);
  REAL *Ap = A;
  for (long j = 0; j < n; j++) {
    for (long i = 0; i < m; i+=DWIDTH) {
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
  assert(m % DWIDTH == 0);
  REAL *Ap = A;
  for (long j = 0; j < n; j++) {
    for (long i = 0; i < m; i+= DWIDTH) {
      _mm512_storeu_ps(&A[i+j*lda], _mm512_loadu_ps(&buf[i+j*m]));
    }
  }

  return m*n;
}

long base_float_unpackA(REAL *A, long lda, REAL *buf)
{
  return base_float_unpack_gen(g.basesize.x, g.basesize.y, A, lda, buf);
}


// Kernel1 (based on Rucci's)
// This kernel can be used for pivot computation 
// m must be a multiple of DWIDTH
int kernel_pivot_float_simd(long m, long n, long k, REAL *A, REAL *B, REAL *C, long lda)
{
  // assert(m % DWIDTH == 0);
  for (long l = 0; l < k; l ++) {
    for (long i = 0; i < m; i += DWIDTH) {
      __m512 va = _mm512_loadu_ps(&A[i+l*lda]);
#pragma unroll
      for (long j = 0; j < n; j++) {
	__m512 vc = _mm512_loadu_ps(&C[i+j*lda]);
	__m512 vb = _mm512_set1_ps(B[l+j*lda]);
	__m512 vsum = _mm512_add_ps(va, vb);
	__mmask16 mask = _mm512_cmp_ps_mask(vsum, vc, _CMP_LT_OQ);
	_mm512_mask_storeu_ps(&C[i+j*lda], mask, vsum);
      }
    }
  }

  return 0;
}

// definitions used in kernels
#define ONE_COL(L, J, CNAME) \
  {							\
    __m512 vb = _mm512_set1_ps(B[(L)+(J)*lda]);		\
    __m512 vsum = _mm512_add_ps(va, vb);			\
    __mmask16 mask = _mm512_cmp_ps_mask(vsum, CNAME, _CMP_LT_OQ);		\
    CNAME = _mm512_mask_blend_ps(mask, CNAME, vsum);	\
  }	

#ifdef USE_ZMM_ARRAY

#define ONE_STEP(I, J, L)				\
  {							\
    __m512 va = _mm512_loadu_ps(&A[(I)+(L)*lda]);	\
    for (long jj = 0; jj < 16; jj++) {			\
      ONE_COL(L, J+jj, vcs[jj]);			\
    }							\
  }

#else // !USE_ZMM_ARRAY

#define ONE_STEP(I, J, L)				\
  {							\
    __m512 va = _mm512_loadu_ps(&A[(I)+(L)*lda]);	\
    ONE_COL(L, J+0, vc00);				\
    ONE_COL(L, J+1, vc01);				\
    ONE_COL(L, J+2, vc02);				\
    ONE_COL(L, J+3, vc03);				\
    ONE_COL(L, J+4, vc04);				\
    ONE_COL(L, J+5, vc05);				\
    ONE_COL(L, J+6, vc06);				\
    ONE_COL(L, J+7, vc07);				\
    ONE_COL(L, J+8, vc08);				\
    ONE_COL(L, J+9, vc09);				\
    ONE_COL(L, J+10, vc0a);				\
    ONE_COL(L, J+11, vc0b);				\
    ONE_COL(L, J+12, vc0c);				\
    ONE_COL(L, J+13, vc0d);				\
    ONE_COL(L, J+14, vc0e);				\
    ONE_COL(L, J+15, vc0f);				\
  }
#endif // !USE_ZMM_ARRAY

#define CCLEAR(val) (_mm512_set1_ps(val))
#define CUPDATE(I0, J0, CNAME)				\
  {							\
    __m512 v;						\
    __mmask16 mask;					\
    v = _mm512_loadu_ps(&C[(I0)+(J0)*lda]);		\
    mask = _mm512_cmp_ps_mask(CNAME, v, _CMP_LT_OQ);	\
    v = _mm512_mask_blend_ps(mask, v, CNAME);		\
    _mm512_storeu_ps(&C[(I0)+(J0)*lda], v);		\
  } 


// Kernel2
// This kernel is only for nonpivot computation, where A, B, C are distinct.
int kernel_nonpivot_float_simd(long m, long n, long k, REAL *A, REAL *B, REAL *C, long lda)
{
  long i, j;
  const long sbs = DWIDTH; // small block
  for (j = 0; j < n; j += DWIDTH) {
#pragma unroll
    for (i = 0; i < m; i += DWIDTH) {
      // designed for DWIDTHxDWIDTHx(mul of 4)
      const REAL infval = 1.0e+8;

#ifdef USE_ZMM_ARRAY
      __m512 vcs[16];
      for (long jj = 0; jj < 16; jj++) {
	vcs[jj] = CCLEAR(infval);
      }
#else // !USE_ZMM_ARRAY      
      __m512 vc00 = CCLEAR(infval);
      __m512 vc01 = CCLEAR(infval);
      __m512 vc02 = CCLEAR(infval);
      __m512 vc03 = CCLEAR(infval);
      __m512 vc04 = CCLEAR(infval);
      __m512 vc05 = CCLEAR(infval);
      __m512 vc06 = CCLEAR(infval);
      __m512 vc07 = CCLEAR(infval);
      __m512 vc08 = CCLEAR(infval);
      __m512 vc09 = CCLEAR(infval);
      __m512 vc0a = CCLEAR(infval);
      __m512 vc0b = CCLEAR(infval);
      __m512 vc0c = CCLEAR(infval);
      __m512 vc0d = CCLEAR(infval);
      __m512 vc0e = CCLEAR(infval);
      __m512 vc0f = CCLEAR(infval);
#endif // !USE_ZMM_ARRAY      
      
      // incrementing 2 looks best. 1 or 4 is slower.
      for (long l = 0; l < k; l += 2) {
	ONE_STEP(i, j, l+0);
	ONE_STEP(i, j, l+1);
      }

#ifdef USE_ZMM_ARRAY
      for (long jj = 0; jj < 16; jj++) {
	CUPDATE(i, j+jj, vcs[jj]);
      }
#else      
      CUPDATE(i, j+0, vc00);
      CUPDATE(i, j+1, vc01);
      CUPDATE(i, j+2, vc02);
      CUPDATE(i, j+3, vc03);
      CUPDATE(i, j+4, vc04);
      CUPDATE(i, j+5, vc05);
      CUPDATE(i, j+6, vc06);
      CUPDATE(i, j+7, vc07);
      CUPDATE(i, j+8, vc08);
      CUPDATE(i, j+9, vc09);
      CUPDATE(i, j+10, vc0a);
      CUPDATE(i, j+11, vc0b);
      CUPDATE(i, j+12, vc0c);
      CUPDATE(i, j+13, vc0d);
      CUPDATE(i, j+14, vc0e);
      CUPDATE(i, j+15, vc0f);
#endif
    }
  }

  //printf("after kernel (%ld,%ld,%ld): last of C is %lf\n", v0.x, v0.y, v0.z, C[15+15*ldc]);
  
  return 0;
}


#undef ONE_COL
#undef ONE_STEP
#undef CUPDATE
#undef CCLEAR


//////////////////////////////////////////
int base_float_simd(bool onpivot, vec3 v0, vec3 v1)
{
  REAL *A, *B, *C;
  long lda;
  
  if (g.use_pack_mat) {
    // overwrite lda, ldb, ldc
    const long bs = g.basesize.x; // large block
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

  if (onpivot) {
    kernel_pivot_float_simd(v1.x-v0.x, v1.y-v0.y, v1.z-v0.z, A, B, C, lda);
  }
  else {
    kernel_nonpivot_float_simd(v1.x-v0.x, v1.y-v0.y, v1.z-v0.z, A, B, C, lda);
  }

  return 0;
}

