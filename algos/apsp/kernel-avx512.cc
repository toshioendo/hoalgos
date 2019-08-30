#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/time.h>
#include <immintrin.h>
#include <assert.h>

#include <homm.h>
#include "apsp.h"

// SIMD width of float type in AVX512
#define DWIDTH 16

vec3 basesize_float_simd()
{
  return vec3(DWIDTH*KERNEL_MAG, DWIDTH*KERNEL_MAG, DWIDTH*KERNEL_MAG);
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

// definitions used in kernels
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

#define CREAD(I0, J0) (_mm512_loadu_ps(&C[(I0)+(J0)*lda]))
#define CWRITE(I0, J0, CNAME)				\
  _mm512_storeu_ps(&C[(I0)+(J0)*lda], CNAME)

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


// Kernel2
// This kernel is for nonpivot computation 
// for size [DWIDTH,DWIDTH,k]
int kernel_nonpivot_float_simd(long m, long n, long k, REAL *A, REAL *B, REAL *C, long lda)
{
  // designed for DWIDTHxDWIDTHx(mul of 4)
  __m512 vc00, vc01, vc02, vc03, vc04, vc05, vc06, vc07;
  __m512 vc08, vc09, vc0a, vc0b, vc0c, vc0d, vc0e, vc0f;

  const REAL infval = 1.0e+8;

  vc00 = CCLEAR(infval);
  vc01 = CCLEAR(infval);
  vc02 = CCLEAR(infval);
  vc03 = CCLEAR(infval);
  vc04 = CCLEAR(infval);
  vc05 = CCLEAR(infval);
  vc06 = CCLEAR(infval);
  vc07 = CCLEAR(infval);
  vc08 = CCLEAR(infval);
  vc09 = CCLEAR(infval);
  vc0a = CCLEAR(infval);
  vc0b = CCLEAR(infval);
  vc0c = CCLEAR(infval);
  vc0d = CCLEAR(infval);
  vc0e = CCLEAR(infval);
  vc0f = CCLEAR(infval);

  // incrementing 2 looks best. 1 or 4 is slower.
  for (long l = 0; l < k; l += 2) {
    ONE_STEP(l+0);
    ONE_STEP(l+1);
  }

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

  //printf("after kernel (%ld,%ld,%ld): last of C is %lf\n", v0.x, v0.y, v0.z, C[15+15*ldc]);
  
  return 0;
}


#undef ONE_COL
#undef ONE_STEP
#undef CREAD
#undef CWRITE
#undef CUPDATE
#undef CCLEAR


//////////////////////////////////////////
int base_float_simd(bool onpivot, vec3 v0, vec3 v1)
{
  assert(vec3eq(g.basesize, vec3sub(v1, v0)));

  REAL *A, *B, *C;
  long lda;

  const long sbs = DWIDTH; // small block
  const long bs = DWIDTH*KERNEL_MAG; // large block
  
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

  size_t coffs = sbs*lda;
  size_t roffs = sbs;

  //#define PARAMS(ib, jb, kb) A+roffs*(ib)+coffs*(kb), B+roffs*(kb)+coffs*(jb), C+roffs*(ib)+coffs*(jb)

#ifndef USE_SECOND_KERNEL
  onpivot = true;
#endif

  if (onpivot) {
    kernel_pivot_float_simd(bs, bs, bs, A, B, C, lda);
  }
  else {
    // divide task into KERNEL_MAG^2 (as large as possible)
    long ib, jb;
    for (jb = 0; jb < KERNEL_MAG; jb++) {
#pragma unroll
      for (ib = 0; ib < KERNEL_MAG; ib++) {
	kernel_nonpivot_float_simd(sbs, sbs, bs, A+roffs*ib, B+coffs*jb, C+roffs*ib+coffs*jb, lda);
      }
    }
  }


  return 0;
}

