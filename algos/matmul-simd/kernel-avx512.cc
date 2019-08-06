#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/time.h>
#include <immintrin.h>
#include <assert.h>

#include <homm.h>
#include "matmul.h"

vec3 basesize_double_simd()
{
  return vec3(16, 8, 256/*64*/);
}

#ifdef USE_PACK_MAT
long base_double_pack_gen(long m, long n, REAL *A, long lda, REAL *buf)
{
  assert(m % 8 == 0 && n % 4 == 0);
  REAL *Ap = A;
#if 0
  for (long j = 0; j < n; j++) {
    for (long i = 0; i < m; i+=8) {
      _mm512_storeu_pd(&buf[i+j*m], _mm512_loadu_pd(&A[i+j*lda]));
    }
  }
#else
  for (long j = 0; j < n; j += 4) {
    for (long i = 0; i < m; i+=8) {
      _mm512_storeu_pd(&buf[i+(j+0)*m], _mm512_loadu_pd(&A[i+(j+0)*lda]));
      _mm512_storeu_pd(&buf[i+(j+1)*m], _mm512_loadu_pd(&A[i+(j+1)*lda]));
      _mm512_storeu_pd(&buf[i+(j+2)*m], _mm512_loadu_pd(&A[i+(j+2)*lda]));
      _mm512_storeu_pd(&buf[i+(j+3)*m], _mm512_loadu_pd(&A[i+(j+3)*lda]));
    }
  }
#endif

  return m*n;
}

long base_double_packA(REAL *A, long lda, REAL *buf)
{
  return base_double_pack_gen(g.basesize.x, g.basesize.z, A, lda, buf);
}

long base_double_packB(REAL *B, long ldb, REAL *buf)
{
  return base_double_pack_gen(g.basesize.z, g.basesize.y, B, ldb, buf);
}

long base_double_packC(REAL *C, long ldc, REAL *buf)
{
  return base_double_pack_gen(g.basesize.x, g.basesize.y, C, ldc, buf);
}

long base_double_unpack_gen(long m, long n, REAL *A, long lda, REAL *buf)
{
  assert(m % 8 == 0 && n % 4 == 0);
  REAL *Ap = A;
#if 0
  for (long j = 0; j < n; j++) {
    for (long i = 0; i < m; i+=8) {
      _mm512_storeu_pd(&A[i+j*lda], _mm512_loadu_pd(&buf[i+j*m]));
    }
  }
#else
  for (long j = 0; j < n; j += 4) {
    for (long i = 0; i < m; i+=8) {
      _mm512_storeu_pd(&A[i+(j+0)*lda], _mm512_loadu_pd(&buf[i+(j+0)*m]));
      _mm512_storeu_pd(&A[i+(j+1)*lda], _mm512_loadu_pd(&buf[i+(j+1)*m]));
      _mm512_storeu_pd(&A[i+(j+2)*lda], _mm512_loadu_pd(&buf[i+(j+2)*m]));
      _mm512_storeu_pd(&A[i+(j+3)*lda], _mm512_loadu_pd(&buf[i+(j+3)*m]));
    }
  }
#endif

  return m*n;
}

long base_double_unpackC(REAL *C, long ldc, REAL *buf)
{
  return base_double_unpack_gen(g.basesize.x, g.basesize.y, C, ldc, buf);
}


#endif

int base_double_simd(vec3 v0, vec3 v1, REAL *Am, long lda, REAL *Bm, long ldb, REAL *Cm, long ldc)
{
  const long m = g.basesize.x;
  const long n = g.basesize.y;
  const long k = g.basesize.z;
  // designed for 16x4x(mul of 4)

#ifdef USE_PACK_MAT
  long ib = v0.x/m; // g.basesize.x
  long jb = v0.y/n; // g.basesize.y
  long lb = v0.z/k; // g.basesize.z

  REAL *A = &g.Abuf[(ib+lb*g.mb)*m*k];
  REAL *B = &g.Bbuf[(lb+jb*g.kb)*k*n];
  REAL *C = &g.Cbuf[(ib+jb*g.mb)*m*n];
  // overwrite lda, ldb, ldc
  lda = 16; // g.basesize.x;
  ldb = k; // g.basesize.x;
  ldc = 16; // g.basesize.x;
#else
  REAL *A = &Am[v0.x + v0.z * lda];
  REAL *B = &Bm[v0.z + v0.y * ldb];
  REAL *C = &Cm[v0.x + v0.y * lda];
#endif

  __m512d vc00, vc01, vc02, vc03, vc04, vc05, vc06, vc07;
  __m512d vc10, vc11, vc12, vc13, vc14, vc15, vc16, vc17;
  vc00 = _mm512_setzero_pd();
  vc01 = _mm512_setzero_pd();
  vc02 = _mm512_setzero_pd();
  vc03 = _mm512_setzero_pd();
  vc04 = _mm512_setzero_pd();
  vc05 = _mm512_setzero_pd();
  vc06 = _mm512_setzero_pd();
  vc07 = _mm512_setzero_pd();
  vc10 = _mm512_setzero_pd();
  vc11 = _mm512_setzero_pd();
  vc12 = _mm512_setzero_pd();
  vc13 = _mm512_setzero_pd();
  vc14 = _mm512_setzero_pd();
  vc15 = _mm512_setzero_pd();
  vc16 = _mm512_setzero_pd();
  vc17 = _mm512_setzero_pd();

  if (k % 8 != 0) {
    fprintf(stderr, "ERROR: k=%d is invalid\n", k);
    exit(1);
  }

#define LOADA_STEP(L0)					\
  va0 = _mm512_loadu_pd(&A[0+(L0)*lda]);		\
  va1 = _mm512_loadu_pd(&A[8+(L0)*lda]);

#define ONE_STEP(L0)					\
  vb = _mm512_set1_pd(B[(L0)+0*ldb]);			\
  vc00 = _mm512_fmadd_pd(va0, vb, vc00);		\
  vc10 = _mm512_fmadd_pd(va1, vb, vc10);		\
  							\
  vb = _mm512_set1_pd(B[(L0)+1*ldb]);			\
  vc01 = _mm512_fmadd_pd(va0, vb, vc01);		\
  vc11 = _mm512_fmadd_pd(va1, vb, vc11);		\
  							\
  vb = _mm512_set1_pd(B[(L0)+2*ldb]);			\
  vc02 = _mm512_fmadd_pd(va0, vb, vc02);		\
  vc02 = _mm512_fmadd_pd(va1, vb, vc12);		\
  							\
  vb = _mm512_set1_pd(B[(L0)+3*ldb]);			\
  vc03 = _mm512_fmadd_pd(va0, vb, vc03);		\
  vc13 = _mm512_fmadd_pd(va1, vb, vc13);		\
  							\
  vb = _mm512_set1_pd(B[(L0)+4*ldb]);			\
  vc04 = _mm512_fmadd_pd(va0, vb, vc04);		\
  vc14 = _mm512_fmadd_pd(va1, vb, vc14);		\
  							\
  vb = _mm512_set1_pd(B[(L0)+5*ldb]);			\
  vc05 = _mm512_fmadd_pd(va0, vb, vc05);		\
  vc15 = _mm512_fmadd_pd(va1, vb, vc15);		\
  							\
  vb = _mm512_set1_pd(B[(L0)+6*ldb]);			\
  vc06 = _mm512_fmadd_pd(va0, vb, vc06);		\
  vc16 = _mm512_fmadd_pd(va1, vb, vc16);		\
  							\
  vb = _mm512_set1_pd(B[(L0)+7*ldb]);			\
  vc07 = _mm512_fmadd_pd(va0, vb, vc07);		\
  vc17 = _mm512_fmadd_pd(va1, vb, vc17);		

  long l;
  for (l = 0; l < k; l += 8) {
    __m512d va0, va1;
    __m512d vb;
    
    LOADA_STEP(l+0);
    ONE_STEP(l+0);
    LOADA_STEP(l+1);
    ONE_STEP(l+1);
    LOADA_STEP(l+2);
    ONE_STEP(l+2);
    LOADA_STEP(l+3);
    ONE_STEP(l+3);
    LOADA_STEP(l+4);
    ONE_STEP(l+4);
    LOADA_STEP(l+5);
    ONE_STEP(l+5);
    LOADA_STEP(l+6);
    ONE_STEP(l+6);
    LOADA_STEP(l+7);
    ONE_STEP(l+7);
  }

#undef LOADA_STEP
#undef ONE_STEP
  
  _mm512_storeu_pd(&C[0+0*ldc], _mm512_add_pd(_mm512_loadu_pd(&C[0+0*ldc]), vc00));
  _mm512_storeu_pd(&C[8+0*ldc], _mm512_add_pd(_mm512_loadu_pd(&C[8+0*ldc]), vc10));

  _mm512_storeu_pd(&C[0+1*ldc], _mm512_add_pd(_mm512_loadu_pd(&C[0+1*ldc]), vc01));
  _mm512_storeu_pd(&C[8+1*ldc], _mm512_add_pd(_mm512_loadu_pd(&C[8+1*ldc]), vc11));

  _mm512_storeu_pd(&C[0+2*ldc], _mm512_add_pd(_mm512_loadu_pd(&C[0+2*ldc]), vc02));
  _mm512_storeu_pd(&C[8+2*ldc], _mm512_add_pd(_mm512_loadu_pd(&C[8+2*ldc]), vc12));

  _mm512_storeu_pd(&C[0+3*ldc], _mm512_add_pd(_mm512_loadu_pd(&C[0+3*ldc]), vc03));
  _mm512_storeu_pd(&C[8+3*ldc], _mm512_add_pd(_mm512_loadu_pd(&C[8+3*ldc]), vc13));
  
  _mm512_storeu_pd(&C[0+4*ldc], _mm512_add_pd(_mm512_loadu_pd(&C[0+4*ldc]), vc04));
  _mm512_storeu_pd(&C[8+4*ldc], _mm512_add_pd(_mm512_loadu_pd(&C[8+4*ldc]), vc14));
  
  _mm512_storeu_pd(&C[0+5*ldc], _mm512_add_pd(_mm512_loadu_pd(&C[0+5*ldc]), vc05));
  _mm512_storeu_pd(&C[8+5*ldc], _mm512_add_pd(_mm512_loadu_pd(&C[8+5*ldc]), vc15));
  
  _mm512_storeu_pd(&C[0+6*ldc], _mm512_add_pd(_mm512_loadu_pd(&C[0+6*ldc]), vc06));
  _mm512_storeu_pd(&C[8+6*ldc], _mm512_add_pd(_mm512_loadu_pd(&C[8+6*ldc]), vc16));
  
  _mm512_storeu_pd(&C[0+7*ldc], _mm512_add_pd(_mm512_loadu_pd(&C[0+7*ldc]), vc07));
  _mm512_storeu_pd(&C[8+7*ldc], _mm512_add_pd(_mm512_loadu_pd(&C[8+7*ldc]), vc17));

  //printf("after kernel (%ld,%ld,%ld): last of C is %lf\n", v0.x, v0.y, v0.z, C[15+7*ldc]);
  
  return 0;
}

