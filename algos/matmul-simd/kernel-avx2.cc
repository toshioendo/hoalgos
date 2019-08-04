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
  return vec3(8, 4, 64);
}

int base_double_simd(vec3 v0, vec3 v1, REAL *Am, long lda, REAL *Bm, long ldb, REAL *Cm, long ldc)
{
  int k = g.basesize.z;
  // designed for 8x4x(mul of 4)

  REAL *A = &Am[v0.x + v0.z * lda];
  REAL *B = &Bm[v0.z + v0.y * ldb];
  REAL *C = &Cm[v0.x + v0.y * lda];

  __m256d vc00, vc01, vc02, vc03;
  __m256d vc10, vc11, vc12, vc13;
  vc00 = _mm256_setzero_pd();
  vc01 = _mm256_setzero_pd();
  vc02 = _mm256_setzero_pd();
  vc03 = _mm256_setzero_pd();
  vc10 = _mm256_setzero_pd();
  vc11 = _mm256_setzero_pd();
  vc12 = _mm256_setzero_pd();
  vc13 = _mm256_setzero_pd();

  if (k % 4 != 0) {
    fprintf(stderr, "ERROR: k=%d is invalid\n", k);
    exit(1);
  }

  int l;
  for (l = 0; l < k; l += 4) {
    __m256d va0, va1;
    __m256d vb;
    // broadcast based
#define ONE_STEP(K0)				\
    va0 = _mm256_load_pd(&A[0+(l+K0)*lda]);	\
    va1 = _mm256_load_pd(&A[4+(l+K0)*lda]);	\
    vb = _mm256_set1_pd(B[(l+K0)+0*ldb]);	\
    vc00 = _mm256_fmadd_pd(va0, vb, vc00);		\
    vc10 = _mm256_fmadd_pd(va1, vb, vc10);		\
    						\
    vb = _mm256_set1_pd(B[(l+K0)+1*ldb]);	\
    vc01 = _mm256_fmadd_pd(va0, vb, vc01);		\
    vc11 = _mm256_fmadd_pd(va1, vb, vc11);		\
    						\
    vb = _mm256_set1_pd(B[(l+K0)+2*ldb]);	\
    vc02 = _mm256_fmadd_pd(va0, vb, vc02);		\
    vc02 = _mm256_fmadd_pd(va1, vb, vc12);		\
    						\
    vb = _mm256_set1_pd(B[(l+K0)+3*ldb]);	\
    vc03 = _mm256_fmadd_pd(va0, vb, vc03);	\
    vc13 = _mm256_fmadd_pd(va1, vb, vc13);

    ONE_STEP(0);
    ONE_STEP(1);
    ONE_STEP(2);
    ONE_STEP(3);

#undef ONE_STEP
  }

  _mm256_store_pd(&C[0+0*ldc], _mm256_add_pd(_mm256_load_pd(&C[0+0*ldc]), vc00));
  _mm256_store_pd(&C[0+1*ldc], _mm256_add_pd(_mm256_load_pd(&C[0+1*ldc]), vc01));
  _mm256_store_pd(&C[0+2*ldc], _mm256_add_pd(_mm256_load_pd(&C[0+2*ldc]), vc02));
  _mm256_store_pd(&C[0+3*ldc], _mm256_add_pd(_mm256_load_pd(&C[0+3*ldc]), vc03));

  _mm256_store_pd(&C[4+0*ldc], _mm256_add_pd(_mm256_load_pd(&C[4+0*ldc]), vc10));
  _mm256_store_pd(&C[4+1*ldc], _mm256_add_pd(_mm256_load_pd(&C[4+1*ldc]), vc11));
  _mm256_store_pd(&C[4+2*ldc], _mm256_add_pd(_mm256_load_pd(&C[4+2*ldc]), vc12));
  _mm256_store_pd(&C[4+3*ldc], _mm256_add_pd(_mm256_load_pd(&C[4+3*ldc]), vc13));

  return 0;
}

