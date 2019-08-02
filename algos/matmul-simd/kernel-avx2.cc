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

int base_double_simd(vec3 v0, vec3 v1)
{
  int k = g.basesize.z;
  // designed for 8x4x(mul of 4)
  long lda = g.size.x; // m
  long ldb = g.size.z; // k
  long ldc = g.size.x; // m

  REAL *A = &g.A[v0.x + v0.z * lda];
  REAL *B = &g.B[v0.z + v0.y * ldb];
  REAL *C = &g.C[v0.x + v0.y * lda];

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
    __m256d vb0, vb1, vb2, vb3;
    vb0 = _mm256_load_pd(&B[l+0*ldb]);
    vb1 = _mm256_load_pd(&B[l+1*ldb]);
    vb2 = _mm256_load_pd(&B[l+2*ldb]);
    vb3 = _mm256_load_pd(&B[l+3*ldb]);
    
    const int factor = (1 << 6) | (1 << 4) | (1 << 2) | (1 << 0); // for broadcast by permute
    
#define ONE_STEP(K0)				\
    va0 = _mm256_load_pd(&A[0+(l+K0)*lda]);	\
    va1 = _mm256_load_pd(&A[4+(l+K0)*lda]);	\
    vb = _mm256_permute4x64_pd(vb0, K0*factor);	\
    vc00 = _mm256_fmadd_pd(va0, vb, vc00);		\
    vc10 = _mm256_fmadd_pd(va1, vb, vc10);		\
    						\
    vb = _mm256_permute4x64_pd(vb1, K0*factor);	\
    vc01 = _mm256_fmadd_pd(va0, vb, vc01);		\
    vc11 = _mm256_fmadd_pd(va1, vb, vc11);		\
    						\
    vb = _mm256_permute4x64_pd(vb2, K0*factor);	\
    vc02 = _mm256_fmadd_pd(va0, vb, vc02);		\
    vc02 = _mm256_fmadd_pd(va1, vb, vc12);		\
    						\
    vb = _mm256_permute4x64_pd(vb3, K0*factor);	\
    vc03 = _mm256_fmadd_pd(va0, vb, vc03);	\
    vc13 = _mm256_fmadd_pd(va1, vb, vc13);

    ONE_STEP(0);
    ONE_STEP(1);
    ONE_STEP(2);
    ONE_STEP(3);

#undef ONE_STEP
  }

  __m256d vr0, vr1, vr2, vr3;
  vr0 = _mm256_load_pd(&C[0+0*ldc]);
  vr1 = _mm256_load_pd(&C[0+1*ldc]);
  vr2 = _mm256_load_pd(&C[0+2*ldc]);
  vr3 = _mm256_load_pd(&C[0+3*ldc]);

  vr0 = _mm256_add_pd(vr0, vc00);
  vr1 = _mm256_add_pd(vr1, vc01);
  vr2 = _mm256_add_pd(vr2, vc02);
  vr3 = _mm256_add_pd(vr3, vc03);

  _mm256_store_pd(&C[0+0*ldc], vr0);
  _mm256_store_pd(&C[0+1*ldc], vr1);
  _mm256_store_pd(&C[0+2*ldc], vr2);
  _mm256_store_pd(&C[0+3*ldc], vr3);

  vr0 = _mm256_load_pd(&C[4+0*ldc]);
  vr1 = _mm256_load_pd(&C[4+1*ldc]);
  vr2 = _mm256_load_pd(&C[4+2*ldc]);
  vr3 = _mm256_load_pd(&C[4+3*ldc]);

  vr0 = _mm256_add_pd(vr0, vc00);
  vr1 = _mm256_add_pd(vr1, vc01);
  vr2 = _mm256_add_pd(vr2, vc02);
  vr3 = _mm256_add_pd(vr3, vc03);

  _mm256_store_pd(&C[4+0*ldc], vr0);
  _mm256_store_pd(&C[4+1*ldc], vr1);
  _mm256_store_pd(&C[4+2*ldc], vr2);
  _mm256_store_pd(&C[4+3*ldc], vr3);

  return 0;
}

