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
  return vec3(16, 16, 16);
}

// This kernel can be used both for pivot computation and nonpivot computation
// since L-loop is outermost
int base_gen_float_simd(vec3 v0, vec3 v1, REAL *Am, long lda)
{
  const long m = g.basesize.x;
  const long n = g.basesize.y;
  const long k = g.basesize.z;
  // designed for 16x16x(mul of 4)

#if 1
  const long ldb = lda;
  const long ldc = lda;
  REAL *A = &Am[v0.x + v0.z * lda];
  REAL *B = &Am[v0.z + v0.y * ldb];
  REAL *C = &Am[v0.x + v0.y * ldc];
#endif

  __m512 vc00, vc01, vc02, vc03, vc04, vc05, vc06, vc07;
  __m512 vc08, vc09, vc0a, vc0b, vc0c, vc0d, vc0e, vc0f;
  const REAL infval = 1.0e+8;
#define INITC() (_mm512_set1_ps(infval))
  vc00 = INITC();
  vc01 = INITC();
  vc02 = INITC();
  vc03 = INITC();
  vc04 = INITC();
  vc05 = INITC();
  vc06 = INITC();
  vc07 = INITC();
  vc08 = INITC();
  vc09 = INITC();
  vc0a = INITC();
  vc0b = INITC();
  vc0c = INITC();
  vc0d = INITC();
  vc0e = INITC();
  vc0f = INITC();

#define LOADA_STEP(L0)					\
  va0 = _mm512_loadu_ps(&A[0+(L0)*lda]);


#ifdef USE_TRANSB

#define BELEM(i,j) (B[(j)+(i)*ldb])

#else // !USE_TRANSB

#define BELEM(i,j) (B[(i)+(j)*ldb])

#endif // USE_TRANSB

#define ONE_COL(L0, J0, CNAME) \
  {							\
    __m512 vb;						\
    __m512 vsum;					\
    __mmask16 mask;					\
    vb = _mm512_set1_ps(BELEM(L0, J0));			\
    vsum = _mm512_add_ps(va0, vb);			\
    mask = _mm512_cmp_ps_mask(vsum, CNAME, _CMP_LT_OQ);	\
    CNAME = _mm512_mask_blend_ps(mask, CNAME, vsum);	\
  }	
  

#define ONE_STEP(L0)					\
  ONE_COL(L0, 0, vc00);					\
  ONE_COL(L0, 1, vc01);					\
  ONE_COL(L0, 2, vc02);					\
  ONE_COL(L0, 3, vc03);					\
  ONE_COL(L0, 4, vc04);					\
  ONE_COL(L0, 5, vc05);					\
  ONE_COL(L0, 6, vc06);					\
  ONE_COL(L0, 7, vc07);					\
  ONE_COL(L0, 8, vc08);					\
  ONE_COL(L0, 9, vc09);					\
  ONE_COL(L0, 10, vc0a);				\
  ONE_COL(L0, 11, vc0b);				\
  ONE_COL(L0, 12, vc0c);				\
  ONE_COL(L0, 13, vc0d);				\
  ONE_COL(L0, 14, vc0e);				\
  ONE_COL(L0, 15, vc0f);

  long l;
  for (l = 0; l < k; l += 2) {
    __m512 va0;
    
    LOADA_STEP(l+0);
    ONE_STEP(l+0);
    LOADA_STEP(l+1);
    ONE_STEP(l+1);
  }

#undef LOADA_STEP
#undef ONE_COL
#undef ONE_STEP

#define CUPDATE(I0, J0, CNAME)				\
  {							\
    __m512 v;						\
    __mmask16 mask;					\
    v = _mm512_loadu_ps(&C[(I0)+(J0)*ldc]);		\
    mask = _mm512_cmp_ps_mask(CNAME, v, _CMP_LT_OQ);	\
    v = _mm512_mask_blend_ps(mask, v, CNAME);		\
    _mm512_storeu_ps(&C[(I0)+(J0)*ldc], v);		\
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

#undef CUPDATE

  //printf("after kernel (%ld,%ld,%ld): last of C is %lf\n", v0.x, v0.y, v0.z, C[15+15*ldc]);
  
  return 0;
}

