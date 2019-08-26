#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/time.h>
#include <immintrin.h>
#include <assert.h>

#include <homm.h>
#include "apsp.h"

#ifdef USE_COALESCED_KERNEL
#define BLOCK_MAG 2
#else
#define BLOCK_MAG 1
#endif

vec3 basesize_float_simd()
{
  return vec3(16*BLOCK_MAG, 16*BLOCK_MAG, 16*BLOCK_MAG);
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




// definitions used both for pivot/nonpivot kernels
#define ONE_COL(L0, J0, CNAME) \
  {							\
    __m512 vb;						\
    __m512 vsum;					\
    __mmask16 mask;					\
    vb = _mm512_set1_ps(B[(L0)+(J0)*ldb]);		\
    vsum = _mm512_add_ps(va0, vb);			\
    mask = _mm512_cmp_ps_mask(vsum, CNAME, _CMP_LT_OQ);	\
    CNAME = _mm512_mask_blend_ps(mask, CNAME, vsum);	\
  }	
  
#define ONE_STEP(L0)					\
  {							\
    __m512 va0;						\
    va0 = _mm512_loadu_ps(&A[0+(L0)*lda]);		\
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


// This kernel is for nonpivot computation 
// for size [16,16,k]
int base_s_nonpivot_float_simd(vec3 v0, vec3 v1, REAL *Am, long lda)
{
  const long m = v1.x-v0.x;
  const long n = v1.y-v0.y;
  const long k = v1.z-v0.z;
  // designed for 16x16x(mul of 4)

  long ldb, ldc;
  long ib, jb, lb;
  REAL *A, *B, *C;
  
  if (g.use_pack_mat) {
    // overwrite lda, ldb, ldc
    lda = 16*BLOCK_MAG;
    ldb = 16*BLOCK_MAG;
    ldc = 16*BLOCK_MAG;
    
    // block idx
    ib = v0.x/lda;
    jb = v0.y/lda;
    lb = v0.z/lda;
  
    // index in a block (nonzero with USE_COALESCED_KERNEL
    const long ii = v0.x % lda;
    const long jj = v0.y % lda;
    const long ll = v0.z % lda;
#ifndef USE_COALESCED_KERNEL
    assert(ii == 0 && jj == 0 && ll == 0);
#endif
    const long bs = lda*lda;

    A = &g.Abuf[(ib+lb*g.nb)*bs + (ii+ll*lda)];
    B = &g.Abuf[(lb+jb*g.nb)*bs + (ll+jj*lda)];
    C = &g.Abuf[(ib+jb*g.nb)*bs + (ii+jj*lda)];
  }
  else {
    ldb = lda;
    ldc = lda;
    A = &Am[v0.x + v0.z * lda];
    B = &Am[v0.z + v0.y * lda];
    C = &Am[v0.x + v0.y * lda];
  }

  __m512 vc00, vc01, vc02, vc03, vc04, vc05, vc06, vc07;
  __m512 vc08, vc09, vc0a, vc0b, vc0c, vc0d, vc0e, vc0f;
  const REAL infval = 1.0e+8;
#define CCLEAR() (_mm512_set1_ps(infval))

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

  long l;
  for (l = 0; l < k; l += 2) {
    ONE_STEP(l+0);
    ONE_STEP(l+1);
  }

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

// for every k, results must be updated on array
int base_s_pivot_float_simd(vec3 v0, vec3 v1, REAL *Am, long lda)
{
  const long m = v1.x-v0.x;
  const long n = v1.y-v0.y;
  const long k = v1.z-v0.z;
  // designed for 16x16x(mul of 4)

  long ldb, ldc;
  long ib, jb, lb;
  REAL *A, *B, *C;
  
  if (g.use_pack_mat) {
    // overwrite lda, ldb, ldc
    lda = 16*BLOCK_MAG;
    ldb = 16*BLOCK_MAG;
    ldc = 16*BLOCK_MAG;
    
    // block idx
    ib = v0.x/lda;
    jb = v0.y/lda;
    lb = v0.z/lda;
  
    // index in a block (nonzero with USE_COALESCED_KERNEL
    const long ii = v0.x % lda;
    const long jj = v0.y % lda;
    const long ll = v0.z % lda;
#ifndef USE_COALESCED_KERNEL
    assert(ii == 0 && jj == 0 && ll == 0);
#endif
    const long bs = lda*lda;

    A = &g.Abuf[(ib+lb*g.nb)*bs + (ii+ll*lda)];
    B = &g.Abuf[(lb+jb*g.nb)*bs + (ll+jj*lda)];
    C = &g.Abuf[(ib+jb*g.nb)*bs + (ii+jj*lda)];
  }
  else {
    ldb = lda;
    ldc = lda;
    A = &Am[v0.x + v0.z * lda];
    B = &Am[v0.z + v0.y * lda];
    C = &Am[v0.x + v0.y * lda];
  }
    
  __m512 vc00, vc01, vc02, vc03, vc04, vc05, vc06, vc07;
  __m512 vc08, vc09, vc0a, vc0b, vc0c, vc0d, vc0e, vc0f;
  const REAL infval = 1.0e+8;
  //#define CCLEAR() (_mm512_set1_ps(infval))
#define CREAD(I0, J0) (_mm512_loadu_ps(&C[(I0)+(J0)*ldc]))

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

#define CWRITE(I0, J0, CNAME)				\
  _mm512_storeu_ps(&C[(I0)+(J0)*ldc], CNAME)		\

  long l;
  for (l = 0; l < k; l ++) {
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
  }


#undef CREAD
#undef CWRITE

  //printf("after kernel (%ld,%ld,%ld): last of C is %lf\n", v0.x, v0.y, v0.z, C[15+15*ldc]);
  
  return 0;
}

#undef ONE_COL
#undef ONE_STEP


//////////////////////////////////////////
int base_pivot_float_simd(vec3 v0, vec3 v1, REAL *Am, long lda)
{
  assert(vec3eq(g.basesize, vec3sub(v1, v0)));
  long chunklen = 16;
  long x0 = v0.x;
  long x1 = v1.x;
  long y0 = v0.y;
  long y1 = v1.y;
  long z0 = v0.z;
  long z1 = v1.z;

  long xm = x0+chunklen;
  long ym = y0+chunklen;
  long zm = z0+chunklen;

  // divide task into 8
  // 1
  base_s_pivot_float_simd(vec3(x0, y0, z0), vec3(xm, ym, zm), Am, lda);
  // 2
  base_s_pivot_float_simd(vec3(xm, y0, z0), vec3(x1, ym, zm), Am, lda);
  // 3
  base_s_pivot_float_simd(vec3(x0, ym, z0), vec3(xm, y1, zm), Am, lda);
  // 4
  base_s_nonpivot_float_simd(vec3(xm, ym, z0), vec3(x1, y1, zm), Am, lda);
  // 5
  base_s_pivot_float_simd(vec3(xm, ym, zm), vec3(x1, y1, z1), Am, lda);
  // 6
  base_s_pivot_float_simd(vec3(x0, ym, zm), vec3(xm, y1, z1), Am, lda);
  // 7
  base_s_pivot_float_simd(vec3(xm, y0, zm), vec3(x1, ym, z1), Am, lda);
  // 8
  base_s_nonpivot_float_simd(vec3(x0, y0, zm), vec3(xm, ym, z1), Am, lda);
  return 0;
}

int base_nonpivot_float_simd(vec3 v0, vec3 v1, REAL *Am, long lda)
{
  assert(vec3eq(g.basesize, vec3sub(v1, v0)));
  long chunklen = 16;
  long x0 = v0.x;
  long x1 = v1.x;
  long y0 = v0.y;
  long y1 = v1.y;
  long z0 = v0.z;
  long z1 = v1.z;

  long xm = x0+chunklen;
  long ym = y0+chunklen;

  // divide task into 4 (as large as possible)
  // 1
  base_s_nonpivot_float_simd(vec3(x0, y0, z0), vec3(xm, ym, z1), Am, lda);
  // 2
  base_s_nonpivot_float_simd(vec3(xm, y0, z0), vec3(x1, ym, z1), Am, lda);
  // 3
  base_s_nonpivot_float_simd(vec3(x0, ym, z0), vec3(xm, y1, z1), Am, lda);
  // 4
  base_s_nonpivot_float_simd(vec3(xm, ym, z0), vec3(x1, y1, z1), Am, lda);
  return 0;
}


