#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#include <homm.h>


#if 0
#  define REAL float
#  define TYPENAME "float"
#  define USE_FLOAT
#  define GEMM sgemm_
#else
#  define REAL double
#  define TYPENAME "double"
#  define USE_DOUBLE
#  define GEMM dgemm_
#endif

#define FCHAR char *
#define BLASINT int
#define FINT BLASINT *

extern "C" {
void GEMM(FCHAR, FCHAR, FINT, FINT, FINT, \
	      const REAL *, const REAL *, FINT, const REAL *, FINT,	\
	      const REAL *, REAL *, FINT);
};

#define USE_PACK_MAT

#define VERBOSE 10

#include "vec3.h"

int init_algo();
vec3 basesize_double_simd();
int base_double_simd(vec3 v0, vec3 v1, REAL *Am, long lda, REAL *Bm, long ldb, REAL *Cm, long ldc);
int algo(long m, long n, long k, REAL *Am, long lda, REAL *Bm, long ldb, REAL *Cm, long ldc);

#ifdef USE_PACK_MAT
long base_double_packA(REAL *A, long lda, REAL *buf);
long base_double_packB(REAL *B, long ldb, REAL *buf);
long base_double_packC(REAL *C, long ldc, REAL *buf);
long base_double_unpackC(REAL *C, long ldc, REAL *buf);
#endif


/* walltime clock (sync if GPU is used) */
static double Wtime()
{
  struct timeval tv;
#ifdef USE_CUDA
  cudaDeviceSynchronize();
#endif
  gettimeofday(&tv, NULL);
  return (double)tv.tv_sec + (double)tv.tv_usec * 0.000001;
}

#define roundup(i, align) (((size_t)(i)+(size_t)(align)-1)/(size_t)(align)*(size_t)(align))

#define isaligned(i, align) (((size_t)(i) % (size_t)(align)) == 0)

struct global {
  long ndiv;
  vec3 basesize; // preferable base case size

  long bufsize; // size in words
  REAL *buf;

  REAL *Abuf; //packed A
  REAL *Bbuf; //packed B
  REAL *Cbuf; //packed C
  long mb;
  long nb;
  long kb;
};

extern global g;
