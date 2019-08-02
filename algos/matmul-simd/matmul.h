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

#define VERBOSE 10

#include "vec3.h"

vec3 basesize_double_simd();
int base_double_simd(vec3 v0, vec3 v1);

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

struct global {
  vec3 size;
  long stopsizemb;
  REAL *A;
  REAL *B;
  REAL *C;

  vec3 basesize; // preferable base case size

  double logtime;
};

extern global g;
