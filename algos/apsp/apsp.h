#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#include <homm.h>


#if 1
#  define REAL float
#  define TYPENAME "float"
#  define USE_FLOAT
#else
#  error NOT IMPLEMENTED YET
#  define REAL double
#  define TYPENAME "double"
#  define USE_DOUBLE
#endif

#define FCHAR char *
#define BLASINT int
#define FINT BLASINT *

extern "C" {
void GEMM(FCHAR, FCHAR, FINT, FINT, FINT, \
	      const REAL *, const REAL *, FINT, const REAL *, FINT,	\
	      const REAL *, REAL *, FINT);
};

// kernel configuration
#define KERNEL_MAG 4 //2
#define USE_SECOND_KERNEL

#define VERBOSE 10

#include "vec3.h"

vec3 basesize_float_simd();

int base_float_simd(bool onpivot, vec3 v0, vec3 v1);

// for pack_mat
long base_float_packA(REAL *A, long lda, REAL *buf);
long base_float_unpackA(REAL *A, long lda, REAL *buf);

// algo.cc
int init_algo(int *argcp, char ***argvp);
int algo(long n, REAL *Am, long lda);



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
  vec3 basesize; // preferable base case size
  long task_thre;
  bool use_recursive;
  bool use_pack_mat;

  // buffer information allocated by user
  // valid during APSP computation
  REAL *Amat;
  long lda;
  long n;

  // buffer allocated by APSP library
  // valid if use_pack_mat
  long bufsize; // size in words
  REAL *buf;
  REAL *Abuf; //packed A
  long nb;
};

extern global g;
