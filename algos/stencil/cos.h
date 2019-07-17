#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

#ifdef USE_OMP
#include <omp.h>
#endif

#include "vec3.h"

#if 1
#  define REAL float
#else
#  define REAL double
#endif

//#define VERBOSE 30
#define VERBOSE 10
//#define VERBOSE 0

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

// to be defined in compXXX.cX
int update(REAL *afrom, REAL *ato, vec3 v0, vec3 v1);

// defined in algo.cc
int algo(long t0, long t1, vec3 c0, vec3 c1);


extern vec3 n3d; // world size
extern long nt;

extern long bt; // temporal blocking factor
extern int recflag; // 1 if recursive algorithm is used
extern int paraflag; // 1 if parallelogram block is used. 0 means trapezoid only
extern int divdims; // 1: div Z, 2: div Y, Z
extern long stopsizemb;

extern REAL *arrays[2];

extern double logtime;

#define IDX(ix, iy, iz, nx, ny)			\
  ((long)(ix)+(iy)*((nx)+2)+(iz)*((nx)+2)*(((ny)+2)))

#define KERNEL(ix, iy, iz, nx, ny, afrom, ato) {	\
  long idx = IDX(ix, iy, iz, nx, ny);			\
  long ox = 1;						\
  long oy = (nx)+2;					\
  long oz = ((nx)+2)*((ny)+2);				\
  ato[idx] =  \
    (REAL)0.1 * (afrom[idx-ox] +			\
		 afrom[idx+ox] +			\
		 afrom[idx-oy] +			\
		 afrom[idx+oy] +			\
		 afrom[idx-oz] +			\
		 afrom[idx+oz]) +			\
    (REAL)0.4 * afrom[idx]; \
}
