// Hierarchy Oblivious Memory Management header
// (Single address space version)

// following symbols should be predefined if necessary
//   USE_OMP
//   USE_CUDA

#ifndef __HOMM_H
#define __HOMM_H

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#ifdef USE_CUDA
#  include <cuda_runtime.h>
#endif

#ifdef USE_OMP
#  include <omp.h>
#endif

#ifndef HOMM_VERBOSE
#define HOMM_VERBOSE 10
#endif

#define piadd(p, i) (void*)((char*)(p) + (size_t)(i))
#define ppsub(p1, p2) (size_t)((char*)(p1) - (char*)(p2))

#define roundup(i, align) (((size_t)(i)+(size_t)(align)-1)/(size_t)(align)*(size_t)(align))

#ifndef USE_CUDA

inline int homm_init()
{
  return 0;
}

inline int homm_gfree(void *gp)
{
  free(gp);
  return 0;
}

inline void *homm_galloc(size_t size)
{
  return malloc(size);
}

inline int homm_finish()
{
  return 0;
}

#else // USE_CUDA

#define CERR(cstat)	    \
  {								\
    cudaError_t crc0 = cstat;						\
    if ((crc0) != cudaSuccess) {					\
      fprintf(stderr, "[mm(%s:%d)] CUDA ERROR %d\n", __FILE__, __LINE__, (crc0)); \
      exit(1);								\
    }									\
  }

inline int homm_init()
{
#ifdef USE_OMP // if both USE_CUDA/USE_OMP are defined, muti-GPU are used
  cudaError_t crc;
  /* one GPU per OMP thread */
  char *envstr;
  int ngpu;
  crc = cudaGetDeviceCount(&ngpu);
  CERR(crc);
  
  int nt = -1;
  envstr = getenv("OMP_NUM_THREADS");
  if (envstr != NULL) nt = atoi(envstr);
  
  if (nt < 0 || nt > ngpu) nt = ngpu;
#if HOMM_VERBOSE >= 10
  printf("[mm_init] %d GPUs (out of %d) are used\n", nt, ngpu);
#endif
  
  assert(ngpu >= 0);
  omp_set_num_threads(nt);
  
#pragma omp parallel
  {
    int tid;
    tid = omp_get_thread_num();
    assert(tid >= 0 && tid < nt);
    crc = cudaSetDevice(tid);
    CERR(crc);
#if HOMM_VERBOSE >= 10
    printf("[mm_init] start to use GPU %d\n", tid);
#endif
  }
#endif // USE_OMP
  return 0;
}  
    
inline int homm_gfree(void *gp)
{
  cudaError_t crc;
  crc = cudaFree(gp);
  CERR(crc);
  return 0;
}
 
inline void *homm_galloc(size_t size)
{
  void *gp;
  cudaError_t crc;
  crc = cudaMallocManaged(&gp, size, cudaMemAttachGlobal);
  CERR(crc);
  return gp;
}

inline int homm_finish()
{
  return 0;
}

#endif // USE_CUDA

#endif // __HOMM_H
