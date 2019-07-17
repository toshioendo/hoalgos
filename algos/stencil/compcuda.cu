#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/time.h>
#include <assert.h>
#include <cuda_runtime.h>

#include <homm.h>
#include "cos.h"

#ifdef USE_CUDA

__global__ void gpu_kernel(REAL *afrom, REAL *ato, long nx, long ny, long lz0, long lz1)
{
  long ix = blockIdx.x*blockDim.x + threadIdx.x;
  long iy = blockIdx.y*blockDim.y + threadIdx.y;
  if (ix < 1 || ix >= nx+1) {
    return; // do nothing
  }

  if (iy < 1 || iy >= ny+1) {
    return; // do nothing
  }

  long iz;
  for (iz = lz0; iz < lz1; iz++) {
    // update one point
    KERNEL(ix, iy, iz, nx, ny, afrom, ato);
  }
  
  return;
}
    
int update(REAL *afrom, REAL *ato, vec3 v0, vec3 v1)
{
  dim3 bs = dim3(32, 32, 1);
  dim3 gs = dim3(((v1.x-v0.x+2)+(bs.x-1))/bs.x, ((v1.y-v0.y+2)+(bs.y-1))/bs.y);

  gpu_kernel<<<gs, bs>>>(afrom, ato, n3d.x, n3d.y, v0.z, v1.z);
#if 0
  cudaError_t crc = cudaDeviceSynchronize();
  if (crc != cudaSuccess) {
    fprintf(stderr, "Error in update(): afrom=%p, ato=%p, z=[%ld,%ld)\n",
	    afrom, ato, v0.z, v1.z);
  }
  CERR(crc);
#endif
  return 0;
}

#endif // GPU
