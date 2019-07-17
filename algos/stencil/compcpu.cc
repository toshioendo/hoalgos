#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/time.h>
#include <assert.h>

#include <homm.h>
#include "stencil.h"


int update(REAL *afrom, REAL *ato, vec3 v0, vec3 v1/*long lz0, long lz1*/)
{
#if defined USE_OMP && !defined USE_OMPTASK
#pragma omp parallel for collapse(2)
#endif
  for (long iz = v0.z; iz < v1.z; iz++) {
    for (long iy = v0.y; iy < v1.y; iy++) {
      for (long ix = v0.x; ix < v1.x; ix++) {
	KERNEL(ix, iy, iz, n3d.x, n3d.y, afrom, ato);
      }
    }
  }

  return 0;
}

