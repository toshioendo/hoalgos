// hierarchy oblivious 7p-stencil
//   single address space version
// main function
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/time.h>
#include <assert.h>

#include <homm.h>
#include "cos.h"

// 3D (x,y,z)
vec3 n3d;
long nt;
long bt; // temporal blocking factor
int recflag; // 1 if recursive algorithm is used
int paraflag; // 1 if parallelogram block is used. 0 means trapezoid only
int divdims; // 1: div Z, 2: div Y&Z

long stopsizemb;
REAL *arrays[2];

double logtime = 0.0;

int simple(long t0, long t1, long z0, long z1)
{
  for (long it = t0; it < t1; it++) {
    int from = it%2;
    int to = (it+1)%2;
#pragma omp parallel for
    for (long iz = z0; iz < z1; iz++) {
      for (long iy = 1; iy < n3d.y+1; iy++) {
	for (long ix = 1; ix < n3d.x+1; ix++) {
	  KERNEL(ix, iy, iz, n3d.x, n3d.y, arrays[from], arrays[to]);
	}
      }
    }
#if VERBOSE >= 10
    // print periodically
    double wt = Wtime();
    if (wt > logtime+1.0) {
      printf("  t=%d done\n", it);
      logtime = wt;
    }
#endif
  }
  return 0;
}

int init()
{
  int i;
  size_t size;
  size = sizeof(REAL)*(n3d.x+2)*(n3d.y+2)*(n3d.z+2);
  for (i = 0; i < 2; i++) {
    arrays[i] = (REAL *)homm_galloc(size);
  }

#if VERBOSE >= 10
  printf("[init] allocated %ldMiB x 2 = %ldMiB\n",
	 size >> 20L, 2*size >> 20L);
#endif

  /* initial values */
  double st = Wtime(), et;

  long nsum = (n3d.x+n3d.y+n3d.z);
#pragma omp parallel for
  for (long iz = 0; iz < n3d.z+2; iz++) {
    for (long iy = 0; iy < n3d.y+2; iy++) {
      for (long ix = 0; ix < n3d.x+2; ix++) {
	REAL v = 0.0;
	int i;

	if ((ix+iy+iz) < nsum/8) v = 1.0;
	else if ((ix+iy+iz) >= nsum*7/8) v = -1.0;
	long idx = IDX(ix, iy, iz, n3d.x, n3d.y);
	for (i = 0; i < 2; i++) {
	  arrays[i][idx] = v;
	}
      }
    }
  }

  et = Wtime();
#if VERBOSE >= 10
  printf("[init] array initialization took %.3lfsec\n", et-st);
#endif

  return 0;
}



int main(int argc, char *argv[])
{

  /* default setting option */
  bt = -1;
  nt = 512;
  n3d = vec3(256, 256, 256);
  stopsizemb = 16; //128;
  recflag = 1;
  paraflag = 0;
  divdims = 1;
  
  while (argc >= 2 && argv[1][0] == '-') {
    if (strcmp(argv[1], "-bt") == 0) {
      bt = atoi(argv[2]);
      assert(bt >= 1);
      recflag = 0; // if bt is specified, we do not use recursive algorithm
      argc -= 2;
      argv += 2;
    }
    else if (strcmp(argv[1], "-nt") == 0) {
      nt = atoi(argv[2]);
      argc -= 2;
      argv += 2;
    }
    else if (strcmp(argv[1], "-ss") == 0) {
      stopsizemb = atol(argv[2]);
      argc -= 2;
      argv += 2;
    }
    else if (strcmp(argv[1], "-para") == 0) {
      paraflag = 1;
      argc -= 1;
      argv += 1;
    }
    else if (strcmp(argv[1], "-d0d") == 0) {
      divdims = 0;
      argc -= 1;
      argv += 1;
    }
    else if (strcmp(argv[1], "-d1d") == 0) {
      divdims = 1;
      argc -= 1;
      argv += 1;
    }
    else if (strcmp(argv[1], "-d2d") == 0) {
      divdims = 2;
      argc -= 1;
      argv += 1;
    }
    else break;
  }
  
  if (argc >= 3) {
    n3d = vec3(atol(argv[1]), atol(argv[2]), atol(argv[3]));
  }

  printf("Cache Oblivious 3D-Stencil sample (Single Address Space version)\n");
  char use_cuda = 'N';
#ifdef USE_CUDA
  use_cuda = 'Y';
#endif
  char use_omp = 'N';
#ifdef USE_OMP
  use_omp = 'Y';
#endif
  char use_omptask = 'N';
#ifdef USE_OMPTASK
  use_omptask = 'Y';
#endif

  printf("  Compile time options: USE_CUDA %c, USE_OMP %c, USE_OMPTASK %c\n",
	 use_cuda, use_omp, use_omptask);
  printf("nx=%ld, ny=%ld, nz=%ld, nt=%d, rec %s, bt=%d, para=%s, divdims=%d, stopsize=%ldMiB\n",
	 n3d.x, n3d.y, n3d.z, nt, (recflag)?"USED":"UNUSED", bt, (paraflag)?"USED":"UNUSED",
	 divdims, stopsizemb);

  homm_init();

  init();

  // main computation
  {
    double st, et;
    double nupd = (double)n3d.x*n3d.y*n3d.z*nt;
    st = Wtime();

    algo(0, nt, vec3(1,1,1), vec3(n3d.x+1, n3d.y+1, n3d.z+1));

    //simple(0, nt, 1, nz+1);
    et = Wtime();

    printf("%.3lf sec -> %lf Mupd/s\n",
	   (et-st), nupd/(et-st)/1000000.0);
  }

  homm_finish();

#if 0
  // test print
#define DIST2(x0,y0,z0,x1,y1,z1) \
  (((x0)-(x1))*((x0)-(x1))+((y0)-(y1))*((y0)-(y1))+((z0)-(z1))*((z0)-(z1)))
  {
    int to = (nt%2);
    // answers are at arrays[to]
    long ix, iy, iz;
    long rad2 = 2*2;
    for (iz = 0; iz < nz+2; iz++) {
      for (iy = 0; iy < ny+2; iy++) {
	for (ix = 0; ix < nx+2; ix++) {
	  if (DIST2(ix,iy,iz,0,0,0) < rad2 ||
	      DIST2(ix,iy,iz,nx+1,ny+1,nz+1) < rad2 ||
	      DIST2(ix,iy,iz,nx/2,ny/2,nz/2) < rad2) {
	    printf("a[%ld][%ld][%ld]=%f ", iz, iy, ix, (float)arrays[to][IDX(ix,iy,iz,nx,ny)]);
	  }
	}
      }
    }
    printf("\n");
  }
#endif

}
