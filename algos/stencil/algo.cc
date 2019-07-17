// cache oblivious 7p-stencil 
// recusvie algorithm
// Z-dimension is divided
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/time.h>
#include <assert.h>

#include <homm.h>
#include "stencil.h"

vec3 adjustindexW(long t0, long t1, vec3 vt1, vec3 rv)
{
  // vector at time t0
  // compute vt1 - (t1-t0)*rv
  vec3 vt0(vt1.x - (t1-t0)*rv.x,
	   vt1.y - (t1-t0)*rv.y,
	   vt1.z - (t1-t0)*rv.z);

  // adjust vector elements within [(1,1,1), (nx+1,ny+1,nz+1))

  if (vt0.x < 1) vt0.x = 1;
  if (vt0.y < 1) vt0.y = 1;
  if (vt0.z < 1) vt0.z = 1;

  if (vt0.x > n3d.x+1) vt0.x = n3d.x+1;
  if (vt0.y > n3d.y+1) vt0.y = n3d.y+1;
  if (vt0.z > n3d.z+1) vt0.z = n3d.z+1;

  return vt0;
}

vec3 adjustindexR(long t0, long t1, vec3 vt1, vec3 rv)
{
  // vector at time t0
  // compute vt1 - (t1-t0)*rv
  vec3 vt0(vt1.x - (t1-t0)*rv.x,
	   vt1.y - (t1-t0)*rv.y,
	   vt1.z - (t1-t0)*rv.z);

  // adjust vector elements within [(0,0,0), (nx+2,ny+2,nz+2))

  if (vt0.x < 0) vt0.x = 0;
  if (vt0.y < 0) vt0.y = 0;
  if (vt0.z < 0) vt0.z = 0;

  if (vt0.x > n3d.x+2) vt0.x = n3d.x+2;
  if (vt0.y > n3d.y+2) vt0.y = n3d.y+2;
  if (vt0.z > n3d.z+2) vt0.z = n3d.z+2;

  return vt0;
}

int getAccessRegion(long t0, long t1, vec3 v0, vec3 v1, vec3 vz0, vec3 vz1,
		    vec3 *pv0z, vec3 *pv1z)
{
  vec3 v0z = vec3(v0.x-1, v0.y-1, v0.z-1);
  vec3 v1z = vec3(v1.x+1, v1.y+1, v1.z+1);

  v0z = vec3min(v0z, adjustindexR(t0, t1, v0, vz0)); // lower index of footprint
  v1z = vec3max(v1z, adjustindexR(t0, t1, v1, vz1)); // upper index of footprint

  *pv0z = v0z;
  *pv1z = v1z;
  return 0;
}
    
int base(long t0, long t1, vec3 v0, vec3 v1, vec3 rv0, vec3 rv1)
{
  double st, et;
  // base case
#if VERBOSE >= 20
  printf("[base] START t=[%d,%d], [(%ld,%ld,%ld), (%ld,%ld,%ld))\n", t0, t1,
	 v0.x, v0.y, v0.z, v1.x, v1.y, v1.z);
#endif

  vec3 v0z, v1z;
  getAccessRegion(t0, t1, v0, v1, rv0, rv1, &v0z, &v1z);
  size_t accsize = vec3sub(v1z, v0z).vol() * sizeof(REAL);
#if VERBOSE >= 30
  printf("[base]   access region is [(%ld,%ld,%ld), (%ld,%ld,%ld)) -> %ldMiB\n",
	 v0z.x, v0z.y, v0z.z, v1z.x, v1z.y, v1z.z,
	 accsize>>20L);
#endif



  REAL *darrays[2];
  {
    int i;
    for (i = 0; i < 2; i++) {
      darrays[i] = &arrays[i][IDX(v0z.x, v0z.y, v0z.z, n3d.x, n3d.y)];
    }
  }


  st = Wtime();
  long it;
  for (it = t0; it < t1; it++) {
    vec3 v0a = adjustindexW(it+1, t1, v0, rv0); // lower index
    vec3 v1a = adjustindexW(it+1, t1, v1, rv1); // upper index

#if VERBOSE >= 40
    printf("        compute t=%d->%d, [(%d,%d,%d),(%d,%d,%d))\n", it, it+1, 
	   v0a.x, v0a.y, v0a.z, v1a.x, v1a.y, v1a.z);
#endif
    int from = it%2;
    int to = (it+1)%2;
    
    assert(v0z.z <= v0a.z);
    assert(v1a.z <= v1z.z);
    // call arch specific update funtion
    update(darrays[from], darrays[to], vec3sub(v0a, v0z), vec3sub(v1a, v0z));
  }

  // print periodically
  et = Wtime();
  double wt = et;
#if VERBOSE >= 10
#if VERBOSE >= 30
  if (1)
#else
  if (wt > logtime+1.0)
#endif
    {
      int tid = 0;
      int nt = 1;
#ifdef USE_OMP
      tid = omp_get_thread_num();
      nt = omp_get_num_threads();
#endif
      printf("[base] END@thr%d/%d: t=[%d,%d], [(%ld,%ld,%ld), (%ld,%ld,%ld)) -> %.3lfsec\n", 
	     tid, nt, t0, t1,
	     v0.x, v0.y, v0.z, v1.x, v1.y, v1.z, et-st);
      logtime = wt;
    }
#endif

  return 0;
}

int recTB(long t0, long t1, vec3 v0, vec3 v1, vec3 rv0, vec3 rv1);

int timecut(long t0, long t1, long tm, vec3 v0, vec3 v1, vec3 rv0, vec3 rv1)
{
  // time cut
  if (t1-t0 <= 1) {
    base(t0, t1, v0, v1, rv0, rv1);
    return 0;
  }

#if VERBOSE >= 30
  printf(" --> time cut [%d,%d) at %d\n", t0, t1, tm);
#endif

  // divide point
  vec3 v0m = adjustindexW(tm, t1, v0, rv0);
  vec3 v1m = adjustindexW(tm, t1, v1, rv1);

  // first half
  recTB(t0, tm, v0m, v1m, rv0, rv1);
  // later half
  recTB(tm, t1, v0, v1, rv0, rv1);
  return 0;
}

// returns <0 if we cannot space cut
int canSpacecutDim(long t0, long t1, vec3 v0, vec3 v1, vec3 rv0, vec3 rv1, char dim)
{
  long dt = t1-t0;
  long z0 = v0.get(dim);
  long z1 = v1.get(dim);
  long dz = z1-z0;
  if (paraflag) {
    return (dz > dt/2);
  }
  else {
    assert(rv0.get(dim) * rv1.get(dim) < 0);
    if (rv0.get(dim) > 0) {  //  /~~~~\ block
      if (dz >= 2*dt+3) return (int)dim;
      else return -1;
    }
    else {  //  \___/ block
      if (dz >= 4*dt+3) return (int)dim;
      else return -1;
    }
  }
}

int canSpacecut(long t0, long t1, vec3 v0, vec3 v1, vec3 rv0, vec3 rv1)
{
  int rc;
  if (divdims == 0) {
    // no space cut
    return -1;
  }
  else if (divdims == 1) {
    rc = canSpacecutDim(t0, t1, v0, v1, rv0, rv1, 'Z');
  }
  else if (divdims == 2) {
    // Y or Z
    int ylarger = ((v1.y-v0.y) > (v1.z-v0.z));
    if (ylarger) {
      rc = canSpacecutDim(t0, t1, v0, v1, rv0, rv1, 'Y');
    }
    else {
      rc = canSpacecutDim(t0, t1, v0, v1, rv0, rv1, 'Z');
    }
  }
  return rc;
}

int spacecutDim(long t0, long t1, vec3 v0, vec3 v1, vec3 rv0, vec3 rv1, char dim)
{
  long dt = t1-t0;
  long z0 = v0.get(dim);
  long z1 = v1.get(dim);
  long dz = z1-z0;

  if (canSpacecut(t0, t1, v0, v1, rv0, rv1) < 0) {
    fprintf(stderr, "ERROR: Cannot spacecut, algorithm failed. t=[%d,%d], z=[%d,%d)\n",
	    t0, t1, z0, z1);
    exit(1);
  }


  if (paraflag) {
    long zm = (z0+z1)/2;
#if VERBOSE >= 30
    printf("  --> space cut [%ld,%ld) at %ld\n", z0, z1, zm);
#endif
    recTB(t0, t1, v0, vec3mod(v1, dim, zm), rv0, vec3mod(rv1, dim, -1));
    recTB(t0, t1, vec3mod(v0, dim, zm), v1, vec3mod(rv0, dim, -1), rv1);
  }
  else {
    assert(rv0.get(dim) * rv1.get(dim) < 0);
    if (rv0.get(dim) > 0) { // /~~~~\ block
      long gap = (dz-2*dt)/3;
      long zm0 = z0+gap;
      long zm1 = z0+2*(dt+gap);
#if VERBOSE >= 30
      printf("  --> space cut [%ld,%ld) at %ld and %ld\n", z0, z1, zm0, zm1);
#endif

      // left
#ifdef USE_OMPTASK
#pragma omp task
#endif
      recTB(t0, t1, v0, vec3mod(v1, dim, zm0), rv0, rv1);

#ifdef USE_OMPTASK
#pragma omp task
#endif
      // right 
      recTB(t0, t1, vec3mod(v0, dim, zm1), v1, rv0, rv1);

#ifdef USE_OMPTASK
#pragma omp taskwait
#endif

      // center
      vec3 rcv0 = vec3mod(rv0, dim, rv1.get(dim));
      vec3 rcv1 = vec3mod(rv1, dim, rv0.get(dim));
      recTB(t0, t1, vec3mod(v0, dim, zm0), vec3mod(v1, dim, zm1), rcv0, rcv1);
    }
    else {  //  \___/ block
      long gap = (dz-4*dt)/3;
      long zm0 = z0+gap+2*dt;
      long zm1 = z0+2*(dt+gap);
#if VERBOSE >= 30
      printf("  --> space cut [%ld,%ld) at %ld and %ld\n", z0, z1, zm0, zm1);
#endif
      // center
      vec3 rcv0 = vec3mod(rv0, dim, rv1.get(dim));
      vec3 rcv1 = vec3mod(rv1, dim, rv0.get(dim));
      recTB(t0, t1, vec3mod(v0, dim, zm0), vec3mod(v1, dim, zm1), rcv0, rcv1);

      // left
#ifdef USE_OMPTASK
#pragma omp task
#endif
      recTB(t0, t1, v0, vec3mod(v1, dim, zm0), rv0, rv1);

#ifdef USE_OMPTASK
#pragma omp task
#endif
      // right
      recTB(t0, t1, vec3mod(v0, dim, zm1), v1, rv0, rv1);

#ifdef USE_OMPTASK
#pragma omp taskwait
#endif

    }
  }
  return 0;
}

int spacecut(long t0, long t1, vec3 v0, vec3 v1, vec3 rv0, vec3 rv1)
{
  spacecutDim(t0, t1, v0, v1, rv0, rv1, 'Z');
  return 0;
}

int recTB(long t0, long t1, vec3 v0, vec3 v1, vec3 rv0, vec3 rv1)
{
  long dt = t1-t0;
  assert(dt > 0);
  assert(v0.z >= 1 && v1.z <= n3d.z+1);
#if VERBOSE >= 30
  printf("[recTB] t=[%d,%d] space=[(%d,%d,%d)+(%+d,%+d,%+d), (%d,%d,%d)+(%+d,%+d,%+d))\n",
	 t0, t1, v0.x, v0.y, v0.z, rv0.x, rv0.y, rv0.z, v1.x, v1.y, v1.z, rv1.x, rv1.y, rv1.z);
#endif
  if (v0.x >= v1.x || v0.y >= v1.y || v0.z >= v1.z) {
    printf("[recTB] (do nothing)\n");
    return 0;
  }

  if (paraflag == 0) assert(rv0.z*rv1.z < 0); // trazezoid check

  vec3 v0z, v1z;
  int dataflag = 1;
  REAL *darrays[2];
  {
    getAccessRegion(t0, t1, v0, v1, rv0, rv1, &v0z, &v1z);
    int i;
    for (i = 0; i < 2; i++) {
      darrays[i] = &arrays[i][IDX(v0z.x, v0z.y, v0z.z, n3d.x, n3d.y)];
    }
  }

  size_t accsize = vec3sub(v1z, v0z).vol() * sizeof(REAL);

  size_t stopsize = stopsizemb * 1024L*1024L;
  size_t small;
  if (divdims == 0) {
    small = (n3d.x+2)*(n3d.y+2)*(n3d.z+2);
  }
  else if (divdims == 1) {
    small = 5*(n3d.x+2)*(n3d.y+2);
  }
  else {
    small = 5*5*(n3d.x+2);
  }
  if (stopsize < small) stopsize = small;

  if (recflag) {
    // recursive TB
    int dim;
    if (dataflag && accsize <= stopsize) {
      // base case
      base(t0, t1, v0, v1, rv0, rv1);
    }
    else if ((dim = canSpacecut(t0, t1, v0, v1, rv0, rv1)) >= 0) {
      // space cut
      spacecutDim(t0, t1, v0, v1, rv0, rv1, dim);
    }
    else {
      // time cut
      long tm = (t1+t0)/2;
      timecut(t0, t1, tm, v0, v1, rv0, rv1);
    }
  }
  else {
    // flat TB. bt is used
    if (dt > bt) {
      // time cut at bt
      long tm = t0+bt;
      if (tm > t1) t1 = tm;
      timecut(t0, t1, tm, v0, v1, rv0, rv1);
    }
    else if (dataflag && accsize <= stopsize) {
      // base case
      base(t0, t1, v0, v1, rv0, rv1);
    }
    else {
      // space cut
      int dim = canSpacecut(t0, t1, v0, v1, rv0, rv1);
      if (dim < 0) {
	fprintf(stderr, "[recTB] ERROR: flat TB: cannot spacecut (t0=%d,t1=%d,v0=(%d,%d,%d),v1=(%d,%d,%d). bt=%d is too large?\n", t0, t1, v0.x, v0.y, v0.z, v1.x, v1.y, v1.z, bt);
	exit(0);
      }
      spacecutDim(t0, t1, v0, v1, rv0, rv1, dim);
    }
  }

  return 0;
}




int algo(long t0, long t1, vec3 v0, vec3 v1)
{
#ifdef USE_OMPTASK
#pragma omp parallel
#pragma omp single
#endif // USE_OMPTASK
  recTB(t0, t1, v0, v1, vec3(1, 1, 1), vec3(-1, -1, -1));

  return 0;
}
