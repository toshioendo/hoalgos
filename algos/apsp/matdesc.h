// description and transformation of 2D matrix data
#include <stdio.h>

#ifndef REAL
#define REAL float // how do we do?
#endif

struct MatDesc {
  long rows = 0;
  long cols = 0;

  static int copy2d(long m, long n, REAL *tgtp, long tgtld, REAL *srcp, long srcld) {
    for (long jj = 0; jj < n; jj++) {
      for (long ii = 0; ii < m; ii++) {
	// TODO: implement SIMD version
	tgtp[ii+jj*tgtld] = srcp[ii+jj*srcld];
      }
    }
    return 0;
  };


};

struct ColMajorDesc: MatDesc {
  long ld;
};

struct BlockDesc: MatDesc {
  long bsm;
  long bsn;

  long nbm; // assigned in pack()
  long nbn; // assigned in pack()

  BlockDesc(long bsm0, long bsn0) {
    bsm = bsm0;
    bsn = bsn0;
  };

  // compute data size (words) 
  size_t getSize(MatDesc *orgdesc) {
    long nbm = (orgdesc->rows+bsm-1)/bsm;
    long nbn = (orgdesc->cols+bsn-1)/bsn;
    long blocklen = bsm*bsn;
    return nbm*nbn*blocklen;
  };

  // data buffer must be prepared by caller
  // it must have sufficient size computed by getSize()
  int pack(REAL *orgp, MatDesc *orgdesc0, REAL *packp) {
    // currently orgdesc must be ColMajorDesc
    ColMajorDesc *orgdesc = static_cast<ColMajorDesc *>(orgdesc0);
    rows = orgdesc->rows;
    cols = orgdesc->cols;
    long ld = orgdesc->ld;

    nbm = (rows+bsm-1)/bsm;
    nbn = (cols+bsn-1)/bsn;
    long blocklen = bsm*bsn;

#ifdef USE_OMP
#pragma omp parallel for
#endif
    for (long jb = 0; jb < nbn; jb ++) {
      long j = jb*bsn;
      for (long ib = 0; ib < nbm; ib ++) {
	long i = ib*bsm;
	assert(i+bsm <= rows);
	assert(j+bsn <= cols); // TODO: support indivislbe cases
#if 0
	printf("[pack] i=%ld j=%ld: packp[%ld] <- orgp[%ld]\n",
	       i, j, (ib+jb*nbm)*blocklen, i+j*ld);
#endif
	copy2d(bsm, bsn, &packp[(ib+jb*nbm)*blocklen], bsm, &orgp[i+j*ld], ld);
      }
    }
    return 0;
  };

  int unpack(REAL *orgp, MatDesc *orgdesc0, REAL *packp) {
    // currently orgdesc must be ColMajorDesc
    ColMajorDesc *orgdesc = static_cast<ColMajorDesc *>(orgdesc0);
    assert(rows == orgdesc->rows);
    assert(cols == orgdesc->cols);
    long ld = orgdesc->ld;

    long blocklen = bsm*bsn;

#ifdef USE_OMP
#pragma omp parallel for
#endif
    for (long jb = 0; jb < nbn; jb ++) {
      long j = jb*bsn;
      for (long ib = 0; ib < nbm; ib ++) {
	long i = ib*bsm;
	assert(i+bsm <= rows);
	assert(j+bsn <= cols); // TODO: support indivislbe cases
#if 0
	printf("[unpack] i=%ld j=%ld: packp[%ld] -> orgp[%ld]\n",
	       i, j, (ib+jb*nbm)*blocklen, i+j*ld);
#endif
	copy2d(bsm, bsn, &orgp[i+j*ld], ld, &packp[(ib+jb*nbm)*blocklen], bsm);
      }
    }
    return 0;
  };
};

struct TransBlockDesc: MatDesc {
};


