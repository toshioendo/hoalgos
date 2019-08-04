// 3D long vector
#ifndef __VEC3_H
#define __VEC3_H

struct vec3 {
  // constructors
  vec3() {x = 0; y = 0; z = 0;};
  vec3(long x0, long y0, long z0) {x = x0; y = y0; z = z0;};

  // methods
  long get(char dim) {
    if (dim == 'Z') return z;
    else if (dim == 'Y') return y;
    else if (dim == 'X') return x;
    else {
      fprintf(stderr, "[vec3::get] dim=%c is INVALID\n", dim);
      return 0;
    }
  };
  void set(char dim, long val) {
    if (dim == 'Z') z = val;
    else if (dim == 'Y') y = val;
    else if (dim == 'X') x = val;
    else {
      fprintf(stderr, "[vec3::set] dim=%c is INVALID\n", dim);
      return;
    }
  };
  long vol() {return x*y*z;};

  // fields
  long x;
  long y;
  long z;
};

// element wise min/max
inline vec3 vec3min(vec3 va, vec3 vb)
{
  vec3 vr((va.x < vb.x)? va.x: vb.x,
	  (va.y < vb.y)? va.y: vb.y,
	  (va.z < vb.z)? va.z: vb.z);
  return vr;
}

inline vec3 vec3max(vec3 va, vec3 vb)
{
  vec3 vr((va.x > vb.x)? va.x: vb.x,
	  (va.y > vb.y)? va.y: vb.y,
	  (va.z > vb.z)? va.z: vb.z);
  return vr;
}

inline vec3 vec3sub(vec3 va, vec3 vb)
{
  vec3 vr(va.x - vb.x, va.y - vb.y, va.z - vb.z);
  return vr;
}

// returns a vector whose "dim" dimension to val
// other dimensions are same as v
inline vec3 vec3mod(vec3 v, char dim, long val)
{
  vec3 vr(v.x, v.y, v.z);
  vr.set(dim, val);
  return vr;
}

static int vec3eq(vec3 v1, vec3 v2)
{
  return (v1.x == v2.x && v1.y == v2.y && v1.z == v2.z);
}

#endif // __VEC3_H
