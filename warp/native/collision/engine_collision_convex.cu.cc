// Copyright 2024 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "engine_collision_convex.h"  // mjx/cuda

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>

#include <cstddef>
#include <cstdio>
#include <string>

#include <driver_types.h>  // cuda
#include <vector_types.h>  // cuda
#include "engine_collision_common.h"  // mjx/cuda
#include "engine_util_blas.cu.h"  // mjx/cuda


namespace mujoco::mjx::cuda {

inline constexpr uint kGjkMultiContactCount = 4;
inline constexpr uint kMaxEpaBestCount = 12;
inline constexpr uint kMaxMultiPolygonCount = 8;

namespace {

bool assert_cuda(const char* msg) {
  cudaError_t last_error = cudaGetLastError();
  if (last_error != cudaSuccess) {
    printf("[CUDA-ERROR] [%s] (%d:%s) \n", msg ? msg : "", (int)last_error,
           cudaGetErrorString(last_error));
  }
  return (last_error == cudaSuccess);
}

__device__ __forceinline__ void xposmat_to_float4(const float* xpos,
                                                  const float* xmat, int gi,
                                                  float4* mat) {
  mat[0] = make_float4(xmat[gi * 9 + 0], xmat[gi * 9 + 1], xmat[gi * 9 + 2],
                       xpos[gi * 3 + 0]);
  mat[1] = make_float4(xmat[gi * 9 + 3], xmat[gi * 9 + 4], xmat[gi * 9 + 5],
                       xpos[gi * 3 + 1]);
  mat[2] = make_float4(xmat[gi * 9 + 6], xmat[gi * 9 + 7], xmat[gi * 9 + 8],
                       xpos[gi * 3 + 2]);
}

}  // namespace

template <class T>
__device__ __forceinline__ T
get_info(const float* __restrict size, const int gid, const int dataid,
         const uint* __restrict convex_vert_offset, const float* xpos,
         const float* xmat);

template <>
__device__ __forceinline__ GeomType_PLANE get_info<GeomType_PLANE>(
    const float* __restrict size, const int gid, const int dataid,
    const uint* __restrict convex_vert_offset, const float* xpos,
    const float* xmat) {
  GeomType_PLANE x;
  xposmat_to_float4(xpos, xmat, gid, x.mat);
  return x;
}

template <>
__device__ __forceinline__ GeomType_SPHERE get_info<GeomType_SPHERE>(
    const float* __restrict size, const int gid, const int dataid,
    const uint* __restrict convex_vert_offset, const float* xpos,
    const float* xmat) {
  GeomType_SPHERE x;
  x.radius = size[0];
  xposmat_to_float4(xpos, xmat, gid, x.mat);
  return x;
}

template <>
__device__ __forceinline__ GeomType_CAPSULE get_info<GeomType_CAPSULE>(
    const float* __restrict size, const int gid, const int dataid,
    const uint* __restrict convex_vert_offset, const float* xpos,
    const float* xmat) {
  GeomType_CAPSULE x;
  x.radius = size[0];
  x.halfsize = size[1];
  xposmat_to_float4(xpos, xmat, gid, x.mat);
  return x;
}

template <>
__device__ __forceinline__ GeomType_ELLIPSOID
get_info<GeomType_ELLIPSOID>(const float* __restrict size, const int gid,
                             const int dataid,
                             const uint* __restrict convex_vert_offset,
                             const float* xpos, const float* xmat) {
  GeomType_ELLIPSOID x;
  x.size = make_float3(size[0], size[1], size[2]);
  xposmat_to_float4(xpos, xmat, gid, x.mat);
  return x;
}

template <>
__device__ __forceinline__ GeomType_CYLINDER
get_info<GeomType_CYLINDER>(const float* __restrict size, const int gid,
                            const int dataid,
                            const uint* __restrict convex_vert_offset,
                            const float* xpos, const float* xmat) {
  GeomType_CYLINDER x;
  x.radius = size[0];
  x.halfsize = size[1];
  xposmat_to_float4(xpos, xmat, gid, x.mat);
  return x;
}

template <>
__device__ __forceinline__ GeomType_BOX get_info<GeomType_BOX>(
    const float* __restrict size, const int gid, const int dataid,
    const uint* __restrict convex_vert_offset, const float* xpos,
    const float* xmat) {
  GeomType_BOX x;
  x.size = make_float3(size[0], size[1], size[2]);
  xposmat_to_float4(xpos, xmat, gid, x.mat);
  return x;
}

template <>
__device__ __forceinline__ GeomType_CONVEX get_info<GeomType_CONVEX>(
    const float* __restrict size, const int gid, const int dataid,
    const uint* __restrict convex_vert_offset, const float* xpos,
    const float* xmat) {
  GeomType_CONVEX x;
  xposmat_to_float4(xpos, xmat, gid, x.mat);
  if (!convex_vert_offset || (dataid < 0)) {
    x.vert_offset = x.vert_count = 0;
    return x;
  }
  x.vert_offset = convex_vert_offset[dataid];
  x.vert_count = convex_vert_offset[dataid + 1] - x.vert_offset;
  return x;
}

// Returns the distance from the support point to the origin. The support point
// is the point on the surface of the object in the direction of `dir`.
template <class T>
__device__ __forceinline__ float gjk_support(
    const T& info, const float3& dir, const float* __restrict convex_vert,
    float3& support_pt);

template <>
__device__ __forceinline__ float gjk_support<GeomType_PLANE>(
    const GeomType_PLANE& info, const float3& dir,
    const float* __restrict convex_vert, float3& support_pt) {
  float3 local_dir;
  mulMatTVec3(local_dir, info.mat, dir);
  float norm = sqrtf(local_dir.x * local_dir.x + local_dir.y * local_dir.y);
  float nx = (norm > 0) ? local_dir.x / norm : 1.0f;
  float ny = (norm > 0) ? local_dir.y / norm : 0.0f;
  float nz = (local_dir.z < 0) ? -1.0f : 0.0f;
  float largeSize = 5.0f;
  float3 res = {nx * largeSize, ny * largeSize, nz * largeSize};
  mulMatVec3(support_pt, info.mat, res);
  support_pt += make_float3(info.mat[0].w, info.mat[1].w, info.mat[2].w);
  return dot(support_pt, dir);
}

template <>
__device__ __forceinline__ float gjk_support<GeomType_SPHERE>(
    const GeomType_SPHERE& info, const float3& dir,
    const float* __restrict convex_vert, float3& support_pt) {
  float3 xpos = make_float3(info.mat[0].w, info.mat[1].w, info.mat[2].w);
  support_pt = xpos + info.radius * dir;
  return dot(support_pt, dir);
}

template <>
__device__ __forceinline__ float gjk_support<GeomType_CAPSULE>(
    const GeomType_CAPSULE& info, const float3& dir,
    const float* __restrict convex_vert, float3& support_pt) {
  float3 local_dir;
  mulMatTVec3(local_dir, info.mat, dir);
  float3 xpos = make_float3(info.mat[0].w, info.mat[1].w, info.mat[2].w);
  // start with sphere
  float3 res = local_dir * info.radius;
  // add cylinder contribution
  res.z += sign(local_dir.z) * info.halfsize;
  mulMatVec3(support_pt, info.mat, res);
  support_pt += xpos;
  return dot(support_pt, dir);
}

template <>
__device__ __forceinline__ float gjk_support<GeomType_ELLIPSOID>(
    const GeomType_ELLIPSOID& info, const float3& dir,
    const float* __restrict convex_vert, float3& support_pt) {
  float3 local_dir;
  mulMatTVec3(local_dir, info.mat, dir);
  float3 xpos = make_float3(info.mat[0].w, info.mat[1].w, info.mat[2].w);
  // find support point on unit sphere: scale dir by ellipsoid sizes and
  // renormalize
  float3 res = local_dir * info.size;
  normalize(res);
  // transform to ellipsoid
  res = res * info.size;
  mulMatVec3(support_pt, info.mat, res);
  support_pt += xpos;
  return dot(support_pt, dir);
}

template <>
__device__ __forceinline__ float gjk_support<GeomType_CYLINDER>(
    const GeomType_CYLINDER& info, const float3& dir,
    const float* __restrict convex_vert, float3& support_pt) {
  float3 local_dir;
  mulMatTVec3(local_dir, info.mat, dir);
  float3 xpos = make_float3(info.mat[0].w, info.mat[1].w, info.mat[2].w);
  float3 res = make_float3(0.0f, 0.0f, 0.0f);
  // set result in XY plane: support on circle
  float d = sqrt(dot(local_dir, local_dir));
  if (d > mjMINVAL) {
    res.x = local_dir.x / d * info.radius;
    res.y = local_dir.y / d * info.radius;
  } else {
    res.x = res.y = 0.0;
  }
  // set result in Z direction
  res.z = sign(local_dir.z) * info.halfsize;
  mulMatVec3(support_pt, info.mat, res);
  support_pt += xpos;
  return dot(support_pt, dir);
}

template <>
__device__ __forceinline__ float gjk_support<GeomType_BOX>(
    const GeomType_BOX& info, const float3& dir,
    const float* __restrict convex_vert, float3& support_pt) {
  float3 local_dir;
  mulMatTVec3(local_dir, info.mat, dir);
  float3 xpos = make_float3(info.mat[0].w, info.mat[1].w, info.mat[2].w);
  float3 res = sign(local_dir) * info.size;
  mulMatVec3(support_pt, info.mat, res);
  support_pt += xpos;
  return dot(support_pt, dir);
}

template <>
__device__ __forceinline__ float gjk_support<GeomType_CONVEX>(
    const GeomType_CONVEX& info, const float3& dir,
    const float* __restrict convex_vert, float3& support_pt) {
  float3 local_dir;
  mulMatTVec3(local_dir, info.mat, dir);
  const float* vertdata = convex_vert + 3 * info.vert_offset;
  float3 res = make_float3(0.0f, 0.0f, 0.0f);
  float d = -1e10f;
  // exhaustive search over all vertices
  // TODO(robotics-simulation): consider hill-climb over graphdata.
  for (int i = 0; i < info.vert_count; i++) {
    float3 p =
        make_float3(vertdata[3 * i], vertdata[3 * i + 1], vertdata[3 * i + 2]);
    float vdot = dot(local_dir, p);
    // update best
    if (vdot > d) {
      d = vdot;
      res = p;
    }
  }
  mulMatVec3(support_pt, info.mat, res);
  support_pt += make_float3(info.mat[0].w, info.mat[1].w, info.mat[2].w);
  return dot(support_pt, dir);
}

// Returns the distance between support points on two geoms. Negative distance
// means objects are not intersecting along direction `dir`. Positive distance
// means objects are intersecting along the given direction `dir`.
template <class T1, class T2>
__device__ __forceinline__ float gjk_support(
    const T1& info1, const T2& info2, const float3& dir,
    const float* __restrict convex_vert, float3& support_pt) {
  float3 s1, s2;
  float dist1 = gjk_support<T1>(info1, dir, convex_vert, s1);
  float3 dir_n = make_float3(-1.0 * dir.x, -1.0 * dir.y, -1.0 * dir.z);
  float dist2 = gjk_support<T2>(info2, dir_n, convex_vert, s2);

  support_pt = s1 - s2;
  return (dist1 + dist2);
}

__forceinline__ __device__ bool gjk_normalize(float3& a) {
  float norm = sqrtf(a.x * a.x + a.y * a.y + a.z * a.z);
  if ((norm > 1e-8f) && (norm < 1e12f)) {
    a /= norm;
    return true;
  }
  return false;
}

inline __device__ void orthonormal(const float3 normal, float3& dir) {
  if ((fabs(normal.x) < fabs(normal.y)) && (fabs(normal.x) < fabs(normal.z))) {
    dir = make_float3(1.0f - normal.x * normal.x, -normal.x * normal.y,
                      -normal.x * normal.z);
  } else if (fabs(normal.y) < fabs(normal.z)) {
    dir = make_float3(-normal.y * normal.x, 1.0f - normal.y * normal.y,
                      -normal.y * normal.z);
  } else {
    dir = make_float3(-normal.z * normal.x, -normal.z * normal.y,
                      1.0f - normal.z * normal.z);
  }
  gjk_normalize(dir);
}

// Initialize the contact distance to a large value.
__global__ void gjk_epa_init(const uint npair, const uint nenv, const int ncon,
                             float* __restrict dist) {
  uint tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= npair * nenv) return;
  for (int i = 0; i < ncon; i++) {
    dist[tid * ncon + i] = 1e12f;
  }
}

// Calculates whether two objects intersect.
template <class T1, class T2>
__device__ __forceinline__ void _gjk(
    const uint tid, const uint env_id, const uint model_id, const int g1,
    const int g2, const uint ngeom, const float* __restrict size,
    const float* __restrict xpos, const float* __restrict xmat,
    const int* __restrict dataid, const float* __restrict convex_vert,
    const uint* __restrict convex_vert_offset, const int gjk_iteration_count,
    float4 simplex_f4[3], float3& normal) {
  const int dataid1 = dataid ? dataid[g1] : -1;
  const int dataid2 = dataid ? dataid[g2] : -1;
  const float* size1 = size + (model_id * ngeom + g1) * 3;
  const float* size2 = size + (model_id * ngeom + g2) * 3;
  const int tg1 = env_id * ngeom + g1;
  const int tg2 = env_id * ngeom + g2;
  T1 info1 = get_info<T1>(size1, tg1, dataid1, convex_vert_offset, xpos, xmat);
  T2 info2 = get_info<T2>(size2, tg2, dataid2, convex_vert_offset, xpos, xmat);

  float3 dir = make_float3(0.0f, 0.0f, 1.0f), simplex[4];
  float3 dir_n = -dir;
  float depth = 1e30f;

  float max = gjk_support<T1, T2>(info1, info2, dir, convex_vert, simplex[0]);
  float min = gjk_support<T1, T2>(info1, info2, dir_n, convex_vert, simplex[1]);
  if (max < min) {
    depth = max, normal = dir;
  } else {
    depth = min, normal = dir_n;
  }

  float3 d = simplex[0] - simplex[1];
  normalize(d);
  orthonormal(d, dir);

  max = gjk_support<T1, T2>(info1, info2, dir, convex_vert, simplex[3]);
  // Initialize a 2-simplex with simplex[2]==simplex[1]. This ensures the
  // correct winding order for face normals defined below. Face 0 and face 3
  // are degenerate, and face 1 and 2 have opposing normals.
  simplex[2] = simplex[1];

  if (max < depth) depth = max, normal = dir;
  if (min < depth) depth = min, normal = dir_n;

  for (int i = 0; i < gjk_iteration_count; i++) {
    float3 plane[4];
    float d[4];

    // Winding orders: plane[0] ccw, plane[1] cw, plane[2] ccw, plane[3] cw.
    plane[0] = cross(simplex[3] - simplex[2], simplex[1] - simplex[2]);
    plane[1] = cross(simplex[3] - simplex[0], simplex[2] - simplex[0]);
    plane[2] = cross(simplex[3] - simplex[1], simplex[0] - simplex[1]);
    plane[3] = cross(simplex[2] - simplex[0], simplex[1] - simplex[0]);

    // Compute distance of each face halfspace to the origin. If d<0, then the
    // origin is outside the halfspace. If d>0 then the origin is inside
    // the halfspace defined by the face plane.
    d[0] = (gjk_normalize(plane[0])) ? dot(plane[0], simplex[2]) : 1e30f;
    d[1] = (gjk_normalize(plane[1])) ? dot(plane[1], simplex[0]) : 1e30f;
    d[2] = (gjk_normalize(plane[2])) ? dot(plane[2], simplex[1]) : 1e30f;
    d[3] = (gjk_normalize(plane[3])) ? dot(plane[3], simplex[0]) : 1e30f;

    // Pick the plane normal with minimum distance to the origin.
    int i1 = (d[0] < d[1]) ? 0 : 1, i2 = (d[2] < d[3]) ? 2 : 3;
    int index = (d[i1] < d[i2]) ? i1 : i2;
    if (d[index] > 0.0f) {
      // Origin is inside the simplex, objects are intersecting.
      break;
    }

    // Add new support point to the simplex.
    float dist = gjk_support<T1, T2>(info1, info2, plane[index], convex_vert,
                                     simplex[index]);
    if (dist < depth) depth = dist, normal = plane[index];

    // Preserver winding order of the simplex faces.
    int index1 = (index + 1) & 3, index2 = (index + 2) & 3;
    float3 swap = simplex[index1];
    simplex[index1] = simplex[index2];
    simplex[index2] = swap;
    if (dist < 0) break;  // Objects are likely non-intersecting.
  }

  float4* simplex_f4_ = (float4*)simplex;
  simplex_f4[0] = simplex_f4_[0];
  simplex_f4[1] = simplex_f4_[1];
  simplex_f4[2] = simplex_f4_[2];
}

template <class T1, class T2>
__global__ void gjk_dense(
    const uint npair, const uint nenv, const uint ngeom, const uint nmodel,
    const uint ncon, const int* __restrict geom_pair,
    const float* __restrict size, const float* __restrict xpos,
    const float* __restrict xmat, const int* __restrict dataid,
    const float* __restrict convex_vert,
    const uint* __restrict convex_vert_offset, const int gjk_iteration_count,
    float* __restrict contact_normal, float4* __restrict d_simplex) {
  uint tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= npair * nenv) return;

  uint pair_id = tid % npair;
  uint env_id = tid / npair;
  uint model_id = env_id % nmodel;

  int g1 = geom_pair[pair_id * 2], g2 = geom_pair[pair_id * 2 + 1];
  if (g1 < 0 || g2 < 0) return;

  float3 normal;
  float4 simplex_f4[3];
  _gjk<T1, T2>(tid, env_id, model_id, g1, g2, ngeom, size, xpos, xmat, dataid,
               convex_vert, convex_vert_offset, gjk_iteration_count, simplex_f4,
               normal);

  // Align to 16 byte boundary.
  d_simplex[tid * 3 + 0] = simplex_f4[0];
  d_simplex[tid * 3 + 1] = simplex_f4[1];
  d_simplex[tid * 3 + 2] = simplex_f4[2];

  if (contact_normal) {
    contact_normal[tid * 3] = normal.x;
    contact_normal[tid * 3 + 1] = normal.y;
    contact_normal[tid * 3 + 2] = normal.z;
  }
}

// Returns the normal and penetration depth for object intersections. If objects
//  intersect, the depth is the smallest penetration. For non-intersecting
//  objects, the depth is of the closest points.
// epa_iteration_count: the number of iterations (increase for more accurate
//  results).
// epa_best_count: the number of best candidates in each iteration, must be less
//  or equal to maxepa_best_count. Minimize epa_best_count for performance.
// exactNegDistance: if true, the algorithm calculates the distance of
//  non-intersecting objects exactly. For intersecting, the closest point on the
//  Minkowski surface to the origin is always on the face.
//  For non-intersecting, the closest point is on the edge or in the corner and
//  additional tests have to be done.
// NOTE: depth_extension should be > 0. If exactly 0, we may miss
// contacts when the origin is on the surface of the initial simplex.
template <class T1, class T2>
__device__ __forceinline__ void _epa(
    const uint tid, const uint env_id, const uint model_id, const int g1,
    const int g2, const uint ngeom, const uint ncon,
    const float* __restrict size, const float* xpos, const float* xmat,
    const int* __restrict dataid, const float* __restrict convex_vert,
    const uint* __restrict convex_vert_offset, const float depth_extension,
    const int epa_iteration_count, const int epa_best_count,
    const float4* __restrict d_simplex, float3& normal, float& depth) {
  bool exactNegDistance = true;
  const int dataid1 = dataid ? dataid[g1] : -1;
  const int dataid2 = dataid ? dataid[g2] : -1;
  const float* size1 = size + (model_id * ngeom + g1) * 3;
  const float* size2 = size + (model_id * ngeom + g2) * 3;
  const int tg1 = env_id * ngeom + g1;
  const int tg2 = env_id * ngeom + g2;
  T1 info1 = get_info<T1>(size1, tg1, dataid1, convex_vert_offset, xpos, xmat);
  T2 info2 = get_info<T2>(size2, tg2, dataid2, convex_vert_offset, xpos, xmat);

  float3 simplex[4];
  // Get the support. If less than 0, objects are not intersecting.
  depth = gjk_support<T1, T2>(info1, info2, normal, convex_vert, simplex[0]);

  if (depth < -depth_extension) {
    // Objects are not intersecting, and we do not obtain the closest points as
    // specified by depth_extension.
    return;
  }

  // Use simplex from GJK.
  float4* simplex_f4 = (float4*)simplex;
  simplex_f4[0] = d_simplex[0];
  simplex_f4[1] = d_simplex[1];
  simplex_f4[2] = d_simplex[2];

  if (exactNegDistance) {
    // Check closest points to all edges of the simplex, rather than just the
    // face normals. This gives the exact depth/normal for the non-intersecting
    // case.
    for (int i = 0; i < 6; i++) {
      int i1 = (i < 3) ? 0 : ((i < 5) ? 1 : 2);
      int i2 = (i < 3) ? i + 1 : (i < 5) ? i - 1 : 3;
      if ((simplex[i1].x != simplex[i2].x) ||
          (simplex[i1].y != simplex[i2].y) ||
          (simplex[i1].z != simplex[i2].z)) {
        float3 v = simplex[i1] - simplex[i2];
        float alpha = dot(simplex[i1], v) / (v.x * v.x + v.y * v.y + v.z * v.z);
        // p0 is the closest segment point to the origin.
        float3 p0 =
            ((alpha < 0.0f) ? 0.0f : ((alpha > 1.0f) ? 1.0f : alpha)) * v -
            simplex[i1];
        if (gjk_normalize(p0)) {
          float dist2 = gjk_support<T1, T2>(info1, info2, p0, convex_vert, v);
          if (dist2 < depth) depth = dist2, normal = p0;
        }
      }
    }
  }

  float3 tr[3 * kMaxEpaBestCount], tr2[3 * kMaxEpaBestCount];
  float3 p[kMaxEpaBestCount];  // supporting points for each triangle.
  float3* tris = tr;
  float3* nexTtris = tr2;
  // Distance to the origin for candidate triangles.
  float dists[kMaxEpaBestCount * 3];

  tris[0] = simplex[2], tris[1] = simplex[1], tris[2] = simplex[3];
  tris[3] = simplex[0], tris[4] = simplex[2], tris[5] = simplex[3];
  tris[6] = simplex[1], tris[7] = simplex[0], tris[8] = simplex[3];
  tris[9] = simplex[0], tris[10] = simplex[1], tris[11] = simplex[2];

  int count = 4;
  for (int q = 0; q < epa_iteration_count; q++) {
    for (int i = 0; i < count; i++) {
      // Loop through all triangles, and obtain distances to the origin for each
      // new triangle candidate.
      float3* triangle = tris + 3 * i;
      float3 n = cross(triangle[2] - triangle[0], triangle[1] - triangle[0]);

      if (!gjk_normalize(n)) {
        for (int j = 0; j < 3; j++) dists[i * 3 + j] = 2e30f;
        continue;
      }

      float dist = gjk_support<T1, T2>(info1, info2, n, convex_vert, p[i]);
      if (dist < depth) depth = dist, normal = n;
      // Loop through all edges, and get distance using support point p[i].
      for (int j = 0; j < 3; j++) {
        if (exactNegDistance) {
          // Obtain the closest point between the new triangle edge and the
          // origin.
          if ((p[i].x != triangle[j].x) || (p[i].y != triangle[j].y) ||
              (p[i].z != triangle[j].z)) {
            float3 v = p[i] - triangle[j];
            float alpha = dot(p[i], v) / (v.x * v.x + v.y * v.y + v.z * v.z);
            float3 p0 =
                ((alpha < 0.0f) ? 0.0f : ((alpha > 1.0f) ? 1.0f : alpha)) * v -
                p[i];
            if (gjk_normalize(p0)) {
              float dist2 =
                  gjk_support<T1, T2>(info1, info2, p0, convex_vert, v);
              if (dist2 < depth) depth = dist2, normal = p0;
            }
          }
        }
        float3 plane =
            cross(p[i] - triangle[j], triangle[((j + 1) % 3)] - triangle[j]);
        float d = (gjk_normalize(plane)) ? dot(plane, triangle[j]) : 1e30f;

        dists[i * 3 + j] = (((d < 0) && (depth >= 0)) ||
                            ((triangle[((j + 2) % 3)].x == p[i].x) &&
                             (triangle[((j + 2) % 3)].y == p[i].y) &&
                             (triangle[((j + 2) % 3)].z == p[i].z)))
                               ? 1e30f
                               : d;
      }
    }
    int prevCount = count;
    count = (count * 3 < epa_best_count) ? count * 3 : epa_best_count;

    // Expand the polytope greedily.
    for (int j = 0; j < count; j++) {
      int bestIndex = 0;
      float d = dists[0];
      for (int i = 1; i < 3 * prevCount; i++) {
        if (dists[i] < d) d = dists[i], bestIndex = i;
      }
      dists[bestIndex] = 2e30f;

      int parentIndex = bestIndex / 3, childIndex = bestIndex % 3;
      float3* triangle = nexTtris + j * 3;
      triangle[0] = tris[parentIndex * 3 + childIndex];
      triangle[1] = tris[parentIndex * 3 + ((childIndex + 1) % 3)];
      triangle[2] = p[parentIndex];
    }
    float3* swap = tris;
    tris = nexTtris, nexTtris = swap;
  }
}

template <class T1, class T2>
__global__ void epa_dense(
    const uint npair, const uint nenv, const uint ngeom, const uint nmodel,
    const uint ncon, const int* __restrict geom_pair,
    const float* __restrict size, const float* xpos, const float* xmat,
    const int* __restrict dataid, const float* __restrict convex_vert,
    const uint* __restrict convex_vert_offset, const float depth_extension,
    const int epa_iteration_count, const int epa_best_count,
    const float4* __restrict d_simplex, float* __restrict dist,
    float* __restrict contact_normal) {
  uint tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= npair * nenv) return;

  uint pair_id = tid % npair;
  uint env_id = tid / npair;
  uint model_id = env_id % nmodel;

  int g1 = geom_pair[pair_id * 2], g2 = geom_pair[pair_id * 2 + 1];
  if (g1 < 0 || g2 < 0) return;

  float depth = -__FLT_MAX__;
  float3 normal =
      make_float3(contact_normal[tid * 3], contact_normal[tid * 3 + 1],
                  contact_normal[tid * 3 + 2]);
  float4 simplex_f4[3];
  simplex_f4[0] = d_simplex[tid * 3 + 0];
  simplex_f4[1] = d_simplex[tid * 3 + 1];
  simplex_f4[2] = d_simplex[tid * 3 + 2];
  _epa<T1, T2>(tid, env_id, model_id, g1, g2, ngeom, ncon, size, xpos, xmat,
               dataid, convex_vert, convex_vert_offset, depth_extension,
               epa_iteration_count, epa_best_count, simplex_f4, normal, depth);

  if (depth < -depth_extension) return;

  for (int i = 0; i < ncon; i++) {
    dist[tid * ncon + i] = -depth;
  }

  contact_normal[tid * 3] = normal.x;
  contact_normal[tid * 3 + 1] = normal.y;
  contact_normal[tid * 3 + 2] = normal.z;
}

// Calculates multiple contact points given the normal from EPA.
//  1. Calculates the polygon on each shape by tiling the normal
//     "multi_tilt_angle" degrees in the orthogonal componenet of the normal.
//     The "multi_tilt_angle" can be changed to depend on the depth of the
//     contact, in a future version.
//  2. The normal is tilted "multi_polygon_count" times in the directions evenly
//    spaced in the orthogonal component of the normal.
//    (works well for >= 6, default is 8).
//  3. The intersection between these two polygons is calculated in 2D space
//    (complement to the normal). If they intersect, extreme points in both
//    directions are found. This can be modified to the extremes in the
//    direction of eigenvectors of the variance of points of each polygon. If
//    they do not intersect, the closest points of both polygons are found.
template <class T1, class T2>
__device__ __forceinline__ void _get_multiple_contacts(
    const uint tid, const uint env_id, const uint model_id, const int g1,
    const int g2, const uint ngeom, const uint ncon,
    const float* __restrict size, const float* __restrict xpos,
    const float* __restrict xmat, const int* __restrict dataid,
    const float* __restrict convex_vert,
    const uint* __restrict convex_vert_offset, const float depth_extension,
    const int multi_polygon_count, const float multi_tilt_angle,
    const float depth, const float3 normal, float* pos, int& contact_count) {
  const int dataid1 = dataid ? dataid[g1] : -1;
  const int dataid2 = dataid ? dataid[g2] : -1;
  const float* size1 = size + (model_id * ngeom + g1) * 3;
  const float* size2 = size + (model_id * ngeom + g2) * 3;
  const int tg1 = env_id * ngeom + g1;
  const int tg2 = env_id * ngeom + g2;
  T1 info1 = get_info<T1>(size1, tg1, dataid1, convex_vert_offset, xpos, xmat);
  T2 info2 = get_info<T2>(size2, tg2, dataid2, convex_vert_offset, xpos, xmat);

  if (depth < -depth_extension) return;

  float3 dir;
  orthonormal(normal, dir);
  float3 dir2 = cross(normal, dir);

  float angle = multi_tilt_angle * PI / 180.0f;
  float c = (float)cos(angle), s = (float)sin(angle), t = 1 - c;
  float3 v1[kMaxMultiPolygonCount], v2[kMaxMultiPolygonCount];

  // Obtain points on the polygon determined by the support and tilt angle,
  // in the basis of the contact frame.
  int v1count = 0, v2count = 0;
  for (int i = 0; i < multi_polygon_count; i++) {
    float3 axis = cos(2 * i * PI / multi_polygon_count) * dir +
                  sin(2 * i * PI / multi_polygon_count) * dir2;

    float mat[12];  // 12-element 3x3 matrix
    // Axis-angle rotation matrix. See
    // https://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle
    mat[0] = c + axis.x * axis.x * t;
    mat[5] = c + axis.y * axis.y * t;
    mat[10] = c + axis.z * axis.z * t;
    float t1 = axis.x * axis.y * t, t2 = axis.z * s;
    mat[4] = t1 + t2, mat[1] = t1 - t2;
    t1 = axis.x * axis.z * t, t2 = axis.y * s;
    mat[8] = t1 - t2, mat[2] = t1 + t2;
    t1 = axis.y * axis.z * t, t2 = axis.x * s;
    mat[9] = t1 + t2, mat[6] = t1 - t2;

    float3 n, p;

    n.x = mat[0] * normal.x + mat[1] * normal.y + mat[2] * normal.z;
    n.y = mat[4] * normal.x + mat[5] * normal.y + mat[6] * normal.z;
    n.z = mat[8] * normal.x + mat[9] * normal.y + mat[10] * normal.z;

    gjk_support<T1>(info1, n, convex_vert, p);
    v1[v1count] = make_float3(dot(p, dir), dot(p, dir2), dot(p, normal));
    if ((!i) || (v1[v1count].x != v1[v1count - 1].x) ||
        (v1[v1count].y != v1[v1count - 1].y) ||
        (v1[v1count].z != v1[v1count - 1].z))
      v1count++;

    n = -n;
    gjk_support<T2>(info2, n, convex_vert, p);
    v2[v2count] = make_float3(dot(p, dir), dot(p, dir2), dot(p, normal));
    if ((!i) || (v2[v2count].x != v2[v2count - 1].x) ||
        (v2[v2count].y != v2[v2count - 1].y) ||
        (v2[v2count].z != v2[v2count - 1].z))
      v2count++;
  }

  // Remove duplicate vertices on the array boundary.
  if ((v1count > 1) && (v1[v1count - 1].x == v1[0].x) &&
      (v1[v1count - 1].y == v1[0].y) && (v1[v1count - 1].z == v1[0].z))
    v1count--;
  if ((v2count > 1) && (v2[v2count - 1].x == v2[0].x) &&
      (v2[v2count - 1].y == v2[0].y) && (v2[v2count - 1].z == v2[0].z))
    v2count--;

  // Find an intersecting polygon between v1 and v2 in the 2D plane.
  float3 out[4];
  int candCount = 0;
  if (v2count > 1) {
    for (int i = 0; i < v1count; i++) {
      float3 m1a = v1[i];
      bool in = true;

      // Check if point m1a is inside the v2 polygon on the 2D plane.
      for (int j = 0; j < v2count; j++) {
        int j2 = (j + 1) % v2count;
        // Checks that orientation of the triangle (v2[j], v2[j2], m1a) is
        // counter-clockwise. If so, point m1a is inside the v2 polygon.
        in &= ((v2[j2].x - v2[j].x) * (m1a.y - v2[j].y) -
                   (v2[j2].y - v2[j].y) * (m1a.x - v2[j].x) >=
               0.0f);
        if (!in) break;
      }
      if (in) {
        if ((!candCount) || (m1a.x < out[0].x)) out[0] = m1a;
        if ((!candCount) || (m1a.x > out[1].x)) out[1] = m1a;
        if ((!candCount) || (m1a.y < out[2].y)) out[2] = m1a;
        if ((!candCount) || (m1a.y > out[3].y)) out[3] = m1a;
        candCount++;
      }
    }
  }
  if (v1count > 1) {
    for (int i = 0; i < v2count; i++) {
      float3 m1a = v2[i];
      bool in = true;

      for (int j = 0; j < v1count; j++) {
        int j2 = (j + 1) % v1count;
        in &= ((v1[j2].x - v1[j].x) * (m1a.y - v1[j].y) -
                   (v1[j2].y - v1[j].y) * (m1a.x - v1[j].x) >=
               0.0f);
        if (!in) break;
      }
      if (in) {
        if ((!candCount) || (m1a.x < out[0].x)) out[0] = m1a;
        if ((!candCount) || (m1a.x > out[1].x)) out[1] = m1a;
        if ((!candCount) || (m1a.y < out[2].y)) out[2] = m1a;
        if ((!candCount) || (m1a.y > out[3].y)) out[3] = m1a;
        candCount++;
      }
    }
  }
  if ((v1count > 1) && (v2count > 1)) {
    // Check all edge pairs, and store line segment intersections if they are
    // on the edge of the boundary.
    for (int i = 0; i < v1count; i++)
      for (int j = 0; j < v2count; j++) {
        float3 m1a = v1[i], m1b = v1[(i + 1) % v1count];
        float3 m2a = v2[j], m2b = v2[(j + 1) % v2count];

        float det = (m2a.y - m2b.y) * (m1b.x - m1a.x) -
                    (m1a.y - m1b.y) * (m2b.x - m2a.x);
        if (fabs(det) > 1e-12f) {
          float a12 = (m2b.x - m2a.x) / det, a22 = (m1b.x - m1a.x) / det;
          float a21 = (m1a.y - m1b.y) / det, a11 = (m2a.y - m2b.y) / det;
          float b1 = m2a.x - m1a.x, b2 = m2a.y - m1a.y;

          float alpha = a11 * b1 + a12 * b2, beta = a21 * b1 + a22 * b2;
          if ((alpha >= 0.0f) && (alpha <= 1.0f) && (beta >= 0.0f) &&
              (beta <= 1.0f)) {
            float3 m0 = make_float3(m1a.x + alpha * (m1b.x - m1a.x),
                                    m1a.y + alpha * (m1b.y - m1a.y),
                                    (m1a.z + alpha * (m1b.z - m1a.z) + m2a.z +
                                     beta * (m2b.z - m2a.z)) *
                                        0.5f);
            if ((!candCount) || (m0.x < out[0].x)) out[0] = m0;
            if ((!candCount) || (m0.x > out[1].x)) out[1] = m0;
            if ((!candCount) || (m0.y < out[2].y)) out[2] = m0;
            if ((!candCount) || (m0.y > out[3].y)) out[3] = m0;
            candCount++;
          }
        }
      }
  }

  float3 var_rx;
  if (candCount > 0) {
    // Polygon intersection was found.
    // TODO(btaba): replace the above routine with the manifold point routine
    // from MJX. Deduplicate the points properly.
    float3 last_pt = make_float3(__FLT_MAX__, __FLT_MAX__, __FLT_MAX__);
#pragma unroll
    for (int k = 0; k < kGjkMultiContactCount; k++) {
      float3 pt = out[k].x * dir + out[k].y * dir2 + out[k].z * normal;
      if (norm(pt - last_pt) <= 1e-6f) continue;
      pos[3 * contact_count] = pt.x;
      pos[3 * contact_count + 1] = pt.y;
      pos[3 * contact_count + 2] = pt.z;
      last_pt = pt;
      contact_count++;
    }
  } else {
    // Polygon intersection was not found. Loop through all vertex pairs and
    // calculate an approximate contact point.
    float minDist = 0.0f;
    for (int i = 0; i < v1count; i++)
      for (int j = 0; j < v2count; j++) {
        // Find the closest vertex pair. Calculate a contact point var_rx as the
        // midpoint between the closest vertex pair.
        float3 m1 = v1[i], m2 = v2[j];
        float d = (m1.x - m2.x) * (m1.x - m2.x) + (m1.y - m2.y) * (m1.y - m2.y);
        if (((!i) && (!j)) || (d < minDist)) {
          minDist = d;
          var_rx = ((m1.x + m2.x) * dir + (m1.y + m2.y) * dir2 +
                    (m1.z + m2.z) * normal) *
                   0.5f;
        }
        // Check for a closer point between a point on v2 and an edge on v1.
        float3 m1b = v1[(i + 1) % v1count], m2b = v2[(j + 1) % v2count];
        if (v1count > 1) {
          float d =
              (m1b.x - m1.x) * (m1b.x - m1.x) + (m1b.y - m1.y) * (m1b.y - m1.y);
          float t = ((m2.y - m1.y) * (m1b.x - m1.x) -
                     (m2.x - m1.x) * (m1b.y - m1.y)) /
                    d;
          float dx = m2.x + (m1b.y - m1.y) * t, dy = m2.y - (m1b.x - m1.x) * t;
          float dist = (dx - m2.x) * (dx - m2.x) + (dy - m2.y) * (dy - m2.y);

          if ((dist < minDist) &&
              ((dx - m1.x) * (m1b.x - m1.x) + (dy - m1.y) * (m1b.y - m1.y) >=
               0) &&
              ((dx - m1b.x) * (m1.x - m1b.x) + (dy - m1b.y) * (m1.y - m1b.y) >=
               0)) {
            float alpha = (float)sqrt(
                ((dx - m1.x) * (dx - m1.x) + (dy - m1.y) * (dy - m1.y)) / d);
            minDist = dist;
            float3 w = ((1 - alpha) * m1 + alpha * m1b + m2) * 0.5f;
            var_rx = w.x * dir + w.y * dir2 + w.z * normal;
          }
        }
        // Check for a closer point between a point on v1 and an edge on v2.
        if (v2count > 1) {
          float d =
              (m2b.x - m2.x) * (m2b.x - m2.x) + (m2b.y - m2.y) * (m2b.y - m2.y);
          float t = ((m1.y - m2.y) * (m2b.x - m2.x) -
                     (m1.x - m2.x) * (m2b.y - m2.y)) /
                    d;
          float dx = m1.x + (m2b.y - m2.y) * t, dy = m1.y - (m2b.x - m2.x) * t;
          float dist = (dx - m1.x) * (dx - m1.x) + (dy - m1.y) * (dy - m1.y);

          if ((dist < minDist) &&
              ((dx - m2.x) * (m2b.x - m2.x) + (dy - m2.y) * (m2b.y - m2.y) >=
               0) &&
              ((dx - m2b.x) * (m2.x - m2b.x) + (dy - m2b.y) * (m2.y - m2b.y) >=
               0)) {
            float alpha = (float)sqrt(
                ((dx - m2.x) * (dx - m2.x) + (dy - m2.y) * (dy - m2.y)) / d);
            minDist = dist;
            float3 w = (m1 + (1 - alpha) * m2 + alpha * m2b) * 0.5f;
            var_rx = w.x * dir + w.y * dir2 + w.z * normal;
          }
        }
      }
#pragma unroll
    for (int k = 0; k < kGjkMultiContactCount; k++) {
      pos[3 * k + 0] = var_rx.x;
      pos[3 * k + 1] = var_rx.y;
      pos[3 * k + 2] = var_rx.z;
    }
    contact_count = 1;
  }
}

template <class T1, class T2>
__global__ void multiple_contacts_dense(
    const uint npair, const uint nenv, const uint ngeom, const uint nmodel,
    const uint ncon, const int* __restrict geom_pair,
    const float* __restrict size, const float* xpos, const float* xmat,
    const int* __restrict dataid, const float* __restrict convex_vert,
    const uint* __restrict convex_vert_offset, const float depth_extension,
    const int multi_polygon_count, const float multi_tilt_angle,
    const float* __restrict dist, const float* __restrict contact_normal,
    float* __restrict pos) {
  uint tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= npair * nenv) return;

  uint pair_id = tid % npair;
  uint env_id = tid / npair;
  uint model_id = env_id % nmodel;

  int g1 = geom_pair[pair_id * 2], g2 = geom_pair[pair_id * 2 + 1];
  if (g1 < 0 || g2 < 0) return;

  float3 normal =
      make_float3(contact_normal[tid * 3], contact_normal[tid * 3 + 1],
                  contact_normal[tid * 3 + 2]);
  float depth = -dist[tid * ncon];

  int contact_count = 0;
  float pos_[12];
  _get_multiple_contacts<T1, T2>(
      tid, env_id, model_id, g1, g2, ngeom, ncon, size, xpos, xmat, dataid,
      convex_vert, convex_vert_offset, depth_extension, multi_polygon_count,
      multi_tilt_angle, depth, normal, pos_, contact_count);

  for (int i = 0; i < ncon; i++) {
    const int offset = (tid * ncon + i) * 3;
    pos[offset + 0] = pos_[3 * (i % contact_count) + 0];
    pos[offset + 1] = pos_[3 * (i % contact_count) + 1];
    pos[offset + 2] = pos_[3 * (i % contact_count) + 2];
  }
}

template <class T1, class T2>
bool _gjk_epa_dense(cudaStream_t stream, gjk_input& input, gjk_output& output) {
  if (input.npair == 0) return true;

  const int blockSize = 256;
  const int gridSize = (input.npair * input.nenv + blockSize - 1) / blockSize;
  gjk_epa_init<<<gridSize, blockSize, 0, stream>>>(input.npair, input.nenv,
                                                   input.ncon, output.dist);
  assert_cuda("kernel::gjk_epa_init");

  // run gjk
  gjk_dense<T1, T2><<<gridSize, blockSize, 0, stream>>>(
      input.npair, input.nenv, input.ngeom, input.nmodel, input.ncon,
      input.geom_pair, input.size, input.xpos, input.xmat, input.dataid,
      input.convex_vert, input.convex_vert_offset, input.gjk_iteration_count,
      output.normal, output.simplex);
  assert_cuda("kernel::GJK");

  // run epa
  epa_dense<T1, T2><<<gridSize, blockSize, 0, stream>>>(
      input.npair, input.nenv, input.ngeom, input.nmodel, input.ncon,
      input.geom_pair, input.size, input.xpos, input.xmat, input.dataid,
      input.convex_vert, input.convex_vert_offset, input.depth_extension,
      input.epa_iteration_count, input.epa_best_count, output.simplex,
      output.dist, output.normal);
  assert_cuda("kernel::EPA");

  multiple_contacts_dense<T1, T2><<<gridSize, blockSize, 0, stream>>>(
      input.npair, input.nenv, input.ngeom, input.nmodel, input.ncon,
      input.geom_pair, input.size, input.xpos, input.xmat, input.dataid,
      input.convex_vert, input.convex_vert_offset, input.depth_extension,
      input.multi_polygon_count, input.multi_tilt_angle, output.dist,
      output.normal, output.pos);
  assert_cuda("kernel::GetMultipleContacts");

  return true;
}

bool gjk_epa_dense(cudaStream_t stream, gjk_input& in, gjk_output& out) {
  if (in.npair == 0) return true;

  const int t1 = in.geom_type0;
  const int t2 = in.geom_type1;

  // NOLINTBEGIN
  if (t1 == mjxGEOM_PLANE && t2 == mjxGEOM_SPHERE)
    return _gjk_epa_dense<GeomType_PLANE, GeomType_SPHERE>(stream, in, out);
  if (t1 == mjxGEOM_PLANE && t2 == mjxGEOM_CAPSULE)
    return _gjk_epa_dense<GeomType_PLANE, GeomType_CAPSULE>(stream, in, out);
  if (t1 == mjxGEOM_PLANE && t2 == mjxGEOM_BOX)
    return _gjk_epa_dense<GeomType_PLANE, GeomType_BOX>(stream, in, out);
  if (t1 == mjxGEOM_PLANE && t2 == mjxGEOM_ELLIPSOID)
    return _gjk_epa_dense<GeomType_PLANE, GeomType_ELLIPSOID>(stream, in, out);
  if (t1 == mjxGEOM_PLANE && t2 == mjxGEOM_CYLINDER)
    return _gjk_epa_dense<GeomType_PLANE, GeomType_CYLINDER>(stream, in, out);
  if (t1 == mjxGEOM_PLANE && t2 == mjxGEOM_MESH)
    return _gjk_epa_dense<GeomType_PLANE, GeomType_CONVEX>(stream, in, out);
  if (t1 == mjxGEOM_SPHERE && t2 == mjxGEOM_SPHERE)
    return _gjk_epa_dense<GeomType_SPHERE, GeomType_SPHERE>(stream, in, out);
  if (t1 == mjxGEOM_SPHERE && t2 == mjxGEOM_CAPSULE)
    return _gjk_epa_dense<GeomType_SPHERE, GeomType_CAPSULE>(stream, in, out);
  if (t1 == mjxGEOM_SPHERE && t2 == mjxGEOM_BOX)
    return _gjk_epa_dense<GeomType_SPHERE, GeomType_BOX>(stream, in, out);
  if (t1 == mjxGEOM_SPHERE && t2 == mjxGEOM_MESH)
    return _gjk_epa_dense<GeomType_SPHERE, GeomType_CONVEX>(stream, in, out);
  if (t1 == mjxGEOM_CAPSULE && t2 == mjxGEOM_CAPSULE)
    return _gjk_epa_dense<GeomType_CAPSULE, GeomType_CAPSULE>(stream, in, out);
  if (t1 == mjxGEOM_CAPSULE && t2 == mjxGEOM_BOX)
    return _gjk_epa_dense<GeomType_CAPSULE, GeomType_BOX>(stream, in, out);
  if (t1 == mjxGEOM_CAPSULE && t2 == mjxGEOM_ELLIPSOID)
    return _gjk_epa_dense<GeomType_CAPSULE, GeomType_ELLIPSOID>(stream, in,
                                                                out);
  if (t1 == mjxGEOM_CAPSULE && t2 == mjxGEOM_CYLINDER)
    return _gjk_epa_dense<GeomType_CAPSULE, GeomType_CYLINDER>(stream, in, out);
  if (t1 == mjxGEOM_CAPSULE && t2 == mjxGEOM_MESH)
    return _gjk_epa_dense<GeomType_CAPSULE, GeomType_CONVEX>(stream, in, out);
  if (t1 == mjxGEOM_ELLIPSOID && t2 == mjxGEOM_ELLIPSOID)
    return _gjk_epa_dense<GeomType_ELLIPSOID, GeomType_ELLIPSOID>(stream, in,
                                                                  out);
  if (t1 == mjxGEOM_ELLIPSOID && t2 == mjxGEOM_CYLINDER)
    return _gjk_epa_dense<GeomType_ELLIPSOID, GeomType_CYLINDER>(stream, in,
                                                                 out);
  if (t1 == mjxGEOM_CYLINDER && t2 == mjxGEOM_CYLINDER)
    return _gjk_epa_dense<GeomType_CYLINDER, GeomType_CYLINDER>(stream, in,
                                                                out);
  if (t1 == mjxGEOM_BOX && t2 == mjxGEOM_BOX)
    return _gjk_epa_dense<GeomType_BOX, GeomType_BOX>(stream, in, out);
  if (t1 == mjxGEOM_BOX && t2 == mjxGEOM_MESH)
    return _gjk_epa_dense<GeomType_BOX, GeomType_CONVEX>(stream, in, out);
  if (t1 == mjxGEOM_MESH && t2 == mjxGEOM_MESH)
    return _gjk_epa_dense<GeomType_CONVEX, GeomType_CONVEX>(stream, in, out);
  // NOLINTEND

  return true;
}

// Runs GJK and EPA on a set of sparse geom pairs per env.
template <class T1, class T2>
__global__ void gjk_epa_sparse(
    const uint group_key, const uint nenv, const uint ngeom, const uint nmodel,
    const uint ncon, const uint max_contact_points_per_env,
    const uint* __restrict type_pair_env_id,
    const uint* __restrict type_pair_geom_id,
    const uint* __restrict type_pair_count,
    const uint* __restrict type_pair_offset, const float* __restrict size,
    const float* __restrict xpos, const float* __restrict xmat,
    const int* __restrict dataid, const float* __restrict convex_vert,
    const uint* __restrict convex_vert_offset, const int gjk_iteration_count,
    const int epa_iteration_count, const int epa_best_count,
    const float depth_extension, const int multi_polygon_count,
    const float multi_tilt_angle, uint* __restrict env_contact_counter,
    int* __restrict geom1, int* __restrict geom2, float* __restrict dist,
    float* __restrict pos, float* __restrict contact_normal) {
  const uint tid = blockIdx.x * blockDim.x + threadIdx.x;
  const uint npair = type_pair_count[group_key];
  if (tid >= npair) return;

  const int type_pair_id = type_pair_offset[group_key] * nenv + tid;
  const uint env_id = type_pair_env_id[type_pair_id];

  // Check if we generated max contacts for this env.
  // TODO(btaba): move max_contact_points_per_env culling to a point later
  // in the pipline, where we can do a sort on penetration depth per env.
  if (env_contact_counter[env_id] > max_contact_points_per_env) return;

  const uint model_id = env_id % nmodel;
  const int g1 = type_pair_geom_id[type_pair_id * 2];
  const int g2 = type_pair_geom_id[type_pair_id * 2 + 1];

  float3 normal;
  float4 simplex_f4[3];
  _gjk<T1, T2>(tid, env_id, model_id, g1, g2, ngeom, size, xpos, xmat, dataid,
               convex_vert, convex_vert_offset, gjk_iteration_count, simplex_f4,
               normal);
  // TODO(btaba): get depth from GJK, conditionally run EPA.
  float depth = -__FLT_MAX__;
  _epa<T1, T2>(tid, env_id, model_id, g1, g2, ngeom, ncon, size, xpos, xmat,
               dataid, convex_vert, convex_vert_offset, depth_extension,
               epa_iteration_count, epa_best_count, simplex_f4, normal, depth);

  // TODO(btaba): add support for margin here.
  if (depth < 0.0) return;

  // TODO(btaba): split get_multiple_contacts into a separate kernel.
  int contact_count = 0;
  float pos_[12];
  _get_multiple_contacts<T1, T2>(
      tid, env_id, model_id, g1, g2, ngeom, ncon, size, xpos, xmat, dataid,
      convex_vert, convex_vert_offset, depth_extension, multi_polygon_count,
      multi_tilt_angle, depth, normal, pos_, contact_count);

  contact_count = contact_count > max_contact_points_per_env
                      ? max_contact_points_per_env
                      : contact_count;
  int cid = atomicAdd(&env_contact_counter[env_id], contact_count);
  cid = cid + env_id * max_contact_points_per_env;
  for (int i = 0; i < contact_count; i++) {
    dist[cid + i] = -depth;
    geom1[cid + i] = g1;
    geom2[cid + i] = g2;
    contact_normal[(cid + i) * 3] = normal.x;
    contact_normal[(cid + i) * 3 + 1] = normal.y;
    contact_normal[(cid + i) * 3 + 2] = normal.z;
    pos[(cid + i) * 3] = pos_[i * 3];
    pos[(cid + i) * 3 + 1] = pos_[i * 3 + 1];
    pos[(cid + i) * 3 + 2] = pos_[i * 3 + 2];
  }
}

template <class T1, class T2>
void _narrowphase(cudaStream_t stream, CollisionInput& input,
                  CollisionOutput& output, const int t1, const int t2) {
  const uint group_key = t1 + t2 * input.n_geom_types;
  thrust::device_ptr<uint> type_pair_count(output.type_pair_count);
  const uint npair = type_pair_count[group_key];
  if (npair == 0) return;

  const int blockSize = 256;
  const int gridSize = (npair + blockSize - 1) / blockSize;
  const int ncon = maxContactPointsMap[t1][t2];
  gjk_epa_sparse<T1, T2><<<gridSize, blockSize, 0, stream>>>(
      group_key, input.nenv, input.ngeom, input.nmodel, ncon,
      /*max_contact_points_per_env=*/input.max_contact_points,
      output.type_pair_env_id, output.type_pair_geom_id, output.type_pair_count,
      input.type_pair_offset, input.geom_size, input.geom_xpos, input.geom_xmat,
      input.geom_dataid, input.convex_vert, input.convex_vert_offset,
      input.gjk_iteration_count, input.epa_iteration_count,
      input.epa_best_count, input.depth_extension, input.multi_polygon_count,
      input.multi_tilt_angle, output.env_counter, output.g1, output.g2,
      output.dist, output.pos, output.normal);
  assert_cuda("kernel::gjk_epa_sparse");
}

void narrowphase(cudaStream_t s, CollisionInput& in, CollisionOutput& out) {
  for (int t2 = 0; t2 < mjxGEOM_size; t2++) {
    for (int t1 = 0; t1 <= t2; t1++) {
      // NOLINTBEGIN
      // TODO(btaba): move functions into lookup table.
      if (t1 == mjxGEOM_PLANE && t2 == mjxGEOM_SPHERE)
        _narrowphase<GeomType_PLANE, GeomType_SPHERE>(s, in, out, t1, t2);
      else if (t1 == mjxGEOM_PLANE && t2 == mjxGEOM_CAPSULE)
        _narrowphase<GeomType_PLANE, GeomType_CAPSULE>(s, in, out, t1, t2);
      else if (t1 == mjxGEOM_PLANE && t2 == mjxGEOM_BOX)
        _narrowphase<GeomType_PLANE, GeomType_BOX>(s, in, out, t1, t2);
      else if (t1 == mjxGEOM_PLANE && t2 == mjxGEOM_ELLIPSOID)
        _narrowphase<GeomType_PLANE, GeomType_ELLIPSOID>(s, in, out, t1, t2);
      else if (t1 == mjxGEOM_PLANE && t2 == mjxGEOM_CYLINDER)
        _narrowphase<GeomType_PLANE, GeomType_CYLINDER>(s, in, out, t1, t2);
      else if (t1 == mjxGEOM_PLANE && t2 == mjxGEOM_MESH)
        _narrowphase<GeomType_PLANE, GeomType_CONVEX>(s, in, out, t1, t2);
      else if (t1 == mjxGEOM_SPHERE && t2 == mjxGEOM_SPHERE)
        _narrowphase<GeomType_SPHERE, GeomType_SPHERE>(s, in, out, t1, t2);
      else if (t1 == mjxGEOM_SPHERE && t2 == mjxGEOM_CAPSULE)
        _narrowphase<GeomType_SPHERE, GeomType_CAPSULE>(s, in, out, t1, t2);
      else if (t1 == mjxGEOM_SPHERE && t2 == mjxGEOM_BOX)
        _narrowphase<GeomType_SPHERE, GeomType_BOX>(s, in, out, t1, t2);
      else if (t1 == mjxGEOM_SPHERE && t2 == mjxGEOM_MESH)
        _narrowphase<GeomType_SPHERE, GeomType_CONVEX>(s, in, out, t1, t2);
      else if (t1 == mjxGEOM_CAPSULE && t2 == mjxGEOM_CAPSULE)
        _narrowphase<GeomType_CAPSULE, GeomType_CAPSULE>(s, in, out, t1, t2);
      else if (t1 == mjxGEOM_CAPSULE && t2 == mjxGEOM_BOX)
        _narrowphase<GeomType_CAPSULE, GeomType_BOX>(s, in, out, t1, t2);
      else if (t1 == mjxGEOM_CAPSULE && t2 == mjxGEOM_ELLIPSOID)
        _narrowphase<GeomType_CAPSULE, GeomType_ELLIPSOID>(s, in, out, t1, t2);
      else if (t1 == mjxGEOM_CAPSULE && t2 == mjxGEOM_CYLINDER)
        _narrowphase<GeomType_CAPSULE, GeomType_CYLINDER>(s, in, out, t1, t2);
      else if (t1 == mjxGEOM_CAPSULE && t2 == mjxGEOM_MESH)
        _narrowphase<GeomType_CAPSULE, GeomType_CONVEX>(s, in, out, t1, t2);
      else if (t1 == mjxGEOM_ELLIPSOID && t2 == mjxGEOM_ELLIPSOID)
        _narrowphase<GeomType_ELLIPSOID, GeomType_ELLIPSOID>(s, in, out, t1,
                                                             t2);
      else if (t1 == mjxGEOM_ELLIPSOID && t2 == mjxGEOM_CYLINDER)
        _narrowphase<GeomType_ELLIPSOID, GeomType_CYLINDER>(s, in, out, t1, t2);
      else if (t1 == mjxGEOM_CYLINDER && t2 == mjxGEOM_CYLINDER)
        _narrowphase<GeomType_CYLINDER, GeomType_CYLINDER>(s, in, out, t1, t2);
      else if (t1 == mjxGEOM_BOX && t2 == mjxGEOM_BOX)
        _narrowphase<GeomType_BOX, GeomType_BOX>(s, in, out, t1, t2);
      else if (t1 == mjxGEOM_BOX && t2 == mjxGEOM_MESH)
        _narrowphase<GeomType_BOX, GeomType_CONVEX>(s, in, out, t1, t2);
      else if (t1 == mjxGEOM_MESH && t2 == mjxGEOM_MESH)
        _narrowphase<GeomType_CONVEX, GeomType_CONVEX>(s, in, out, t1, t2);
      // NOLINTEND
    }
  }
}

}  // namespace mujoco::mjx::cuda



/**
 * @brief Narrowphase collision handling via GJK/EPA for convex shapes.
 * 
 * @param geom_pair shape (npair, 2)
 * @param geom_xpos shape (npair, 3), TODO use vec3
 * @param geom_xmat TODO use mat33
 * @param geom_size TODO use vec3
 * @param geom_dataid 
 * @param convex_vert 
 * @param convex_vert_offset 
 * @param ngeom 
 * @param npair 
 * @param ncon 
 * @param geom_type0 
 * @param geom_type1 
 * @param depth_extension 
 * @param gjk_iteration_count 
 * @param epa_iteration_count 
 * @param epa_best_count 
 * @param multi_polygon_count 
 * @param multi_tilt_angle 
 * @param dist output
 * @param pos output
 * @param normal output
 * @param simplex output
 */
WP_API void epa_gjk_device(
    wp::array_t<wp::int32> geom_pair,
    wp::array_t<wp::float32> geom_xpos,
    wp::array_t<wp::float32> geom_xmat,
    wp::array_t<wp::float32> geom_size,
    wp::array_t<wp::int32> geom_dataid,
    wp::array_t<wp::float32> convex_vert,
    wp::array_t<wp::uint32> convex_vert_offset,
    wp::uint32 ngeom,
    wp::uint32 npair,
    wp::uint32 ncon,
    wp::uint32 geom_type0,
    wp::uint32 geom_type1,
    wp::float32 depth_extension,
    wp::uint32 gjk_iteration_count,
    wp::uint32 epa_iteration_count,
    wp::uint32 epa_best_count,
    wp::uint32 multi_polygon_count,
    wp::float32 multi_tilt_angle,
    // outputs
    wp::array_t<wp::float32> dist,
    wp::array_t<wp::float32> pos,
    wp::array_t<wp::float32> normal,
    wp::array_t<wp::float32> simplex
) {
  mujoco::mjx::cuda::gjk_input input;
  input.ngeom = ngeom;
  input.ncon = ncon;
  input.npair = npair;

  // Get the batch size of mjx.Data.
  input.nenv = 1;
  for (int i = 0; i < geom_xpos.ndim - 1; i++) {
    input.nenv *= geom_xpos.shape[i];
  }
  input.nenv /= ngeom;
  if (input.nenv == 0) {
    throw std::runtime_error(
            "Batch size of mjx.Data calculated in LaunchKernel_GJK_EPA "
            "is 0.");
  }

  // Get the batch size of mjx.Model.
  input.nmodel = 1;
  for (int i = 0; i < geom_size.ndim - 1; i++) {
    input.nmodel *= geom_size.shape[i];
  }
  input.nmodel /= ngeom;
  if (input.nmodel == 0) {
    throw std::runtime_error(
            "Batch size of mjx.Model calculated in LaunchKernel_GJK_EPA "
            "is 0.");
  }
  if (input.nmodel > 1 && input.nmodel != input.nenv) {
    throw std::runtime_error(
            "Batch size of mjx.Model is greater than 1 and does not match the "
            "batch size of mjx.Data in LaunchKernel_GJK_EPA.");
  }

  // if (geom_dataid.shape[0] != ngeom) {
  //   throw std::runtime_error(
  //           "Dimensions of geom_dataid in LaunchKernel_GJK_EPA "
  //           "do not match (ngeom,).");
  // }

  // TODO(btaba): assert all input/output shapes.
  input.geom_pair = geom_pair.data;
  input.depth_extension = depth_extension;
  input.gjk_iteration_count = gjk_iteration_count;
  input.epa_iteration_count = epa_iteration_count;
  input.epa_best_count = epa_best_count;
  input.multi_polygon_count = multi_polygon_count;
  input.multi_tilt_angle = multi_tilt_angle;
  input.xpos = geom_xpos.data;
  input.xmat = geom_xmat.data;
  input.size = geom_size.data;
  input.dataid = geom_dataid.data;
  input.geom_type0 = geom_type0;
  input.geom_type1 = geom_type1;
  input.convex_vert = convex_vert.data;
  input.convex_vert_offset = convex_vert_offset.data;

  mujoco::mjx::cuda::gjk_output output;
  output.dist = dist.data;
  output.pos = pos.data;
  output.normal = normal.data;
  output.simplex = (float4*)simplex.data;

  cudaStream_t stream = static_cast<cudaStream_t>(cuda_stream_get_current());

  if (!mujoco::mjx::cuda::gjk_epa_dense(stream, input, output)) {
    throw std::runtime_error("CUDA error::gjk_epa");
  }
}