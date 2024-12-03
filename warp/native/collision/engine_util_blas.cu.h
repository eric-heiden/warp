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

#ifndef MUJOCO_PYTHON_MJX_CUDA_ENGINE_UTIL_BLAS_CU_H_
#define MUJOCO_PYTHON_MJX_CUDA_ENGINE_UTIL_BLAS_CU_H_

#include <cstring>

namespace mujoco::mjx::cuda {

#define FULL_MASK 0xFFFFFFFF
#define PI 3.14159265358979f
#define MINVAL 1E-15f

inline __device__ void zero(float* x, int n) {
  memset(x, 0, n * sizeof(float));
}

inline __device__ void copy(float* dst, const float* src, int n) {
  memcpy(dst, src, n * sizeof(float));
}

inline __device__ void addTo(float* res, const float* vec, int n) {
  for (int i = 0; i < n; ++i) {
    res[i] += vec[i];
  }
}

inline __device__ void addToScl(float* res, const float* vec, float scl,
                                int n) {
  for (int i = 0; i < n; ++i) {
    res[i] += vec[i] * scl;
  }
}

// res = vec1 - vec2
inline __device__ void sub(float* res, const float* vec1, const float* vec2,
                           int n) {
  for (int i = 0; i < n; ++i) {
    res[i] = vec1[i] - vec2[i];
  }
}

inline __device__ float sign(float x) { return x < 0.0 ? -1.0 : 1.0; }

inline __device__ float3 sign(float3 x) {
  return make_float3(sign(x.x), sign(x.y), sign(x.z));
}

// ------------------------ Matrix Multiplication. -----------------------------

// multiply 3-by-3 matrix by vector
__forceinline__ __device__ void mulMatVec3(float res[3], const float mat[9],
                                           const float vec[3]) {
  res[0] = mat[0] * vec[0] + mat[1] * vec[1] + mat[2] * vec[2];
  res[1] = mat[3] * vec[0] + mat[4] * vec[1] + mat[5] * vec[2];
  res[2] = mat[6] * vec[0] + mat[7] * vec[1] + mat[8] * vec[2];
}

__forceinline__ __device__ void mulMatVec3(float3& res, const float4* mat,
                                           const float3& vec) {
  res.x = mat[0].x * vec.x + mat[0].y * vec.y + mat[0].z * vec.z;
  res.y = mat[1].x * vec.x + mat[1].y * vec.y + mat[1].z * vec.z;
  res.z = mat[2].x * vec.x + mat[2].y * vec.y + mat[2].z * vec.z;
}

// multiply transposed 3-by-3 matrix by vector
__forceinline__ __device__ void mulMatTVec3(float3& res, const float4* mat,
                                            const float3& vec) {
  res.x = mat[0].x * vec.x + mat[1].x * vec.y + mat[2].x * vec.z;
  res.y = mat[0].y * vec.x + mat[1].y * vec.y + mat[2].y * vec.z;
  res.z = mat[0].z * vec.x + mat[1].z * vec.y + mat[2].z * vec.z;
}

// ----------------------------- Normalize. ------------------------------------
__forceinline__ __device__ void normalize(float* __restrict x, int n) {
  float norm = 0.0f;
  for (int i = 0; i < n; ++i) {
    norm += x[i] * x[i];
  }
  norm = sqrt(norm);
  if (norm < MINVAL) {
    x[0] = 1.0;
    for (int i = 1; i < n; ++i) {
      x[i] = 0.0;
    }
  } else {
    for (int i = 0; i < n; ++i) {
      x[i] /= norm;
    }
  }
}

__forceinline__ __device__ void safe_norm(float3& x) {
  float norm = sqrt(x.x * x.x + x.y * x.y + x.z * x.z);
  if (norm < MINVAL) {
    x.x = 1.0;
    x.y = 0.0;
    x.z = 0.0;
  } else {
    x.x /= norm;
    x.y /= norm;
    x.z /= norm;
  }
}

__forceinline__ __device__ float norm(const float3& x) {
  return sqrtf(x.x * x.x + x.y * x.y + x.z * x.z);
}

}  // namespace mujoco::mjx::cuda

#endif  // MUJOCO_PYTHON_MJX_CUDA_ENGINE_UTIL_BLAS_CU_H_
