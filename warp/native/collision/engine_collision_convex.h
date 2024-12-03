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

#ifndef MUJOCO_PYTHON_MJX_CUDA_ENGINE_COLLISION_CONVEX_H_
#define MUJOCO_PYTHON_MJX_CUDA_ENGINE_COLLISION_CONVEX_H_

#include <array>

#include <driver_types.h>  // cuda
#include <vector_types.h>  // cuda
#include "helper_math.h"  // cuda_samples
#include "engine_collision_common.h"  // mjx/cuda
#include "engine_util_blas.cu.h"  // mjx/cuda

#define mjMINVAL 1E-15f


namespace mujoco::mjx::cuda {

static const int mjxGEOM_PLANE = 0;
static const int mjxGEOM_HFIELD = 1;
static const int mjxGEOM_SPHERE = 2;
static const int mjxGEOM_CAPSULE = 3;
static const int mjxGEOM_ELLIPSOID = 4;
static const int mjxGEOM_CYLINDER = 5;
static const int mjxGEOM_BOX = 6;
static const int mjxGEOM_MESH = 7;
static const int mjxGEOM_size = 8;

static constexpr std::array<std::array<int, mjxGEOM_size>, mjxGEOM_size>
maxContactPointsMap = {{
    /*PLANE  HFIELD SPHERE CAPSULE ELLIPSOID CYLINDER BOX  MESH*/
    {0,     0,     1,     2,       1,        3,      4,    4},  // PLANE
    {0,     0,     1,     2,       1,        3,      4,    4},  // HFIELD
    {0,     0,     1,     1,       1,        1,      1,    4},  // SPHERE
    {0,     0,     0,     1,       1,        2,      2,    2},  // CAPSULE
    {0,     0,     0,     0,       1,        1,      1,    1},  // ELLIPSOID
    {0,     0,     0,     0,       0,        3,      3,    3},  // CYLINDER
    {0,     0,     0,     0,       0,        0,      4,    4},  // BOX
    {0,     0,     0,     0,       0,        0,      0,    4},  // MESH
}};

struct gjk_input {
  unsigned int npair = 0;  // number of geom pairs
  unsigned int ngeom = 0;  // ngeom
  unsigned int nenv = 0;   // nenv, the batch size of mjx.Data
  unsigned int nmodel = 0;  // nmodel, the batch size of mjx.Model
                            // either set to 1 or to nenv
  int* geom_pair = NULL;   // geom_pair tuples
  float* size = NULL;      // geom_size
  int* dataid = NULL;      // geom_dataid
  float* xpos = NULL;      // geom_xpos
  float* xmat = NULL;      // geom_xmat

  uint geom_type0 = NULL;  // geom_type[0]
  uint geom_type1 = NULL;  // geom_type[1]

  float* convex_vert = NULL;        // convex mesh vertices
  uint* convex_vert_offset = NULL;  // convex mesh vertex offsets
  unsigned int ncon = 0;  // max number of points to generate per pair.

  // for sparse collisions per env
  int* type_pair_env_id = NULL;   // the env id for each pair
  int* type_pair_geom_id = NULL;  // the geom id for each pair
  int max_contact_points = 0;     // the max number of contacts per env

  // gjk/epa parameters
  float depth_extension = 0.0f;
  int gjk_iteration_count = 0;
  int epa_iteration_count = 0;
  int epa_best_count = 0;
  int multi_polygon_count = 0;
  float multi_tilt_angle = 0.0f;
};

struct gjk_output {
  float* dist = NULL;
  float* pos = NULL;
  float* normal = NULL;
  float4* simplex = NULL;
  // for sparse collisions per env
  int* env_contact_counter = NULL;  // the number of contacts for each env
};

struct GeomType_PLANE {
  float4 mat[3];
};

struct GeomType_SPHERE {
  float4 mat[3];
  float radius = 0.0f;
};

struct GeomType_CAPSULE {
  float4 mat[3];
  float radius = 0.0f;
  float halfsize = 0.0f;
};

struct GeomType_ELLIPSOID {
  float4 mat[3];
  float3 size;
};

struct GeomType_CYLINDER {
  float4 mat[3];
  float radius = 0.0f;
  float halfsize = 0.0f;
};

struct GeomType_BOX {
  float4 mat[3];
  float3 size;
};

struct GeomType_CONVEX {
  float4 mat[3];
  int vert_offset = 0;
  int vert_count = 0;
};

}  // namespace mujoco::mjx::cuda

#endif  // MUJOCO_PYTHON_MJX_CUDA_ENGINE_COLLISION_CONVEX_H_
