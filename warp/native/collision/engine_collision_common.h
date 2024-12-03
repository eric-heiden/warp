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

#ifndef MUJOCO_PYTHON_MJX_CUDA_ENGINE_COLLISION_COMMON_H_
#define MUJOCO_PYTHON_MJX_CUDA_ENGINE_COLLISION_COMMON_H_


struct CollisionInput {
  // Data.
  float* geom_xpos = NULL;
  float* geom_xmat = NULL;
  float* geom_size = NULL;
  // Model.
  int* geom_type = NULL;
  int* geom_contype = NULL;
  int* geom_conaffinity = NULL;
  int* geom_priority = NULL;
  float* geom_margin = NULL;
  float* geom_gap = NULL;
  float* geom_solmix = NULL;
  float* geom_friction = NULL;
  float* geom_solref = NULL;
  float* geom_solimp = NULL;
  float* geom_aabb = NULL;
  float* geom_rbound = NULL;
  int* geom_dataid = NULL;
  int* geom_bodyid = NULL;
  int* body_parentid = NULL;
  int* body_weldid = NULL;
  int* body_contype = NULL;
  int* body_conaffinity = NULL;
  int* body_geomadr = NULL;
  uint* body_geomnum = NULL;
  uint* body_has_plane = NULL;
  int* pair_geom1 = NULL;
  int* pair_geom2 = NULL;
  int* exclude_signature = NULL;
  float* pair_margin = NULL;
  float* pair_gap = NULL;
  float* pair_friction = NULL;
  float* pair_solref = NULL;
  float* pair_solimp = NULL;
  float* convex_vert = NULL;
  uint* convex_vert_offset = NULL;
  uint* type_pair_offset = NULL;
  // Static arguments.
  uint ngeom = 0;
  uint npair = 0;
  uint nbody = 0;
  uint nexclude = 0;
  uint max_contact_points = 0;
  uint n_geom_pair = 0;
  uint n_geom_types = 0;
  bool filter_parent = false;
  // Derived arguments.
  uint nenv = 0;
  uint nmodel = 0;
  // GJK/EPA arguments.
  float depth_extension = 0.0f;
  uint gjk_iteration_count = 0;
  uint epa_iteration_count = 0;
  uint epa_best_count = 0;
  uint multi_polygon_count = 0;
  float multi_tilt_angle = 0.0f;
};

struct CollisionOutput {
  float* dist = NULL;
  float* pos = NULL;
  float* normal = NULL;
  int* g1 = NULL;
  int* g2 = NULL;
  float* includemargin = NULL;
  float* friction = NULL;
  float* solref = NULL;
  float* solreffriction = NULL;
  float* solimp = NULL;
  // Output buffers used internally.
  float* dyn_body_aamm = NULL;
  int* col_body_pair = NULL;
  uint* env_counter = NULL;
  uint* env_counter2 = NULL;
  uint* env_offset = NULL;
  float* dyn_geom_aabb = NULL;
  int* col_geom_pair = NULL;
  uint* type_pair_env_id = NULL;
  uint* type_pair_geom_id = NULL;
  uint* type_pair_count = NULL;
};

#endif  // MUJOCO_PYTHON_MJX_CUDA_ENGINE_COLLISION_COMMON_H_
