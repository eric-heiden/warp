# Copyright 2024 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Convex collision cuda functions."""

from typing import Tuple
import jax
from jax import numpy as jp
from jax.extend import ffi
# pylint: disable=g-importing-member
from mujoco.mjx._src.types import Data
from mujoco.mjx._src.types import Model
# pylint: enable=g-importing-member
# from mujoco.mjx._src.cuda import _engine_collision_convex
import numpy as np

# ffi.register_ffi_target(
#     'gjk_epa_cuda', _engine_collision_convex.gjk_epa(), platform='CUDA'
# )


def get_convex_vert(m: Model) -> Tuple[jax.Array, jax.Array]:
  convex_vert, convex_vert_offset = [], [0]
  nvert = 0
  for mesh in m.mesh_convex:
    if mesh is not None:
      nvert += mesh.vert.shape[0]
      convex_vert.append(mesh.vert)
    convex_vert_offset.append(nvert)

  convex_vert = jp.concatenate(convex_vert) if nvert else jp.array([])
  convex_vert_offset = jp.array(convex_vert_offset, dtype=jp.uint32)
  return convex_vert, convex_vert_offset


def gjk_epa(
    m: Model,
    d: Data,
    geom_pair: jax.Array,
    types: Tuple[int, int],
    ncon: int,
    ngeom: int,
    depth_extension: float,
    gjk_iter: int,
    epa_iter: int,
    epa_best_count: int,
    multi_polygon_count: int,
    multi_tilt_angle: float,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
  """GJK/EPA narrowphase routine."""
  if ngeom <= 0:
    raise ValueError(f'ngeom should be positive, got "{ngeom}".')
  if ncon <= 0:
    raise ValueError(f'ncon should be positive, got "{ncon}".')
  if len(d.geom_xpos.shape) != 2:
    raise ValueError(
        f'd.geom_xpos should have 2d shape, got "{len(d.geom_xpos.shape)}".'
    )
  if len(d.geom_xmat.shape) != 3:
    raise ValueError(
        f'd.geom_xmat should have 3d shape, got "{len(d.geom_xmat.shape)}".'
    )
  if m.geom_size.shape[0] != ngeom:
    raise ValueError(
        f'm.geom_size.shape[0] should be ngeom ({ngeom}), '
        f'got "{m.geom_size.shape[0]}".'
    )
  if m.geom_dataid.shape != (ngeom,):
    raise ValueError(
        f'm.geom_dataid.shape should be (ngeom,) == ({ngeom},), got'
        f' "({m.geom_dataid.shape[0]},)".'
    )
  if len(geom_pair.shape) != 2:
    raise ValueError('Expecting 2D geom_pair.')
  if geom_pair.shape[1] != 2:
    raise ValueError(
        f'geom_pair.shape[1] should be 2, got "{geom_pair.shape[1]}".'
    )

  npair = geom_pair.shape[0]
  n_points = ncon * npair
  out_types = (
      jax.ShapeDtypeStruct((n_points,), dtype=jp.float32),  # dist
      jax.ShapeDtypeStruct((n_points, 3), dtype=jp.float32),  # pos
      jax.ShapeDtypeStruct((npair, 3), dtype=jp.float32),  # normal
      jax.ShapeDtypeStruct((npair, 12), dtype=jp.float32),  # simplex
  )

  # TODO(btaba): consider passing in sliced geom_xpos/xmat instead for perf.
  convex_vert, convex_vert_offset = get_convex_vert(m)
  dist, pos, normal, _ = ffi.ffi_call(
      'gjk_epa_cuda',
      out_types,
      geom_pair,
      d.geom_xpos,
      d.geom_xmat,
      m.geom_size,
      m.geom_dataid,
      convex_vert,
      convex_vert_offset,
      ngeom=np.uint32(ngeom),
      npair=np.uint32(npair),
      ncon=np.uint32(ncon),
      geom_type0=np.uint32(types[0]),
      geom_type1=np.uint32(types[1]),
      depth_extension=np.float32(depth_extension),
      gjk_iteration_count=np.uint32(gjk_iter),
      epa_iteration_count=np.uint32(epa_iter),
      epa_best_count=np.uint32(epa_best_count),
      multi_polygon_count=np.uint32(multi_polygon_count),
      multi_tilt_angle=np.float32(multi_tilt_angle),
      vectorized=True,
  )

  return dist, pos, normal
