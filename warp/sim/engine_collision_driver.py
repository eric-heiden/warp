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
"""Collision driver in CUDA."""

import collections
import itertools
from typing import Iterator, Tuple, Union

import jax
from jax import numpy as jp
from jax.extend import ffi
import mujoco
from mujoco.mjx._src import math
# pylint: disable=g-importing-member
from mujoco.mjx._src.types import Contact
from mujoco.mjx._src.types import Data
from mujoco.mjx._src.types import DisableBit
from mujoco.mjx._src.types import GeomType
from mujoco.mjx._src.types import Model

#from convex import gjk_epa_dense
from convex import _narrowphase

# pylint: enable=g-importing-member
# from mujoco.mjx._src.cuda import _engine_collision_driver
# from mujoco.mjx._src.cuda import engine_collision_convex
# import engine_collision_convex

import convex

import numpy as np

import warp as wp




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





@wp.kernel
def init_buffers(
    nenv: int,
    col_body_pair_count: wp.array(dtype=int),
    col_body_pair_offset: wp.array(dtype=int),
    col_geom_pair_count: wp.array(dtype=int),
):
    tid = wp.tid()
    if tid >= nenv:
        return

    col_body_pair_offset[tid] = 0
    col_body_pair_count[tid] = 0
    col_geom_pair_count[tid] = 0


@wp.kernel
def get_dyn_body_aamm(
    nenv: int, nbody: int, nmodel: int, ngeom: int,
    body_geomnum: wp.array(dtype=int),
    body_geomadr: wp.array(dtype=int),
    geom_margin: wp.array(dtype=float),
    geom_xpos: wp.array(dtype=wp.vec3),
    geom_rbound: wp.array(dtype=float),
    dyn_body_aamm: wp.array(dtype=float),
):
    tid = wp.tid()
    if tid >= nenv * nbody:
        return

    bid = tid % nbody
    env_id = tid // nbody
    model_id = env_id % nmodel

    # Initialize AAMM with extreme values
    aamm_min = wp.vec3(1000000000.0, 1000000000.0, 1000000000.0)
    aamm_max = wp.vec3(-1000000000.0, -1000000000.0, -1000000000.0)

    # Iterate over all geometries associated with the body
    for i in range(body_geomnum[bid]):
        g = body_geomadr[bid] + i

        for j in range(3):
            pos = geom_xpos[(env_id * ngeom + g)][j]
            rbound = geom_rbound[model_id * ngeom + g]
            margin = geom_margin[model_id * ngeom + g]

            min_val = pos - rbound - margin
            max_val = pos + rbound + margin           

            aamm_min[j] = wp.min(aamm_min[j], min_val)
            aamm_max[j] = wp.max(aamm_max[j], max_val)


    # Write results to output
    dyn_body_aamm[tid * 6 + 0] = aamm_min[0]
    dyn_body_aamm[tid * 6 + 1] = aamm_min[1]
    dyn_body_aamm[tid * 6 + 2] = aamm_min[2]
    dyn_body_aamm[tid * 6 + 3] = aamm_max[0]
    dyn_body_aamm[tid * 6 + 4] = aamm_max[1]
    dyn_body_aamm[tid * 6 + 5] = aamm_max[2]


# @wp.kernel
# def get_dyn_body_aamm(
#     nenv: int, nbody: int, nmodel: int, ngeom: int,
#     body_geomnum: wp.array(dtype=int),
#     body_geomadr: wp.array(dtype=int),
#     geom_margin: wp.array(dtype=float),
#     geom_xpos: wp.array(dtype=wp.vec3),
#     geom_rbound: wp.array(dtype=float),
#     dyn_body_aamm: wp.array(dtype=float),
# ):
#     tid = wp.tid()
#     if tid >= nenv * nbody:
#         return

#     bid = tid % nbody
#     env_id = tid // nbody
#     model_id = env_id % nmodel

#     # Initialize AAMM with extreme values
#     # aamm_min = wp.vec3(1000000000.0, 1000000000.0, 1000000000.0)
#     # aamm_max = wp.vec3(-1000000000.0, -1000000000.0, -1000000000.0)

#     for j in range(3):
#         dyn_body_aamm[tid * 6 + j] = 1000000000.0
#     for j in range(3):
#         dyn_body_aamm[tid * 6 + j + 3] = -1000000000.0

#     # Iterate over all geometries associated with the body
#     for i in range(body_geomnum[bid]):
#         g = body_geomadr[bid] + i

#         for j in range(3):
#             pos = geom_xpos[(env_id * ngeom + g)][j]
#             rbound = geom_rbound[model_id * ngeom + g]
#             margin = geom_margin[model_id * ngeom + g]

#             min_val = pos - rbound - margin
#             max_val = pos + rbound + margin
#             # print(min_val)
#             # print(max_val)

#             dyn_body_aamm[tid * 6 + j] = wp.min(dyn_body_aamm[tid * 6 + j], min_val)
#             dyn_body_aamm[tid * 6 + j + 3] = wp.max(dyn_body_aamm[tid * 6 + j + 3], max_val)




@wp.func
def map_body_pair_nxn(tid: int, nenv: int, nbody: int) -> int:
    
    if tid >= nenv * nbody * nbody:
        return -1
    
    body_pair_id = tid % (nbody * nbody)
    body1 = body_pair_id // nbody
    body2 = body_pair_id % nbody
    
    return wp.select(body1 < body2, body1 + body2 * nbody, -1)


@wp.kernel
def get_body_pairs_nxn(
    nenv: int,
    nbody: int,
    filter_parent: bool,
    nexclude: int,
    body_parentid: wp.array(dtype=int),
    body_weldid: wp.array(dtype=int),
    body_contype: wp.array(dtype=int),
    body_conaffinity: wp.array(dtype=int),
    body_has_plane: wp.array(dtype=int),
    exclude_signature: wp.array(dtype=int),
    dyn_body_aamm: wp.array(dtype=float),
    col_body_pair: wp.array(dtype=int),
    col_body_pair_count: wp.array(dtype=int),
):
    tid = wp.tid()
    map_coord = map_body_pair_nxn(tid, nenv, nbody)

    if map_coord == -1:  # Assuming `FULL_MASK` translates to -1
        return

    env_id = tid // (nbody * nbody)
    body1 = map_coord % nbody
    body2 = map_coord // nbody

    if (body_contype[body1] == 0 and body_conaffinity[body1] == 0) or \
       (body_contype[body2] == 0 and body_conaffinity[body2] == 0):
        return

    signature = (body1 << 16) + body2
    for i in range(nexclude):
        if exclude_signature[i] == signature:
            return

    w1 = body_weldid[body1]
    w2 = body_weldid[body2]
    if w1 == w2:
        return

    w1_p = body_weldid[body_parentid[w1]]
    w2_p = body_weldid[body_parentid[w2]]
    if filter_parent and w1 != 0 and w2 != 0 and (w1 == w2_p or w2 == w1_p):
        return

    # aamm1 = dyn_body_aamm[(env_id * nbody + body1) * 6 : (env_id * nbody + body1) * 6 + 6]
    # aamm2 = dyn_body_aamm[(env_id * nbody + body2) * 6 : (env_id * nbody + body2) * 6 + 6]
    # separating = (
    #     (aamm1[0] > aamm2[3]) or
    #     (aamm1[1] > aamm2[4]) or
    #     (aamm1[2] > aamm2[5]) or
    #     (aamm2[0] > aamm1[3]) or
    #     (aamm2[1] > aamm1[4]) or
    #     (aamm2[2] > aamm1[5])
    # )
    separating = (
        (dyn_body_aamm[(env_id * nbody + body1) * 6 + 0] > dyn_body_aamm[(env_id * nbody + body2) * 6 + 3]) or (dyn_body_aamm[(env_id * nbody + body1) * 6 + 1] > dyn_body_aamm[(env_id * nbody + body2) * 6 + 4]) or
        (dyn_body_aamm[(env_id * nbody + body1) * 6 + 2] > dyn_body_aamm[(env_id * nbody + body2) * 6 + 5]) or (dyn_body_aamm[(env_id * nbody + body2) * 6 + 0] > dyn_body_aamm[(env_id * nbody + body1) * 6 + 3]) or
        (dyn_body_aamm[(env_id * nbody + body2) * 6 + 1] > dyn_body_aamm[(env_id * nbody + body1) * 6 + 4]) or (dyn_body_aamm[(env_id * nbody + body2) * 6 + 2] > dyn_body_aamm[(env_id * nbody + body1) * 6 + 5])
    )

    if separating and not (body_has_plane[body1] or body_has_plane[body2]):
        return

    idx = wp.atomic_add(col_body_pair_count, env_id, 1)
    nbody_pair = ((nbody * (nbody - 1) // 2 + 15) // 16) * 16
    col_body_pair[(env_id * nbody_pair + idx) * 2] = body1
    col_body_pair[(env_id * nbody_pair + idx) * 2 + 1] = body2

@wp.struct
class Mat3x4:
    row0 : wp.vec4
    row1 : wp.vec4
    row2 : wp.vec4

# @wp.func
# def xposmat_to_float4(xpos: wp.array(dtype=wp.vec3), xmat: wp.array(dtype=wp.float32), gi: int) -> Mat3x4:
#     result = Mat3x4()
#     pos = xpos[gi]
#     result.row0 = wp.vec4(xmat[gi * 9 + 0], xmat[gi * 9 + 1], xmat[gi * 9 + 2],
#                           pos.x)
#     result.row1 = wp.vec4(xmat[gi * 9 + 3], xmat[gi * 9 + 4], xmat[gi * 9 + 5],
#                           pos.y)
#     result.row2 = wp.vec4(xmat[gi * 9 + 6], xmat[gi * 9 + 7], xmat[gi * 9 + 8],
#                           pos.z)
#     return result
@wp.func
def xposmat_to_float4(xpos: wp.array(dtype=wp.vec3), xmat: wp.array(dtype=wp.mat33), gi: int) -> Mat3x4:
    result = Mat3x4()
    pos = xpos[gi]
    mat = xmat[gi]

    result.row0 = wp.vec4(mat[0][0], mat[0][1], mat[0][2], pos.x)
    result.row1 = wp.vec4(mat[1][0], mat[1][1], mat[1][2], pos.y)
    result.row2 = wp.vec4(mat[2][0], mat[2][1], mat[2][2], pos.z)

    return result

@wp.func
def transform_point(mat : Mat3x4, pos : wp.vec3)->wp.vec3:
    x = wp.dot(wp.vec3(mat.row0[0], mat.row0[1], mat.row0[2]), pos) + mat.row0[3]
    y = wp.dot(wp.vec3(mat.row1[0], mat.row1[1], mat.row1[2]), pos) + mat.row1[3]
    z = wp.dot(wp.vec3(mat.row2[0], mat.row2[1], mat.row2[2]), pos) + mat.row2[3]
    return wp.vec3(x, y, z)

@wp.kernel
def get_dyn_geom_aabb(nenv: int, nmodel: int, ngeom: int,
                      geom_xpos: wp.array(dtype=wp.vec3),
                      geom_xmat: wp.array(dtype=wp.mat33),
                      geom_aabb: wp.array(dtype=float),
                      dyn_aabb: wp.array(dtype=float)):
    tid = wp.tid()
    if tid >= nenv * ngeom:
        return

    env_id = tid // ngeom
    gid = tid % ngeom

    mat = xposmat_to_float4(geom_xpos, geom_xmat, env_id * ngeom + gid)

    aabb = wp.vec3(geom_aabb[gid * 6 + 3], geom_aabb[gid * 6 + 4], geom_aabb[gid * 6 + 5])
    aabb_pos = wp.vec3(geom_aabb[gid * 6], geom_aabb[gid * 6 + 1], geom_aabb[gid * 6 + 2])

    aabb_max = wp.vec3(-1000000000.0, -1000000000.0, -1000000000.0)
    aabb_min = wp.vec3(1000000000.0, 1000000000.0, 1000000000.0)

    for i in range(8):
        corner = wp.vec3(aabb.x, aabb.y, aabb.z)
        if i % 2 == 0:
            corner.x = -corner.x
        if (i // 2) % 2 == 0:
            corner.y = -corner.y
        if i < 4:
            corner.z = -corner.z 
        corner_world = transform_point(mat, corner + aabb_pos)
        aabb_max = wp.max(aabb_max, corner_world)
        aabb_min = wp.min(aabb_min, corner_world)

    dyn_aabb[tid * 6 + 0] = aabb_min[0]
    dyn_aabb[tid * 6 + 1] = aabb_min[1]
    dyn_aabb[tid * 6 + 2] = aabb_min[2]
    dyn_aabb[tid * 6 + 3] = aabb_max[0]
    dyn_aabb[tid * 6 + 4] = aabb_max[1]
    dyn_aabb[tid * 6 + 5] = aabb_max[2]



@wp.func
def bisection(x: wp.array(dtype=int), v: int, a: int, b: int) -> int:
    c = int(0)
    while b - a > 1:
        c = (a + b) // 2
        if x[c] <= v:
            a = c
        else:
            b = c
    c = a
    if c != b and x[b] <= v:
        c = b
    return c

@wp.kernel
def get_geom_pairs_nxn(nenv: int, ngeom: int, nbody: int, n_geom_pair: int,
                       body_geomnum: wp.array(dtype=int),
                       body_geomadr: wp.array(dtype=int),
                       geom_contype: wp.array(dtype=int),
                       geom_conaffinity: wp.array(dtype=int),
                       geom_type: wp.array(dtype=int),
                       geom_margin: wp.array(dtype=float),
                       dyn_geom_aabb: wp.array(dtype=float),
                       col_body_pair: wp.array(dtype=int),
                       col_body_pair_count: wp.array(dtype=int),
                       col_body_pair_offset: wp.array(dtype=int),
                       col_geom_pair: wp.array(dtype=int),
                       col_geom_pair_count: wp.array(dtype=int)):
    
    mjGEOM_PLANE = int(0)
    mjGEOM_HFIELD = int(1)

    tid = wp.tid()
    env_id = bisection(col_body_pair_offset, tid, 0, nenv - 1)
    body_pair_id = tid - col_body_pair_offset[env_id]
    if body_pair_id >= col_body_pair_count[env_id]:
        return

    nbody_pair = ((nbody * (nbody - 1) // 2 + 15) // 16) * 16
    body1 = col_body_pair[(env_id * nbody_pair + body_pair_id) * 2]
    body2 = col_body_pair[(env_id * nbody_pair + body_pair_id) * 2 + 1]

    for g1 in range(body_geomnum[body1]):
        geom1 = body_geomadr[body1] + g1
        for g2 in range(body_geomnum[body2]):
            geom2 = body_geomadr[body2] + g2

            type1 = geom_type[geom1]
            type2 = geom_type[geom2]
            skip_type = ((type1 == mjGEOM_HFIELD or type1 == mjGEOM_PLANE) and
                         (type2 == mjGEOM_HFIELD or type2 == mjGEOM_PLANE))

            skip_con = not ((geom_contype[geom1] & geom_conaffinity[geom2]) or
                            (geom_contype[geom2] & geom_conaffinity[geom1]))

            # aabb1 = dyn_geom_aabb[(env_id * ngeom + geom1) * 6:(env_id * ngeom + geom1) * 6 + 6]
            # aabb2 = dyn_geom_aabb[(env_id * ngeom + geom2) * 6:(env_id * ngeom + geom2) * 6 + 6]
            # separating = ((aabb1[0] > aabb2[3]) or (aabb1[1] > aabb2[4]) or
            #               (aabb1[2] > aabb2[5]) or (aabb2[0] > aabb1[3]) or
            #               (aabb2[1] > aabb1[4]) or (aabb2[2] > aabb1[5]))

            separating = ((dyn_geom_aabb[(env_id * ngeom + geom1) * 6 + 0] > dyn_geom_aabb[(env_id * ngeom + geom2) * 6 + 3]) or (dyn_geom_aabb[(env_id * ngeom + geom1) * 6 + 1] > dyn_geom_aabb[(env_id * ngeom + geom2) * 6 + 4]) or
                          (dyn_geom_aabb[(env_id * ngeom + geom1) * 6 + 2] > dyn_geom_aabb[(env_id * ngeom + geom2) * 6 + 5]) or (dyn_geom_aabb[(env_id * ngeom + geom2) * 6 + 0] > dyn_geom_aabb[(env_id * ngeom + geom1) * 6 + 3]) or
                          (dyn_geom_aabb[(env_id * ngeom + geom2) * 6 + 1] > dyn_geom_aabb[(env_id * ngeom + geom1) * 6 + 4]) or (dyn_geom_aabb[(env_id * ngeom + geom2) * 6 + 2] > dyn_geom_aabb[(env_id * ngeom + geom1) * 6 + 5]))

            if separating or skip_con or skip_type:
                continue

            if type1 > type2:
                tmp = geom1
                geom1 = geom2
                geom2 = tmp
                # geom1, geom2 = geom2, geom1
            pair_id = wp.atomic_add(col_geom_pair_count, env_id, 1)
            col_geom_pair[(env_id * n_geom_pair + pair_id) * 2] = geom1
            col_geom_pair[(env_id * n_geom_pair + pair_id) * 2 + 1] = geom2



@wp.kernel
def group_contacts_by_type(nenv: int, n_geom_pair: int, n_geom_types: int,
                           geom_type: wp.array(dtype=int),
                           col_geom_pair: wp.array(dtype=int),
                           col_geom_pair_count: wp.array(dtype=int),
                           col_geom_pair_offset: wp.array(dtype=int),
                           type_pair_offset: wp.array(dtype=int),
                           type_pair_env_id: wp.array(dtype=int),
                           type_pair_geom_id: wp.array(dtype=int),
                           type_pair_count: wp.array(dtype=int)):
    tid = wp.tid()
    env_id = bisection(col_geom_pair_offset, tid, 0, nenv - 1)
    pair_id = tid - col_geom_pair_offset[env_id]
    if pair_id >= col_geom_pair_count[env_id]:
        return

    geom1 = col_geom_pair[(env_id * n_geom_pair + pair_id) * 2]
    geom2 = col_geom_pair[(env_id * n_geom_pair + pair_id) * 2 + 1]

    type1 = geom_type[geom1]
    type2 = geom_type[geom2]
    group_key = type1 + type2 * n_geom_types

    n_type_pair = wp.atomic_add(type_pair_count, group_key, 1)
    type_pair_id = type_pair_offset[group_key] * nenv + n_type_pair
    type_pair_env_id[type_pair_id] = env_id
    type_pair_geom_id[type_pair_id * 2] = geom1
    type_pair_geom_id[type_pair_id * 2 + 1] = geom2


@wp.kernel
def set_zero(count: int, ptr: wp.array(dtype=int)):
    tid = wp.tid()
    if tid >= count:
        return
    ptr[tid] = 0



@wp.kernel
def get_contact_solver_params(nenv: int, nmodel: int, ngeom: int,
                              max_contact_pts: int, n_contact_pts: int,
                              geom1: wp.array(dtype=int),
                              geom2: wp.array(dtype=int),
                              geom_priority: wp.array(dtype=int),
                              geom_solmix: wp.array(dtype=float),
                              geom_friction: wp.array(dtype=float),
                              geom_solref: wp.array(dtype=float),
                              geom_solimp: wp.array(dtype=float),
                              geom_margin: wp.array(dtype=float),
                              geom_gap: wp.array(dtype=float),
                              env_contact_offset: wp.array(dtype=int),
                              includemargin: wp.array(dtype=float),
                              friction: wp.array(dtype=float),
                              solref: wp.array(dtype=float),
                              solreffriction: wp.array(dtype=float),
                              solimp: wp.array(dtype=float)):
    tid = wp.tid()
    if tid >= n_contact_pts:
        return

    mjNREF = int(2) 
    mjNIMP = int(5) 
    mjMINVAL = float(1E-15)

    env_id = bisection(env_contact_offset, tid, 0, nenv - 1)
    model_id = env_id % nmodel
    pt_id = env_id * max_contact_pts + tid - env_contact_offset[env_id]

    g1 = geom1[pt_id] + model_id * ngeom
    g2 = geom2[pt_id] + model_id * ngeom

    margin = wp.max(geom_margin[g1], geom_margin[g2])
    gap = wp.max(geom_gap[g1], geom_gap[g2])
    solmix1 = geom_solmix[g1]
    solmix2 = geom_solmix[g2]
    mix = solmix1 / (solmix1 + solmix2)
    mix = wp.select((solmix1 < mjMINVAL) and (solmix2 < mjMINVAL), 0.5, mix)
    mix = wp.select((solmix1 < mjMINVAL)and (solmix2 >= mjMINVAL), 0.0, mix)
    mix = wp.select((solmix1 >= mjMINVAL) and (solmix2 < mjMINVAL), 1.0, mix)

    p1 = geom_priority[g1]
    p2 = geom_priority[g2]
    mix = wp.select(p1 == p2, mix, wp.select(p1 > p2, 1.0, 0.0))
    is_standard = (geom_solref[0 + g1 * mjNREF] > 0) and (geom_solref[0 + g2 * mjNREF] > 0)

    # Hard code mjNREF = 2
    solref_ = wp.vec2(0.0, 0.0) # wp.zeros(mjNREF, dtype=float)
    for i in range(mjNREF):
        solref_[i] = mix * geom_solref[i + g1 * mjNREF] + (float(1) - mix) * geom_solref[i + g2 * mjNREF]
        solref_[i] = wp.select(is_standard, solref_[i], wp.min(geom_solref[i + g1 * mjNREF], geom_solref[i + g2 * mjNREF]))

    # solimp_ = wp.zeros(mjNIMP, dtype=float)
    # for i in range(mjNIMP):
    #     solimp_[i] = mix * geom_solimp[i + g1 * mjNIMP] + (1 - mix) * geom_solimp[i + g2 * mjNIMP]

    friction_ = wp.vec3(0.0, 0.0, 0.0)   # wp.zeros(3, dtype=float)
    for i in range(3):
        friction_[i] = wp.max(geom_friction[i + g1 * 3], geom_friction[i + g2 * 3])

    includemargin[tid] = margin - gap
    friction[tid * 5] = friction_[0]
    friction[tid * 5 + 1] = friction_[0]
    friction[tid * 5 + 2] = friction_[1]
    friction[tid * 5 + 3] = friction_[2]
    friction[tid * 5 + 4] = friction_[2]

    for i in range(mjNREF):
        solref[tid * mjNREF + i] = solref_[i]

    for i in range(mjNIMP):
        solimp[tid * mjNIMP + i] = mix * geom_solimp[i + g1 * mjNIMP] + (float(1) - mix) * geom_solimp[i + g2 * mjNIMP] #solimp_[i]












# ffi.register_ffi_target(
#     'collision_driver_cuda',
#     _engine_collision_driver.collision(),
#     platform='CUDA',
# )


def _get_body_has_plane(m: Model) -> np.ndarray:
  body_has_plane = [False] * m.nbody
  for i in range(m.nbody):
    start = m.body_geomadr[i]
    end = m.body_geomadr[i] + m.body_geomnum[i]
    for g in range(start, end):
      if m.geom_type[g] == GeomType.PLANE:
        body_has_plane[i] = True
        break
  return np.array(body_has_plane, dtype=np.uint32)


def _body_pairs(
    m: Union[Model, mujoco.MjModel],
) -> Iterator[Tuple[int, int]]:
  """Yields body pairs to check for collision."""
  # TODO(btaba): merge logic back into collision driver.
  exclude_signature = set(m.exclude_signature)
  geom_con = m.geom_contype | m.geom_conaffinity
  filterparent = not (m.opt.disableflags & DisableBit.FILTERPARENT)
  b_start = m.body_geomadr
  b_end = b_start + m.body_geomnum

  for b1 in range(m.nbody):
    if not geom_con[b_start[b1] : b_end[b1]].any():
      continue
    w1 = m.body_weldid[b1]
    w1_p = m.body_weldid[m.body_parentid[w1]]

    for b2 in range(b1, m.nbody):
      if not geom_con[b_start[b2] : b_end[b2]].any():
        continue
      signature = (b1 << 16) + (b2)
      if signature in exclude_signature:
        continue
      w2 = m.body_weldid[b2]
      # ignore self-collisions
      if w1 == w2:
        continue
      w2_p = m.body_weldid[m.body_parentid[w2]]
      # ignore parent-child collisions
      if filterparent and w1 != 0 and w2 != 0 and (w1 == w2_p or w2 == w1_p):
        continue
      yield b1, b2


def _geom_pairs(
    m: Union[Model, mujoco.MjModel],
) -> Iterator[Tuple[int, int, int, int]]:
  """Yields geom pairs to check for collision."""
  geom_con = m.geom_contype | m.geom_conaffinity
  b_start = m.body_geomadr
  b_end = b_start + m.body_geomnum
  for b1, b2 in _body_pairs(m):
    g1_range = [g for g in range(b_start[b1], b_end[b1]) if geom_con[g]]
    g2_range = [g for g in range(b_start[b2], b_end[b2]) if geom_con[g]]
    for g1, g2 in itertools.product(g1_range, g2_range):
      t1, t2 = m.geom_type[g1], m.geom_type[g2]
      # order pairs by geom_type for correct function mapping
      if t1 > t2:
        g1, g2, t1, t2 = g2, g1, t2, t1
      # ignore plane<>plane and plane<>hfield
      if (t1, t2) == (GeomType.PLANE, GeomType.PLANE):
        continue
      if (t1, t2) == (GeomType.PLANE, GeomType.HFIELD):
        continue
      # geoms must match contype and conaffinity on some bit
      mask = m.geom_contype[g1] & m.geom_conaffinity[g2]
      mask |= m.geom_contype[g2] & m.geom_conaffinity[g1]
      if not mask:
        continue
      yield g1, g2, t1, t2


def _get_ngeom_pair(m: Model) -> int:
  """Returns an upper bound on the number of colliding geom pairs."""
  n_geom_pair = 0
  for (*_,) in _geom_pairs(m):
    n_geom_pair += 1
  return n_geom_pair


def _get_ngeom_pair_type_offset(m: Model) -> np.ndarray:
  """Returns offsets into geom pair types."""
  geom_pair_type_count = collections.defaultdict(int)
  for *_, t1, t2 in _geom_pairs(m):
    geom_pair_type_count[(t1, t2)] += 1

  offsets = [0]
  # order according to sequential id = t1 + t2 * n_geom_types
  for t2 in range(len(GeomType)):
    for t1 in range(len(GeomType)):
      if t1 > t2:
        offsets.append(0)  # upper triangle only
        continue
      if (t1, t2) not in geom_pair_type_count:
        offsets.append(0)
      else:
        offsets.append(geom_pair_type_count[(t1, t2)])

  assert sum(offsets) == _get_ngeom_pair(m)
  return np.cumsum(offsets)[:-1]





class CollisionInput:
    def __init__(self, m: Model, d: Data, nenv : int, nmodel : int,                 
      depth_extension: float,
      gjk_iter: int,
      epa_iter: int,
      epa_best_count: int,
      multi_polygon_count: int,
      multi_tilt_angle: float,
      device           
      ):
        
        max_contact_points = d.contact.pos.shape[0]
        # n_pts = max_contact_points
        #  = int((m.nbody * (m.nbody - 1) / 2 + 15) / 16) * 16
        n_geom_pair = _get_ngeom_pair(m)

        # n_geom_types = len(GeomType)
        # n_geom_type_pairs = n_geom_types * n_geom_types
        # type_pair_offset = _get_ngeom_pair_type_offset(m)
        # type_pair_count = np.zeros(n_geom_type_pairs, dtype=np.uint32)
        convex_vert, convex_vert_offset = get_convex_vert(m)

        self.geom_xpos = wp.from_jax(d.geom_xpos, dtype = wp.vec3) #.reshape((-1,))
        self.geom_xmat = wp.from_jax(d.geom_xmat, dtype=wp.mat33) #.reshape((-1,))
        self.geom_size = wp.from_jax(m.geom_size, dtype = wp.vec3)
        self.geom_type = wp.array(m.geom_type, dtype=wp.int32)
        self.geom_contype = wp.array(m.geom_contype, dtype=wp.int32)
        self.geom_conaffinity = wp.array(m.geom_conaffinity, dtype=wp.int32)
        self.geom_priority = wp.array(m.geom_priority, dtype=wp.int32)
        self.geom_margin = wp.array(m.geom_margin, dtype=wp.float32)
        self.geom_gap = wp.array(m.geom_gap, dtype=wp.float32)
        self.geom_solmix = wp.array(m.geom_solmix, dtype=wp.float32)
        self.geom_friction = wp.array(m.geom_friction, dtype=wp.float32).reshape((-1,))
        self.geom_solref = wp.array(m.geom_solref, dtype=wp.float32).reshape((-1,))
        self.geom_solimp = wp.array(m.geom_solimp, dtype=wp.float32).reshape((-1,))
        self.geom_aabb = wp.array(m.geom_aabb, dtype=wp.float32).reshape((-1,))
        self.geom_rbound = wp.array(m.geom_rbound, dtype=wp.float32)
        self.geom_dataid = wp.array(m.geom_dataid, dtype=wp.int32)
        self.geom_bodyid = wp.array(m.geom_bodyid, dtype=wp.int32)
        self.body_parentid = wp.array(m.body_parentid, dtype=wp.int32)
        self.body_weldid = wp.array(m.body_weldid, dtype=wp.int32)
        self.body_contype = wp.array(m.body_contype, dtype=wp.int32)
        self.body_conaffinity = wp.array(m.body_conaffinity, dtype=wp.int32)
        self.body_geomadr = wp.array(m.body_geomadr, dtype=wp.int32)
        self.body_geomnum = wp.array(m.body_geomnum, dtype=wp.int32)
        self.body_has_plane = wp.array(_get_body_has_plane(m), dtype=wp.int32)
        self.pair_geom1 = wp.array(m.pair_geom1, dtype=wp.int32)
        self.pair_geom2 = wp.array(m.pair_geom2, dtype=wp.int32)
        self.exclude_signature = wp.array(m.exclude_signature, dtype=wp.int32)
        self.pair_margin = wp.array(m.pair_margin, dtype=wp.float32)
        self.pair_gap = wp.array(m.pair_gap, dtype=wp.float32)
        self.pair_friction = wp.array(m.pair_friction, dtype=wp.float32)
        self.pair_solref = wp.array(m.pair_solref, dtype=wp.float32)
        self.pair_solimp = wp.array(m.pair_solimp, dtype=wp.float32)
        self.convex_vert = wp.from_jax(convex_vert, dtype = wp.vec3)
        self.convex_vert_offset = wp.from_jax(convex_vert_offset, dtype=wp.int32)
        self.type_pair_offset = wp.array(_get_ngeom_pair_type_offset(m), dtype=wp.int32)
        self.ngeom = len(m.geom_type)
        self.npair = m.npair
        self.nbody = m.nbody
        self.nexclude = m.nexclude
        self.max_contact_points = max_contact_points
        self.n_geom_pair = n_geom_pair
        self.n_geom_types = len(GeomType)
        self.filter_parent = not (m.opt.disableflags & DisableBit.FILTERPARENT)
        self.depth_extension = depth_extension
        self.gjk_iteration_count = gjk_iter
        self.epa_iteration_count = epa_iter
        self.epa_best_count = epa_best_count
        self.multi_polygon_count = multi_polygon_count
        self.multi_tilt_angle = wp.float32(multi_tilt_angle)
        self.nenv = nenv
        self.nmodel = nmodel

    def __str__(self):
        # Return a string representation of all the members
        return (
            f"geom_xpos: {self.geom_xpos.numpy()}\n"
            f"geom_xmat: {self.geom_xmat.numpy()}\n"
            f"geom_size: {self.geom_size.numpy()}\n"
            f"geom_type: {self.geom_type.numpy()}\n"
            f"geom_contype: {self.geom_contype.numpy()}\n"
            f"geom_conaffinity: {self.geom_conaffinity.numpy()}\n"
            f"geom_priority: {self.geom_priority.numpy()}\n"
            f"geom_margin: {self.geom_margin.numpy()}\n"
            f"geom_gap: {self.geom_gap.numpy()}\n"
            f"geom_solmix: {self.geom_solmix.numpy()}\n"
            f"geom_friction: {self.geom_friction.numpy()}\n"
            f"geom_solref: {self.geom_solref.numpy()}\n"
            f"geom_solimp: {self.geom_solimp.numpy()}\n"
            f"geom_aabb: {self.geom_aabb.numpy()}\n"
            f"geom_rbound: {self.geom_rbound.numpy()}\n"
            f"geom_dataid: {self.geom_dataid.numpy()}\n"
            f"geom_bodyid: {self.geom_bodyid.numpy()}\n"
            f"body_parentid: {self.body_parentid.numpy()}\n"
            f"body_weldid: {self.body_weldid.numpy()}\n"
            f"body_contype: {self.body_contype.numpy()}\n"
            f"body_conaffinity: {self.body_conaffinity.numpy()}\n"
            f"body_geomadr: {self.body_geomadr.numpy()}\n"
            f"body_geomnum: {self.body_geomnum.numpy()}\n"
            f"body_has_plane: {self.body_has_plane.numpy()}\n"
            f"pair_geom1: {self.pair_geom1.numpy()}\n"
            f"pair_geom2: {self.pair_geom2.numpy()}\n"
            f"exclude_signature: {self.exclude_signature.numpy()}\n"
            f"pair_margin: {self.pair_margin.numpy()}\n"
            f"pair_gap: {self.pair_gap.numpy()}\n"
            f"pair_friction: {self.pair_friction.numpy()}\n"
            f"pair_solref: {self.pair_solref.numpy()}\n"
            f"pair_solimp: {self.pair_solimp.numpy()}\n"
            f"convex_vert: {self.convex_vert.numpy()}\n"
            f"convex_vert_offset: {self.convex_vert_offset.numpy()}\n"
            f"type_pair_offset: {self.type_pair_offset.numpy()}\n"
            # f"ngeom: {self.ngeom.numpy()}\n"
            # f"npair: {self.npair.numpy()}\n"
            # f"nbody: {self.nbody.numpy()}\n"
            # f"nexclude: {self.nexclude.numpy()}\n"
            # f"max_contact_points: {self.max_contact_points.numpy()}\n"
            # f"n_geom_pair: {self.n_geom_pair.numpy()}\n"
            # f"n_geom_types: {self.n_geom_types.numpy()}\n"
            # f"filter_parent: {self.filter_parent}\n"
            # f"depth_extension: {self.depth_extension.numpy()}\n"
            # f"gjk_iteration_count: {self.gjk_iteration_count.numpy()}\n"
            # f"epa_iteration_count: {self.epa_iteration_count.numpy()}\n"
            # f"epa_best_count: {self.epa_best_count.numpy()}\n"
            # f"multi_polygon_count: {self.multi_polygon_count.numpy()}\n"
            # f"multi_tilt_angle: {self.multi_tilt_angle.numpy()}\n"
            # f"nenv: {self.nenv.numpy()}\n"
            # f"nmodel: {self.nmodel.numpy()}"
        )   


class CollisionOutput:
    def __init__(self, n_points : int, nenv : int, nbody : int, ngeom : int, n_geom_pair : int, n_geom_types : int, mjNREF : int, mjNIMP : int, device):
        self.dist = wp.empty(n_points, dtype=wp.float32)
        self.pos = wp.empty(n_points, dtype=wp.vec3)
        self.normal = wp.empty(n_points, dtype=wp.vec3)
        self.g1 = wp.empty(n_points, dtype=wp.int32)
        self.g2 = wp.empty(n_points, dtype=wp.int32)
        self.includemargin = wp.empty(n_points, dtype=wp.float32)
        self.friction = wp.empty(n_points * 5, dtype=wp.float32)
        self.solref = wp.empty(n_points * mjNREF, dtype=wp.float32)
        self.solreffriction = wp.empty(n_points * mjNREF, dtype=wp.float32)
        self.solimp = wp.empty(n_points * mjNIMP, dtype=wp.float32)
        self.dyn_body_aamm = wp.empty(nenv * nbody * 6, dtype=wp.float32)
        self.col_body_pair = wp.empty(nenv * nbody * nbody * 2, dtype=wp.int32)
        self.env_counter = wp.empty(nenv, dtype=wp.int32)
        self.env_counter2 = wp.empty(nenv, dtype=wp.int32)
        self.env_offset = wp.empty(nenv, dtype=wp.int32)
        self.dyn_geom_aabb = wp.empty(nenv * ngeom * 6, dtype=wp.float32)
        self.col_geom_pair = wp.empty(n_geom_pair * 2, dtype=wp.int32)
        self.type_pair_env_id = wp.empty(n_geom_pair, dtype=wp.int32)
        self.type_pair_geom_id = wp.empty(n_geom_pair * 2, dtype=wp.int32)
        self.type_pair_count = wp.empty(n_geom_types * n_geom_types, dtype=wp.int32)
        self.tmp_count = wp.empty(1, dtype=wp.int32)
        # self.simplex = wp.empty(n_points, dtype=wp.types.matrix(shape=(4, 3), dtype=float))


@wp.kernel
def finalize_sum(
    nenv: int,
    scan: wp.array(dtype=wp.int32),
    data_before_scan: wp.array(dtype=wp.int32),
    sum: wp.array(dtype=wp.int32)):
    tid = wp.tid()
    if tid == 0:
        sum[0] = scan[nenv - 1] + data_before_scan[nenv - 1]

@wp.kernel
def testKernel(num : int):
    tid = wp.tid()
    if tid >= num:
        return

@wp.kernel
def init(
    max_contact_points: int,
    nenv: int,
    dist: wp.array(dtype=wp.float32),
    pos: wp.array(dtype=wp.vec3),
    normal: wp.array(dtype=wp.vec3),
    g1: wp.array(dtype=wp.int32),
    g2: wp.array(dtype=wp.int32),
    includemargin: wp.array(dtype=wp.float32),
    friction: wp.array(dtype=wp.float32),
    solref: wp.array(dtype=wp.float32),
    solreffriction: wp.array(dtype=wp.float32),
    solimp: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    if tid >= nenv * max_contact_points:
        return

    dist[tid] = 1e12
    pos[tid ] = wp.vec3(0.0, 0.0, 0.0)
    normal[tid] = wp.vec3(0.0, 0.0, 0.0)
    g1[tid] = -1
    g2[tid] = -1
    includemargin[tid] = 0.0
    solref[tid * 2 + 0] = 0.02
    solref[tid * 2 + 1] = 1.0
    solimp[tid * 5 + 0] = 0.9
    solimp[tid * 5 + 1] = 0.95
    solimp[tid * 5 + 2] = 0.001
    solimp[tid * 5 + 3] = 0.5
    solimp[tid * 5 + 4] = 2.0
    friction[tid * 5 + 0] = 1.0
    friction[tid * 5 + 1] = 1.0
    friction[tid * 5 + 2] = 0.005
    friction[tid * 5 + 3] = 0.0001
    friction[tid * 5 + 4] = 0.0001
    solreffriction[tid * 2 + 0] = 0.0
    solreffriction[tid * 2 + 1] = 0.0


@wp.struct
class OrthoBasis:
    b: wp.vec3
    c: wp.vec3

@wp.func
def orthogonals(a: wp.vec3) -> OrthoBasis:
    y = wp.vec3(0.0, 1.0, 0.0)
    z = wp.vec3(0.0, 0.0, 1.0)
    b = wp.select((-0.5 < a[1]) and (a[1] < 0.5), y, z)
    b = b - a * wp.dot(a, b)
    b = wp.normalize(b)
    if (a == wp.vec3(0.0, 0.0, 0.0)):
        b = wp.vec3(0.0, 0.0, 0.0)
    c = wp.cross(a, b)
    
    result = OrthoBasis(b=b, c=c)
    return result


@wp.kernel
def make_frame(n_frames: int, a: wp.array(dtype=wp.vec3), frame: wp.array(dtype=wp.vec3)):
    tid = wp.tid()
    if tid >= n_frames:
        return
    
    a_normalized = wp.normalize(a[tid])
    basis = orthogonals(a_normalized)
    
    frame[3 * tid + 0] = a_normalized
    frame[3 * tid + 1] = basis.b
    frame[3 * tid + 2] = basis.c



# # TODO
# def narrowphase(input : CollisionInput, output : CollisionOutput):
#     pass




def _narrowphase2(s, input, output, t1, t2):


    mjxGEOM_size = 8  # Replace this with the actual value of mjxGEOM_size if known.

    #   PLANE  HFIELD SPHERE CAPSULE ELLIPSOID CYLINDER BOX  MESH
    maxContactPointsMap = [       
        [0,     0,     1,     2,       1,        3,      4,    4],  # PLANE
        [0,     0,     1,     2,       1,        3,      4,    4],  # HFIELD
        [0,     0,     1,     1,       1,        1,      1,    4],  # SPHERE
        [0,     0,     0,     1,       1,        2,      2,    2],  # CAPSULE
        [0,     0,     0,     0,       1,        1,      1,    1],  # ELLIPSOID
        [0,     0,     0,     0,       0,        3,      3,    3],  # CYLINDER
        [0,     0,     0,     0,       0,        0,      4,    4],  # BOX
        [0,     0,     0,     0,       0,        0,      0,    4],  # MESH
    ]


    # Assuming maxContactPointsMap is a 2D array or list accessible in Python
    ncon = maxContactPointsMap[t1][t2]


    _narrowphase(
        t1,
        t2,
        input.gjk_iteration_count,
        input.epa_iteration_count,
        input.nenv,
        input.ngeom,
        input.nmodel,
        input.n_geom_types,
        ncon,
        output.type_pair_env_id,
        output.type_pair_geom_id,
        output.type_pair_count,
        input.type_pair_offset,
        input.geom_xpos,
        input.geom_xmat,
        input.geom_size,
        input.geom_dataid,
        input.convex_vert,
        input.convex_vert_offset,
        input.epa_best_count,
        input.depth_extension,
        input.multi_polygon_count,
        input.multi_tilt_angle,
        # outputs
        output.env_counter,
        output.g1,
        output.g2,
        output.dist,
        output.pos,
        output.normal,
    )

    wp.synchronize()

    return



   


    group_key = t1 + t2 * input.n_geom_types

    type_pair_count = output.type_pair_count.numpy()

    # Accessing the value from the device array
    npair = int(type_pair_count[group_key])
    if npair == 0:
        return

    # Assuming maxContactPointsMap is a 2D array or list accessible in Python
    ncon = maxContactPointsMap[t1][t2]

    print("Calling gjk_epa_dense")
    # gjk_epa_dense(
    #     geom_pair=output.type_pair_geom_id.reshape((-1, 2)),
    #     geom_xpos=input.geom_xpos,
    #     geom_xmat=input.geom_xmat,
    #     geom_size=input.geom_size,
    #     geom_dataid=input.geom_dataid,
    #     convex_vert=input.convex_vert,
    #     convex_vert_offset=input.convex_vert_offset,
    #     ngeom=input.ngeom,
    #     npair=npair,
    #     ncon=ncon,
    #     geom_type0=t1,
    #     geom_type1=t2,
    #     depth_extension=input.depth_extension,
    #     gjk_iteration_count=input.gjk_iteration_count,
    #     epa_iteration_count=input.epa_iteration_count,
    #     epa_best_count=input.epa_best_count,
    #     multi_polygon_count=input.multi_polygon_count,
    #     multi_tilt_angle=input.multi_tilt_angle,

    #     dist=output.dist,
    #     pos=output.pos,
    #     normal=output.normal,
    #     simplex=output.simplex
    # )
    gjk_epa_sparse(
        geom_pair=output.type_pair_geom_id.reshape((-1, 2)),
        geom_xpos=input.geom_xpos,
        geom_xmat=input.geom_xmat,
        geom_size=input.geom_size,
        geom_dataid=input.geom_dataid,
        convex_vert=input.convex_vert,
        convex_vert_offset=input.convex_vert_offset,
        ngeom=input.ngeom,
        npair=npair,
        ncon=ncon,
        geom_type0=t1,
        geom_type1=t2,
        depth_extension=input.depth_extension,
        gjk_iteration_count=input.gjk_iteration_count,
        epa_iteration_count=input.epa_iteration_count,
        epa_best_count=input.epa_best_count,
        multi_polygon_count=input.multi_polygon_count,
        multi_tilt_angle=input.multi_tilt_angle,
        
        dist=output.dist,
        pos=output.pos,
        normal=output.normal,
        simplex=output.simplex
    )


def narrowphase(s, input, output):
    """
    Perform the narrowphase collision detection based on geometry types.

    Args:
        s: CUDA stream.
        input (CollisionInput): Input collision data.
        output (CollisionOutput): Output collision data.
    """

    mjxGEOM_PLANE = 0
    mjxGEOM_HFIELD = 1
    mjxGEOM_SPHERE = 2
    mjxGEOM_CAPSULE = 3
    mjxGEOM_ELLIPSOID = 4
    mjxGEOM_CYLINDER = 5
    mjxGEOM_BOX = 6
    mjxGEOM_MESH = 7
    mjxGEOM_size = 8

    mjxGEOM_size = 8  # Replace this with the actual value of mjxGEOM_size if known.

    for t2 in range(mjxGEOM_size):
        for t1 in range(t2 + 1):
            # Mapping types to narrowphase functions
            if t1 == mjxGEOM_PLANE and t2 == mjxGEOM_SPHERE:
                _narrowphase2(s, input, output, t1, t2)
            elif t1 == mjxGEOM_PLANE and t2 == mjxGEOM_CAPSULE:
                _narrowphase2(s, input, output, t1, t2)
            elif t1 == mjxGEOM_PLANE and t2 == mjxGEOM_BOX:
                _narrowphase2(s, input, output, t1, t2)
            elif t1 == mjxGEOM_PLANE and t2 == mjxGEOM_ELLIPSOID:
                _narrowphase2(s, input, output, t1, t2)
            elif t1 == mjxGEOM_PLANE and t2 == mjxGEOM_CYLINDER:
                _narrowphase2(s, input, output, t1, t2)
            elif t1 == mjxGEOM_PLANE and t2 == mjxGEOM_MESH:
                _narrowphase2(s, input, output, t1, t2)
            elif t1 == mjxGEOM_SPHERE and t2 == mjxGEOM_SPHERE:
                _narrowphase2(s, input, output, t1, t2)
            elif t1 == mjxGEOM_SPHERE and t2 == mjxGEOM_CAPSULE:
                _narrowphase2(s, input, output, t1, t2)
            elif t1 == mjxGEOM_SPHERE and t2 == mjxGEOM_BOX:
                _narrowphase2(s, input, output, t1, t2)
            elif t1 == mjxGEOM_SPHERE and t2 == mjxGEOM_MESH:
                _narrowphase2(s, input, output, t1, t2)
            elif t1 == mjxGEOM_CAPSULE and t2 == mjxGEOM_CAPSULE:
                _narrowphase2(s, input, output, t1, t2)
            elif t1 == mjxGEOM_CAPSULE and t2 == mjxGEOM_BOX:
                _narrowphase2(s, input, output, t1, t2)
            elif t1 == mjxGEOM_CAPSULE and t2 == mjxGEOM_ELLIPSOID:
                _narrowphase2(s, input, output, t1, t2)
            elif t1 == mjxGEOM_CAPSULE and t2 == mjxGEOM_CYLINDER:
                _narrowphase2(s, input, output, t1, t2)
            elif t1 == mjxGEOM_CAPSULE and t2 == mjxGEOM_MESH:
                _narrowphase2(s, input, output, t1, t2)
            elif t1 == mjxGEOM_ELLIPSOID and t2 == mjxGEOM_ELLIPSOID:
                _narrowphase2(s, input, output, t1, t2)
            elif t1 == mjxGEOM_ELLIPSOID and t2 == mjxGEOM_CYLINDER:
                _narrowphase2(s, input, output, t1, t2)
            elif t1 == mjxGEOM_CYLINDER and t2 == mjxGEOM_CYLINDER:
                _narrowphase2(s, input, output, t1, t2)
            elif t1 == mjxGEOM_BOX and t2 == mjxGEOM_BOX:
                _narrowphase2(s, input, output, t1, t2)
            elif t1 == mjxGEOM_BOX and t2 == mjxGEOM_MESH:
                _narrowphase2(s, input, output, t1, t2)
            elif t1 == mjxGEOM_MESH and t2 == mjxGEOM_MESH:
                _narrowphase2(s, input, output, t1, t2)




def _collision(input : CollisionInput, output : CollisionOutput, device):
    if input.ngeom == 0:
        return True

    #device = 'cpu'

    gridSize = input.max_contact_points

    wp.launch(testKernel, dim=1, inputs = [1])

    # Initialize the output data
    wp.launch(init, 
              dim=[input.nenv * input.max_contact_points], 
              inputs=[input.max_contact_points, input.nenv, output.dist, output.pos,
                      output.normal, output.g1, output.g2, output.includemargin,
                      output.friction, output.solref, output.solreffriction, output.solimp], device = device)

    wp.synchronize()

    # Initialize environment buffers
    wp.launch(init_buffers, 
              dim=[input.nenv], 
              inputs=[input.nenv, output.env_counter, output.env_offset, output.env_counter2], device = device)

    wp.synchronize()

    #print(output.dyn_body_aamm.numpy())

    # Generate body AAMMs
    wp.launch(get_dyn_body_aamm, 
              dim=[input.nenv * input.nbody], 
              inputs=[input.nenv, input.nbody, input.nmodel, input.ngeom, input.body_geomnum,
                                                       input.body_geomadr, input.geom_margin, input.geom_xpos, input.geom_rbound,
                                                       output.dyn_body_aamm], device = device)


    wp.synchronize()
    #print(input.body_geomnum.numpy())
    #print(output.dyn_body_aamm.numpy())

    wp.synchronize()

    # Generate body pairs (broadphase)
    col_body_pair_count = output.env_counter
    wp.launch(get_body_pairs_nxn, 
              dim=[input.nenv * input.nbody * input.nbody], 
              inputs=[input.nenv, input.nbody, input.filter_parent, input.nexclude,
                      input.body_parentid, input.body_weldid, input.body_contype,
                      input.body_conaffinity, input.body_has_plane, input.exclude_signature,
                      output.dyn_body_aamm, output.col_body_pair, col_body_pair_count], device = device)

    wp.synchronize()

    #print(output.env_counter.numpy())

    wp.synchronize()

    # Get geom AABBs in global frame
    wp.launch(get_dyn_geom_aabb, 
              dim=[input.nenv * input.ngeom], 
              inputs=[input.nenv, input.nmodel, input.ngeom, input.geom_xpos, input.geom_xmat,
                      input.geom_aabb, output.dyn_geom_aabb], device = device)

    # Generate geom pairs (midphase)
    wp.synchronize()
    #print(output.dyn_geom_aabb.numpy())
    wp.synchronize()


    body_pair_offset = output.env_offset
    wp.utils.array_scan(col_body_pair_count, body_pair_offset, False)

    wp.launch(finalize_sum,
              dim = 1,
              inputs = [                 
                input.nenv,
                body_pair_offset,
                col_body_pair_count,
                output.tmp_count
              ], 
              device = device)
    
    wp.synchronize() #This is not for debugging - keep it
    total_body_pairs = output.tmp_count.numpy()[0]
  

  
    col_geom_pair_count = output.env_counter2    

    wp.synchronize()
    print(input.geom_type.numpy())
    wp.synchronize()

    wp.launch(get_geom_pairs_nxn, 
              dim=[total_body_pairs], 
              inputs=[input.nenv, input.ngeom, input.nbody, input.n_geom_pair,
                                                        input.body_geomnum, input.body_geomadr, input.geom_contype,
                                                        input.geom_conaffinity, input.geom_type, input.geom_margin,
                                                        output.dyn_geom_aabb, output.col_body_pair, output.env_counter,
                                                        body_pair_offset, output.col_geom_pair, col_geom_pair_count], device = device)


    wp.synchronize()
    #print(col_geom_pair_count.numpy())
    wp.synchronize()


    # Initialize type pair count
    wp.launch(set_zero, 
              dim=[input.n_geom_types * input.n_geom_types], 
              inputs=[input.n_geom_types * input.n_geom_types, output.type_pair_count], device = device)

    wp.synchronize()
    wp.synchronize()

    # total_geom_pairs = np.sum(col_geom_pair_count.numpy())
    # col_geom_pair_offset = np.cumsum(col_geom_pair_count.numpy())


    # Use Warp's array_sum to compute the total number of geom pairs
    #total_geom_pairs = wp.empty(shape=(1,), dtype=wp.int32)
    #wp.utils.array_sum(col_geom_pair_count, out=total_geom_pairs)

    # Use Warp's array_scan to compute the geom pair offsets
    #col_geom_pair_offset = wp.empty_like(col_geom_pair_count)
    #wp.utils.array_scan(col_geom_pair_count, out=col_geom_pair_offset, inclusive=False)



    col_geom_pair_offset = output.env_offset
    wp.utils.array_scan(col_geom_pair_count, col_geom_pair_offset, False)

    wp.launch(finalize_sum,
              dim = 1,
              inputs = [                 
                input.nenv,
                col_geom_pair_offset,
                col_geom_pair_count,
                output.tmp_count
              ], 
              device = device)
    total_geom_pairs = output.tmp_count.numpy()[0]



    wp.synchronize()
    #print(total_geom_pairs)
    wp.synchronize()



    wp.launch(group_contacts_by_type, 
              dim=[total_geom_pairs], 
              inputs=[input.nenv, input.n_geom_pair, input.n_geom_types, input.geom_type,
                      output.col_geom_pair, col_geom_pair_count, col_geom_pair_offset,
                      input.type_pair_offset, output.type_pair_env_id, output.type_pair_geom_id,
                      output.type_pair_count], 
                      device = device)
    

    wp.synchronize()
    print(output.type_pair_count.numpy())
    wp.synchronize()


    # Initialize the env contact counter
    env_contact_count = output.env_counter
    wp.launch(set_zero,
              dim=[input.nenv * input.ngeom], 
              inputs=[input.nenv, env_contact_count],
              device = device)


    wp.synchronize()
    wp.synchronize()


    # Dispatch to narrowphase collision functions
    narrowphase(None, input, output)

    # Generate contact solver params
    wp.synchronize()
    wp.synchronize()




    env_contact_offset = output.env_offset
    wp.utils.array_scan(env_contact_count, env_contact_offset, False)

    wp.launch(finalize_sum,
              dim = 1,
              inputs = [                 
                input.nenv,
                env_contact_offset,
                env_contact_count,
                output.tmp_count
              ], 
              device = device)
    n_contact_pts = output.tmp_count.numpy()[0]


    # n_contact_pts = np.sum(env_contact_count.numpy())
    # env_contact_offset = np.cumsum(env_contact_count.numpy())


    wp.synchronize()
    wp.synchronize()


    wp.launch(get_contact_solver_params, 
              dim=[n_contact_pts], 
              inputs=[input.nenv, input.nmodel, input.ngeom, input.max_contact_points,
                      n_contact_pts, output.g1, output.g2, input.geom_priority,
                      input.geom_solmix, input.geom_friction, input.geom_solref,
                      input.geom_solimp, input.geom_margin, input.geom_gap,
                      env_contact_offset, output.includemargin, output.friction,
                      output.solref, output.solreffriction, output.solimp], 
                      device = device)

    wp.synchronize()
    wp.synchronize()

    return True



def collision2(
    m: Model,
    d: Data,
    depth_extension: float,
    gjk_iter: int,
    epa_iter: int,
    epa_best_count: int,
    multi_polygon_count: int,
    multi_tilt_angle: float,
) -> Contact:
    """GJK/EPA narrowphase routine."""
    ngeom = m.ngeom

    if not (m.geom_condim[0] == m.geom_condim).all():
        raise NotImplementedError(
            'm.geom_condim should be the same for all geoms. Different condim per'
            ' geom is not supported yet.'
        )
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
    if m.npair > 0:
        raise NotImplementedError('m.npair > 0 is not supported.')


    device = 'cpu'

    max_contact_points = d.contact.pos.shape[0]
    n_pts = max_contact_points
    body_pair_size = int((m.nbody * (m.nbody - 1) / 2 + 15) / 16) * 16
    n_geom_pair = _get_ngeom_pair(m)

    n_geom_types = len(GeomType)
    n_geom_type_pairs = n_geom_types * n_geom_types
    type_pair_offset = _get_ngeom_pair_type_offset(m)
    type_pair_count = wp.zeros(n_geom_type_pairs, dtype=wp.int32, device=device)
    #convex_vert, convex_vert_offset = engine_collision_convex.get_convex_vert(m)

    nenv = 1
    nmodel = 1
    mjNREF = int(2) 
    mjNIMP = int(5) 
    mjMINVAL = float(1E-15)

    # Initialize input and output structures
    input = CollisionInput(m, d, nenv, nmodel, depth_extension, gjk_iter, epa_iter, epa_best_count, multi_polygon_count, multi_tilt_angle, device)
    output = CollisionOutput(n_pts, nenv=nenv, nbody=m.nbody, ngeom=ngeom, n_geom_pair=n_geom_pair, n_geom_types=n_geom_types, mjNREF=mjNREF, mjNIMP=mjNIMP, device = device)

    # Call the collision function using Warp kernels
    _collision(input, output, device)


    # Assuming output.normal is a Warp array of wp.vec3
    n_frames = len(output.normal)

    # Allocate memory for the frame
    frame = wp.empty((n_frames * 3,), dtype=wp.vec3, device=device)

    # Launch the make_frame kernel
    wp.launch(make_frame, dim=n_frames, inputs=[n_frames, output.normal, frame], device=device)

    # Synchronize to ensure the kernel has completed
    wp.synchronize()


    c = Contact(
        dist=output.dist.numpy(),
        pos=output.pos.numpy(),
        frame=frame.numpy(),
        includemargin=output.includemargin.numpy(),
        friction=output.friction.numpy(),
        solref=output.solref.numpy(),
        solreffriction=output.solreffriction.numpy(),
        solimp=output.solimp.numpy(),
        geom1=output.g1.numpy(),
        geom2=output.g2.numpy(),
        geom=None,  # jp.array([g1, g2]).T,
        efc_address=None,  # d.contact.efc_address,
        dim=None,  # d.contact.dim,
    )

    return c








# def collision(
#     m: Model,
#     d: Data,
#     depth_extension: float,
#     gjk_iter: int,
#     epa_iter: int,
#     epa_best_count: int,
#     multi_polygon_count: int,
#     multi_tilt_angle: float,
# ) -> Contact:
#   """GJK/EPA narrowphase routine."""
#   ngeom = m.ngeom

#   if not (m.geom_condim[0] == m.geom_condim).all():
#     raise NotImplementedError(
#         'm.geom_condim should be the same for all geoms. Different condim per'
#         ' geom is not supported yet.'
#     )
#   if len(d.geom_xpos.shape) != 2:
#     raise ValueError(
#         f'd.geom_xpos should have 2d shape, got "{len(d.geom_xpos.shape)}".'
#     )
#   if len(d.geom_xmat.shape) != 3:
#     raise ValueError(
#         f'd.geom_xmat should have 3d shape, got "{len(d.geom_xmat.shape)}".'
#     )
#   if m.geom_size.shape[0] != ngeom:
#     raise ValueError(
#         f'm.geom_size.shape[0] should be ngeom ({ngeom}), '
#         f'got "{m.geom_size.shape[0]}".'
#     )
#   if m.geom_dataid.shape != (ngeom,):
#     raise ValueError(
#         f'm.geom_dataid.shape should be (ngeom,) == ({ngeom},), got'
#         f' "({m.geom_dataid.shape[0]},)".'
#     )
#   if m.npair > 0:
#     raise NotImplementedError('m.npair > 0 is not supported.')

#   # TODO(btaba): geom_margin/gap are not supported, we should throw an error in
#   #   put_model.

#   max_contact_points = d.contact.pos.shape[0]
#   n_pts = max_contact_points
#   body_pair_size = int((m.nbody * (m.nbody - 1) / 2 + 15) / 16) * 16
#   n_geom_pair = _get_ngeom_pair(m)
#   # out_types = (
#   #     # Output buffers.
#   #     jax.ShapeDtypeStruct((n_pts,), dtype=jp.float32),  # dist
#   #     jax.ShapeDtypeStruct((n_pts, 3), dtype=jp.float32),  # pos
#   #     jax.ShapeDtypeStruct((n_pts, 3), dtype=jp.float32),  # normal
#   #     jax.ShapeDtypeStruct((n_pts,), dtype=jp.int32),  # g1
#   #     jax.ShapeDtypeStruct((n_pts,), dtype=jp.int32),  # g2
#   #     jax.ShapeDtypeStruct((n_pts,), dtype=jp.float32),  # includemargin
#   #     jax.ShapeDtypeStruct((n_pts, 5), dtype=jp.float32),  # friction
#   #     jax.ShapeDtypeStruct((n_pts, mujoco.mjNREF), dtype=jp.float32),  # solref
#   #     jax.ShapeDtypeStruct(
#   #         (n_pts, mujoco.mjNREF), dtype=jp.float32
#   #     ),  # solreffriction
#   #     jax.ShapeDtypeStruct((n_pts, mujoco.mjNIMP), dtype=jp.float32),  # solimp
#   #     # Buffers used for intermediate results.
#   #     # TODO(btaba): combine and re-use buffers instead of having so many.
#   #     jax.ShapeDtypeStruct((m.nbody, 6), dtype=jp.float32),  # dyn_body_aamm
#   #     jax.ShapeDtypeStruct(
#   #         (body_pair_size, 2), dtype=jp.int32
#   #     ),  # col_body_pair
#   #     jax.ShapeDtypeStruct((1,), dtype=jp.uint32),  # env_counter
#   #     jax.ShapeDtypeStruct((1,), dtype=jp.uint32),  # env_counter2
#   #     jax.ShapeDtypeStruct((1,), dtype=jp.uint32),  # env_offset
#   #     jax.ShapeDtypeStruct((ngeom, 6), dtype=jp.float32),  # dyn_geom_aabb
#   #     jax.ShapeDtypeStruct((n_geom_pair, 2), dtype=jp.int32),  # col_geom_pair
#   #     jax.ShapeDtypeStruct((n_geom_pair,), dtype=jp.uint32),  # type_pair_env_id
#   #     jax.ShapeDtypeStruct(
#   #         (n_geom_pair * 2,), dtype=jp.uint32
#   #     ),  # type_pair_geom_id
#   # )

#   n_geom_types = len(GeomType)
#   n_geom_type_pairs = n_geom_types * n_geom_types
#   type_pair_offset = _get_ngeom_pair_type_offset(m)
#   type_pair_count = np.zeros(n_geom_type_pairs, dtype=np.uint32)
#   convex_vert, convex_vert_offset = engine_collision_convex.get_convex_vert(m)





#   # (
#   #     dist,
#   #     pos,
#   #     normal,
#   #     g1,
#   #     g2,
#   #     includemargin,
#   #     friction,
#   #     solref,
#   #     solreffriction,
#   #     solimp,
#   #     *_,
#   # ) = ffi.ffi_call(
#   #     'collision_driver_cuda',
#   #     out_types,
#   #     d.geom_xpos,
#   #     d.geom_xmat,
#   #     m.geom_size,
#   #     m.geom_type,
#   #     m.geom_contype,
#   #     m.geom_conaffinity,
#   #     m.geom_priority,
#   #     m.geom_margin,
#   #     m.geom_gap,
#   #     m.geom_solmix,
#   #     m.geom_friction,
#   #     m.geom_solref,
#   #     m.geom_solimp,
#   #     # TODO(btaba): allow vmapping over sizes via geom_aabb/rbound jax.Array.
#   #     m.geom_aabb,
#   #     m.geom_rbound,
#   #     m.geom_dataid,
#   #     m.geom_bodyid,
#   #     m.body_parentid,
#   #     m.body_weldid,
#   #     m.body_contype,
#   #     m.body_conaffinity,
#   #     m.body_geomadr,
#   #     m.body_geomnum.astype(np.uint32),
#   #     _get_body_has_plane(m),
#   #     m.pair_geom1,
#   #     m.pair_geom2,
#   #     m.exclude_signature,
#   #     m.pair_margin,
#   #     m.pair_gap,
#   #     m.pair_friction,
#   #     m.pair_solref,
#   #     m.pair_solimp,
#   #     convex_vert,
#   #     convex_vert_offset,
#   #     type_pair_offset.astype(np.uint32),
#   #     type_pair_count,
#   #     ngeom=np.uint32(ngeom),
#   #     npair=np.uint32(m.npair),
#   #     nbody=np.uint32(m.nbody),
#   #     nexclude=np.uint32(m.nexclude),
#   #     max_contact_points=np.uint32(max_contact_points),
#   #     n_geom_pair=np.uint32(n_geom_pair),
#   #     n_geom_types=np.uint32(n_geom_types),
#   #     filter_parent=not (m.opt.disableflags & DisableBit.FILTERPARENT),
#   #     depth_extension=np.float32(depth_extension),
#   #     gjk_iteration_count=np.uint32(gjk_iter),
#   #     epa_iteration_count=np.uint32(epa_iter),
#   #     epa_best_count=np.uint32(epa_best_count),
#   #     multi_polygon_count=np.uint32(multi_polygon_count),
#   #     multi_tilt_angle=np.float32(multi_tilt_angle),
#   #     vectorized=True,
#   # )


#   (
#       dist,
#       pos,
#       normal,
#       g1,
#       g2,
#       includemargin,
#       friction,
#       solref,
#       solreffriction,
#       solimp
#   ) = warp_collide(
#       wp.from_jax(d.geom_xpos),
#       wp.from_jax(d.geom_xmat),
#     )

#   frame = warp_normal_to_frame(normal)

#   c = Contact(
#       dist=dist.numpy(),
#       pos=pos.numpy(),
#       frame=frame.numpy(),
#       includemargin=includemargin.numpy(),
#       friction=friction.numpy(),
#       solref=solref.numpy(),
#       solreffriction=solreffriction.numpy(),
#       solimp=solimp.numpy(),
#       geom1=g1.numpy(),
#       geom2=g2.numpy(),
#       geom=None, #jp.array([g1, g2]).T,
#       efc_address=None, #d.contact.efc_address,
#       dim=None, #d.contact.dim,
#   )

#   return c
