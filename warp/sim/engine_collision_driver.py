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
import mujoco

# pylint: enable=g-importing-member
import numpy as np

# from convex import gjk_epa_dense
from convex import narrowphase, where

# pylint: disable=g-importing-member
from mujoco.mjx._src.types import DisableBit, GeomType, Model

import warp as wp


@wp.kernel
def get_dyn_body_aamm(
    nenv: int,
    nbody: int,
    nmodel: int,
    ngeom: int,
    body_geomnum: wp.array(dtype=int),
    body_geomadr: wp.array(dtype=int),
    geom_margin: wp.array(dtype=float),
    geom_xpos: wp.array(dtype=wp.vec3),
    geom_rbound: wp.array(dtype=float),
    dyn_body_aamm: wp.array(dtype=float, ndim=2),
):
    tid = wp.tid()

    bid = tid % nbody
    env_id = tid // nbody
    model_id = env_id % nmodel

    # Initialize AAMM with extreme values
    aamm_min = wp.vec3(1000000000.0, 1000000000.0, 1000000000.0)
    aamm_max = wp.vec3(-1000000000.0, -1000000000.0, -1000000000.0)

    # Iterate over all geometries associated with the body
    for i in range(body_geomnum[bid]):
        g = body_geomadr[bid] + i
        pos = geom_xpos[(env_id * ngeom + g)]
        rbound = geom_rbound[model_id * ngeom + g]
        margin = geom_margin[model_id * ngeom + g]

        for j in range(3):
            min_val = pos[j] - rbound - margin
            max_val = pos[j] + rbound + margin

            aamm_min[j] = wp.min(aamm_min[j], min_val)
            aamm_max[j] = wp.max(aamm_max[j], max_val)

    # Write results to output
    dyn_body_aamm[tid, 0] = aamm_min[0]
    dyn_body_aamm[tid, 1] = aamm_min[1]
    dyn_body_aamm[tid, 2] = aamm_min[2]
    dyn_body_aamm[tid, 3] = aamm_max[0]
    dyn_body_aamm[tid, 4] = aamm_max[1]
    dyn_body_aamm[tid, 5] = aamm_max[2]


@wp.func
def map_body_pair_nxn(tid: int, nenv: int, nbody: int) -> int:
    if tid >= nenv * nbody * nbody:
        return -1

    body_pair_id = tid % (nbody * nbody)
    body1 = body_pair_id // nbody
    body2 = body_pair_id % nbody

    return where(body1 < body2, body1 + body2 * nbody, -1)


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
    body_has_plane: wp.array(dtype=bool),
    exclude_signature: wp.array(dtype=int),
    dyn_body_aamm: wp.array(dtype=float, ndim=2),
    # outputs
    col_body_pair: wp.array(dtype=int, ndim=2),
    col_body_pair_count: wp.array(dtype=int),
):
    tid = wp.tid()
    map_coord = map_body_pair_nxn(tid, nenv, nbody)

    if map_coord == -1:  # Assuming `FULL_MASK` translates to -1
        return

    env_id = tid // (nbody * nbody)
    body1 = map_coord % nbody
    body2 = map_coord // nbody

    if (body_contype[body1] == 0 and body_conaffinity[body1] == 0) or (
        body_contype[body2] == 0 and body_conaffinity[body2] == 0
    ):
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

    b1 = env_id * nbody + body1
    b2 = env_id * nbody + body2
    separating = (
        (dyn_body_aamm[b1, 0] > dyn_body_aamm[b2, 3])
        or (dyn_body_aamm[b1, 1] > dyn_body_aamm[b2, 4])
        or (dyn_body_aamm[b1, 2] > dyn_body_aamm[b2, 5])
        or (dyn_body_aamm[b2, 0] > dyn_body_aamm[b1, 3])
        or (dyn_body_aamm[b2, 1] > dyn_body_aamm[b1, 4])
        or (dyn_body_aamm[b2, 2] > dyn_body_aamm[b1, 5])
    )

    if separating and not (body_has_plane[body1] or body_has_plane[body2]):
        return

    idx = wp.atomic_add(col_body_pair_count, env_id, 1)
    nbody_pair = ((nbody * (nbody - 1) // 2 + 15) // 16) * 16
    col_body_pair[env_id * nbody_pair + idx, 0] = body1
    col_body_pair[env_id * nbody_pair + idx, 1] = body2


@wp.struct
class Mat3x4:
    row0: wp.vec4
    row1: wp.vec4
    row2: wp.vec4


@wp.func
def transform_point(mat: Mat3x4, pos: wp.vec3) -> wp.vec3:
    x = wp.dot(wp.vec3(mat.row0[0], mat.row0[1], mat.row0[2]), pos) + mat.row0[3]
    y = wp.dot(wp.vec3(mat.row1[0], mat.row1[1], mat.row1[2]), pos) + mat.row1[3]
    z = wp.dot(wp.vec3(mat.row2[0], mat.row2[1], mat.row2[2]), pos) + mat.row2[3]
    return wp.vec3(x, y, z)


@wp.func
def max3(a: wp.vec3, b: wp.vec3):
    return wp.vec3(wp.max(a.x, b.x), wp.max(a.y, b.y), wp.max(a.z, b.z))


@wp.func
def min3(a: wp.vec3, b: wp.vec3):
    return wp.vec3(wp.min(a.x, b.x), wp.min(a.y, b.y), wp.min(a.z, b.z))


@wp.kernel
def get_dyn_geom_aabb(
    nenv: int,
    nmodel: int,
    ngeom: int,
    geom_xpos: wp.array(dtype=wp.vec3),
    geom_xmat: wp.array(dtype=wp.mat33),
    geom_aabb: wp.array(dtype=float, ndim=2),
    # outputs
    dyn_aabb: wp.array(dtype=float, ndim=2),
):
    tid = wp.tid()
    if tid >= nenv * ngeom:
        return

    env_id = tid // ngeom
    gid = tid % ngeom

    pos = geom_xpos[env_id * ngeom + gid]
    rot = geom_xmat[env_id * ngeom + gid]

    aabb = wp.vec3(geom_aabb[gid, 3], geom_aabb[gid, 4], geom_aabb[gid, 5])
    aabb_pos = wp.vec3(geom_aabb[gid, 0], geom_aabb[gid, 1], geom_aabb[gid, 2])

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
        # corner_world = transform_point(mat, corner + aabb_pos)
        # corner_world = pos + rot @ (corner + aabb_pos)
        corner_world = rot @ (corner + aabb_pos)
        aabb_max = max3(aabb_max, corner_world)
        aabb_min = min3(aabb_min, corner_world)

    dyn_aabb[tid, 0] = pos[0] + aabb_min[0]
    dyn_aabb[tid, 1] = pos[1] + aabb_min[1]
    dyn_aabb[tid, 2] = pos[2] + aabb_min[2]
    dyn_aabb[tid, 3] = pos[0] + aabb_max[0]
    dyn_aabb[tid, 4] = pos[1] + aabb_max[1]
    dyn_aabb[tid, 5] = pos[2] + aabb_max[2]


@wp.func
def bisection(x: wp.array(dtype=int), v: int, a_: int, b_: int) -> int:
    # Binary search for the largest index i such that x[i] <= v
    # x is a sorted array
    # a and b are the start and end indices within x to search
    a = int(a_)
    b = int(b_)
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
def get_geom_pairs_nxn(
    nenv: int,
    ngeom: int,
    nbody: int,
    n_geom_pair: int,
    body_geomnum: wp.array(dtype=int),
    body_geomadr: wp.array(dtype=int),
    geom_contype: wp.array(dtype=int),
    geom_conaffinity: wp.array(dtype=int),
    geom_type: wp.array(dtype=int),
    geom_margin: wp.array(dtype=float),
    dyn_geom_aabb: wp.array(dtype=float, ndim=2),
    col_body_pair: wp.array(dtype=int, ndim=2),
    col_body_pair_count: wp.array(dtype=int),
    col_body_pair_offset: wp.array(dtype=int),
    # outputs
    col_geom_pair: wp.array(dtype=int, ndim=2),
    col_geom_pair_count: wp.array(dtype=int),
):
    mjGEOM_PLANE = int(0)
    mjGEOM_HFIELD = int(1)

    tid = wp.tid()
    env_id = bisection(col_body_pair_offset, tid, 0, nenv - 1)
    body_pair_id = tid - col_body_pair_offset[env_id]
    if body_pair_id >= col_body_pair_count[env_id]:
        return

    nbody_pair = ((nbody * (nbody - 1) // 2 + 15) // 16) * 16
    body1 = col_body_pair[env_id * nbody_pair + body_pair_id, 0]
    body2 = col_body_pair[env_id * nbody_pair + body_pair_id, 1]

    for g1 in range(body_geomnum[body1]):
        geom1 = body_geomadr[body1] + g1
        for g2 in range(body_geomnum[body2]):
            geom2 = body_geomadr[body2] + g2

            type1 = geom_type[geom1]
            type2 = geom_type[geom2]
            skip_type = (type1 == mjGEOM_HFIELD or type1 == mjGEOM_PLANE) and (
                type2 == mjGEOM_HFIELD or type2 == mjGEOM_PLANE
            )

            skip_con = not (
                (geom_contype[geom1] & geom_conaffinity[geom2]) or (geom_contype[geom2] & geom_conaffinity[geom1])
            )

            eg1 = env_id * ngeom + geom1
            eg2 = env_id * ngeom + geom2
            separating = (
                (dyn_geom_aabb[eg1, 0] > dyn_geom_aabb[eg2, 3])
                or (dyn_geom_aabb[eg1, 1] > dyn_geom_aabb[eg2, 4])
                or (dyn_geom_aabb[eg1, 2] > dyn_geom_aabb[eg2, 5])
                or (dyn_geom_aabb[eg2, 0] > dyn_geom_aabb[eg1, 3])
                or (dyn_geom_aabb[eg2, 1] > dyn_geom_aabb[eg1, 4])
                or (dyn_geom_aabb[eg2, 2] > dyn_geom_aabb[eg1, 5])
            )

            if separating or skip_con or skip_type:
                continue

            if type1 > type2:
                tmp = geom1
                geom1 = geom2
                geom2 = tmp
                # geom1, geom2 = geom2, geom1
            pair_id = wp.atomic_add(col_geom_pair_count, env_id, 1)
            col_geom_pair[env_id * n_geom_pair + pair_id, 0] = geom1
            col_geom_pair[env_id * n_geom_pair + pair_id, 1] = geom2


@wp.kernel
def group_contacts_by_type(
    nenv: int,
    n_geom_pair: int,
    n_geom_types: int,
    geom_type: wp.array(dtype=int),
    col_geom_pair: wp.array(dtype=int, ndim=2),
    col_geom_pair_count: wp.array(dtype=int),
    col_geom_pair_offset: wp.array(dtype=int),
    type_pair_offset: wp.array(dtype=int),
    # outputs
    type_pair_env_id: wp.array(dtype=int),
    type_pair_geom_id: wp.array(dtype=int, ndim=2),
    type_pair_count: wp.array(dtype=int),
):
    tid = wp.tid()
    env_id = bisection(col_geom_pair_offset, tid, 0, nenv - 1)
    pair_id = tid - col_geom_pair_offset[env_id]
    if pair_id >= col_geom_pair_count[env_id]:
        return

    pid = env_id * n_geom_pair + pair_id
    geom1 = col_geom_pair[pid, 0]
    geom2 = col_geom_pair[pid, 1]

    type1 = geom_type[geom1]
    type2 = geom_type[geom2]
    group_key = type1 + type2 * n_geom_types

    n_type_pair = wp.atomic_add(type_pair_count, group_key, 1)
    type_pair_id = type_pair_offset[group_key] * nenv + n_type_pair
    type_pair_env_id[type_pair_id] = env_id
    type_pair_geom_id[type_pair_id, 0] = geom1
    type_pair_geom_id[type_pair_id, 1] = geom2


@wp.kernel
def get_contact_solver_params(
    nenv: int,
    nmodel: int,
    ngeom: int,
    max_contact_pts: int,
    n_contact_pts: int,
    geom1: wp.array(dtype=int),
    geom2: wp.array(dtype=int),
    geom_priority: wp.array(dtype=int),
    geom_solmix: wp.array(dtype=float),
    geom_friction: wp.array(dtype=float, ndim=2),
    geom_solref: wp.array(dtype=float, ndim=2),
    geom_solimp: wp.array(dtype=float, ndim=2),
    geom_margin: wp.array(dtype=float),
    geom_gap: wp.array(dtype=float),
    env_contact_offset: wp.array(dtype=int),
    # outputs
    includemargin: wp.array(dtype=float),
    friction: wp.array(dtype=float, ndim=2),
    solref: wp.array(dtype=float, ndim=2),
    solreffriction: wp.array(dtype=float, ndim=2),
    solimp: wp.array(dtype=float, ndim=2),
):
    tid = wp.tid()
    if tid >= n_contact_pts:
        return

    mjNIMP = int(5)
    mjMINVAL = float(1e-15)

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
    mix = where((solmix1 < mjMINVAL) and (solmix2 < mjMINVAL), 0.5, mix)
    mix = where((solmix1 < mjMINVAL) and (solmix2 >= mjMINVAL), 0.0, mix)
    mix = where((solmix1 >= mjMINVAL) and (solmix2 < mjMINVAL), 1.0, mix)

    p1 = geom_priority[g1]
    p2 = geom_priority[g2]
    mix = where(p1 == p2, mix, where(p1 > p2, 1.0, 0.0))
    is_standard = (geom_solref[g1, 0] > 0) and (geom_solref[g2, 0] > 0)

    # Hard code mjNREF = 2
    solref_ = wp.vec2(0.0, 0.0)  # wp.zeros(mjNREF, dtype=float)
    for i in range(2):
        solref_[i] = mix * geom_solref[g1, i] + (1.0 - mix) * geom_solref[g2, i]
        solref_[i] = where(is_standard, solref_[i], wp.min(geom_solref[g1, i], geom_solref[g2, i]))

    # solimp_ = wp.zeros(mjNIMP, dtype=float)
    # for i in range(mjNIMP):
    #     solimp_[i] = mix * geom_solimp[i + g1 * mjNIMP] + (1 - mix) * geom_solimp[i + g2 * mjNIMP]

    friction_ = wp.vec3(0.0, 0.0, 0.0)  # wp.zeros(3, dtype=float)
    for i in range(3):
        friction_[i] = wp.max(geom_friction[g1, i], geom_friction[g2, i])

    includemargin[tid] = margin - gap
    friction[tid, 0] = friction_[0]
    friction[tid, 1] = friction_[0]
    friction[tid, 2] = friction_[1]
    friction[tid, 3] = friction_[2]
    friction[tid, 4] = friction_[2]

    for i in range(2):
        solref[tid, i] = solref_[i]

    for i in range(mjNIMP):
        solimp[tid, i] = mix * geom_solimp[g1, i] + (1.0 - mix) * geom_solimp[g2, i]  # solimp_[i]


# ffi.register_ffi_target(
#     'collision_driver_cuda',
#     _engine_collision_driver.collision(),
#     platform='CUDA',
# )


def _get_body_has_plane(m: Model) -> np.ndarray:
    # Determine which bodies have plane geoms
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


def squeeze_array(a: jax.Array, target_ndim: int) -> jax.Array:
    # remove batch dimension
    if a.ndim == target_ndim:
        return a
    if a.ndim == target_ndim + 1:
        return a.reshape(-1, *a.shape[2:])
    raise ValueError(f"Invalid array shape: {a.shape}, expected {target_ndim} or {target_ndim + 1}")


def unsqueeze_array(a: Union[np.ndarray, jax.Array], batch_dim) -> jax.Array:
    # add batch dimension
    shape_div_batch = a.shape[0] // batch_dim
    new_shape = (batch_dim, shape_div_batch, *a.shape[1:])
    return a.reshape(*new_shape)


@wp.kernel
def finalize_sum(
    nenv: int, scan: wp.array(dtype=wp.int32), data_before_scan: wp.array(dtype=wp.int32), sum: wp.array(dtype=wp.int32)
):
    tid = wp.tid()
    if tid == 0:
        sum[0] = scan[nenv - 1] + data_before_scan[nenv - 1]


@wp.kernel
def init(
    max_contact_points: int,
    nenv: int,
    # outputs
    dist: wp.array(dtype=wp.float32),
    pos: wp.array(dtype=wp.vec3),
    normal: wp.array(dtype=wp.vec3),
    g1: wp.array(dtype=wp.int32),
    g2: wp.array(dtype=wp.int32),
    includemargin: wp.array(dtype=wp.float32),
    friction: wp.array(dtype=wp.float32, ndim=2),
    solref: wp.array(dtype=wp.float32, ndim=2),
    solreffriction: wp.array(dtype=wp.float32, ndim=2),
    solimp: wp.array(dtype=wp.float32, ndim=2),
):
    tid = wp.tid()
    if tid >= nenv * max_contact_points:
        return

    dist[tid] = 1e12
    pos[tid] = wp.vec3(0.0, 0.0, 0.0)
    normal[tid] = wp.vec3(0.0, 0.0, 0.0)
    g1[tid] = -1
    g2[tid] = -1
    includemargin[tid] = 0.0
    solref[tid, 0] = 0.02
    solref[tid, 1] = 1.0
    solimp[tid, 0] = 0.9
    solimp[tid, 1] = 0.95
    solimp[tid, 2] = 0.001
    solimp[tid, 3] = 0.5
    solimp[tid, 4] = 2.0
    friction[tid, 0] = 1.0
    friction[tid, 1] = 1.0
    friction[tid, 2] = 0.005
    friction[tid, 3] = 0.0001
    friction[tid, 4] = 0.0001
    solreffriction[tid, 0] = 0.0
    solreffriction[tid, 1] = 0.0


@wp.struct
class OrthoBasis:
    b: wp.vec3
    c: wp.vec3


@wp.func
def orthogonals(a: wp.vec3) -> OrthoBasis:
    y = wp.vec3(0.0, 1.0, 0.0)
    z = wp.vec3(0.0, 0.0, 1.0)
    b = where((-0.5 < a[1]) and (a[1] < 0.5), y, z)
    b = b - a * wp.dot(a, b)
    b = wp.normalize(b)
    if wp.length(a) == 0.0:
        b = wp.vec3(0.0, 0.0, 0.0)
    c = wp.cross(a, b)

    result = OrthoBasis(b=b, c=c)
    return result


@wp.kernel
def make_frame(
    n_frames: int,
    a: wp.array(dtype=wp.vec3),
    # outputs
    frame: wp.array(dtype=wp.mat33),
):
    tid = wp.tid()
    if tid >= n_frames:
        return

    a_normalized = wp.normalize(a[tid])
    basis = orthogonals(a_normalized)

    # fmt: off
    m = wp.mat33(
        a_normalized.x, a_normalized.y, a_normalized.z,
        basis.b.x, basis.b.y, basis.b.z,
        basis.c.x, basis.c.y, basis.c.z
    )
    # fmt: on

    frame[tid] = m


def collision(
    geom_xpos: wp.array(dtype=wp.vec3),
    geom_xmat: wp.array(dtype=wp.mat33),
    geom_size: wp.array(dtype=wp.vec3),
    geom_type: wp.array(dtype=int),
    geom_contype: wp.array(dtype=int),
    geom_conaffinity: wp.array(dtype=int),
    geom_priority: wp.array(dtype=int),
    geom_margin: wp.array(dtype=float),
    geom_gap: wp.array(dtype=float),
    geom_solmix: wp.array(dtype=float),
    geom_friction: wp.array(dtype=float, ndim=2),
    geom_solref: wp.array(dtype=float, ndim=2),
    geom_solimp: wp.array(dtype=float, ndim=2),
    # TODO(btaba): allow vmapping over sizes via geom_aabb/rbound jax.Array.
    geom_aabb: wp.array(dtype=float, ndim=2),
    geom_rbound: wp.array(dtype=float),
    geom_dataid: wp.array(dtype=int),
    # geom_bodyid,
    body_parentid: wp.array(dtype=int),
    body_weldid: wp.array(dtype=int),
    body_contype: wp.array(dtype=int),
    body_conaffinity: wp.array(dtype=int),
    body_geomadr: wp.array(dtype=int),
    body_geomnum: wp.array(dtype=int),
    body_has_plane: wp.array(dtype=bool),
    # pair_geom1,
    # pair_geom2,
    exclude_signature: wp.array(dtype=int),
    # pair_margin,
    # pair_gap,
    # pair_friction,
    # pair_solref,
    # pair_solimp,
    convex_vert: wp.array(dtype=wp.vec3),
    convex_vert_offset: wp.array(dtype=int),
    type_pair_offset: wp.array(dtype=int),
    type_pair_count: wp.array(dtype=int),
    # nenv: int,
    # nmodel: int,
    ngeom: int,
    # npair: int,
    nbody: int,
    nexclude: int,
    max_contact_points: int,
    n_geom_pair: int,
    n_geom_types: int,
    filter_parent: bool,
    depth_extension: float,
    gjk_iteration_count: int,
    epa_iteration_count: int,
    epa_best_count: int,
    multi_polygon_count: int,
    multi_tilt_angle: float,
    # outputs
    contact_geom1: wp.array(dtype=int),
    contact_geom2: wp.array(dtype=int),
    contact_dist: wp.array(dtype=float),
    contact_pos: wp.array(dtype=wp.vec3),
    contact_normal: wp.array(dtype=wp.vec3),
    includemargin: wp.array(dtype=float),
    friction: wp.array(dtype=float, ndim=2),
    solref: wp.array(dtype=float, ndim=2),
    solreffriction: wp.array(dtype=float, ndim=2),
    solimp: wp.array(dtype=float, ndim=2),
):
    if ngeom == 0:
        return True

    device = contact_dist.device

    # XXX this is annoying
    geom_xpos = geom_xpos.reshape(-1)
    geom_xmat = geom_xmat.reshape(-1)
    geom_size = geom_size.reshape(-1)
    geom_type = geom_type.reshape(-1)
    geom_contype = geom_contype.reshape(-1)
    geom_conaffinity = geom_conaffinity.reshape(-1)
    geom_priority = geom_priority.reshape(-1)
    geom_margin = geom_margin.reshape(-1)
    geom_gap = geom_gap.reshape(-1)
    geom_solmix = geom_solmix.reshape(-1)
    geom_friction = geom_friction.reshape((-1, 3))
    geom_solref = geom_solref.reshape((-1, 2))
    geom_solimp = geom_solimp.reshape((-1, 5))
    geom_aabb = geom_aabb.reshape((-1, 6))
    geom_rbound = geom_rbound.reshape(-1)
    geom_dataid = geom_dataid.reshape(-1)
    body_parentid = body_parentid.reshape(-1)
    body_weldid = body_weldid.reshape(-1)
    body_contype = body_contype.reshape(-1)
    body_conaffinity = body_conaffinity.reshape(-1)
    body_geomadr = body_geomadr.reshape(-1)
    body_geomnum = body_geomnum.reshape(-1)
    body_has_plane = body_has_plane.reshape(-1)
    exclude_signature = exclude_signature.reshape(-1)
    convex_vert = convex_vert.reshape(-1)
    convex_vert_offset = convex_vert_offset.reshape(-1)
    type_pair_offset = type_pair_offset.reshape(-1)
    type_pair_count = type_pair_count.reshape(-1)

    contact_geom1 = contact_geom1.reshape(-1)
    contact_geom2 = contact_geom2.reshape(-1)
    contact_dist = contact_dist.reshape(-1)
    contact_pos = contact_pos.reshape(-1)
    contact_normal = contact_normal.reshape(-1)
    includemargin = includemargin.reshape(-1)
    friction = friction.reshape((-1, 5))
    solref = solref.reshape((-1, 2))
    solreffriction = solreffriction.reshape((-1, 2))
    solimp = solimp.reshape((-1, 5))

    # Get the batch size of mjx.Data.
    nenv = 1
    for i in range(geom_xpos.ndim):  # note: geom_xpos is 2D in JAX, 1D in Warp for the unbatched case
        nenv *= geom_xpos.shape[i]
    nenv //= ngeom
    if nenv == 0:
        raise RuntimeError("Batch size of mjx.Data calculated in LaunchKernel_GJK_EPA is 0.")

    # Get the batch size of mjx.Model.
    nmodel = 1
    for i in range(geom_size.ndim):  # note: geom_size is 2D in JAX, 1D in Warp for the unbatched case
        nmodel *= geom_size.shape[i]
    nmodel //= ngeom
    if nmodel == 0:
        raise RuntimeError("Batch size of mjx.Model calculated in LaunchKernel_GJK_EPA is 0.")

    # Initialize the output data
    wp.launch(
        init,
        dim=[nenv * max_contact_points],
        inputs=[
            max_contact_points,
            nenv,
        ],
        outputs=[
            contact_dist,
            contact_pos,
            contact_normal,
            contact_geom1,
            contact_geom2,
            includemargin,
            friction,
            solref,
            solreffriction,
            solimp,
        ],
        device=device,
    )

    dyn_body_aamm = wp.empty((nenv * nbody, 6), dtype=wp.float32)
    dyn_geom_aabb = wp.empty((nenv * ngeom, 6), dtype=wp.float32)
    nbody_pair_buf_size = ((nbody * (nbody - 1) // 2 + 15) // 16) * 16
    col_body_pair = wp.empty((nenv * nbody_pair_buf_size, 2), dtype=wp.int32)
    # col_body_pair = wp.empty((nenv * nbody * nbody, 2), dtype=wp.int32)
    col_geom_pair = wp.empty((nenv * n_geom_pair, 2), dtype=wp.int32)

    type_pair_env_id = wp.zeros(nenv * n_geom_pair, dtype=wp.int32)
    type_pair_geom_id = wp.zeros((nenv * n_geom_pair, 2), dtype=wp.int32)
    type_pair_count = wp.zeros(n_geom_types * n_geom_types, dtype=wp.int32)
    tmp_count = wp.zeros(1, dtype=wp.int32)

    # Generate body AAMMs
    wp.launch(
        get_dyn_body_aamm,
        dim=[nenv * nbody],
        inputs=[
            nenv,
            nbody,
            nmodel,
            ngeom,
            body_geomnum,
            body_geomadr,
            geom_margin,
            geom_xpos,
            geom_rbound,
        ],
        outputs=[
            dyn_body_aamm,
        ],
        device=device,
    )

    # Generate body pairs (broadphase)
    col_body_pair_count = wp.zeros(nenv, dtype=wp.int32)
    wp.launch(
        get_body_pairs_nxn,
        dim=[nenv * nbody * nbody],
        inputs=[
            nenv,
            nbody,
            filter_parent,
            nexclude,
            body_parentid,
            body_weldid,
            body_contype,
            body_conaffinity,
            body_has_plane,
            exclude_signature,
            dyn_body_aamm,
        ],
        outputs=[
            col_body_pair,
            col_body_pair_count,
        ],
        device=device,
    )

    # Get geom AABBs in global frame
    wp.launch(
        get_dyn_geom_aabb,
        dim=[nenv * ngeom],
        inputs=[
            nenv,
            nmodel,
            ngeom,
            geom_xpos,
            geom_xmat,
            geom_aabb,
        ],
        outputs=[
            dyn_geom_aabb,
        ],
        device=device,
    )

    body_pair_offset = wp.zeros(nenv, dtype=wp.int32)
    wp.utils.array_scan(col_body_pair_count, body_pair_offset, False)

    wp.launch(
        finalize_sum,
        dim=1,
        inputs=[
            nenv,
            body_pair_offset,
            col_body_pair_count,
        ],
        outputs=[tmp_count],
        device=device,
    )

    total_body_pairs = int(tmp_count.numpy()[0])
    col_geom_pair_count = wp.zeros(nenv, dtype=wp.int32)

    wp.launch(
        get_geom_pairs_nxn,
        dim=[total_body_pairs],
        inputs=[
            nenv,
            ngeom,
            nbody,
            n_geom_pair,
            body_geomnum,
            body_geomadr,
            geom_contype,
            geom_conaffinity,
            geom_type,
            geom_margin,
            dyn_geom_aabb,
            col_body_pair,
            col_body_pair_count,
            body_pair_offset,
        ],
        outputs=[
            col_geom_pair,
            col_geom_pair_count,
        ],
        device=device,
    )

    # Initialize type pair count
    type_pair_count.zero_()

    col_geom_pair_offset = wp.zeros(nenv, dtype=wp.int32)
    wp.utils.array_scan(col_geom_pair_count, col_geom_pair_offset, False)

    wp.launch(
        finalize_sum,
        dim=1,
        inputs=[
            nenv,
            col_geom_pair_offset,
            col_geom_pair_count,
        ],
        outputs=[tmp_count],
        device=device,
    )

    total_geom_pairs = int(tmp_count.numpy()[0])
    assert total_geom_pairs > 0

    wp.launch(
        group_contacts_by_type,
        dim=[total_geom_pairs],
        inputs=[
            nenv,
            n_geom_pair,
            n_geom_types,
            geom_type,
            col_geom_pair,
            col_geom_pair_count,
            col_geom_pair_offset,
            type_pair_offset,
        ],
        outputs=[
            type_pair_env_id,
            type_pair_geom_id,
            type_pair_count,
        ],
        device=device,
    )

    # Initialize the env contact counter
    env_contact_counter = wp.zeros(nenv, dtype=wp.int32)

    # Dispatch to narrowphase collision functions
    max_contact_points_per_env = max_contact_points
    narrowphase(
        gjk_iteration_count,
        epa_iteration_count,
        nenv,
        ngeom,
        nmodel,
        n_geom_types,
        max_contact_points_per_env,
        type_pair_env_id,
        type_pair_geom_id,
        type_pair_count,
        type_pair_offset,
        geom_xpos,
        geom_xmat,
        geom_size,
        geom_dataid,
        convex_vert,
        convex_vert_offset,
        epa_best_count,
        depth_extension,
        multi_polygon_count,
        multi_tilt_angle,
        env_contact_counter,
        contact_geom1,
        contact_geom2,
        contact_dist,
        contact_pos,
        contact_normal,
    )

    env_contact_offset = wp.zeros(nenv, dtype=wp.int32)
    wp.utils.array_scan(env_contact_counter, env_contact_offset, False)

    wp.launch(
        finalize_sum,
        dim=1,
        inputs=[
            nenv,
            env_contact_offset,
            env_contact_counter,
        ],
        outputs=[tmp_count],
        device=device,
    )

    n_contact_pts = tmp_count.numpy()[0]

    wp.launch(
        get_contact_solver_params,
        dim=[n_contact_pts],
        inputs=[
            nenv,
            nmodel,
            ngeom,
            max_contact_points,
            n_contact_pts,
            contact_geom1,
            contact_geom2,
            geom_priority,
            geom_solmix,
            geom_friction,
            geom_solref,
            geom_solimp,
            geom_margin,
            geom_gap,
            env_contact_offset,
        ],
        outputs=[
            includemargin,
            friction,
            solref,
            solreffriction,
            solimp,
        ],
        device=device,
    )

    return True
