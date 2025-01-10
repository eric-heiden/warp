from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import copy

import jax
import mujoco
import numpy as np
from absl.testing import absltest
from jax import numpy as jp
from mujoco import mjx
from mujoco.mjx._src.types import Data, Model

import warp as wp
import warp.sim

# wp.set_device("cpu")

# wp.config.verify_cuda = True
# wp.config.verify_fp = True

wp.config.enable_backward = False
wp.set_module_options({"enable_backward": False})

mjxGEOM_PLANE = 0
mjxGEOM_HFIELD = 1
mjxGEOM_SPHERE = 2
mjxGEOM_CAPSULE = 3
mjxGEOM_ELLIPSOID = 4
mjxGEOM_CYLINDER = 5
mjxGEOM_BOX = 6
mjxGEOM_CONVEX = 7
mjxGEOM_size = 8

mjMINVAL = 1e-15

FLOAT_MIN = -1e30
FLOAT_MAX = 1e30

kGjkMultiContactCount = 4
kMaxEpaBestCount = 12
kMaxMultiPolygonCount = 8


@wp.struct
class GeomType_PLANE:
    pos: wp.vec3
    rot: wp.mat33


@wp.struct
class GeomType_SPHERE:
    pos: wp.vec3
    rot: wp.mat33
    radius: float


@wp.struct
class GeomType_CAPSULE:
    pos: wp.vec3
    rot: wp.mat33
    radius: float
    halfsize: float


@wp.struct
class GeomType_ELLIPSOID:
    pos: wp.vec3
    rot: wp.mat33
    size: wp.vec3


@wp.struct
class GeomType_CYLINDER:
    pos: wp.vec3
    rot: wp.mat33
    radius: float
    halfsize: float


@wp.struct
class GeomType_BOX:
    pos: wp.vec3
    rot: wp.mat33
    size: wp.vec3


@wp.struct
class GeomType_CONVEX:
    pos: wp.vec3
    rot: wp.mat33
    vert_offset: int
    vert_count: int


def get_info(t):
    @wp.func
    def _get_info(
        gid: int,
        dataid: int,
        geom_xpos: wp.array(dtype=wp.vec3),
        geom_xmat: wp.array(dtype=wp.mat33),
        size: wp.vec3,
        convex_vert_offset: wp.array(dtype=int),
    ):
        pos = geom_xpos[gid]
        rot = geom_xmat[gid]
        if wp.static(t == mjxGEOM_SPHERE):
            sphere = GeomType_SPHERE()
            sphere.pos = pos
            sphere.rot = rot
            sphere.radius = size[0]
            return sphere
        elif wp.static(t == mjxGEOM_BOX):
            box = GeomType_BOX()
            box.pos = pos
            box.rot = rot
            box.size = size
            return box
        elif wp.static(t == mjxGEOM_PLANE):
            plane = GeomType_PLANE()
            plane.pos = pos
            plane.rot = rot
            return plane
        elif wp.static(t == mjxGEOM_CAPSULE):
            capsule = GeomType_CAPSULE()
            capsule.pos = pos
            capsule.rot = rot
            capsule.radius = size[0]
            capsule.halfsize = size[1]
            return capsule
        elif wp.static(t == mjxGEOM_ELLIPSOID):
            ellipsoid = GeomType_ELLIPSOID()
            ellipsoid.pos = pos
            ellipsoid.rot = rot
            ellipsoid.size = size
            return ellipsoid
        elif wp.static(t == mjxGEOM_CYLINDER):
            cylinder = GeomType_CYLINDER()
            cylinder.pos = pos
            cylinder.rot = rot
            cylinder.radius = size[0]
            cylinder.halfsize = size[1]
            return cylinder
        elif wp.static(t == mjxGEOM_CONVEX):
            convex = GeomType_CONVEX()
            convex.pos = pos
            convex.rot = rot
            if convex_vert_offset and dataid >= 0:
                convex.vert_offset = convex_vert_offset[dataid]
                convex.vert_count = convex_vert_offset[dataid + 1] - convex.vert_offset
            else:
                convex.vert_offset = 0
                convex.vert_count = 0
            return convex
        else:
            wp.static(RuntimeError("Unsupported type", t))

    return _get_info


@wp.func
def gjk_support_plane(
    info: GeomType_PLANE,
    dir: wp.vec3,
    convex_vert: wp.array(dtype=wp.vec3),
):
    local_dir = wp.transpose(info.rot) @ dir
    norm = wp.sqrt(local_dir[0] * local_dir[0] + local_dir[1] * local_dir[1])
    if norm > 0.0:
        nx = local_dir[0] / norm
        ny = local_dir[1] / norm
    else:
        nx = 1.0
        ny = 0.0
    nz = -float(int(local_dir[2] < 0))
    largeSize = 5.0
    res = wp.vec3(nx * largeSize, ny * largeSize, nz * largeSize)
    support_pt = info.rot @ res + info.pos
    return wp.dot(support_pt, dir), support_pt


@wp.func
def gjk_support_sphere(
    info: GeomType_SPHERE,
    dir: wp.vec3,
    convex_vert: wp.array(dtype=wp.vec3),
):
    support_pt = info.pos + info.radius * dir
    return wp.dot(support_pt, dir), support_pt


@wp.func
def sign(x: wp.vec3):
    return wp.vec3(wp.sign(x[0]), wp.sign(x[1]), wp.sign(x[2]))


@wp.func
def gjk_support_box(
    info: GeomType_BOX,
    dir: wp.vec3,
    convex_vert: wp.array(dtype=wp.vec3),
):
    local_dir = wp.transpose(info.rot) @ dir
    res = wp.cw_mul(sign(local_dir), info.size)
    support_pt = info.rot @ res + info.pos
    return wp.dot(support_pt, dir), support_pt


@wp.func
def gjk_support_capsule(
    info: GeomType_CAPSULE,
    dir: wp.vec3,
    convex_vert: wp.array(dtype=wp.vec3),
):
    local_dir = wp.transpose(info.rot) @ dir
    # start with sphere
    res = local_dir * info.radius
    # add cylinder contribution
    res[2] += wp.sign(local_dir[2]) * info.halfsize
    support_pt = info.rot @ res + info.pos
    return wp.dot(support_pt, dir), support_pt


@wp.func
def gjk_support_ellipsoid(
    info: GeomType_ELLIPSOID,
    dir: wp.vec3,
    convex_vert: wp.array(dtype=wp.vec3),
):
    local_dir = wp.transpose(info.rot) @ dir
    # find support point on unit sphere: scale dir by ellipsoid sizes and
    # renormalize
    res = local_dir * info.size
    res = wp.normalize(res)
    # transform to ellipsoid
    res = wp.cw_mul(res, info.size)
    support_pt = info.rot @ res + info.pos
    return wp.dot(support_pt, dir), support_pt


@wp.func
def gjk_support_cylinder(
    info: GeomType_CYLINDER,
    dir: wp.vec3,
    convex_vert: wp.array(dtype=wp.vec3),
):
    local_dir = wp.transpose(info.rot) @ dir
    res = wp.vec3(0.0, 0.0, 0.0)
    # set result in XY plane: support on circle
    d = wp.sqrt(wp.dot(local_dir, local_dir))
    if d > mjMINVAL:
        res[0] = local_dir[0] / d * info.radius
        res[1] = local_dir[1] / d * info.radius

    # set result in Z direction
    res[2] = wp.sign(local_dir[2]) * info.halfsize
    support_pt = info.rot @ res + info.pos
    return wp.dot(support_pt, dir), support_pt


@wp.func
def gjk_support_convex(
    info: GeomType_CONVEX,
    dir: wp.vec3,
    convex_vert: wp.array(dtype=wp.vec3),
):
    local_dir = wp.transpose(info.rot) @ dir
    support_pt = wp.vec3(0.0, 0.0, 0.0)
    max_dist = float(FLOAT_MIN)
    # exhaustive search over all vertices
    # TODO(robotics-simulation): consider hill-climb over graphdata.
    for i in range(info.vert_count):
        vert = convex_vert[info.vert_offset + i]
        dist = wp.dot(vert, local_dir)
        if dist > max_dist:
            max_dist = dist
            support_pt = vert
    support_pt = info.rot @ support_pt + info.pos
    return wp.dot(support_pt, dir), support_pt


support_functions = {
    mjxGEOM_PLANE: gjk_support_plane,
    mjxGEOM_SPHERE: gjk_support_sphere,
    mjxGEOM_BOX: gjk_support_box,
    mjxGEOM_CAPSULE: gjk_support_capsule,
    mjxGEOM_ELLIPSOID: gjk_support_ellipsoid,
    mjxGEOM_CYLINDER: gjk_support_cylinder,
    mjxGEOM_CONVEX: gjk_support_convex,
}


def gjk_support(type1, type2):
    @wp.func
    def _gjk_support(
        info1: Any,
        info2: Any,
        dir: wp.vec3,
        convex_vert: wp.array(dtype=wp.vec3),
    ):
        # Returns the distance between support points on two geoms, and the support point.
        # Negative distance means objects are not intersecting along direction `dir`.
        # Positive distance means objects are intersecting along the given direction `dir`.

        dist1, s1 = wp.static(support_functions[type1])(info1, dir, convex_vert)
        dist2, s2 = wp.static(support_functions[type2])(info2, -dir, convex_vert)

        support_pt = s1 - s2
        return dist1 + dist2, support_pt

    return _gjk_support


@wp.func
def gjk_normalize(a: wp.vec3):
    norm = wp.length(a)
    if norm > 1e-8 and norm < 1e12:
        a /= norm
        return a, True
    return a, False


@wp.func
def orthonormal(normal: wp.vec3) -> wp.vec3:
    if wp.abs(normal[0]) < wp.abs(normal[1]) and wp.abs(normal[0]) < wp.abs(normal[2]):
        dir = wp.vec3(1.0 - normal[0] * normal[0], -normal[0] * normal[1], -normal[0] * normal[2])
    elif wp.abs(normal[1]) < wp.abs(normal[2]):
        dir = wp.vec3(-normal[1] * normal[0], 1.0 - normal[1] * normal[1], -normal[1] * normal[2])
    else:
        dir = wp.vec3(-normal[2] * normal[0], -normal[2] * normal[1], 1.0 - normal[2] * normal[2])
    dir, _ = gjk_normalize(dir)
    return dir


@wp.func
def where(condition: bool, ret_true: Any, ret_false: Any):
    if condition:
        return ret_true
    return ret_false


@wp.func
def all_same(a: wp.vec3, b: wp.vec3):
    return a[0] == b[0] and a[1] == b[1] and a[2] == b[2]


@wp.func
def any_different(a: wp.vec3, b: wp.vec3):
    return a[0] != b[0] or a[1] != b[1] or a[2] != b[2]


mat43 = wp.types.matrix(shape=(4, 3), dtype=float)


class GjkEpaPipeline:
    def __init__(
        self,
        type1,
        type2,
        gjk_dense: wp.Kernel,
        epa_dense: wp.Kernel,
        multiple_contacts_dense: wp.Kernel,
        gjk_epa_sparse: wp.Kernel,
    ):
        self.type1 = type1
        self.type2 = type2
        self.gjk_dense = gjk_dense
        self.epa_dense = epa_dense
        self.multiple_contacts_dense = multiple_contacts_dense
        self.gjk_epa_sparse = gjk_epa_sparse


def gjk_epa_pipeline(
    type1: int,
    type2: int,
    gjk_iteration_count: int,
    epa_iteration_count: int,
    max_epa_best_count: int = kMaxEpaBestCount,
    epa_exact_neg_distance: bool = True,
    kMaxMultiPolygonCount: int = kMaxMultiPolygonCount,
    kGjkMultiContactCount: int = kGjkMultiContactCount,
) -> GjkEpaPipeline:
    type1 = int(type1)
    type2 = int(type2)

    # Calculates whether two objects intersect.
    # Returns simplex and normal.
    @wp.func
    def _gjk(
        env_id: int,
        model_id: int,
        g1: int,
        g2: int,
        ngeom: int,
        geom_xpos: wp.array(dtype=wp.vec3),
        geom_xmat: wp.array(dtype=wp.mat33),
        geom_size: wp.array(dtype=wp.vec3),
        geom_dataid: wp.array(dtype=wp.int32),
        convex_vert: wp.array(dtype=wp.vec3),
        convex_vert_offset: wp.array(dtype=int),
    ):
        dataid1 = -1
        dataid2 = -1
        if geom_dataid:
            dataid1 = geom_dataid[g1]
            dataid2 = geom_dataid[g2]
        size1 = geom_size[model_id * ngeom + g1]
        size2 = geom_size[model_id * ngeom + g2]
        gid1 = env_id * ngeom + g1
        gid2 = env_id * ngeom + g2
        info1 = wp.static(get_info(type1))(gid1, dataid1, geom_xpos, geom_xmat, size1, convex_vert_offset)
        info2 = wp.static(get_info(type2))(gid2, dataid2, geom_xpos, geom_xmat, size2, convex_vert_offset)

        dir = wp.vec3(0.0, 0.0, 1.0)
        dir_n = -dir
        depth = 1e30

        dist_max, simplex0 = wp.static(gjk_support(type1, type2))(info1, info2, dir, convex_vert)
        dist_min, simplex1 = wp.static(gjk_support(type1, type2))(info1, info2, dir_n, convex_vert)
        if dist_max < dist_min:
            depth = dist_max
            normal = dir
        else:
            depth = dist_min
            normal = dir_n

        # sd = wp.normalize(simplex0 - simplex1)
        sd = simplex0 - simplex1
        dir = orthonormal(sd)

        dist_max, simplex3 = wp.static(gjk_support(type1, type2))(info1, info2, dir, convex_vert)
        # Initialize a 2-simplex with simplex[2]==simplex[1]. This ensures the
        # correct winding order for face normals defined below. Face 0 and face 3
        # are degenerate, and face 1 and 2 have opposing normals.
        simplex = mat43()
        simplex[0] = simplex0
        simplex[1] = simplex1
        simplex[2] = simplex[1]
        simplex[3] = simplex3

        if dist_max < depth:
            depth = dist_max
            normal = dir
        if dist_min < depth:
            depth = dist_min
            normal = dir_n

        plane = mat43()
        for _ in range(gjk_iteration_count):
            # Winding orders: plane[0] ccw, plane[1] cw, plane[2] ccw, plane[3] cw.
            plane[0] = wp.cross(simplex[3] - simplex[2], simplex[1] - simplex[2])
            plane[1] = wp.cross(simplex[3] - simplex[0], simplex[2] - simplex[0])
            plane[2] = wp.cross(simplex[3] - simplex[1], simplex[0] - simplex[1])
            plane[3] = wp.cross(simplex[2] - simplex[0], simplex[1] - simplex[0])

            # Compute distance of each face halfspace to the origin. If d<0, then the
            # origin is outside the halfspace. If d>0 then the origin is inside
            # the halfspace defined by the face plane.
            d = wp.vec4(1e30)
            plane0, p0 = gjk_normalize(plane[0])
            plane[0] = plane0  # XXX currently cannot assign directly from multiple-return functions
            if p0:
                d[0] = wp.dot(plane[0], simplex[2])
            plane1, p1 = gjk_normalize(plane[1])
            plane[1] = plane1
            if p1:
                d[1] = wp.dot(plane[1], simplex[0])
            plane2, p2 = gjk_normalize(plane[2])
            plane[2] = plane2
            if p2:
                d[2] = wp.dot(plane[2], simplex[1])
            plane3, p3 = gjk_normalize(plane[3])
            plane[3] = plane3
            if p3:
                d[3] = wp.dot(plane[3], simplex[0])

            # Pick the plane normal with minimum distance to the origin.
            i1 = where(d[0] < d[1], 0, 1)
            i2 = where(d[2] < d[3], 2, 3)
            index = where(d[i1] < d[i2], i1, i2)
            if d[index] > 0.0:
                # Origin is inside the simplex, objects are intersecting.
                break

            # Add new support point to the simplex.
            dist, simplex_i = wp.static(gjk_support(type1, type2))(info1, info2, plane[index], convex_vert)
            simplex[index] = simplex_i
            if dist < depth:
                depth = dist
                normal = plane[index]
            # wp.printf("dist: %f\n", dist)

            # Preserve winding order of the simplex faces.
            index1 = (index + 1) & 3
            index2 = (index + 2) & 3
            swap = simplex[index1]
            simplex[index1] = simplex[index2]
            simplex[index2] = swap
            # wp.printf("simplex[0]: %f %f %f\n", simplex[0, 0], simplex[0, 1], simplex[0, 2])
            # wp.printf("simplex[1]: %f %f %f\n", simplex[1, 0], simplex[1, 1], simplex[1, 2])
            # wp.printf("simplex[2]: %f %f %f\n", simplex[2, 0], simplex[2, 1], simplex[2, 2])
            # wp.printf("simplex[3]: %f %f %f\n", simplex[3, 0], simplex[3, 1], simplex[3, 2])
            if dist < 0.0:
                break  # Objects are likely non-intersecting.

        return simplex, normal

    @wp.kernel
    def gjk_dense(
        npair: int,
        nenv: int,
        ngeom: int,
        nmodel: int,
        geom_pair: wp.array(dtype=int, ndim=2),
        geom_xpos: wp.array(dtype=wp.vec3),
        geom_xmat: wp.array(dtype=wp.mat33),
        geom_size: wp.array(dtype=wp.vec3),
        geom_dataid: wp.array(dtype=wp.int32),
        convex_vert: wp.array(dtype=wp.vec3),
        convex_vert_offset: wp.array(dtype=int),
        contact_normal: wp.array(dtype=wp.vec3),
        contact_simplex: wp.array(dtype=mat43),
    ):
        tid = wp.tid()
        if tid >= npair * nenv:
            return

        pair_id = tid % npair
        env_id = tid // npair
        model_id = env_id % nmodel

        g1, g2 = geom_pair[pair_id, 0], geom_pair[pair_id, 1]
        if g1 < 0 or g2 < 0:
            return

        simplex, normal = _gjk(
            env_id,
            model_id,
            g1,
            g2,
            ngeom,
            geom_xpos,
            geom_xmat,
            geom_size,
            geom_dataid,
            convex_vert,
            convex_vert_offset,
        )
        contact_simplex[tid] = simplex

        if contact_normal:
            contact_normal[tid] = normal

    matc3 = wp.types.matrix(shape=(max_epa_best_count, 3), dtype=float)
    vecc3 = wp.types.vector(max_epa_best_count * 3, dtype=float)

    # Matrix definition for the `tris` scratch space which is used to store the
    # triangles of the polytope. Note that the first dimension is 2, as we need
    # to store the previous and current polytope. But since Warp doesn't support
    # 3D matrices yet, we use 2 * 3 * max_epa_best_count as the first dimension.
    tris_dim = 3 * max_epa_best_count
    mat2c3 = wp.types.matrix(shape=(2 * tris_dim, 3), dtype=float)

    # computes contact normal and depth
    @wp.func
    def _epa(
        env_id: int,
        model_id: int,
        g1: int,
        g2: int,
        ngeom: int,
        geom_xpos: wp.array(dtype=wp.vec3),
        geom_xmat: wp.array(dtype=wp.mat33),
        geom_size: wp.array(dtype=wp.vec3),
        geom_dataid: wp.array(dtype=wp.int32),
        convex_vert: wp.array(dtype=wp.vec3),
        convex_vert_offset: wp.array(dtype=int),
        depth_extension: float,
        epa_best_count: int,
        simplex: mat43,
        input_normal: wp.vec3,
    ):
        dataid1 = -1
        dataid2 = -1
        if geom_dataid:
            dataid1 = geom_dataid[g1]
            dataid2 = geom_dataid[g2]

        size1 = geom_size[model_id * ngeom + g1]
        size2 = geom_size[model_id * ngeom + g2]
        tg1 = env_id * ngeom + g1
        tg2 = env_id * ngeom + g2
        info1 = wp.static(get_info(type1))(tg1, dataid1, geom_xpos, geom_xmat, size1, convex_vert_offset)
        info2 = wp.static(get_info(type2))(tg2, dataid2, geom_xpos, geom_xmat, size2, convex_vert_offset)

        normal = input_normal

        # Get the support. If less than 0, objects are not intersecting.
        depth, _simplex = wp.static(gjk_support(type1, type2))(info1, info2, normal, convex_vert)

        if depth < -depth_extension:
            # Objects are not intersecting, and we do not obtain the closest points as
            # specified by depth_extension.
            return wp.nan, wp.vec3(wp.nan, wp.nan, wp.nan)

        if wp.static(epa_exact_neg_distance):
            # Check closest points to all edges of the simplex, rather than just the
            # face normals. This gives the exact depth/normal for the non-intersecting
            # case.
            for i in range(6):
                if i < 3:
                    i1 = 0
                    i2 = i + 1
                elif i < 5:
                    i1 = 1
                    i2 = i - 1
                else:
                    i1 = 2
                    i2 = 3
                si1 = simplex[i1]
                si2 = simplex[i2]
                if si1[0] != si2[0] or si1[1] != si2[1] or si1[2] != si2[2]:
                    v = si1 - si2
                    alpha = wp.dot(si1, v) / wp.dot(v, v)
                    # p0 is the closest segment point to the origin.
                    p0 = wp.clamp(alpha, 0.0, 1.0) * v - si1
                    p0, pf = gjk_normalize(p0)
                    if pf:
                        depth2, _ = wp.static(gjk_support(type1, type2))(info1, info2, p0, convex_vert)
                        if depth2 < depth:
                            depth = depth2
                            normal = p0

        # TODO do we need to allocate p?
        p = matc3()  # supporting points for each triangle.
        # Distance to the origin for candidate triangles.
        dists = vecc3()

        tris = mat2c3()
        tris[0] = simplex[2]
        tris[1] = simplex[1]
        tris[2] = simplex[3]

        tris[3] = simplex[0]
        tris[4] = simplex[2]
        tris[5] = simplex[3]

        tris[6] = simplex[1]
        tris[7] = simplex[0]
        tris[8] = simplex[3]

        tris[9] = simplex[0]
        tris[10] = simplex[1]
        tris[11] = simplex[2]

        count = int(4)
        for _iter in range(wp.static(epa_iteration_count)):
            for i in range(count):
                # Loop through all triangles, and obtain distances to the origin for each
                # new triangle candidate.
                ti = 3 * i
                n = wp.cross(tris[ti + 2] - tris[ti + 0], tris[ti + 1] - tris[ti + 0])

                n, nf = gjk_normalize(n)
                if not nf:
                    for j in range(3):
                        dists[i * 3 + j] = 2e30
                    continue

                dist, pi = wp.static(gjk_support(type1, type2))(info1, info2, n, convex_vert)
                p[i] = pi
                if dist < depth:
                    depth = dist
                    normal = n
                # Loop through all edges, and get distance using support point p[i].
                for j in range(3):
                    if wp.static(epa_exact_neg_distance):
                        # Obtain the closest point between the new triangle edge and the origin.
                        tqj = tris[ti + j]
                        if (p[i, 0] != tqj[0]) or (p[i, 1] != tqj[1]) or (p[i, 2] != tqj[2]):
                            v = p[i] - tris[ti + j]
                            alpha = wp.dot(p[i], v) / wp.dot(v, v)
                            p0 = wp.clamp(alpha, 0.0, 1.0) * v - p[i]
                            p0, pf = gjk_normalize(p0)
                            if pf:
                                dist2, v = wp.static(gjk_support(type1, type2))(info1, info2, p0, convex_vert)
                                if dist2 < depth:
                                    depth = dist2
                                    normal = p0

                    plane = wp.cross(p[i] - tris[ti + j], tris[ti + ((j + 1) % 3)] - tris[ti + j])
                    plane, pf = gjk_normalize(plane)
                    if pf:
                        d = wp.dot(plane, tris[ti + j])
                    else:
                        d = 1e30

                    if (d < 0 and depth >= 0) or (
                        tris[ti + ((j + 2) % 3)][0] == p[i][0]
                        and tris[ti + ((j + 2) % 3)][1] == p[i][1]
                        and tris[ti + ((j + 2) % 3)][2] == p[i][2]
                    ):
                        dists[i * 3 + j] = 1e30
                    else:
                        dists[i * 3 + j] = d

            prevCount = count
            count = wp.min(count * 3, epa_best_count)

            # Expand the polytope greedily.
            for j in range(count):
                bestIndex = int(0)
                d = float(dists[0])
                for i in range(1, 3 * prevCount):
                    if dists[i] < d:
                        d = dists[i]
                        bestIndex = i

                dists[bestIndex] = 2e30

                parentIndex = bestIndex // 3
                childIndex = bestIndex % 3
                # fill in the new triangle at the next index
                tris[tris_dim + j * 3 + 0] = tris[parentIndex * 3 + childIndex]
                tris[tris_dim + j * 3 + 1] = tris[parentIndex * 3 + ((childIndex + 1) % 3)]
                tris[tris_dim + j * 3 + 2] = p[parentIndex]

            for r in range(max_epa_best_count * 3):
                # swap triangles
                swap = tris[tris_dim + r]
                tris[tris_dim + r] = tris[r]
                tris[r] = swap

        return depth, normal

    @wp.kernel
    def epa_dense(
        npair: int,
        nenv: int,
        ngeom: int,
        nmodel: int,
        ncon: int,
        geom_pair: wp.array(dtype=int, ndim=2),
        geom_xpos: wp.array(dtype=wp.vec3),
        geom_xmat: wp.array(dtype=wp.mat33),
        geom_size: wp.array(dtype=wp.vec3),
        geom_dataid: wp.array(dtype=wp.int32),
        convex_vert: wp.array(dtype=wp.vec3),
        convex_vert_offset: wp.array(dtype=int),
        contact_simplex: wp.array(dtype=mat43),
        depth_extension: float,
        epa_best_count: int,
        # outputs
        contact_dist: wp.array(dtype=float),
        contact_normal: wp.array(dtype=wp.vec3),
    ):
        tid = wp.tid()
        if tid >= npair * nenv:
            return

        pair_id = tid % npair
        env_id = tid // npair
        model_id = env_id % nmodel

        g1 = geom_pair[pair_id, 0]
        g2 = geom_pair[pair_id, 1]
        if g1 < 0 or g2 < 0:
            return

        simplex = contact_simplex[tid]
        input_normal = contact_normal[tid]
        depth, normal = _epa(
            env_id,
            model_id,
            g1,
            g2,
            ngeom,
            geom_xpos,
            geom_xmat,
            geom_size,
            geom_dataid,
            convex_vert,
            convex_vert_offset,
            depth_extension,
            epa_best_count,
            simplex,
            input_normal,
        )

        if wp.isnan(depth) or depth < -depth_extension:
            return

        contact_normal[tid] = normal

        for i in range(ncon):
            contact_dist[tid * ncon + i] = -depth

    mat3p = wp.types.matrix(shape=(kMaxMultiPolygonCount, 3), dtype=float)

    # allocate maximum number of contact points
    mat3c = wp.types.matrix(shape=(kGjkMultiContactCount, 3), dtype=float)

    @wp.func
    def _get_multiple_contacts(
        env_id: int,
        model_id: int,
        g1: int,
        g2: int,
        ngeom: int,
        geom_xpos: wp.array(dtype=wp.vec3),
        geom_xmat: wp.array(dtype=wp.mat33),
        geom_size: wp.array(dtype=wp.vec3),
        geom_dataid: wp.array(dtype=wp.int32),
        convex_vert: wp.array(dtype=wp.vec3),
        convex_vert_offset: wp.array(dtype=int),
        depth_extension: float,
        multi_polygon_count: int,
        multi_tilt_angle: float,
        depth: float,
        normal: wp.vec3,
    ):
        # Calculates multiple contact points given the normal from EPA.
        #  1. Calculates the polygon on each shape by tiling the normal
        #     "multi_tilt_angle" degrees in the orthogonal componenet of the normal.
        #     The "multi_tilt_angle" can be changed to depend on the depth of the
        #     contact, in a future version.
        #  2. The normal is tilted "multi_polygon_count" times in the directions evenly
        #    spaced in the orthogonal component of the normal.
        #    (works well for >= 6, default is 8).
        #  3. The intersection between these two polygons is calculated in 2D space
        #    (complement to the normal). If they intersect, extreme points in both
        #    directions are found. This can be modified to the extremes in the
        #    direction of eigenvectors of the variance of points of each polygon. If
        #    they do not intersect, the closest points of both polygons are found.
        if geom_dataid:
            dataid1 = geom_dataid[g1]
            dataid2 = geom_dataid[g2]
        else:
            dataid1 = -1
            dataid2 = -1
        size1 = geom_size[model_id * ngeom + g1]
        size2 = geom_size[model_id * ngeom + g2]
        gid1 = env_id * ngeom + g1
        gid2 = env_id * ngeom + g2
        info1 = wp.static(get_info(type1))(gid1, dataid1, geom_xpos, geom_xmat, size1, convex_vert_offset)
        info2 = wp.static(get_info(type2))(gid2, dataid2, geom_xpos, geom_xmat, size2, convex_vert_offset)

        if depth < -depth_extension:
            return

        dir = orthonormal(normal)
        dir2 = wp.cross(normal, dir)

        angle = multi_tilt_angle * wp.pi / 180.0
        c = wp.cos(angle)
        s = wp.sin(angle)
        t = 1.0 - c

        v1 = mat3p()
        v2 = mat3p()

        contact_points = mat3c()

        # Obtain points on the polygon determined by the support and tilt angle,
        # in the basis of the contact frame.
        v1count = int(0)
        v2count = int(0)
        for i in range(multi_polygon_count):
            angle = 2.0 * float(i) * wp.pi / float(multi_polygon_count)
            axis = wp.cos(angle) * dir + wp.sin(angle) * dir2

            # Axis-angle rotation matrix. See
            # https://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle
            mat0 = c + axis[0] * axis[0] * t
            mat5 = c + axis[1] * axis[1] * t
            mat10 = c + axis[2] * axis[2] * t
            t1 = axis[0] * axis[1] * t
            t2 = axis[2] * s
            mat4 = t1 + t2
            mat1 = t1 - t2
            t1 = axis[0] * axis[2] * t
            t2 = axis[1] * s
            mat8 = t1 - t2
            mat2 = t1 + t2
            t1 = axis[1] * axis[2] * t
            t2 = axis[0] * s
            mat9 = t1 + t2
            mat6 = t1 - t2

            n = wp.vec3(
                mat0 * normal[0] + mat1 * normal[1] + mat2 * normal[2],
                mat4 * normal[0] + mat5 * normal[1] + mat6 * normal[2],
                mat8 * normal[0] + mat9 * normal[1] + mat10 * normal[2],
            )

            _, p = wp.static(support_functions[type1])(info1, n, convex_vert)
            v1[v1count] = wp.vec3(wp.dot(p, dir), wp.dot(p, dir2), wp.dot(p, normal))
            if i != 0 or any_different(v1[v1count], v1[v1count - 1]):
                v1count += 1

            n = -n
            _, p = wp.static(support_functions[type2])(info2, n, convex_vert)
            v2[v2count] = wp.vec3(wp.dot(p, dir), wp.dot(p, dir2), wp.dot(p, normal))
            if i != 0 or any_different(v2[v2count], v2[v2count - 1]):
                v2count += 1

        # Remove duplicate vertices on the array boundary.
        if v1count > 1 and all_same(v1[v1count - 1], v1[0]):
            v1count -= 1
        if v2count > 1 and all_same(v2[v2count - 1], v2[0]):
            v2count -= 1

        # Find an intersecting polygon between v1 and v2 in the 2D plane.
        out = mat43()
        candCount = int(0)
        if v2count > 1:
            for i in range(v1count):
                m1a = v1[i]
                is_in = bool(True)

                # Check if point m1a is inside the v2 polygon on the 2D plane.
                for j in range(v2count):
                    j2 = (j + 1) % v2count
                    # Checks that orientation of the triangle (v2[j], v2[j2], m1a) is
                    # counter-clockwise. If so, point m1a is inside the v2 polygon.
                    is_in = is_in and (
                        (v2[j2][0] - v2[j][0]) * (m1a[1] - v2[j][1]) - (v2[j2][1] - v2[j][1]) * (m1a[0] - v2[j][0])
                        >= 0.0
                    )
                    if not is_in:
                        break

                if is_in:
                    if not candCount or m1a[0] < out[0, 0]:
                        out[0] = m1a
                    if not candCount or m1a[0] > out[1, 0]:
                        out[1] = m1a
                    if not candCount or m1a[1] < out[2, 1]:
                        out[2] = m1a
                    if not candCount or m1a[1] > out[3, 1]:
                        out[3] = m1a
                    candCount += 1

        if v1count > 1:
            for i in range(v2count):
                m1a = v2[i]
                is_in = bool(True)

                for j in range(v1count):
                    j2 = (j + 1) % v1count
                    is_in = (
                        is_in
                        and (v1[j2][0] - v1[j][0]) * (m1a[1] - v1[j][1])
                        - (v1[j2][1] - v1[j][1]) * (m1a[0] - v1[j][0])
                        >= 0.0
                    )
                    if not is_in:
                        break

                if is_in:
                    if not candCount or m1a[0] < out[0, 0]:
                        out[0] = m1a
                    if not candCount or m1a[0] > out[1, 0]:
                        out[1] = m1a
                    if not candCount or m1a[1] < out[2, 1]:
                        out[2] = m1a
                    if not candCount or m1a[1] > out[3, 1]:
                        out[3] = m1a
                    candCount += 1

        if v1count > 1 and v2count > 1:
            # Check all edge pairs, and store line segment intersections if they are
            # on the edge of the boundary.
            for i in range(v1count):
                for j in range(v2count):
                    m1a = v1[i]
                    m1b = v1[(i + 1) % v1count]
                    m2a = v2[j]
                    m2b = v2[(j + 1) % v2count]

                    det = (m2a[1] - m2b[1]) * (m1b[0] - m1a[0]) - (m1a[1] - m1b[1]) * (m2b[0] - m2a[0])
                    if wp.abs(det) > 1e-12:
                        a11 = (m2a[1] - m2b[1]) / det
                        a12 = (m2b[0] - m2a[0]) / det
                        a21 = (m1a[1] - m1b[1]) / det
                        a22 = (m1b[0] - m1a[0]) / det
                        b1 = m2a[0] - m1a[0]
                        b2 = m2a[1] - m1a[1]

                        alpha = a11 * b1 + a12 * b2
                        beta = a21 * b1 + a22 * b2
                        if alpha >= 0.0 and alpha <= 1.0 and beta >= 0.0 and beta <= 1.0:
                            m0 = wp.vec3(
                                m1a[0] + alpha * (m1b[0] - m1a[0]),
                                m1a[1] + alpha * (m1b[1] - m1a[1]),
                                (m1a[2] + alpha * (m1b[2] - m1a[2]) + m2a[2] + beta * (m2b[2] - m2a[2])) * 0.5,
                            )
                            if not candCount or m0[0] < out[0, 0]:
                                out[0] = m0
                            if not candCount or m0[0] > out[1, 0]:
                                out[1] = m0
                            if not candCount or m0[1] < out[2, 1]:
                                out[2] = m0
                            if not candCount or m0[1] > out[3, 1]:
                                out[3] = m0
                            candCount += 1

        var_rx = wp.vec3(0.0)
        contact_count = int(0)
        if candCount > 0:
            # Polygon intersection was found.
            # TODO(btaba): replace the above routine with the manifold point routine
            # from MJX. Deduplicate the points properly.
            last_pt = wp.vec3(FLOAT_MAX, FLOAT_MAX, FLOAT_MAX)

            for k in range(wp.static(kGjkMultiContactCount)):
                pt = out[k, 0] * dir + out[k, 1] * dir2 + out[k, 2] * normal
                # Skip contact points that are too close.
                if wp.length(pt - last_pt) <= 1e-6:
                    continue
                contact_points[contact_count] = pt
                last_pt = pt
                contact_count += 1

        else:
            # Polygon intersection was not found. Loop through all vertex pairs and
            # calculate an approximate contact point.
            minDist = float(0.0)
            for i in range(v1count):
                for j in range(v2count):
                    # Find the closest vertex pair. Calculate a contact point var_rx as the
                    # midpoint between the closest vertex pair.
                    m1 = v1[i]
                    m2 = v2[j]
                    d = (m1[0] - m2[0]) * (m1[0] - m2[0]) + (m1[1] - m2[1]) * (m1[1] - m2[1])
                    if i != 0 and j != 0 or d < minDist:
                        minDist = d
                        var_rx = ((m1[0] + m2[0]) * dir + (m1[1] + m2[1]) * dir2 + (m1[2] + m2[2]) * normal) * 0.5

                    # Check for a closer point between a point on v2 and an edge on v1.
                    m1b = v1[(i + 1) % v1count]
                    m2b = v2[(j + 1) % v2count]
                    if v1count > 1:
                        d = (m1b[0] - m1[0]) * (m1b[0] - m1[0]) + (m1b[1] - m1[1]) * (m1b[1] - m1[1])
                        t = ((m2[1] - m1[1]) * (m1b[0] - m1[0]) - (m2[0] - m1[0]) * (m1b[1] - m1[1])) / d
                        dx = m2[0] + (m1b[1] - m1[1]) * t
                        dy = m2[1] - (m1b[0] - m1[0]) * t
                        dist = (dx - m2[0]) * (dx - m2[0]) + (dy - m2[1]) * (dy - m2[1])

                        if (
                            (dist < minDist)
                            and ((dx - m1[0]) * (m1b[0] - m1[0]) + (dy - m1[1]) * (m1b[1] - m1[1]) >= 0)
                            and ((dx - m1b[0]) * (m1[0] - m1b[0]) + (dy - m1b[1]) * (m1[1] - m1b[1]) >= 0)
                        ):
                            alpha = wp.sqrt(((dx - m1[0]) * (dx - m1[0]) + (dy - m1[1]) * (dy - m1[1])) / d)
                            minDist = dist
                            w = ((1.0 - alpha) * m1 + alpha * m1b + m2) * 0.5
                            var_rx = w[0] * dir + w[1] * dir2 + w[2] * normal

                    # Check for a closer point between a point on v1 and an edge on v2.
                    if v2count > 1:
                        d = (m2b[0] - m2[0]) * (m2b[0] - m2[0]) + (m2b[1] - m2[1]) * (m2b[1] - m2[1])
                        t = ((m1[1] - m2[1]) * (m2b[0] - m2[0]) - (m1[0] - m2[0]) * (m2b[1] - m2[1])) / d
                        dx = m1[0] + (m2b[1] - m2[1]) * t
                        dy = m1[1] - (m2b[0] - m2[0]) * t
                        dist = (dx - m1[0]) * (dx - m1[0]) + (dy - m1[1]) * (dy - m1[1])

                        if (
                            dist < minDist
                            and (dx - m2[0]) * (m2b[0] - m2[0]) + (dy - m2[1]) * (m2b[1] - m2[1]) >= 0
                            and (dx - m2b[0]) * (m2[0] - m2b[0]) + (dy - m2b[1]) * (m2[1] - m2b[1]) >= 0
                        ):
                            alpha = wp.sqrt(((dx - m2[0]) * (dx - m2[0]) + (dy - m2[1]) * (dy - m2[1])) / d)
                            minDist = dist
                            w = (m1 + (1.0 - alpha) * m2 + alpha * m2b) * 0.5
                            var_rx = w[0] * dir + w[1] * dir2 + w[2] * normal

            for k in range(wp.static(kGjkMultiContactCount)):
                contact_points[k] = var_rx

            contact_count = 1

        return contact_count, contact_points

    @wp.kernel
    def multiple_contacts_dense(
        npair: int,
        nenv: int,
        ngeom: int,
        nmodel: int,
        ncon: int,
        geom_pair: wp.array(dtype=int, ndim=2),
        geom_xpos: wp.array(dtype=wp.vec3),
        geom_xmat: wp.array(dtype=wp.mat33),
        geom_size: wp.array(dtype=wp.vec3),
        geom_dataid: wp.array(dtype=wp.int32),
        convex_vert: wp.array(dtype=wp.vec3),
        convex_vert_offset: wp.array(dtype=int),
        depth_extension: float,
        multi_polygon_count: int,
        multi_tilt_angle: float,
        contact_dist: wp.array(dtype=float),
        contact_normal: wp.array(dtype=wp.vec3),
        # outputs
        contact_pos: wp.array(dtype=wp.vec3),
    ):
        tid = wp.tid()
        if tid >= npair * nenv:
            return

        pair_id = tid % npair
        env_id = tid // npair
        model_id = env_id % nmodel

        g1 = geom_pair[pair_id, 0]
        g2 = geom_pair[pair_id, 1]
        if g1 < 0 or g2 < 0:
            return

        normal = contact_normal[tid]
        depth = -contact_dist[tid * ncon]

        count, points = _get_multiple_contacts(
            env_id,
            model_id,
            g1,
            g2,
            ngeom,
            geom_xpos,
            geom_xmat,
            geom_size,
            geom_dataid,
            convex_vert,
            convex_vert_offset,
            depth_extension,
            multi_polygon_count,
            multi_tilt_angle,
            depth,
            normal,
        )

        for i in range(ncon):
            contact_pos[tid * ncon + i] = points[i % count]

    # Runs GJK and EPA on a set of sparse geom pairs per env.
    @wp.kernel
    def gjk_epa_sparse(
        group_key: int,
        nenv: int,
        ngeom: int,
        nmodel: int,
        max_contact_points_per_env: int,
        type_pair_env_id: wp.array(dtype=int),
        type_pair_geom_id: wp.array(dtype=int),
        type_pair_count: wp.array(dtype=int),
        type_pair_offset: wp.array(dtype=int),
        geom_xpos: wp.array(dtype=wp.vec3),
        geom_xmat: wp.array(dtype=wp.mat33),
        geom_size: wp.array(dtype=wp.vec3),
        geom_dataid: wp.array(dtype=wp.int32),
        convex_vert: wp.array(dtype=wp.vec3),
        convex_vert_offset: wp.array(dtype=int),
        epa_best_count: int,
        depth_extension: float,
        multi_polygon_count: int,
        multi_tilt_angle: float,
        # outputs
        env_contact_counter: wp.array(dtype=int),
        contact_geom1: wp.array(dtype=int),
        contact_geom2: wp.array(dtype=int),
        contact_dist: wp.array(dtype=float),
        contact_pos: wp.array(dtype=wp.vec3),
        contact_normal: wp.array(dtype=wp.vec3),
    ):
        tid = wp.tid()
        npair = type_pair_count[group_key]
        if tid >= npair:
            return

        type_pair_id = type_pair_offset[group_key] * nenv + tid
        env_id = type_pair_env_id[type_pair_id]

        # Check if we generated max contacts for this env.
        # TODO(btaba): move max_contact_points_per_env culling to a point later
        # in the pipline, where we can do a sort on penetration depth per env.
        if env_contact_counter[env_id] > max_contact_points_per_env:
            return

        model_id = env_id % nmodel
        g1 = type_pair_geom_id[type_pair_id * 2]
        g2 = type_pair_geom_id[type_pair_id * 2 + 1]

        simplex, normal = _gjk(
            env_id,
            model_id,
            g1,
            g2,
            ngeom,
            geom_xpos,
            geom_xmat,
            geom_size,
            geom_dataid,
            convex_vert,
            convex_vert_offset,
        )
        # TODO(btaba): get depth from GJK, conditionally run EPA.
        depth, normal = _epa(
            env_id,
            model_id,
            g1,
            g2,
            ngeom,
            geom_xpos,
            geom_xmat,
            geom_size,
            geom_dataid,
            convex_vert,
            convex_vert_offset,
            depth_extension,
            epa_best_count,
            simplex,
            normal,
        )

        # TODO(btaba): add support for margin here.
        if depth < 0.0:
            return

        # TODO(btaba): split get_multiple_contacts into a separate kernel.
        count, points = _get_multiple_contacts(
            env_id,
            model_id,
            g1,
            g2,
            ngeom,
            geom_xpos,
            geom_xmat,
            geom_size,
            geom_dataid,
            convex_vert,
            convex_vert_offset,
            depth_extension,
            multi_polygon_count,
            multi_tilt_angle,
            depth,
            normal,
        )

        contact_count = min(count, max_contact_points_per_env)
        cid = wp.atomic_add(env_contact_counter, env_id, contact_count)
        cid = cid + env_id * max_contact_points_per_env
        for i in range(contact_count):
            contact_dist[cid + i] = -depth
            contact_geom1[cid + i] = g1
            contact_geom2[cid + i] = g2
            contact_normal[cid + i] = normal
            contact_pos[cid + i] = points[i]

    return GjkEpaPipeline(
        type1,
        type2,
        gjk_dense,
        epa_dense,
        multiple_contacts_dense,
        gjk_epa_sparse,
    )


def gjk_epa_dense(
    geom_pair: wp.array(dtype=int),
    geom_xpos: wp.array(dtype=wp.vec3),
    geom_xmat: wp.array(dtype=wp.mat33),
    geom_size: wp.array(dtype=wp.vec3),
    geom_dataid: wp.array(dtype=wp.int32),
    convex_vert: wp.array(dtype=wp.vec3),
    convex_vert_offset: wp.array(dtype=int),
    ngeom: int,
    npair: int,
    ncon: int,
    geom_type0: int,
    geom_type1: int,
    depth_extension: float,
    gjk_iteration_count: int,
    epa_iteration_count: int,
    epa_best_count: int,
    multi_polygon_count: int,
    multi_tilt_angle: float,
    # outputs
    dist: wp.array(dtype=float),
    pos: wp.array(dtype=wp.vec3),
    normal: wp.array(dtype=wp.vec3),
    simplex: wp.array(dtype=mat43),
):
    # Get the batch size of mjx.Data.
    nenv = 1
    for i in range(geom_xpos.ndim):
        nenv *= geom_xpos.shape[i]
    nenv //= ngeom
    if nenv == 0:
        raise RuntimeError("Batch size of mjx.Data calculated in LaunchKernel_GJK_EPA is 0.")

    # Get the batch size of mjx.Model.
    nmodel = 1
    for i in range(geom_size.ndim):
        nmodel *= geom_size.shape[i]
    nmodel //= ngeom
    if nmodel == 0:
        raise RuntimeError("Batch size of mjx.Model calculated in LaunchKernel_GJK_EPA is 0.")

    if nmodel > 1 and nmodel != nenv:
        raise RuntimeError(
            "Batch size of mjx.Model is greater than 1 and does not match the "
            "batch size of mjx.Data in LaunchKernel_GJK_EPA."
        )

    # if len(geom_dataid) != ngeom:
    #     raise RuntimeError("Dimensions of geom_dataid in LaunchKernel_GJK_EPA do not match (ngeom,).")

    # create kernels
    pipeline = gjk_epa_pipeline(
        geom_type0,
        geom_type1,
        gjk_iteration_count,
        epa_iteration_count,
        epa_best_count,
    )

    # gjk_epa_init
    dist.fill_(1e12)

    grid_size = npair * nenv
    with wp.ScopedTimer("gjk_dense"):
        wp.launch(
            pipeline.gjk_dense,
            dim=grid_size,
            inputs=[
                npair,
                nenv,
                ngeom,
                nmodel,
                geom_pair,
                geom_xpos,
                geom_xmat,
                geom_size,
                geom_dataid,
                convex_vert,
                convex_vert_offset,
            ],
            outputs=[
                normal,
                simplex,
            ],
            device=geom_pair.device,
        )

    # print("normal:")
    # print(normal.numpy())
    # print("simplex:")
    # print(simplex.numpy())
    # print()

    with wp.ScopedTimer("epa_dense"):
        wp.launch(
            pipeline.epa_dense,
            dim=grid_size,
            inputs=[
                npair,
                nenv,
                ngeom,
                nmodel,
                ncon,
                geom_pair,
                geom_xpos,
                geom_xmat,
                geom_size,
                geom_dataid,
                convex_vert,
                convex_vert_offset,
                simplex,
                depth_extension,
                epa_best_count,
            ],
            outputs=[
                dist,
                normal,
            ],
            device=geom_pair.device,
        )

    with wp.ScopedTimer("multiple_contacts_dense"):
        wp.launch(
            pipeline.multiple_contacts_dense,
            dim=grid_size,
            inputs=[
                npair,
                nenv,
                ngeom,
                nmodel,
                ncon,
                geom_pair,
                geom_xpos,
                geom_xmat,
                geom_size,
                geom_dataid,
                convex_vert,
                convex_vert_offset,
                depth_extension,
                multi_polygon_count,
                multi_tilt_angle,
                dist,
                normal,
            ],
            outputs=[pos],
            device=geom_pair.device,
        )


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
    m: mujoco.MjModel,
    d: mujoco.MjData,
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
    nbatch: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """GJK/EPA narrowphase routine."""
    # XXX determine ngeom from geom_size now
    ngeom = m.geom_size.shape[0]
    if ncon <= 0:
        raise ValueError(f'ncon should be positive, got "{ncon}".')
    if len(d.geom_xpos.shape) != 2:
        raise ValueError(f'd.geom_xpos should have 2d shape, got "{len(d.geom_xpos.shape)}".')
    if len(d.geom_xmat.shape) != 3:
        raise ValueError(f'd.geom_xmat should have 3d shape, got "{len(d.geom_xmat.shape)}".')
    if m.geom_dataid.shape != (m.ngeom,):
        raise ValueError(
            f"m.geom_dataid.shape should be (ngeom,) == ({m.ngeom},), got" f' "({m.geom_dataid.shape[0]},)".'
        )
    if len(geom_pair.shape) != 2:
        raise ValueError("Expecting 2D geom_pair.")
    if geom_pair.shape[1] != 2:
        raise ValueError(f'geom_pair.shape[1] should be 2, got "{geom_pair.shape[1]}".')

    npair = geom_pair.shape[0]
    n_points = ncon * npair

    # TODO(btaba): consider passing in sliced geom_xpos/xmat instead for perf.
    convex_vert, convex_vert_offset = get_convex_vert(m)

    device = wp.get_preferred_device()

    wp_geom_pair = wp.from_jax(geom_pair.astype(int), dtype=wp.int32).to(device)
    wp_geom_xpos = wp.from_jax(d.geom_xpos, dtype=wp.vec3).to(device)
    wp_geom_xmat = wp.from_jax(d.geom_xmat, dtype=wp.mat33).to(device)
    wp_geom_size = wp.from_jax(m.geom_size, dtype=wp.vec3).to(device)
    wp_geom_dataid = wp.array(m.geom_dataid, dtype=wp.int32).to(device)
    wp_convex_vert = wp.from_jax(convex_vert.reshape(-1, 3), dtype=wp.vec3).to(device)
    wp_convex_vert_offset = wp.from_jax(convex_vert_offset.astype(int), dtype=wp.int32).to(device)

    with wp.ScopedDevice(device):
        dist = wp.empty((n_points,), dtype=wp.float32)
        pos = wp.empty((n_points,), dtype=wp.vec3)
        normal = wp.empty((n_points,), dtype=wp.vec3)
        simplex = wp.empty((n_points,), dtype=mat43)

        with wp.ScopedTimer("gjk_epa_dense"):
            gjk_epa_dense(
                wp_geom_pair,
                wp_geom_xpos,
                wp_geom_xmat,
                wp_geom_size,
                wp_geom_dataid,
                wp_convex_vert,
                wp_convex_vert_offset,
                ngeom,
                npair,
                ncon,
                types[0],
                types[1],
                wp.float32(depth_extension),
                gjk_iter,
                epa_iter,
                epa_best_count,
                multi_polygon_count,
                wp.float32(multi_tilt_angle),
                dist,
                pos,
                normal,
                simplex,
            )

    return dist.numpy(), pos.numpy(), normal.numpy()


max_contact_points_map = [
    # PLANE  HFIELD SPHERE CAPSULE ELLIPSOID CYLINDER BOX  CONVEX
    [0, 0, 1, 2, 1, 3, 4, 4],  # PLANE
    [0, 0, 1, 2, 1, 3, 4, 4],  # HFIELD
    [0, 0, 1, 1, 1, 1, 1, 4],  # SPHERE
    [0, 0, 0, 1, 1, 2, 2, 2],  # CAPSULE
    [0, 0, 0, 0, 1, 1, 1, 1],  # ELLIPSOID
    [0, 0, 0, 0, 0, 3, 3, 3],  # CYLINDER
    [0, 0, 0, 0, 0, 0, 4, 4],  # BOX
    [0, 0, 0, 0, 0, 0, 0, 4],  # CONVEX
]


def _narrowphase(
    type1: int,
    type2: int,
    gjk_iteration_count: int,
    epa_iteration_count: int,
    nenv: int,
    ngeom: int,
    nmodel: int,
    n_geom_types: int,
    max_contact_points_per_env: int,
    type_pair_env_id: wp.array(dtype=int),
    type_pair_geom_id: wp.array(dtype=int),
    type_pair_count: wp.array(dtype=int),
    type_pair_offset: wp.array(dtype=int),
    geom_xpos: wp.array(dtype=wp.vec3),
    geom_xmat: wp.array(dtype=wp.mat33),
    geom_size: wp.array(dtype=wp.vec3),
    geom_dataid: wp.array(dtype=wp.int32),
    convex_vert: wp.array(dtype=wp.vec3),
    convex_vert_offset: wp.array(dtype=int),
    epa_best_count: int,
    depth_extension: float,
    multi_polygon_count: int,
    multi_tilt_angle: float,
    # outputs
    env_contact_counter: wp.array(dtype=int),
    contact_geom1: wp.array(dtype=int),
    contact_geom2: wp.array(dtype=int),
    contact_dist: wp.array(dtype=float),
    contact_pos: wp.array(dtype=wp.vec3),
    contact_normal: wp.array(dtype=wp.vec3),
):
    group_key = type1 + type2 * n_geom_types
    npair = type_pair_count[group_key]
    if npair == 0:
        return

    blockSize = 256
    grid_size = (npair + blockSize - 1) / blockSize
    ncon = max_contact_points_map[type1][type2]
    pipeline = gjk_epa_pipeline(
        type1,
        type2,
        gjk_iteration_count,
        epa_iteration_count,
    )
    wp.launch(
        pipeline.gjk_epa_sparse,
        grid_size,
        inputs=[
            group_key,
            nenv,
            ngeom,
            nmodel,
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
        ],
        outputs=[
            env_contact_counter,
            contact_geom1,
            contact_geom2,
            contact_dist,
            contact_pos,
            contact_normal,
        ],
        device=geom_xpos.device,
        block_dim=blockSize,
    )


# def narrowphase(cudaStream_t s, CollisionInput& in, CollisionOutput& out) {
#     for t2 in range(mjxGEOM_size):
#         for t1 in range(t1):

#             _narrowphase(t1, t2)(s, in, out, t1, t2)


def _collide(
    mjcf: str,
    assets: Optional[Dict[str, str]] = None,
    geoms: Tuple[int, int] = (0, 1),
    ncon: int = 4,
) -> Tuple[mujoco.MjData, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    m = mujoco.MjModel.from_xml_string(mjcf, assets or {})
    mx = mjx.put_model(m)
    d = mujoco.MjData(m)
    dx = mjx.put_data(m, d)
    kinematics_jit_fn = jax.jit(mjx.kinematics)
    dx = kinematics_jit_fn(mx, dx)

    key_types = [int(m.geom_type[g]) for g in geoms]
    mujoco.mj_step(m, d)

    dist, pos, n = gjk_epa(
        mx,
        dx,
        jp.array([geoms]),
        key_types,
        ncon=ncon,
        ngeom=mx.ngeom,
        depth_extension=1e9,
        gjk_iter=1,
        epa_iter=12,
        epa_best_count=12,
        multi_polygon_count=8,
        multi_tilt_angle=1.0,
    )

    return d, (dist, pos, n[:1])


def vmap(fn: Callable, in_axes: int | None | Sequence[Any] = 0):
    batch_dim = None

    def squeeze_array(a: jax.Array) -> jax.Array:
        # remove batch dimension
        return a.reshape(-1, *a.shape[2:])

    def unsqueeze_array(a: Union[np.ndarray, jax.Array]) -> jax.Array:
        # add batch dimension
        shape_div_batch = a.shape[0] // batch_dim
        new_shape = (batch_dim, shape_div_batch, *a.shape[1:])
        return a.reshape(*new_shape)

    def vec_fn(*args):
        nonlocal batch_dim
        squeezed_args = []
        for i_arg, arg in enumerate(args):
            if in_axes is not None:
                if isinstance(in_axes, int):
                    if i_arg != in_axes:
                        squeezed_args.append(arg)
                        continue
                if i_arg < len(in_axes) and in_axes[i_arg] is None:
                    squeezed_args.append(arg)
                    continue
            if isinstance(arg, mujoco.mjx._src.types.PyTreeNode):
                arg_cpy = copy.copy(arg)
                for key, value in arg.__dict__.items():
                    if isinstance(value, jax.Array):
                        if batch_dim is None:
                            batch_dim = value.shape[0]
                        if value.shape[0] != batch_dim:
                            raise ValueError("All arrays must have the same batch dimension")
                        arg_cpy.__dict__[key] = squeeze_array(value)
                squeezed_args.append(arg_cpy)
            elif isinstance(arg, jax.Array):
                squeezed_args.append(squeeze_array(arg))
            else:
                squeezed_args.append(arg)
        with wp.ScopedTimer(f"vmap_{fn.__name__}"):
            results = fn(*squeezed_args)
        batched_results = []
        for result in results:
            if isinstance(result, np.ndarray) or isinstance(result, jax.Array):
                batched_results.append(unsqueeze_array(result))
            else:
                batched_results.append(result)
        return batched_results

    return vec_fn


class EngineCollisionConvexTest(absltest.TestCase):
    # _BOX_PLANE = """
    #     <mujoco>
    #         <worldbody>
    #         <geom size="40 40 40" type="plane"/>
    #         <body pos="0 0 0.7" euler="45 0 0">
    #             <freejoint/>
    #             <geom size="0.5 0.5 0.5" type="box"/>
    #         </body>
    #         </worldbody>
    #     </mujoco>
    # """

    # def test_box_plane(self):
    #     """Tests box collision with a plane."""
    #     d, (dist, pos, n) = _collide(self._BOX_PLANE)

    #     np.testing.assert_array_less(dist, 0)
    #     np.testing.assert_array_almost_equal(dist[:2], d.contact.dist[:2])
    #     np.testing.assert_array_equal(n, np.array([[0.0, 0.0, 1.0]]))
    #     idx = np.lexsort((pos[:, 0], pos[:, 1]))
    #     pos = pos[idx]
    #     np.testing.assert_array_almost_equal(pos[2:4], d.contact.pos, decimal=2)

    # _FLAT_BOX_PLANE = """
    #     <mujoco>
    #         <worldbody>
    #             <geom size="40 40 40" type="plane"/>
    #             <body pos="0 0 0.45">
    #                 <freejoint/>
    #                 <geom size="0.5 0.5 0.5" type="box"/>
    #             </body>
    #         </worldbody>
    #     </mujoco>
    # """

    # def test_flat_box_plane(self):
    #     """Tests box collision with a plane."""
    #     d, (dist, pos, n) = _collide(self._FLAT_BOX_PLANE)

    #     np.testing.assert_array_less(dist, 0)
    #     np.testing.assert_array_almost_equal(dist, d.contact.dist)
    #     np.testing.assert_array_equal(n, np.array([[0.0, 0.0, 1.0]]))
    #     idx = np.lexsort((pos[:, 0], pos[:, 1]))
    #     pos = pos[idx]
    #     np.testing.assert_array_almost_equal(
    #         pos,
    #         jp.array(
    #             [
    #                 [-0.5, -0.5, -0.05000001],
    #                 [0.5, -0.5, -0.05000001],
    #                 [-0.5, 0.5, -0.05000001],
    #                 [-0.5, 0.5, -0.05000001],
    #             ]
    #         ),
    #     )

    # _BOX_BOX_EDGE = """
    #     <mujoco>
    #         <worldbody>
    #             <body pos="-1.0 -1.0 0.2">
    #                 <joint axis="1 0 0" type="free"/>
    #                 <geom size="0.2 0.2 0.2" type="box"/>
    #             </body>
    #             <body pos="-1.0 -1.2 0.55" euler="0 45 30">
    #                 <joint axis="1 0 0" type="free"/>
    #                 <geom size="0.1 0.1 0.1" type="box"/>
    #             </body>
    #         </worldbody>
    #     </mujoco>
    #   """

    # def test_box_box_edge(self):
    #     """Tests an edge contact for a box-box collision."""
    #     d, (dist, pos, n) = _collide(self._BOX_BOX_EDGE)

    #     np.testing.assert_array_less(dist, 0)
    #     np.testing.assert_array_almost_equal(dist[0], d.contact.dist)
    #     np.testing.assert_array_almost_equal(n.squeeze(), d.contact.frame[0, :3], decimal=5)
    #     idx = np.lexsort((pos[:, 0], pos[:, 1]))
    #     pos = pos[idx]
    #     np.testing.assert_array_almost_equal(pos[0], d.contact.pos[0])

    # _CONVEX_CONVEX = """
    #     <mujoco>
    #         <asset>
    #             <mesh name="poly"
    #             vertex="0.3 0 0  0 0.5 0  -0.3 0 0  0 -0.5 0  0 -1 1  0 1 1"
    #             face="0 1 5  0 5 4  0 4 3  3 4 2  2 4 5  1 2 5  0 2 1  0 3 2"/>
    #         </asset>
    #         <worldbody>
    #             <body pos="0.0 2.0 0.35" euler="0 0 90">
    #                 <freejoint/>
    #                 <geom size="0.2 0.2 0.2" type="mesh" mesh="poly"/>
    #             </body>
    #             <body pos="0.0 2.0 2.281" euler="180 0 0">
    #                 <freejoint/>
    #                 <geom size="0.2 0.2 0.2" type="mesh" mesh="poly"/>
    #             </body>
    #         </worldbody>
    #     </mujoco>
    # """

    # def test_convex_convex(self):
    #     """Tests convex-convex collisions."""
    #     d, (dist, pos, n) = _collide(self._CONVEX_CONVEX)

    #     np.testing.assert_array_less(dist, 0)
    #     np.testing.assert_array_almost_equal(dist[0], d.contact.dist)
    #     np.testing.assert_array_almost_equal(n.squeeze(), d.contact.frame[0, :3], decimal=5)
    #     idx = np.lexsort((pos[:, 0], pos[:, 1]))
    #     pos = pos[idx]
    #     np.testing.assert_array_almost_equal(pos[0], d.contact.pos[0])

    _SPHERE_SPHERE = """
        <mujoco>
            <worldbody>
                <body>
                    <joint type="free"/>
                    <geom pos="0 0 0" size="0.2" type="sphere"/>
                </body>
                <body >
                    <joint type="free"/>
                    <geom pos="0 0.3 0" size="0.11" type="sphere"/>
                </body>
            </worldbody>
        </mujoco>
    """

    def test_call_batched_model_and_data(self):
        m = mujoco.MjModel.from_xml_string(self._SPHERE_SPHERE)
        batch_size = 8

        @jax.vmap
        def make_model_and_data(val):
            dx = mjx.make_data(m)
            mx = mjx.put_model(m)
            size = mx.geom_size
            mx = mx.replace(geom_size=size.at[0, :].set(val * size[0, :]))
            return mx, dx

        # vary the size of body 0.
        mx, dx = make_model_and_data((jp.arange(batch_size) + 1) / batch_size)
        # assert that sizes are scaled appropriately
        self.assertTrue(float(mx.geom_size[0][0, 0]) < float(mx.geom_size[-1][0, 0]))

        kinematics_jit_fn = jax.jit(jax.vmap(mjx.kinematics))
        dx = kinematics_jit_fn(mx, dx)
        key_types = (m.geom_type[0], m.geom_type[1])
        # XXX geom_pair here has to be for the correct IDs of the geoms
        # geom_pair = jp.array(np.tile(np.array([[0, 1]]), (batch_size, 1, 1)))
        geom_pair = jp.array([[[i * 2, i * 2 + 1]] for i in range(batch_size)])

        vec_gjk_epa = vmap(
            gjk_epa,
            in_axes=(
                0,
                0,
                0,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            ),
        )
        dist, pos, n = vec_gjk_epa(mx, dx, geom_pair, key_types, 1, mx.ngeom, 1e9, 12, 12, 12, 8, 1.0, batch_size)

        self.assertTupleEqual(dist.shape, (batch_size, 1))
        self.assertTupleEqual(pos.shape, (batch_size, 1, 3))
        self.assertTupleEqual(n.shape, (batch_size, 1, 3))
        # geom0 is not colliding in env0 since the size of geom0 is small
        self.assertGreater(dist[0, 0], 0.0)  # geom (0, 1)
        # the last env should have a collision since geom0 is scaled to 1x the
        # original size
        self.assertLess(dist[-1, 0], 0.0)  # geom (0, 1)


def profile_gjk_epa(batch_size):
    SPHERE_SPHERE = """
        <mujoco>
            <worldbody>
                <body>
                    <joint type="free"/>
                    <geom pos="0 0 0" size="0.2" type="sphere"/>
                </body>
                <body >
                    <joint type="free"/>
                    <geom pos="0 0.3 0" size="0.11" type="sphere"/>
                </body>
            </worldbody>
        </mujoco>
    """
    m = mujoco.MjModel.from_xml_string(SPHERE_SPHERE)

    @jax.vmap
    def make_model_and_data(val):
        dx = mjx.make_data(m)
        mx = mjx.put_model(m)
        size = mx.geom_size
        mx = mx.replace(geom_size=size.at[0, :].set(val * size[0, :]))
        return mx, dx

    # vary the size of body 0.
    mx, dx = make_model_and_data((jp.arange(batch_size) + 1) / batch_size)

    kinematics_jit_fn = jax.jit(jax.vmap(mjx.kinematics))
    dx = kinematics_jit_fn(mx, dx)
    key_types = (m.geom_type[0], m.geom_type[1])
    # XXX geom_pair here has to be for the correct IDs of the geoms
    geom_pair = jp.array(np.tile(np.array([[0, 1]]), (batch_size, 1, 1)))
    # geom_pair = jp.array([[[i * 2, i * 2 + 1]] for i in range(batch_size)])

    vec_gjk_epa = vmap(
        gjk_epa,
        in_axes=(
            0,
            0,
            0,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        ),
    )
    with warp.ScopedTimer(f"warp_gjk_epa_{batch_size}"):
        dist, pos, n = vec_gjk_epa(mx, dx, geom_pair, key_types, 1, mx.ngeom, 1e9, 12, 12, 12, 8, 1.0, batch_size)
        wp.synchronize()


if __name__ == "__main__":
    wp.init()
    assert wp.is_cuda_available(), "CUDA is not available."

    # absltest.main()
    # test = EngineCollisionConvexTest()
    # test.test_call_batched_model_and_data()

    # profile_gjk_epa(8)
    # profile_gjk_epa(3)
    # profile_gjk_epa(7)
    # profile_gjk_epa(5)
    # profile_gjk_epa(1)
    # profile_gjk_epa(100)
    # profile_gjk_epa(10000)
    profile_gjk_epa(100000)
    # profile_gjk_epa(1000000)
    # profile_gjk_epa(1000000)
    # profile_gjk_epa(10000000)
