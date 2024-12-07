from typing import Any, Dict, Optional, Tuple

import jax
import mujoco
import numpy as np
from absl.testing import absltest
from jax import numpy as jp
from mujoco import mjx
from mujoco.mjx._src.types import Data, Model

import warp as wp
import warp.sim

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

maxContactPointsMap = [
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
        gid: wp.uint32,
        dataid: int,
        geom_xpos: wp.array(dtype=wp.vec3),
        geom_xmat: wp.array(dtype=wp.mat33),
        size: wp.vec3,
        convex_vert_offset: wp.array(dtype=wp.uint32),
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
        else:
            wp.static(RuntimeError("Unsupported type", t))

    return _get_info


@wp.func
def gjk_support_plane(
    info: GeomType_PLANE,
    dir: wp.vec3,
    convex_vert: wp.array(dtype=wp.vec3),
):
    local_dir = info.rot @ dir
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
    local_dir = info.rot @ dir
    res = wp.cw_mul(sign(local_dir), info.size)
    support_pt = info.rot @ res + info.pos
    return wp.dot(support_pt, dir), support_pt


support_functions = {
    mjxGEOM_PLANE: gjk_support_plane,
    mjxGEOM_SPHERE: gjk_support_sphere,
    mjxGEOM_BOX: gjk_support_box,
}


def gjk_support(type1, type2):
    @wp.func
    def _gjk_support(
        info1: Any,
        info2: Any,
        dir: wp.vec3,
        convex_vert: wp.array(dtype=wp.vec3),
    ):
        # Returns the distance between support points on two geoms. Negative distance
        # means objects are not intersecting along direction `dir`. Positive distance
        # means objects are intersecting along the given direction `dir`.

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
def orthonormal(normal: wp.vec3, dir: wp.vec3):
    if wp.abs(normal[0]) < wp.abs(normal[1]) and wp.abs(normal[0]) < wp.abs(normal[2]):
        dir = wp.vec3(1.0 - normal[0] * normal[0], -normal[0] * normal.y, -normal[0] * normal[2])
    elif wp.abs(normal[1]) < wp.abs(normal[2]):
        dir = wp.vec3(-normal[1] * normal[0], 1.0 - normal[1] * normal[1], -normal[1] * normal[2])
    else:
        dir = wp.vec3(-normal[2] * normal[0], -normal[2] * normal[1], 1.0 - normal[2] * normal[2])
    norm, _ = gjk_normalize(dir)
    return norm


@wp.func
def where(condition: bool, ret_true: Any, ret_false: Any):
    if condition:
        return ret_true
    return ret_false


mat43 = wp.types.matrix(shape=(4, 3), dtype=float)


# Calculates whether two objects intersect.
# Returns simplex and normal.
def _gjk(type1, type2):
    @wp.func
    def __gjk(
        env_id: wp.uint32,
        model_id: wp.uint32,
        g1: wp.uint32,
        g2: wp.uint32,
        ngeom: wp.uint32,
        geom_xpos: wp.array(dtype=wp.vec3),
        geom_xmat: wp.array(dtype=wp.mat33),
        geom_size: wp.array(dtype=wp.vec3),
        geom_dataid: wp.array(dtype=wp.int32),
        convex_vert: wp.array(dtype=wp.vec3),
        convex_vert_offset: wp.array(dtype=wp.uint32),
        gjk_iteration_count: int,
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

        sd = wp.normalize(simplex0 - simplex1)
        dir = orthonormal(sd, dir)

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
        for i in range(gjk_iteration_count):
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
            simplex[i] = simplex_i
            if dist < depth:
                depth = dist
                normal = plane[index]

            # Preserve winding order of the simplex faces.
            index1 = (index + 1) & 3
            index2 = (index + 2) & 3
            swap = simplex[index1]
            simplex[index1] = simplex[index2]
            simplex[index2] = swap
            if dist < 0:
                break  # Objects are likely non-intersecting.

        return simplex, normal

    return __gjk


def gjk_dense(type1, type2):
    @wp.kernel
    def _gjk_dense(
        npair: wp.uint32,
        nenv: wp.uint32,
        ngeom: wp.uint32,
        nmodel: wp.uint32,
        geom_pair: wp.array(dtype=wp.uint32, ndim=2),
        geom_xpos: wp.array(dtype=wp.vec3),
        geom_xmat: wp.array(dtype=wp.mat33),
        geom_size: wp.array(dtype=wp.vec3),
        geom_dataid: wp.array(dtype=wp.int32),
        convex_vert: wp.array(dtype=wp.vec3),
        convex_vert_offset: wp.array(dtype=wp.uint32),
        gjk_iteration_count: int,
        contact_normal: wp.array(dtype=wp.vec3),
        d_simplex: wp.array(dtype=mat43),
    ):
        tid = wp.uint32(wp.tid())
        if tid >= npair * nenv:
            return

        pair_id = tid % npair
        env_id = tid / npair
        model_id = env_id % nmodel

        g1, g2 = geom_pair[pair_id, 0], geom_pair[pair_id, 1]
        if g1 < 0 or g2 < 0:
            return

        simplex, normal = wp.static(_gjk(type1, type2))(
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
            gjk_iteration_count,
        )
        d_simplex[tid] = simplex

        # Align to 16 byte boundary.
        # d_simplex[tid * 3 + 0] = simplex_f4[0]
        # d_simplex[tid * 3 + 1] = simplex_f4[1]
        # d_simplex[tid * 3 + 2] = simplex_f4[2]

        if contact_normal:
            contact_normal[tid] = normal

    return _gjk_dense


def gjk_epa_dense(
    geom_pair: wp.array(dtype=wp.uint32),
    geom_xpos: wp.array(dtype=wp.vec3),
    geom_xmat: wp.array(dtype=wp.mat33),
    geom_size: wp.array(dtype=wp.vec3),
    geom_dataid: wp.array(dtype=wp.int32),
    convex_vert: wp.array(dtype=wp.vec3),
    convex_vert_offset: wp.array(dtype=wp.uint32),
    ngeom: wp.uint32,
    npair: wp.uint32,
    ncon: wp.uint32,
    geom_type0: int,
    geom_type1: int,
    depth_extension: wp.float32,
    gjk_iteration_count: wp.uint32,
    epa_iteration_count: wp.uint32,
    epa_best_count: wp.uint32,
    multi_polygon_count: wp.uint32,
    multi_tilt_angle: wp.float32,
    # outputs
    dist: wp.array(dtype=wp.float32),
    pos: wp.array(dtype=wp.vec3),
    normal: wp.array(dtype=wp.vec3),
    simplex: wp.array(dtype=mat43),
):
    # Get the batch size of mjx.Data.
    nenv = 1
    for i in range(geom_xpos.ndim):
        nenv *= geom_xpos.shape[i]

    nenv //= int(ngeom)
    if nenv == 0:
        raise RuntimeError("Batch size of mjx.Data calculated in LaunchKernel_GJK_EPA " "is 0.")

    # Get the batch size of mjx.Model.
    nmodel = 1
    for i in range(geom_size.ndim):
        nmodel *= geom_size.shape[i]

    nmodel //= int(ngeom)
    if nmodel == 0:
        raise RuntimeError("Batch size of mjx.Model calculated in LaunchKernel_GJK_EPA " "is 0.")

    if nmodel > 1 and nmodel != nenv:
        raise RuntimeError(
            "Batch size of mjx.Model is greater than 1 and does not match the "
            "batch size of mjx.Data in LaunchKernel_GJK_EPA."
        )

    if len(geom_dataid) != int(ngeom):
        raise RuntimeError("Dimensions of geom_dataid in LaunchKernel_GJK_EPA " "do not match (ngeom,).")

    # gjk_epa_init
    dist.fill_(1e12)

    grid_size = npair * wp.uint32(nenv)
    wp.launch(
        gjk_dense(geom_type0, geom_type1),
        dim=grid_size,
        inputs=[
            npair,
            wp.uint32(nenv),
            ngeom,
            wp.uint32(nmodel),
            geom_pair,
            geom_xpos,
            geom_xmat,
            geom_size,
            geom_dataid,
            convex_vert,
            convex_vert_offset,
            gjk_iteration_count,
        ],
        outputs=[
            normal,
            simplex,
        ],
        device=geom_pair.device,
    )

    print(normal.numpy())
    print(simplex.numpy())
    print()


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
    geom_pair: wp.array,
    types: Tuple[int, int],
    ncon: int,
    ngeom: int,
    depth_extension: float,
    gjk_iter: int,
    epa_iter: int,
    epa_best_count: int,
    multi_polygon_count: int,
    multi_tilt_angle: float,
) -> Tuple[wp.array, wp.array, wp.array]:
    """GJK/EPA narrowphase routine."""
    if ngeom <= 0:
        raise ValueError(f'ngeom should be positive, got "{ngeom}".')
    if ncon <= 0:
        raise ValueError(f'ncon should be positive, got "{ncon}".')
    if len(d.geom_xpos.shape) != 2:
        raise ValueError(f'd.geom_xpos should have 2d shape, got "{len(d.geom_xpos.shape)}".')
    if len(d.geom_xmat.shape) != 3:
        raise ValueError(f'd.geom_xmat should have 3d shape, got "{len(d.geom_xmat.shape)}".')
    if m.geom_size.shape[0] != ngeom:
        raise ValueError(f"m.geom_size.shape[0] should be ngeom ({ngeom}), " f'got "{m.geom_size.shape[0]}".')
    if m.geom_dataid.shape != (ngeom,):
        raise ValueError(
            f"m.geom_dataid.shape should be (ngeom,) == ({ngeom},), got" f' "({m.geom_dataid.shape[0]},)".'
        )
    if len(geom_pair.shape) != 2:
        raise ValueError("Expecting 2D geom_pair.")
    if geom_pair.shape[1] != 2:
        raise ValueError(f'geom_pair.shape[1] should be 2, got "{geom_pair.shape[1]}".')

    npair = geom_pair.shape[0]
    n_points = ncon * npair

    # TODO(btaba): consider passing in sliced geom_xpos/xmat instead for perf.
    convex_vert, convex_vert_offset = get_convex_vert(m)

    wp_geom_pair = wp.from_jax(geom_pair, dtype=wp.uint32)
    wp_geom_xpos = wp.from_jax(d.geom_xpos, dtype=wp.vec3)
    wp_geom_xmat = wp.from_jax(d.geom_xmat, dtype=wp.mat33)
    wp_geom_size = wp.from_jax(m.geom_size, dtype=wp.vec3)
    wp_geom_dataid = wp.array(m.geom_dataid, dtype=wp.int32)
    wp_convex_vert = wp.from_jax(convex_vert.reshape(-1, 3), dtype=wp.vec3)
    wp_convex_vert_offset = wp.from_jax(convex_vert_offset, dtype=wp.uint32)

    dist = wp.empty((n_points,), dtype=wp.float32)
    pos = wp.empty((n_points,), dtype=wp.vec3)
    normal = wp.empty((n_points,), dtype=wp.vec3)
    simplex = wp.empty((n_points,), dtype=mat43)

    gjk_epa_dense(
        wp_geom_pair,
        wp_geom_xpos,
        wp_geom_xmat,
        wp_geom_size,
        wp_geom_dataid,
        wp_convex_vert,
        wp_convex_vert_offset,
        wp.uint32(ngeom),
        wp.uint32(npair),
        wp.uint32(ncon),
        types[0],
        types[1],
        wp.float32(depth_extension),
        wp.uint32(gjk_iter),
        wp.uint32(epa_iter),
        wp.uint32(epa_best_count),
        wp.uint32(multi_polygon_count),
        wp.float32(multi_tilt_angle),
        dist,
        pos,
        normal,
        simplex,
    )

    return dist.numpy(), pos.numpy(), normal.numpy()[:1]


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
        gjk_iter=12,
        epa_iter=12,
        epa_best_count=12,
        multi_polygon_count=8,
        multi_tilt_angle=1.0,
    )

    return d, (dist, pos, n)


class EngineCollisionConvexTest(absltest.TestCase):
    _BOX_PLANE = """
        <mujoco>
            <worldbody>
            <geom size="40 40 40" type="plane"/>
            <body pos="0 0 0.7" euler="45 0 0">
                <freejoint/>
                <geom size="0.5 0.5 0.5" type="box"/>
            </body>
            </worldbody>
        </mujoco>
    """

    def test_box_plane(self):
        """Tests box collision with a plane."""
        d, (dist, pos, n) = _collide(self._BOX_PLANE)

        np.testing.assert_array_less(dist, 0)
        np.testing.assert_array_almost_equal(dist[:2], d.contact.dist[:2])
        np.testing.assert_array_equal(n, np.array([[0.0, 0.0, 1.0]]))
        idx = np.lexsort((pos[:, 0], pos[:, 1]))
        pos = pos[idx]
        np.testing.assert_array_almost_equal(pos[2:4], d.contact.pos, decimal=2)


#     _FLAT_BOX_PLANE = """
#     <mujoco>
#       <worldbody>
#         <geom size="40 40 40" type="plane"/>
#         <body pos="0 0 0.45">
#           <freejoint/>
#           <geom size="0.5 0.5 0.5" type="box"/>
#         </body>
#       </worldbody>
#     </mujoco>
#   """

#     def test_flat_box_plane(self):
#         """Tests box collision with a plane."""
#         d, (dist, pos, n) = _collide(self._FLAT_BOX_PLANE)

#         np.testing.assert_array_less(dist, 0)
#         np.testing.assert_array_almost_equal(dist, d.contact.dist)
#         np.testing.assert_array_equal(n, np.array([[0.0, 0.0, 1.0]]))
#         idx = np.lexsort((pos[:, 0], pos[:, 1]))
#         pos = pos[idx]
#         np.testing.assert_array_almost_equal(
#             pos,
#             jp.array(
#                 [
#                     [-0.5, -0.5, -0.05000001],
#                     [0.5, -0.5, -0.05000001],
#                     [-0.5, 0.5, -0.05000001],
#                     [-0.5, 0.5, -0.05000001],
#                 ]
#             ),
#         )

#     _BOX_BOX_EDGE = """
#     <mujoco>
#       <worldbody>
#         <body pos="-1.0 -1.0 0.2">
#           <joint axis="1 0 0" type="free"/>
#           <geom size="0.2 0.2 0.2" type="box"/>
#         </body>
#         <body pos="-1.0 -1.2 0.55" euler="0 45 30">
#           <joint axis="1 0 0" type="free"/>
#           <geom size="0.1 0.1 0.1" type="box"/>
#         </body>
#       </worldbody>
#     </mujoco>
#   """

#     def test_box_box_edge(self):
#         """Tests an edge contact for a box-box collision."""
#         d, (dist, pos, n) = _collide(self._BOX_BOX_EDGE)

#         np.testing.assert_array_less(dist, 0)
#         np.testing.assert_array_almost_equal(dist[0], d.contact.dist)
#         np.testing.assert_array_almost_equal(n.squeeze(), d.contact.frame[0, :3], decimal=5)
#         idx = np.lexsort((pos[:, 0], pos[:, 1]))
#         pos = pos[idx]
#         np.testing.assert_array_almost_equal(pos[0], d.contact.pos[0])

#     _CONVEX_CONVEX = """
#     <mujoco>
#       <asset>
#         <mesh name="poly"
#          vertex="0.3 0 0  0 0.5 0  -0.3 0 0  0 -0.5 0  0 -1 1  0 1 1"
#          face="0 1 5  0 5 4  0 4 3  3 4 2  2 4 5  1 2 5  0 2 1  0 3 2"/>
#       </asset>
#       <worldbody>
#         <body pos="0.0 2.0 0.35" euler="0 0 90">
#           <freejoint/>
#           <geom size="0.2 0.2 0.2" type="mesh" mesh="poly"/>
#         </body>
#         <body pos="0.0 2.0 2.281" euler="180 0 0">
#           <freejoint/>
#           <geom size="0.2 0.2 0.2" type="mesh" mesh="poly"/>
#         </body>
#       </worldbody>
#     </mujoco>
#   """

#     def test_convex_convex(self):
#         """Tests convex-convex collisions."""
#         d, (dist, pos, n) = _collide(self._CONVEX_CONVEX)

#         np.testing.assert_array_less(dist, 0)
#         np.testing.assert_array_almost_equal(dist[0], d.contact.dist)
#         np.testing.assert_array_almost_equal(n.squeeze(), d.contact.frame[0, :3], decimal=5)
#         idx = np.lexsort((pos[:, 0], pos[:, 1]))
#         pos = pos[idx]
#         np.testing.assert_array_almost_equal(pos[0], d.contact.pos[0])

#     _CONVEX_CONVEX_MULTI = """
#     <mujoco>
#       <asset>
#         <mesh name="poly"
#          vertex="0.3 0 0  0 0.5 0  -0.3 0 0  0 -0.5 0  0 -1 1  0 1 1"
#          face="0 1 5  0 5 4  0 4 3  3 4 2  2 4 5  1 2 5  0 2 1  0 3 2"/>
#       </asset>
#       <worldbody>
#         <body pos="0.0 2.0 0.35" euler="0 0 90">
#           <freejoint/>
#           <geom size="0.2 0.2 0.2" type="mesh" mesh="poly"/>
#         </body>
#         <body pos="0.0 2.0 2.281" euler="180 0 0">
#           <freejoint/>
#           <geom size="0.2 0.2 0.2" type="mesh" mesh="poly"/>
#         </body>
#         <body pos="0.0 2.0 2.281" euler="180 0 0">
#           <freejoint/>
#           <geom size="0.2 0.2 0.2" type="mesh" mesh="poly"/>
#         </body>
#       </worldbody>
#     </mujoco>
#   """


if __name__ == "__main__":
    wp.init()
    assert wp.is_cuda_available(), "CUDA is not available."

    # absltest.main()

    test = EngineCollisionConvexTest()
    test.test_box_plane()
