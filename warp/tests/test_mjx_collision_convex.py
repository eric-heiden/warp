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
"""Tests engine_collision_convex CUDA kernels.

Note: these tests are not yet comprehensive. Ideally we would run a pipeline
to test against a ground truth (i.e. SAT).
"""

import ctypes
from typing import Dict, Optional, Tuple

import jax
import mujoco
import numpy as np
from absl.testing import absltest
from jax import numpy as jp
from mujoco import mjx
from mujoco.mjx._src.types import Data, Model

import warp as wp

launch_epa_gjk = None


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

    wp_geom_pair = wp.from_jax(geom_pair)
    wp_geom_xpos = wp.from_jax(d.geom_xpos)
    wp_geom_xmat = wp.from_jax(d.geom_xmat)
    wp_geom_size = wp.from_jax(m.geom_size)
    wp_geom_dataid = wp.array(m.geom_dataid, dtype=wp.int32)
    wp_convex_vert = wp.from_jax(convex_vert)
    wp_convex_vert_offset = wp.from_jax(convex_vert_offset, dtype=wp.uint32)

    dist = wp.empty((n_points,), dtype=wp.float32)
    pos = wp.empty((n_points, 3), dtype=wp.float32)
    normal = wp.empty((n_points, 3), dtype=wp.float32)
    simplex = wp.empty((n_points, 4, 3), dtype=wp.float32)

    launch_epa_gjk(
        wp_geom_pair.__ctype__(),
        wp_geom_xpos.__ctype__(),
        wp_geom_xmat.__ctype__(),
        wp_geom_size.__ctype__(),
        wp_geom_dataid.__ctype__(),
        wp_convex_vert.__ctype__(),
        wp_convex_vert_offset.__ctype__(),
        wp.uint32(ngeom),
        wp.uint32(npair),
        wp.uint32(ncon),
        wp.uint32(types[0]),
        wp.uint32(types[1]),
        wp.float32(depth_extension),
        wp.uint32(gjk_iter),
        wp.uint32(epa_iter),
        wp.uint32(epa_best_count),
        wp.uint32(multi_polygon_count),
        wp.float32(multi_tilt_angle),
        dist.__ctype__(),
        pos.__ctype__(),
        normal.__ctype__(),
        simplex.__ctype__(),
    )

    return dist.numpy(), pos.numpy(), normal.numpy()


def _collide(
    mjcf: str,
    assets: Optional[Dict[str, str]] = None,
    geoms: Tuple[int, int] = (0, 1),
    ncon: int = 4,
) -> Tuple[mujoco.MjData, Tuple[wp.array, wp.array, wp.array]]:
    m = mujoco.MjModel.from_xml_string(mjcf, assets or {})
    d = mujoco.MjData(m)

    key_types = (m.geom_type[geoms[0]], m.geom_type[geoms[1]])
    mujoco.mj_step(m, d)

    kinematics_jit_fn = jax.jit(mjx.kinematics)
    mx = mjx.put_model(m)
    dx = mjx.put_data(m, d)
    dx = kinematics_jit_fn(mx, dx)
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

    _FLAT_BOX_PLANE = """
    <mujoco>
      <worldbody>
        <geom size="40 40 40" type="plane"/>
        <body pos="0 0 0.45">
          <freejoint/>
          <geom size="0.5 0.5 0.5" type="box"/>
        </body>
      </worldbody>
    </mujoco>
  """

    def test_flat_box_plane(self):
        """Tests box collision with a plane."""
        d, (dist, pos, n) = _collide(self._FLAT_BOX_PLANE)

        np.testing.assert_array_less(dist, 0)
        np.testing.assert_array_almost_equal(dist, d.contact.dist)
        np.testing.assert_array_equal(n, np.array([[0.0, 0.0, 1.0]]))
        idx = np.lexsort((pos[:, 0], pos[:, 1]))
        pos = pos[idx]
        np.testing.assert_array_almost_equal(
            pos,
            jp.array(
                [
                    [-0.5, -0.5, -0.05000001],
                    [0.5, -0.5, -0.05000001],
                    [-0.5, 0.5, -0.05000001],
                    [-0.5, 0.5, -0.05000001],
                ]
            ),
        )

    _BOX_BOX_EDGE = """
    <mujoco>
      <worldbody>
        <body pos="-1.0 -1.0 0.2">
          <joint axis="1 0 0" type="free"/>
          <geom size="0.2 0.2 0.2" type="box"/>
        </body>
        <body pos="-1.0 -1.2 0.55" euler="0 45 30">
          <joint axis="1 0 0" type="free"/>
          <geom size="0.1 0.1 0.1" type="box"/>
        </body>
      </worldbody>
    </mujoco>
  """

    def test_box_box_edge(self):
        """Tests an edge contact for a box-box collision."""
        d, (dist, pos, n) = _collide(self._BOX_BOX_EDGE)

        np.testing.assert_array_less(dist, 0)
        np.testing.assert_array_almost_equal(dist[0], d.contact.dist)
        np.testing.assert_array_almost_equal(n.squeeze(), d.contact.frame[0, :3], decimal=5)
        idx = np.lexsort((pos[:, 0], pos[:, 1]))
        pos = pos[idx]
        np.testing.assert_array_almost_equal(pos[0], d.contact.pos[0])

    _CONVEX_CONVEX = """
    <mujoco>
      <asset>
        <mesh name="poly"
         vertex="0.3 0 0  0 0.5 0  -0.3 0 0  0 -0.5 0  0 -1 1  0 1 1"
         face="0 1 5  0 5 4  0 4 3  3 4 2  2 4 5  1 2 5  0 2 1  0 3 2"/>
      </asset>
      <worldbody>
        <body pos="0.0 2.0 0.35" euler="0 0 90">
          <freejoint/>
          <geom size="0.2 0.2 0.2" type="mesh" mesh="poly"/>
        </body>
        <body pos="0.0 2.0 2.281" euler="180 0 0">
          <freejoint/>
          <geom size="0.2 0.2 0.2" type="mesh" mesh="poly"/>
        </body>
      </worldbody>
    </mujoco>
  """

    def test_convex_convex(self):
        """Tests convex-convex collisions."""
        d, (dist, pos, n) = _collide(self._CONVEX_CONVEX)

        np.testing.assert_array_less(dist, 0)
        np.testing.assert_array_almost_equal(dist[0], d.contact.dist)
        np.testing.assert_array_almost_equal(n.squeeze(), d.contact.frame[0, :3], decimal=5)
        idx = np.lexsort((pos[:, 0], pos[:, 1]))
        pos = pos[idx]
        np.testing.assert_array_almost_equal(pos[0], d.contact.pos[0])

    _CONVEX_CONVEX_MULTI = """
    <mujoco>
      <asset>
        <mesh name="poly"
         vertex="0.3 0 0  0 0.5 0  -0.3 0 0  0 -0.5 0  0 -1 1  0 1 1"
         face="0 1 5  0 5 4  0 4 3  3 4 2  2 4 5  1 2 5  0 2 1  0 3 2"/>
      </asset>
      <worldbody>
        <body pos="0.0 2.0 0.35" euler="0 0 90">
          <freejoint/>
          <geom size="0.2 0.2 0.2" type="mesh" mesh="poly"/>
        </body>
        <body pos="0.0 2.0 2.281" euler="180 0 0">
          <freejoint/>
          <geom size="0.2 0.2 0.2" type="mesh" mesh="poly"/>
        </body>
        <body pos="0.0 2.0 2.281" euler="180 0 0">
          <freejoint/>
          <geom size="0.2 0.2 0.2" type="mesh" mesh="poly"/>
        </body>
      </worldbody>
    </mujoco>
  """

    def _test_call_batched_data(self):
        m = mujoco.MjModel.from_xml_string(self._CONVEX_CONVEX_MULTI)
        mx = mjx.put_model(m)
        d = mujoco.MjData(m)
        batch_size = 32

        @jax.vmap
        def make_data(val):
            qpos = jp.array(d.qpos)
            # move the first body up or down
            z = (val - batch_size / 2) / batch_size
            dx = mjx.make_data(m).replace(qpos=qpos.at[2].set(z))
            return dx

        # move body 1 incrementally higher in its z-position
        dx = make_data(jp.arange(batch_size))

        kinematics_jit_fn = jax.jit(jax.vmap(mjx.kinematics, in_axes=(None, 0)))
        dx = kinematics_jit_fn(mx, dx)
        key_types = (m.geom_type[0], m.geom_type[1])
        geom_pair = jp.array(np.tile(np.array([[0, 1], [0, 2], [1, 2]]), (batch_size, 1, 1)))

        dist, pos, n = gjk_epa(
            mx,
            dx,
            geom_pair,
            key_types,
            ncon=4,
            ngeom=mx.ngeom,
            depth_extension=1e9,
            gjk_iter=12,
            epa_iter=12,
            epa_best_count=12,
            multi_polygon_count=8,
            multi_tilt_angle=1.0,
        )

        # dist, pos, n = jax.jit(
        #     jax.vmap(
        #         engine_collision_convex.gjk_epa,
        #         in_axes=(
        #             None,
        #             0,
        #             0,
        #             None,
        #             None,
        #             None,
        #             None,
        #             None,
        #             None,
        #             None,
        #             None,
        #             None,
        #         ),
        #     ),
        #     static_argnums=(
        #         3,
        #         4,
        #         5,
        #         6,
        #         7,
        #         8,
        #         9,
        #         10,
        #         11,
        #     ),
        # )(mx, dx, geom_pair, key_types, 4, mx.ngeom, 1e9, 12, 12, 12, 8, 1.0)

        self.assertTupleEqual(dist.shape, (batch_size, 12))
        self.assertTupleEqual(pos.shape, (batch_size, 12, 3))
        self.assertTupleEqual(n.shape, (batch_size, 3, 3))
        # geom0 is not colliding in env0, ~50cm below the initial position
        # geom0 is thus not colliding with geom 1 or 2 in env0
        self.assertGreater(dist[0, 0], 0.0)  # geom (0, 1)
        self.assertGreater(dist[0, 4], 0.0)  # geom (0, 2)
        # the last env has geom0 ~1m higher and it should collide with geom1 and 2
        self.assertLess(dist[-1, 0], 0.0)  # geom (0, 1)
        self.assertLess(dist[-1, 4], 0.0)  # geom (0, 2)
        np.testing.assert_array_equal(-np.sort(-dist[:, 0], stable=True), dist[:, 0])
        np.testing.assert_array_equal(-np.sort(-dist[:, 4], stable=True), dist[:, 4])

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

    def _test_call_batched_model_and_data(self):
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
        self.assertTrue(float(mx.geom_size[0][0, 0]), float(mx.geom_size[-1][0, 0]))

        kinematics_jit_fn = jax.jit(jax.vmap(mjx.kinematics))
        dx = kinematics_jit_fn(mx, dx)
        key_types = (m.geom_type[0], m.geom_type[1])
        geom_pair = jp.array(np.tile(np.array([[0, 1]]), (batch_size, 1, 1)))

        dist, pos, n = gjk_epa(
            mx,
            dx,
            geom_pair,
            key_types,
            ncon=1,
            ngeom=mx.ngeom,
            depth_extension=1e9,
            gjk_iter=12,
            epa_iter=12,
            epa_best_count=12,
            multi_polygon_count=8,
            multi_tilt_angle=1.0,
        )

        # dist, pos, n = jax.jit(
        #     jax.vmap(
        #         engine_collision_convex.gjk_epa,
        #         in_axes=(
        #             0,
        #             0,
        #             0,
        #             None,
        #             None,
        #             None,
        #             None,
        #             None,
        #             None,
        #             None,
        #             None,
        #             None,
        #         ),
        #     ),
        #     static_argnums=(
        #         3,
        #         4,
        #         5,
        #         6,
        #         7,
        #         8,
        #         9,
        #         10,
        #         11,
        #     ),
        # )(mx, dx, geom_pair, key_types, 1, mx.ngeom, 1e9, 12, 12, 12, 8, 1.0)

        self.assertTupleEqual(dist.shape, (batch_size, 1))
        self.assertTupleEqual(pos.shape, (batch_size, 1, 3))
        self.assertTupleEqual(n.shape, (batch_size, 1, 3))
        # geom0 is not colliding in env0 since the size of geom0 is small
        self.assertGreater(dist[0, 0], 0.0)  # geom (0, 1)
        # the last env should have a collision since geom0 is scaled to 1x the
        # original size
        self.assertLess(dist[-1, 0], 0.0)  # geom (0, 1)


if __name__ == "__main__":
    wp.init()
    assert wp.is_cuda_available(), "CUDA is not available."

    # make CUDA implementation available from Warp
    wp.context.runtime.core.epa_gjk_device.argtypes = [
        wp.types.array_t,
        wp.types.array_t,
        wp.types.array_t,
        wp.types.array_t,
        wp.types.array_t,
        wp.types.array_t,
        wp.types.array_t,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_float,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_float,
        wp.types.array_t,
        wp.types.array_t,
        wp.types.array_t,
        wp.types.array_t,
    ]
    wp.context.runtime.core.epa_gjk_device.restype = None

    launch_epa_gjk = wp.context.runtime.core.epa_gjk_device

    absltest.main()
