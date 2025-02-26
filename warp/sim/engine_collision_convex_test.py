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

from absl.testing import absltest
import jax
from jax import numpy as jp
import mujoco
from mujoco import mjx
import numpy as np

from convex import _collide
from engine_collision_interop import gjk_epa_jax


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

    def test_call_batched_data(self):
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

        dist, pos, n = jax.vmap(
            gjk_epa_jax,
            in_axes=(
                None,
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
        )(mx, dx, geom_pair, key_types, 4, mx.ngeom, 1e9, 12, 12, 12, 8, 1.0)

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
        self.assertTrue(float(mx.geom_size[0][0, 0]), float(mx.geom_size[-1][0, 0]))

        kinematics_jit_fn = jax.jit(jax.vmap(mjx.kinematics))
        dx = kinematics_jit_fn(mx, dx)
        key_types = (m.geom_type[0], m.geom_type[1])
        geom_pair = jp.array(np.tile(np.array([[0, 1]]), (batch_size, 1, 1)))

        dist, pos, n = jax.jit(
            jax.vmap(
                gjk_epa_jax,
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
            ),
            static_argnums=(
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
            ),
        )(mx, dx, geom_pair, key_types, 1, mx.ngeom, 1e9, 12, 12, 12, 8, 1.0)

        self.assertTupleEqual(dist.shape, (batch_size, 1))
        self.assertTupleEqual(pos.shape, (batch_size, 1, 3))
        self.assertTupleEqual(n.shape, (batch_size, 1, 3))
        # geom0 is not colliding in env0 since the size of geom0 is small
        self.assertGreater(dist[0, 0], 0.0)  # geom (0, 1)
        # the last env should have a collision since geom0 is scaled to 1x the
        # original size
        self.assertLess(dist[-1, 0], 0.0)  # geom (0, 1)


if __name__ == "__main__":
    absltest.main()
    # test = EngineCollisionConvexTest()
    # test.test_box_plane()
