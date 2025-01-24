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
"""Tests the CUDA collision driver."""

from absl.testing import absltest
from etils import epath
import jax
from jax import numpy as jp
import mujoco
from mujoco import mjx
import engine_collision_driver
import numpy as np

import warp as wp


def _compare_contacts(test_cls, dx, c):
  """Compares JAX and CUDA contacts."""
  for env_id, (g1, g2) in enumerate(zip(dx.contact.geom, c.geom)):
    for g1_key in np.unique(g1, axis=0):
      idx1 = np.where((g1 == g1_key).all(axis=1))
      idx2 = np.where((g2 == g1_key).all(axis=1))
      dist1 = dx.contact.dist[env_id][idx1]
      dist2 = c.dist[env_id][idx2]
      # contacts may appear in JAX with dist>0, but not in CUDA.
      if (dist1 > 0).any():
        if dist2.shape[0]:
          test_cls.assertTrue((dist2 >= 0).any())
        continue
      test_cls.assertTrue((dist1 < 0).all())
      # contact distance in JAX are dynamically calculated, so we only
      # check that CUDA distances are equal to the first JAX distance.
      np.testing.assert_array_almost_equal(dist1[0], dist2, decimal=3)
      # normals should be equal.
      normal1 = dx.contact.frame[env_id, :, 0][idx1]
      normal2 = c.frame[env_id, :, 0][idx2]
      test_cls.assertLess(np.abs(normal1[0] - normal2).max(), 1e-5)
      # contact points are not as accurate in CUDA, the test is rather loose.
      found_point = 0
      pos1 = dx.contact.pos[env_id][idx1]
      pos2 = c.pos[env_id][idx2]
      for pos1_idx in range(pos1.shape[0]):
        pos2_idx = np.abs(pos1[pos1_idx] - pos2).sum(axis=1).argmin()
        found_point += np.abs(pos1[pos1_idx] - pos2[pos2_idx]).max() < 0.11
      test_cls.assertGreater(found_point, 0)


class EngineCollisionDriverTest: #(absltest.TestCase):

  _CONVEX_CONVEX = """
    <mujoco>
      <asset>
        <mesh name="meshbox"
              vertex="-1 -1 -1
                      1 -1 -1
                      1  1 -1
                      1  1  1
                      1 -1  1
                      -1  1 -1
                      -1  1  1
                      -1 -1  1"/>
        <mesh name="poly" scale="0.5 0.5 0.5"
         vertex="0.3 0 0  0 0.5 0  -0.3 0 0  0 -0.5 0  0 -1 1  0 1 1"
         face="0 1 5  0 5 4  0 4 3  3 4 2  2 4 5  1 2 5  0 2 1  0 3 2"/>
        <mesh name="tetrahedron"  scale="0.5 0.5 0.5"
          vertex="1 1 1  -1 -1 1  1 -1 -1  -1 1 -1"
          face="0 1 2  0 3 1  0 2 3  1 3 2"/>
      </asset>
      <custom>
        <numeric data="12" name="max_contact_points"/>
      </custom>
      <worldbody>
        <light pos="-.5 .7 1.5" cutoff="55"/>
        <body pos="0.0 2.0 0.35" euler="0 0 90">
          <freejoint/>
          <geom type="mesh" mesh="meshbox"/>
        </body>
        <body pos="0.0 2.0 1.781" euler="180 0 0">
          <freejoint/>
          <geom type="mesh" mesh="poly"/>
          <geom pos="0.5 0 -0.2" type="sphere" size="0.3"/>
        </body>
        <body pos="0.0 2.0 2.081">
          <freejoint/>
          <geom type="mesh" mesh="tetrahedron"/>
        </body>
        <body pos="0.0 0.0 -2.0">
          <geom type="plane" size="40 40 40"/>
        </body>
      </worldbody>
    </mujoco>
  """

  def test_shapes(self):

    # https://nvidia.github.io/warp/debugging.html
    wp.init()

    wp.config.mode = "debug"
    assert wp.context.runtime.core.is_debug_enabled(), "Warp must be built in debug mode to enable debugging kernels"

    wp.config.print_launches = True   
    wp.config.verify_cuda = True


    """Tests collision driver return shapes."""
    m = mujoco.MjModel.from_xml_string(self._CONVEX_CONVEX)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    batch_size = 3

    def make_model_and_data(val):
        dx = mjx.make_data(m)
        mx = mjx.put_model(m)
        dx = dx.replace(qpos=dx.qpos.at[2].set(val))
        return mx, dx

    # Vary the size of body 0, manually create the batch using a loop.
    mx_list = []
    dx_list = []
    for val in jp.arange(-1, 1, 2 / batch_size):
        mx, dx = make_model_and_data(val)
        mx_list.append(mx)
        dx_list.append(dx)

    # mx = jp.stack(mx_list)
    # dx = jp.stack(dx_list)

    forward_jit_fn = jax.jit(mjx.forward)
    # dx = forward_jit_fn(mx, dx)

    dx = []
    for i in range(batch_size):
       tmp = forward_jit_fn(mx_list[i], dx_list[i])
       dx.append(tmp)

    print("dx")
    print(dx)

    # Manually iterate for the collision function
    c_list = []
    for i in range(batch_size):
        c = engine_collision_driver.collision2(mx_list[i], dx[i], 1e9, 12, 12, 12, 8, 1.0)
        c_list.append(c)

    # Stack the results for consistency
    # c = jp.stack(c_list)

    npts = dx.contact.pos.shape[1]
    self.assertTupleEqual(c.dist.shape, (batch_size, npts))
    self.assertTupleEqual(c.pos.shape, (batch_size, npts, 3))
    self.assertTupleEqual(c.frame.shape, (batch_size, npts, 3, 3))
    self.assertTupleEqual(c.friction.shape, (batch_size, npts, 5))
    self.assertTupleEqual(c.solimp.shape, (batch_size, npts, mujoco.mjNIMP))
    self.assertTupleEqual(c.solref.shape, (batch_size, npts, mujoco.mjNREF))
    self.assertTupleEqual(
        c.solreffriction.shape, (batch_size, npts, mujoco.mjNREF)
    )
    self.assertTupleEqual(c.geom.shape, (batch_size, npts, 2))
    self.assertTupleEqual(c.geom1.shape, (batch_size, npts))
    self.assertTupleEqual(c.geom2.shape, (batch_size, npts))


if __name__ == "__main__":
   instance = EngineCollisionDriverTest()
   instance.test_shapes()
  #absltest.main()
