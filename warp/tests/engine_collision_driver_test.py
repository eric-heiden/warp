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


class EngineCollisionDriverTest(absltest.TestCase):

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
    """Tests collision driver return shapes."""
    m = mujoco.MjModel.from_xml_string(self._CONVEX_CONVEX)
    mx = mjx.put_model(m)
    d = mujoco.MjData(m)
    dx = mjx.put_data(m, d)
    kinematics_jit_fn = jax.jit(mjx.kinematics)
    dx = kinematics_jit_fn(mx, dx)

    mujoco.mj_forward(m, d)
    batch_size = 12

    # @jax.vmap
    # def make_model_and_data(val):
    #   dx = mjx.make_data(m)
    #   mx = mjx.put_model(m)
    #   dx = dx.replace(qpos=dx.qpos.at[2].set(val))
    #   return mx, dx

    # vary the size of body 0.
    # mx, dx = make_model_and_data(jp.arange(-1, 1, 2 / batch_size))

    # forward_jit_fn = jax.jit(jax.vmap(mjx.forward))
    # dx = forward_jit_fn(mx, dx)
    # c = jax.jit(
    #     jax.vmap(
    #         engine_collision_driver.collision,
    #         in_axes=(
    #             0,
    #             0,
    #             None,
    #             None,
    #             None,
    #             None,
    #             None,
    #             None,
    #         ),
    #     ),
    #     static_argnums=(
    #         2,
    #         3,
    #         4,
    #         5,
    #         6,
    #         7,
    #     ),
    # )(mx, dx, 1e9, 12, 12, 12, 8, 1.0)

    print("shape")
    print(len(d.geom_xpos.shape))

    c = engine_collision_driver.collision2(mx, dx, 1e9, 12, 12, 12, 8, 1.0)

    npts = d.contact.pos.shape[1]
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

  # def test_contacts_batched_model_data(self):
  #   """Tests collision driver results."""
  #   m = mujoco.MjModel.from_xml_string(self._CONVEX_CONVEX)
  #   d = mujoco.MjData(m)
  #   mujoco.mj_forward(m, d)
  #   batch_size = 3

  #   #@jax.vmap
  #   def make_model_and_data(val):
  #     dx = mjx.make_data(m)
  #     dx = dx.replace(qpos=dx.qpos.at[2].set(val))
  #     mx = mjx.put_model(m)
  #     return mx, dx

  #   # vary the z-position of body 0.
  #   mx, dx = make_model_and_data(jp.arange(-1, 1, 2 / batch_size))

  #   # forward_jit_fn = jax.jit(jax.vmap(mjx.forward, in_axes=(0, 0)))
  #   # dx = forward_jit_fn(mx, dx)
  #   # c = jax.jit(
  #   #     jax.vmap(
  #   #         engine_collision_driver.collision,
  #   #         in_axes=(
  #   #             0,
  #   #             0,
  #   #             None,
  #   #             None,
  #   #             None,
  #   #             None,
  #   #             None,
  #   #             None,
  #   #         ),
  #   #     ),
  #   #     static_argnums=(
  #   #         2,
  #   #         3,
  #   #         4,
  #   #         5,
  #   #         6,
  #   #         7,
  #   #     ),
  #   # )(mx, dx, 1e9, 12, 12, 12, 8, 1.0)

  #   print("shape")
  #   print(len(d.geom_xpos.shape))

  #   c = engine_collision_driver.collision2(mx, dx, 1e9, 12, 12, 12, 8, 1.0)

  #   # test contact normals and penetration depths.
  #   _compare_contacts(self, dx, c)

  # def test_contacts_batched_data(self):
  #   """Tests collision driver results."""
  #   m = mujoco.MjModel.from_xml_string(self._CONVEX_CONVEX)
  #   mx = mjx.put_model(m)
  #   d = mujoco.MjData(m)
  #   dx = mjx.put_data(m, d)
  #   kinematics_jit_fn = jax.jit(mjx.kinematics)
  #   dx = kinematics_jit_fn(mx, dx)

  #   batch_size = 3

  #   # #@jax.vmap
  #   # def make_model_and_data(val):
  #   #   dx = mjx.make_data(m)
  #   #   dx = dx.replace(qpos=dx.qpos.at[2].set(val))
  #   #   return dx

  #   # # vary the z-position of body 0.
  #   # mx = mjx.put_model(m)
  #   # dx = make_model_and_data(jp.arange(-1, 1, 2 / batch_size))

  #   #forward_jit_fn = jax.jit(jax.vmap(mjx.forward, in_axes=(None, 0)))
  #   #dx = forward_jit_fn(mx, dx)
  #   # c = jax.jit(
  #   #     jax.vmap(
  #   #         engine_collision_driver.collision,
  #   #         in_axes=(
  #   #             None,
  #   #             0,
  #   #             None,
  #   #             None,
  #   #             None,
  #   #             None,
  #   #             None,
  #   #             None,
  #   #         ),
  #   #     ),
  #   #     static_argnums=(
  #   #         2,
  #   #         3,
  #   #         4,
  #   #         5,
  #   #         6,
  #   #         7,
  #   #     ),
  #   # )(mx, dx, 1e9, 12, 12, 12, 8, 1.0)

  #   print("shape")
  #   print(len(d.geom_xpos.shape))

  #   c = engine_collision_driver.collision2(mx, dx, 1e9, 12, 12, 12, 8, 1.0)

  #   # test contact normals and penetration depths.
  #   _compare_contacts(self, dx, c)

  # _CONVEX_CONVEX_2 = """
  #   <mujoco>
  #     <asset>
  #       <mesh name="poly" scale="0.5 0.5 0.5"
  #        vertex="0.3 0 0  0 0.5 0  -0.3 0 0  0 -0.5 0  0 -1 1  0 1 1"
  #        face="0 1 5  0 5 4  0 4 3  3 4 2  2 4 5  1 2 5  0 2 1  0 3 2"/>
  #       <mesh name="tetrahedron"  scale="0.5 0.5 0.5"
  #         vertex="1 1 1  -1 -1 1  1 -1 -1  -1 1 -1"
  #         face="0 1 2  0 3 1  0 2 3  1 3 2"/>
  #     </asset>
  #     <custom>
  #       <numeric data="2" name="max_contact_points"/>
  #     </custom>
  #     <worldbody>
  #       <light pos="-.5 .7 1.5" cutoff="55"/>
  #       <body pos="0.0 2.0 1.781" euler="180 0 0">
  #         <freejoint/>
  #         <geom type="mesh" mesh="poly"/>
  #       </body>
  #       <body pos="0.0 2.0 2.081">
  #         <freejoint/>
  #         <geom type="mesh" mesh="tetrahedron"/>
  #       </body>
  #     </worldbody>
  #   </mujoco>
  # """

  # def test_solparams(self):
  #   """Tests collision driver solparams."""
  #   m = mujoco.MjModel.from_xml_string(self._CONVEX_CONVEX_2)
  #   d = mujoco.MjData(m)
  #   mujoco.mj_forward(m, d)
  #   batch_size = 3

  #   @jax.vmap
  #   def make_model_and_data(val):
  #     dx = mjx.make_data(m)
  #     mx = mjx.put_model(m)
  #     mx = mx.replace(
  #         geom_solref=mx.geom_solref.at[0].set(val),
  #         geom_solimp=mx.geom_solimp.at[0].set(val),
  #         geom_friction=mx.geom_friction.at[0].set(val),
  #     )
  #     return mx, dx

  #   # vary geom contact solver params
  #   mx, dx = make_model_and_data(jp.arange(0.1, 1.1, 1 / batch_size))

  #   forward_jit_fn = jax.jit(jax.vmap(mjx.forward))
  #   dx = forward_jit_fn(mx, dx)
  #   # c = jax.jit(
  #   #     jax.vmap(
  #   #         engine_collision_driver.collision,
  #   #         in_axes=(
  #   #             0,
  #   #             0,
  #   #             None,
  #   #             None,
  #   #             None,
  #   #             None,
  #   #             None,
  #   #             None,
  #   #         ),
  #   #     ),
  #   #     static_argnums=(
  #   #         2,
  #   #         3,
  #   #         4,
  #   #         5,
  #   #         6,
  #   #         7,
  #   #     ),
  #   # )(mx, dx, 1e9, 12, 12, 12, 8, 1.0)
  #   c = engine_collision_driver.collision2(mx, dx, 1e9, 12, 12, 12, 8, 1.0)

  #   np.testing.assert_array_almost_equal(c.solref, dx.contact.solref)
  #   np.testing.assert_array_almost_equal(c.solimp, dx.contact.solimp)
  #   np.testing.assert_array_almost_equal(c.friction, dx.contact.friction)



@wp.kernel
def testKernel(num : int):
    tid = wp.tid()
    if tid >= num:
        return



if __name__ == "__main__":

  wp.launch(testKernel, dim=1, inputs = [1])


  a = wp.zeros(1, dtype=wp.int32)
  b = wp.zeros(1, dtype=wp.int32)

  wp.utils.array_scan(a, b, False)

  absltest.main()
