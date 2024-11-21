# Copyright (c) 2024 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from contextlib import contextmanager
import math
import os
from typing import Any

import numpy as np

import warp as wp
from warp.utils import coerce_float64_types

import warp.examples
import warp.sim
from warp.autograd import gradcheck

@wp.kernel
def assign_kernel(src: wp.array(dtype=Any), dest: wp.array(dtype=Any)):
    tid = wp.tid()
    dest[tid] = src[tid]


def gradcheck_cartpole_semiimplicit():
    def assign(src: wp.array(dtype=Any), dest: wp.array(dtype=Any)):
            wp.launch(
            assign_kernel,
            dim=len(src),
            inputs=[src],
            outputs=[dest],
            device=src.device)

    class CartpoleSim:
        def __init__(self, stage_path="example_cartpole.usd", num_envs=1, num_steps=10, all_states_general_coords=False):
            builder = wp.sim.ModelBuilder()

            self.num_envs = num_envs
            self.num_steps = num_steps
            self.all_states_general_coords = all_states_general_coords

            articulation_builder = wp.sim.ModelBuilder()

            wp.sim.parse_urdf(
                os.path.join(warp.examples.get_asset_directory(), "cartpole.urdf"),
                articulation_builder,
                xform=wp.transform(wp.vec3(), wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), -math.pi * 0.5)),
                floating=False,
                density=100,
                armature=0.1,
                stiffness=0.0,
                damping=0.0,
                limit_ke=1.0e4,
                limit_kd=1.0e1,
                enable_self_collisions=False,
            )

            builder = wp.sim.ModelBuilder()

            self.sim_time = 0.0
            fps = 60
            self.frame_dt = 1.0 / fps

            self.sim_substeps = 10
            self.sim_dt = self.frame_dt / self.sim_substeps

            for i in range(self.num_envs):
                builder.add_builder(
                    articulation_builder, xform=wp.transform(np.array((i * 2.0, 4.0, 0.0)), wp.quat_identity())
                )

            # finalize model
            self.model = builder.finalize(requires_grad=True)
            self.model.ground = False

            self.model.joint_attach_ke = 1600.0
            self.model.joint_attach_kd = 20.0

            #self.integrator = wp.sim.FeatherstoneIntegrator(self.model)
            self.integrator = wp.sim.SemiImplicitIntegrator()
            #self.integrator = wp.sim.XPBDIntegrator(iterations=1)

            self.states = [self.model.state() for _ in range(self.num_steps * self.sim_substeps + 1)]


        def rollout(self):
            if not isinstance(self.integrator, wp.sim.FeatherstoneIntegrator):
                # apply initial generalized coordinates
                wp.sim.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, None, self.states[0])

            i = 0
            for _ in range(self.num_steps):
                for _ in range(self.sim_substeps):
                    self.states[i].clear_forces()
                    self.integrator.simulate(self.model, self.states[i], self.states[i+1], self.sim_dt)
                    if self.all_states_general_coords:
                        if not isinstance(self.integrator, wp.sim.FeatherstoneIntegrator):
                            # compute generalized coordinates
                            wp.sim.eval_ik(self.model, self.states[i], self.states[i].joint_q, self.states[i].joint_qd)
                    i += 1

            if not isinstance(self.integrator, wp.sim.FeatherstoneIntegrator):
                # compute generalized coordinates
                wp.sim.eval_ik(self.model, self.states[-1], self.states[-1].joint_q, self.states[-1].joint_qd)

    def rollout_cartpole(joint_q0, joint_qd0, num_steps=1, all_states_general_coords=False):
        #joint_q0, joint_qd0 = inputs
        
        cart_sim = CartpoleSim(num_envs=1, num_steps=num_steps, all_states_general_coords=all_states_general_coords)

        assign(joint_q0, cart_sim.model.joint_q)
        assign(joint_qd0, cart_sim.model.joint_qd)
        assign(joint_q0, cart_sim.states[0].joint_q)
        assign(joint_qd0, cart_sim.states[0].joint_qd)

        cart_sim.rollout()

        joint_qN = cart_sim.states[-1].joint_q
        joint_qdN = cart_sim.states[-1].joint_qd
        return [joint_qN, joint_qdN]

    with wp.ScopedDevice("cuda:0"):
        joint_q0 = wp.array([0.0, 0.3, 0.0], dtype=float, requires_grad=True)
        joint_qd0 = wp.array([0.0, 0.0, 0.0], dtype=float, requires_grad=True)

        inputs = [joint_q0, joint_qd0]
        outputs = rollout_cartpole(joint_q0, joint_qd0)

        gradcheck(
            rollout_cartpole,
            inputs=inputs,
            outputs=outputs,
            eps=1e-3,
            atol=1e-2,
            rtol=1e-1,
            show_summary=True,
            raise_exception=False)
        
if __name__ == "__main__":
    wp.init()

    print("====================== IN FP32 PRECISION ==============================")
    gradcheck_cartpole_semiimplicit()

    with coerce_float64_types():
        print("====================== IN FP64 PRECISION ==============================")
        gradcheck_cartpole_semiimplicit()