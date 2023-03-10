# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

###########################################################################
# Example Sim Ant
#
# Shows how to set up a simulation of a rigid-body Ant articulation based on
# the OpenAI gym environment using the wp.sim.ModelBuilder() and MCJF
# importer. Note this example does not include a trained policy.
#
###########################################################################

import os
import math

import warp as wp
import warp.sim

from environment import Environment, run_env, IntegratorType

class AntEnvironment(Environment):
    sim_name = "example_sim_ant"
    env_offset=(2.5, 0.0, 2.5)
    tiny_render_settings = dict(scaling=3.0)
    usd_render_settings = dict(scaling=100.0)

    sim_substeps_euler = 32
    sim_substeps_xpbd = 3

    joint_attach_ke: float = 100000.0
    joint_attach_kd: float = 10.0

    use_graph_capture = True

    def create_articulation(self, builder):
        wp.sim.parse_mjcf(
            os.path.join(os.path.dirname(__file__), "../assets/nv_ant.xml"),
            builder,
            stiffness=0.0,
            damping=1.0,
            armature=0.1,
            contact_ke=1.e+4,
            contact_kd=1.e+2,
            contact_kf=1.e+4,
            contact_mu=1.0,
            limit_ke=1.e+4,
            limit_kd=1.e+1,
            enable_self_collisions=False)
        builder.joint_q[7:] = [0.0, 1.0, 0.0, -1.0, 0.0, -1.0, 0.0, 1.0]
        builder.joint_q[:7] = [0.0, 0.7, 0.0, *wp.quat_from_axis_angle((1.0, 0.0, 0.0), -math.pi*0.5)]


if __name__ == "__main__":
    run_env(AntEnvironment)
