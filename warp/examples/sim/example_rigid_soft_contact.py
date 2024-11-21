# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

###########################################################################
# Example Sim Rigid FEM
#
# Shows how to set up a rigid sphere colliding with an FEM beam
# using wp.sim.ModelBuilder().
#
###########################################################################

import warp as wp
import warp.sim
import warp.sim.render

# wp.config.verify_cuda = True
# wp.config.verify_fp = True

from tqdm import trange


class Example:
    def __init__(self, stage_path="example_rigid_soft_contact.usd"):
        self.sim_width = 8
        self.sim_height = 8

        fps = 60
        self.frame_dt = 1.0 / fps
        self.sim_substeps = 32
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.sim_iterations = 1
        self.sim_relaxation = 1.0
        self.profiler = {}

        builder = wp.sim.ModelBuilder()

        partial_builder = wp.sim.ModelBuilder()
        partial_builder.default_particle_radius = 0.01

        partial_builder.add_soft_grid(
            pos=wp.vec3(0.0, 0.0, 0.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=20,
            dim_y=10,
            dim_z=10,
            cell_x=0.1,
            cell_y=0.1,
            cell_z=0.1,
            density=100.0,
            k_mu=50000.0,
            k_lambda=20000.0,
            k_damp=0.0,
        )

        b = partial_builder.add_body(origin=wp.transform((0.5, 2.5, 0.5), wp.quat_identity()))
        partial_builder.add_shape_sphere(body=b, radius=0.75, density=100.0)

        num_envs = 50

        root_envs = int(num_envs ** 0.5)
        for i in range(num_envs):
            builder.add_builder(
                partial_builder,
                xform=wp.transform(
                    wp.vec3((i // root_envs) * 4.0 - root_envs * 2.0, 0.0, (i % root_envs) * 4.0 - root_envs * 2.0), wp.quat_identity()
                ),
            )

        builder.soft_contact_max = 16 * 1024 * num_envs

        self.model = builder.finalize(requires_grad=True)
        self.model.ground = True
        self.model.soft_contact_ke = 1.0e3
        self.model.soft_contact_kd = 0.0
        self.model.soft_contact_kf = 1.0e3

        self.integrator = wp.sim.SemiImplicitIntegrator()

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        if stage_path:
            self.renderer = wp.sim.render.SimRenderer(self.model, stage_path, scaling=1.0)
        else:
            self.renderer = None

        self.use_cuda_graph = wp.get_device().is_cuda
        if self.use_cuda_graph:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def simulate(self):
        for _s in range(self.sim_substeps):
            wp.sim.collide(self.model, self.state_0)

            self.state_0.clear_forces()
            self.state_1.clear_forces()

            self.integrator.simulate(self.model, self.state_0, self.state_1, self.sim_dt)

            # swap states
            (self.state_0, self.state_1) = (self.state_1, self.state_0)

    def step(self):
        with wp.ScopedTimer("step", dict=self.profiler, active=False):
            if self.use_cuda_graph:
                wp.capture_launch(self.graph)
            else:
                self.simulate()
            self.sim_time += self.frame_dt

    def render(self):
        if self.renderer is None:
            return

        with wp.ScopedTimer("render", active=False):
            self.renderer.begin_frame(self.sim_time)
            self.renderer.render(self.state_0)
            self.renderer.end_frame()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument(
        "--stage_path",
        type=lambda x: None if x == "None" else str(x),
        default="example_rigid_soft_contact.usd",
        help="Path to the output USD file.",
    )
    parser.add_argument("--num_frames", type=int, default=100, help="Total number of frames.")

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        example = Example(stage_path=args.stage_path)

        for _ in trange(args.num_frames):
            example.step()
            example.render()

        if example.renderer:
            example.renderer.save()
