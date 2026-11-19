# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmarks for :func:`warp.gemm` on dense-layer shapes, with and without its fused ReLU epilogue.

Each timed call replays a CUDA graph of ``LAUNCHES_PER_SAMPLE`` GEMMs and the timer reads the GPU timeline.
``GemmLayerTrainStep`` times a dense layer's forward and backward pass recorded on a :class:`warp.Tape`.
"""

import numpy as np

import warp as wp

from .benchmarks_utils import LAUNCHES_PER_SAMPLE, CudaEventTimer, capture_launches, setup_once

# (batch, inputs, outputs) of the policy and value network layers of an RL trainer
LAYER_SHAPES = ("4096x64x512", "4096x512x256", "4096x256x128")


def _activation(name):
    return {"none": None, "relu": "relu"}[name]


def _layer(shape, dtype, device, requires_grad=False):
    m, k, n = (int(v) for v in shape.split("x"))
    rng = np.random.default_rng(42)
    x = wp.array(rng.standard_normal((m, k)), dtype=dtype, device=device, requires_grad=requires_grad)
    w = wp.array(rng.standard_normal((k, n)) / np.sqrt(k), dtype=dtype, device=device, requires_grad=requires_grad)
    b = wp.array(rng.standard_normal(n), dtype=dtype, device=device, requires_grad=requires_grad)
    y = wp.zeros((m, n), dtype=dtype, device=device, requires_grad=requires_grad)
    return x, w, b, y


class GemmLayerForward:
    """``y = activation(x @ w + b)`` for dense-layer shapes."""

    params = (LAYER_SHAPES, ("float32", "float16"), ("none", "relu"))
    param_names = ("shape", "dtype", "activation")

    timer = CudaEventTimer()
    repeat = 10
    number = 1

    @setup_once
    def setup(self, shape, dtype, activation):
        wp.init()
        if not wp.is_cublaslt_available():
            raise NotImplementedError("cuBLASLt is not available")
        self.device = wp.get_device("cuda:0")
        x, w, b, y = _layer(shape, getattr(wp, dtype), self.device)
        act = _activation(activation)

        def launch():
            wp.gemm(x, w, y, bias=b, activation=act)

        launch()  # plan, autotune and load modules outside of the capture
        self.graph = capture_launches(launch, LAUNCHES_PER_SAMPLE)
        wp.capture_launch(self.graph)
        wp.synchronize_device(self.device)

    def time_cuda(self, shape, dtype, activation):
        wp.capture_launch(self.graph)


class GemmLayerTrainStep:
    """Forward and backward pass of ``y = activation(x @ w + b)`` recorded on a tape (float32)."""

    params = (LAYER_SHAPES, ("none", "relu"))
    param_names = ("shape", "activation")

    timer = CudaEventTimer()
    repeat = 10
    number = 1

    @setup_once
    def setup(self, shape, activation):
        wp.init()
        if not wp.is_cublaslt_available():
            raise NotImplementedError("cuBLASLt is not available")
        self.device = wp.get_device("cuda:0")
        x, w, b, y = _layer(shape, wp.float32, self.device, requires_grad=True)
        gy = wp.ones_like(y, requires_grad=False)
        act = _activation(activation)

        def step():
            tape = wp.Tape()
            with tape:
                wp.gemm(x, w, y, bias=b, activation=act)
            tape.backward(grads={y: gy})

        step()
        self.graph = capture_launches(step, LAUNCHES_PER_SAMPLE // 8)
        wp.capture_launch(self.graph)
        wp.synchronize_device(self.device)

    def time_cuda(self, shape, activation):
        wp.capture_launch(self.graph)
