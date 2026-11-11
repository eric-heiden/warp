# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests tile element extraction with 512-thread blocks.

Covers one- through four-dimensional element access and its adjoint behavior.
Add tests here when they read individual elements from tiles or validate the
corresponding gradients.
"""

import unittest

import numpy as np

import warp as wp
from warp.tests.unittest_utils import *

TILE_SIZE = 4


@wp.kernel
def tile_extract_1d_kernel(input: wp.array1d[float], output: wp.array1d[float]):
    i = wp.tid()

    t = wp.tile_load(input, shape=TILE_SIZE)

    output[i] = t[i]


@wp.kernel
def tile_extract_2d_kernel(input: wp.array2d[float], output: wp.array2d[float]):
    i, j = wp.tid()

    t = wp.tile_load(input, shape=(TILE_SIZE, TILE_SIZE))

    output[i, j] = t[i, j]


@wp.kernel
def tile_extract_3d_kernel(input: wp.array3d[float], output: wp.array3d[float]):
    i, j, k = wp.tid()

    t = wp.tile_load(input, shape=(TILE_SIZE, TILE_SIZE, TILE_SIZE))

    output[i, j, k] = t[i, j, k]


@wp.kernel
def tile_extract_4d_kernel(input: wp.array4d[float], output: wp.array4d[float]):
    i, j, k, l = wp.tid()

    t = wp.tile_load(input, shape=(TILE_SIZE, TILE_SIZE, TILE_SIZE, TILE_SIZE))

    output[i, j, k, l] = t[i, j, k, l]


def test_tile_extract(kernel, ndim):
    shape = (TILE_SIZE,) * ndim

    def test_run(test, device):
        rng = np.random.default_rng(42)

        input = wp.array(rng.random(shape), dtype=float, requires_grad=True, device=device)
        output = wp.zeros_like(input)

        with wp.Tape() as tape:
            wp.launch(
                kernel,
                dim=shape,
                inputs=[input, output],
                # Debug CUDA backward kernels may be resource-limited below 1024 threads.
                block_dim=512,
                device=device,
            )

        assert_np_equal(output.numpy(), input.numpy())

        output.grad = wp.ones_like(output)
        tape.backward()

        assert_np_equal(input.grad.numpy(), np.ones_like(input.numpy()))

    return test_run


devices = get_test_devices()


class TestTileLoadExtract(unittest.TestCase):
    pass


add_function_test(
    TestTileLoadExtract,
    "test_tile_extract_2d_cpu_blocks",
    test_tile_extract(tile_extract_2d_kernel, 2),
    devices=get_cpu_test_devices(),
    enable_cpu_blocks=True,
)
add_function_test(
    TestTileLoadExtract,
    "test_tile_extract_1d",
    test_tile_extract(tile_extract_1d_kernel, 1),
    devices=devices,
)
add_function_test(
    TestTileLoadExtract,
    "test_tile_extract_2d",
    test_tile_extract(tile_extract_2d_kernel, 2),
    devices=devices,
)
add_function_test(
    TestTileLoadExtract,
    "test_tile_extract_3d",
    test_tile_extract(tile_extract_3d_kernel, 3),
    devices=devices,
)
add_function_test(
    TestTileLoadExtract,
    "test_tile_extract_4d",
    test_tile_extract(tile_extract_4d_kernel, 4),
    devices=devices,
)


# Register tiles read by other threads: single-warp CUDA blocks (block_dim <= 32) read them with warp shuffles,
# larger blocks through shared memory. Each thread reads its own (per-thread) index of a two-register tile and the
# block-wide sum; several threads read the same element, so the adjoint must accumulate all of them.
@wp.kernel(module="unique")
def tile_extract_register_kernel(x: wp.array[float], idx: wp.array[int], y: wp.array[float], s: wp.array[float]):
    _blk, i = wp.tid()
    t = wp.tile_load(x, shape=64, storage="register")
    t2 = t * t
    y[i] = wp.tile_extract(t2, idx[i])
    s[i] = wp.tile_extract(wp.tile_sum(t2), 0)


def test_tile_extract_register(test, device):
    rng = np.random.default_rng(5)
    x_np = rng.standard_normal(64).astype(np.float32)
    for block_dim in (16, 32, 64):
        idx_np = rng.integers(0, 64, block_dim).astype(np.int32)
        idx_np[: block_dim // 2] = 7  # half the threads read the same element
        x = wp.array(x_np, dtype=float, device=device, requires_grad=True)
        y = wp.zeros(block_dim, dtype=float, device=device, requires_grad=True)
        s = wp.zeros(block_dim, dtype=float, device=device, requires_grad=True)
        tape = wp.Tape()
        with tape:
            wp.launch_tiled(
                tile_extract_register_kernel,
                dim=[1],
                inputs=[x, wp.array(idx_np, dtype=int, device=device)],
                outputs=[y, s],
                block_dim=block_dim,
                device=device,
            )
        np.testing.assert_allclose(y.numpy(), x_np[idx_np] ** 2, rtol=1e-6)
        np.testing.assert_allclose(s.numpy(), np.full(block_dim, (x_np**2).sum()), rtol=1e-5)
        y.grad = wp.ones(block_dim, dtype=float, device=device)
        s.grad = wp.ones(block_dim, dtype=float, device=device)
        tape.backward()
        counts = np.bincount(idx_np, minlength=64).astype(np.float32)
        expected = 2.0 * x_np * (counts + block_dim)
        np.testing.assert_allclose(x.grad.numpy(), expected, rtol=1e-5, err_msg=f"block_dim={block_dim}")


add_function_test(
    TestTileLoadExtract,
    "test_tile_extract_register",
    test_tile_extract_register,
    devices=get_selected_cuda_test_devices(),
)


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
