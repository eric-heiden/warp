# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the warp-level reductions :func:`warp.warp_sum` and :func:`warp.warp_max`."""

import unittest
from typing import Any

import numpy as np

import warp as wp
from warp.tests.unittest_utils import *


@wp.kernel
def warp_reduce_kernel(x: wp.array[Any], s: wp.array[Any], m: wp.array[Any]):
    i = wp.tid()
    s[i] = wp.warp_sum(x[i])
    m[i] = wp.warp_max(x[i])


def _expected(x, device, op):
    if not wp.get_device(device).is_cuda:
        return x  # every CPU thread is its own warp
    g = x.reshape(-1, 32)
    r = op(g.astype(np.float64), axis=1)
    return np.repeat(r, 32)


def test_warp_reduce_types(test, device):
    rng = np.random.default_rng(0)
    n = 32 * 37
    for dtype, np_dtype, tol in [
        (wp.float32, np.float32, 1e-5),
        (wp.float64, np.float64, 1e-12),
        (wp.float16, np.float16, 2e-2),
        (wp.int32, np.int32, 0),
        (wp.int64, np.int64, 0),
        (wp.uint32, np.uint32, 0),
        (wp.int8, np.int8, 0),
    ]:
        if np.issubdtype(np_dtype, np.integer):
            hi = 3 if np_dtype == np.int8 else 1000
            x = rng.integers(0, hi, n).astype(np_dtype)
        else:
            x = rng.standard_normal(n).astype(np_dtype)
        xa = wp.array(x, dtype=dtype, device=device)
        s = wp.zeros(n, dtype=dtype, device=device)
        m = wp.zeros(n, dtype=dtype, device=device)
        for block_dim in (32, 128, 256):
            wp.launch(warp_reduce_kernel, dim=n, inputs=[xa], outputs=[s, m], block_dim=block_dim, device=device)
            np.testing.assert_allclose(
                s.numpy().astype(np.float64), _expected(x, device, np.sum), rtol=tol, atol=tol * 32
            )
            np.testing.assert_allclose(m.numpy().astype(np.float64), _expected(x, device, np.max), rtol=0, atol=0)


def test_warp_reduce_bfloat16(test, device):
    x = np.linspace(-2.0, 2.0, 64).astype(np.float32)
    xa = wp.array(x, dtype=wp.bfloat16, device=device)
    s = wp.zeros(64, dtype=wp.bfloat16, device=device)
    m = wp.zeros(64, dtype=wp.bfloat16, device=device)
    wp.launch(warp_reduce_kernel, dim=64, inputs=[xa], outputs=[s, m], device=device)
    xb = xa.numpy().astype(np.float32)
    np.testing.assert_allclose(s.numpy().astype(np.float32), _expected(xb, device, np.sum), atol=0.05)
    np.testing.assert_allclose(m.numpy().astype(np.float32), _expected(xb, device, np.max), atol=1e-6)


@wp.kernel(module="unique")
def warp_rows_kernel(a: wp.array2d[float], out: wp.array[float]):
    # one warp per row, columns strided over the lanes (own module: compiled without the bfloat16 type)
    tid = wp.tid()
    row = tid // 32
    lane = tid % 32
    s = float(0.0)
    for j in range(lane, a.shape[1], 32):
        s += a[row, j] * a[row, j]
    s = wp.warp_sum(s)
    if lane == 0:
        out[row] = s


def test_warp_reduce_rows(test, device):
    if not wp.get_device(device).is_cuda:
        return
    rng = np.random.default_rng(1)
    a = rng.standard_normal((50, 70)).astype(np.float32)
    out = wp.zeros(50, dtype=float, device=device)
    wp.launch(
        warp_rows_kernel, dim=50 * 32, inputs=[wp.array(a, device=device)], outputs=[out], block_dim=128, device=device
    )
    np.testing.assert_allclose(out.numpy(), (a * a).sum(1), rtol=1e-5)


@wp.kernel(module="unique")
def warp_segment_kernel(x: wp.array[float], s16: wp.array[float], m8: wp.array[float], s1: wp.array[float]):
    i = wp.tid()
    s16[i] = wp.warp_sum(x[i], 16)
    m8[i] = wp.warp_max(x[i], width=8)
    s1[i] = wp.warp_sum(x[i], 1)


def test_warp_reduce_width(test, device):
    rng = np.random.default_rng(3)
    n = 32 * 5
    x = rng.standard_normal(n).astype(np.float32)
    s16, m8, s1 = (wp.zeros(n, dtype=float, device=device) for _ in range(3))
    wp.launch(
        warp_segment_kernel,
        dim=n,
        inputs=[wp.array(x, device=device)],
        outputs=[s16, m8, s1],
        block_dim=64,
        device=device,
    )
    if wp.get_device(device).is_cuda:
        e16 = np.repeat(x.reshape(-1, 16).sum(1), 16)
        e8 = np.repeat(x.reshape(-1, 8).max(1), 8)
    else:
        e16, e8 = x, x
    np.testing.assert_allclose(s16.numpy(), e16, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(m8.numpy(), e8, rtol=0, atol=0)
    np.testing.assert_allclose(s1.numpy(), x, rtol=0, atol=0)


def test_warp_reduce_width_check(test, device):
    with test.assertRaisesRegex(ValueError, "power of two"):

        @wp.kernel(module="unique")
        def bad(x: wp.array[float]):
            i = wp.tid()
            x[i] = wp.warp_sum(x[i], 12)

        wp.launch(bad, dim=32, inputs=[wp.zeros(32, dtype=float, device=device)], device=device)


@wp.kernel
def warp_grad_kernel(x: wp.array[float], w: wp.array[float], loss: wp.array[float]):
    i = wp.tid()
    s = wp.warp_sum(x[i])
    m = wp.warp_max(x[i])
    wp.atomic_add(loss, 0, w[i] * s + 2.0 * w[i] * m)


def test_warp_reduce_grad(test, device):
    rng = np.random.default_rng(2)
    n = 64
    x0 = rng.standard_normal(n).astype(np.float32)
    w0 = rng.standard_normal(n).astype(np.float32)
    x = wp.array(x0, dtype=float, device=device, requires_grad=True)
    w = wp.array(w0, dtype=float, device=device)
    loss = wp.zeros(1, dtype=float, device=device, requires_grad=True)
    tape = wp.Tape()
    with tape:
        wp.launch(warp_grad_kernel, dim=n, inputs=[x, w], outputs=[loss], block_dim=64, device=device)
    tape.backward(loss)
    if wp.get_device(device).is_cuda:
        g = x0.reshape(-1, 32)
        wg = w0.reshape(-1, 32)
        expected = np.repeat(wg.sum(1), 32) + 2.0 * (g == g.max(1, keepdims=True)).ravel() * np.repeat(wg.sum(1), 32)
    else:
        expected = w0 + 2.0 * w0
    np.testing.assert_allclose(x.grad.numpy(), expected, rtol=1e-5, atol=1e-5)


class TestWarpReduce(unittest.TestCase):
    pass


devices = get_test_devices(mode="basic")

add_function_test(TestWarpReduce, "test_warp_reduce_types", test_warp_reduce_types, devices=devices)
add_function_test(TestWarpReduce, "test_warp_reduce_bfloat16", test_warp_reduce_bfloat16, devices=devices)
add_function_test(TestWarpReduce, "test_warp_reduce_rows", test_warp_reduce_rows, devices=devices)
add_function_test(TestWarpReduce, "test_warp_reduce_grad", test_warp_reduce_grad, devices=devices)
add_function_test(TestWarpReduce, "test_warp_reduce_width", test_warp_reduce_width, devices=devices)
add_function_test(TestWarpReduce, "test_warp_reduce_width_check", test_warp_reduce_width_check, devices=devices)


if __name__ == "__main__":
    unittest.main(verbosity=2)
