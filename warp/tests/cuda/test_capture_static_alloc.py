# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for graph-owned capture arenas (``wp.ScopedCapture(static_alloc=True)``)."""

import unittest

import numpy as np

import warp as wp
from warp.tests.unittest_utils import *


@wp.kernel
def _scale(src: wp.array[float], dst: wp.array[float], s: float):
    i = wp.tid()
    dst[i] = src[i] * s


@wp.kernel
def _add_into(src: wp.array[float], dst: wp.array[float]):
    i = wp.tid()
    dst[i] = dst[i] + src[i]


def _uses_temporaries(x, out):
    # allocates temporaries during capture, the pattern used by libraries such as MuJoCo Warp
    tmp = wp.empty_like(x)
    wp.launch(_scale, x.shape, [x, tmp, 2.0], device=x.device)
    tmp2 = wp.clone(tmp)
    wp.launch(_add_into, x.shape, [tmp2, out], device=x.device)


def test_static_alloc_no_alloc_nodes(test, device):
    n = 1024
    x = wp.array(np.arange(n, dtype=np.float32), device=device)
    out = wp.zeros(n, dtype=float, device=device)
    with wp.ScopedCapture(device, static_alloc=True) as cap:
        _uses_temporaries(x, out)
    test.assertIsNotNone(cap.graph._arena)
    test.assertGreater(cap.graph._arena.bytes_reserved, 0)
    for _ in range(3):
        wp.capture_launch(cap.graph)
    assert_np_equal(out.numpy(), 3 * 2.0 * np.arange(n, dtype=np.float32))
    # the device allocator is restored after capture
    test.assertIsNot(wp.get_device(device).get_allocator(), cap.graph._arena)


def test_static_alloc_reuses_freed_blocks(test, device):
    n = 4096
    x = wp.array(np.ones(n, dtype=np.float32), device=device)
    out = wp.zeros(n, dtype=float, device=device)
    with wp.ScopedCapture(device, static_alloc=True) as cap:
        for _ in range(8):
            _uses_temporaries(x, out)
    # eight iterations of two same-sized temporaries fit in two blocks thanks to free-list reuse
    test.assertLessEqual(cap.graph._arena.bytes_live, 2 * 4 * n + 2 * 256)
    wp.capture_launch(cap.graph)
    assert_np_equal(out.numpy(), np.full(n, 16.0, dtype=np.float32))


def test_static_alloc_array_outlives_capture(test, device):
    n = 256
    x = wp.array(np.arange(n, dtype=np.float32), device=device)
    with wp.ScopedCapture(device, static_alloc=True) as cap:
        y = wp.empty_like(x)
        wp.launch(_scale, x.shape, [x, y, 3.0], device=device)
    wp.capture_launch(cap.graph)
    graph = cap.graph
    del cap
    assert_np_equal(y.numpy(), 3.0 * np.arange(n, dtype=np.float32))
    del graph  # arena stays alive while y references it
    assert_np_equal(y.numpy(), 3.0 * np.arange(n, dtype=np.float32))


def test_static_alloc_in_conditional_body(test, device):
    if not wp.is_conditional_graph_supported():
        test.skipTest("Conditional graph nodes not supported")
    n = 512
    x = wp.array(np.ones(n, dtype=np.float32), device=device)
    out = wp.zeros(n, dtype=float, device=device)
    cond = wp.array([1], dtype=int, device=device)
    # allocation nodes are not allowed in conditional bodies, the arena makes this capturable
    with wp.ScopedCapture(device, static_alloc=True) as cap:
        wp.capture_if(cond, on_true=lambda: _uses_temporaries(x, out))
    wp.capture_launch(cap.graph)
    cond.fill_(0)
    wp.capture_launch(cap.graph)
    assert_np_equal(out.numpy(), np.full(n, 2.0, dtype=np.float32))


devices = get_selected_cuda_test_devices(mode="basic")


class TestCaptureStaticAlloc(unittest.TestCase):
    pass


add_function_test(
    TestCaptureStaticAlloc,
    "test_static_alloc_no_alloc_nodes",
    test_static_alloc_no_alloc_nodes,
    devices=devices,
    check_output=False,
)
add_function_test(
    TestCaptureStaticAlloc,
    "test_static_alloc_reuses_freed_blocks",
    test_static_alloc_reuses_freed_blocks,
    devices=devices,
    check_output=False,
)
add_function_test(
    TestCaptureStaticAlloc,
    "test_static_alloc_array_outlives_capture",
    test_static_alloc_array_outlives_capture,
    devices=devices,
    check_output=False,
)
add_function_test(
    TestCaptureStaticAlloc,
    "test_static_alloc_in_conditional_body",
    test_static_alloc_in_conditional_body,
    devices=devices,
    check_output=False,
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
