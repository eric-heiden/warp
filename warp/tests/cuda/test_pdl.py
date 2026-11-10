# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for programmatic dependent launch (``"enable_pdl"`` module option / ``wp.config.enable_pdl``)."""

import unittest

import numpy as np

import warp as wp
from warp.tests.unittest_utils import *


def _make_kernels(enable_pdl):
    @wp.kernel(module="unique", module_options={"enable_pdl": enable_pdl})
    def step(a: wp.array[float], b: wp.array[float]):
        i = wp.tid()
        # each kernel depends on the full output of the previous one (neighbor reads)
        n = a.shape[0]
        b[i] = 0.5 * a[i] + 0.25 * (a[(i + 1) % n] + a[(i + n - 1) % n]) + 1.0

    return step


def test_pdl_chain_matches(test, device):
    n = 8192
    x0 = np.random.default_rng(0).standard_normal(n).astype(np.float32)
    results = []
    for enable_pdl in (False, True):
        step = _make_kernels(enable_pdl)
        a = wp.array(x0, device=device)
        b = wp.zeros_like(a)
        with wp.ScopedCapture(device) as cap:
            for _ in range(50):
                wp.launch(step, n, [a, b], device=device)
                wp.launch(step, n, [b, a], device=device)
        for _ in range(4):
            wp.capture_launch(cap.graph)
        for _ in range(10):  # also outside graphs
            wp.launch(step, n, [a, b], device=device)
            wp.launch(step, n, [b, a], device=device)
        results.append(a.numpy())
    np.testing.assert_array_equal(results[0], results[1])


devices = [d for d in get_selected_cuda_test_devices(mode="basic") if wp.get_device(d).arch >= 90]


class TestPDL(unittest.TestCase):
    pass


add_function_test(TestPDL, "test_pdl_chain_matches", test_pdl_chain_matches, devices=devices, check_output=False)


if __name__ == "__main__":
    unittest.main(verbosity=2)
