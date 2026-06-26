# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
import unittest

import numpy as np

import warp as wp
import warp.optim
from warp.tests.unittest_utils import *


@wp.kernel
def step_loss_kernel(x: wp.array(dtype=float), loss: wp.array(dtype=float)):
    if x[0] >= 0.0:
        loss[0] = 1.0
    else:
        loss[0] = 0.0


@wp.kernel
def quadratic_loss_kernel(x: wp.array(dtype=float), loss: wp.array(dtype=float)):
    loss[0] = x[0] * x[0] + 0.5 * x[1] * x[1]


def make_step_loss(device):
    def loss_fn(params):
        loss = wp.zeros(1, dtype=float, requires_grad=True, device=device)
        wp.launch(step_loss_kernel, dim=1, inputs=[params[0], loss], device=device)
        return loss

    return loss_fn


def make_quadratic_loss(device):
    def loss_fn(params):
        loss = wp.zeros(1, dtype=float, requires_grad=True, device=device)
        wp.launch(quadratic_loss_kernel, dim=1, inputs=[params[0], loss], device=device)
        return loss

    return loss_fn


def test_score_function_estimator_detects_branch_gradient(test, device):
    x = wp.array([0.0], dtype=float, requires_grad=True, device=device)

    result = warp.optim.smoothing.estimate_score_function(
        make_step_loss(device),
        [x],
        sigma=0.25,
        samples=20000,
        seed=7,
        antithetic=True,
    )

    expected = 1.0 / (0.25 * math.sqrt(2.0 * math.pi))

    test.assertEqual(result.method, "score_function")
    test.assertEqual(result.samples, 20000)
    test.assertAlmostEqual(result.value, 0.5, delta=0.02)
    test.assertAlmostEqual(float(result.gradients[0].numpy()[0]), expected, delta=0.08)
    test.assertGreater(float(result.gradient_variance[0].numpy()[0]), 0.0)


def test_pathwise_estimator_matches_smooth_quadratic_gradient(test, device):
    x = wp.array([2.0, -3.0], dtype=float, requires_grad=True, device=device)

    result = warp.optim.smoothing.estimate_pathwise(
        make_quadratic_loss(device),
        [x],
        sigma=0.0,
        samples=1,
        seed=4,
    )

    np.testing.assert_allclose(result.gradients[0].numpy(), np.array([4.0, -3.0]), rtol=1.0e-6, atol=1.0e-6)
    test.assertAlmostEqual(result.value, 8.5, places=6)
    test.assertEqual(result.method, "pathwise")


def test_finite_difference_estimator_matches_smooth_quadratic_gradient(test, device):
    x = wp.array([2.0, -3.0], dtype=float, requires_grad=True, device=device)

    result = warp.optim.smoothing.estimate_finite_difference(
        make_quadratic_loss(device),
        [x],
        sigma=0.0,
        epsilon=1.0e-3,
        samples=1,
        seed=4,
    )

    np.testing.assert_allclose(result.gradients[0].numpy(), np.array([4.0, -3.0]), rtol=1.0e-3, atol=1.0e-3)
    test.assertAlmostEqual(result.value, 8.5, places=6)
    test.assertEqual(result.method, "finite_difference")


devices = get_test_devices()


class TestProgramSmoothing(unittest.TestCase):
    pass


add_function_test(
    TestProgramSmoothing,
    "test_score_function_estimator_detects_branch_gradient",
    test_score_function_estimator_detects_branch_gradient,
    devices=devices,
)
add_function_test(
    TestProgramSmoothing,
    "test_pathwise_estimator_matches_smooth_quadratic_gradient",
    test_pathwise_estimator_matches_smooth_quadratic_gradient,
    devices=devices,
)
add_function_test(
    TestProgramSmoothing,
    "test_finite_difference_estimator_matches_smooth_quadratic_gradient",
    test_finite_difference_estimator_matches_smooth_quadratic_gradient,
    devices=devices,
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
