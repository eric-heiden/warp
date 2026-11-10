# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime-indexed reads of vector/matrix fields of struct arguments (``s.vec[i]``, ``s.mat[i, j]``)."""

import unittest

import numpy as np

import warp as wp
from warp.tests.unittest_utils import *

vec29 = wp.types.vector(length=29, dtype=float)
vec29i = wp.types.vector(length=29, dtype=int)
mat56 = wp.types.matrix(shape=(5, 6), dtype=float)


@wp.struct
class Tables:
    scale: vec29
    ids: vec29i
    m: mat56


@wp.kernel(enable_backward=False, module="unique")
def gather_fwd(t: Tables, out: wp.array2d[float]):
    i, j = wp.tid()
    # runtime indices into struct fields, including negative ones
    out[i, j] = t.scale[j] * float(t.ids[(j + i) % 29]) + t.m[j % 5, -1 - (j % 6)]


@wp.kernel(module="unique")
def gather_int_backward_enabled(t: Tables, out: wp.array2d[int]):
    i, j = wp.tid()
    out[i, j] = t.ids[j] + t.ids[-1 - j]


@wp.kernel(module="unique")
def weighted(t: Tables, x: wp.array[float], y: wp.array[float]):
    j = wp.tid()
    # float field with backward enabled: must keep its gradient path
    wp.atomic_add(y, 0, t.scale[j] * x[j])


def _tables():
    t = Tables()
    rng = np.random.default_rng(0)
    scale = rng.standard_normal(29).astype(np.float32)
    ids = rng.integers(0, 100, 29).astype(np.int32)
    m = rng.standard_normal((5, 6)).astype(np.float32)
    t.scale, t.ids, t.m = vec29(*scale), vec29i(*ids), mat56(*m.ravel())
    return t, scale, ids, m


def test_struct_vec_runtime_index(test, device):
    t, scale, ids, m = _tables()
    out = wp.zeros((4, 29), dtype=float, device=device)
    wp.launch(gather_fwd, out.shape, [t, out], device=device)
    i, j = np.meshgrid(np.arange(4), np.arange(29), indexing="ij")
    ref = scale[j] * ids[(j + i) % 29] + m[j % 5, (-1 - (j % 6)) % 6]
    np.testing.assert_allclose(out.numpy(), ref.astype(np.float32), rtol=1e-6, atol=1e-6)

    outi = wp.zeros((2, 29), dtype=int, device=device)
    wp.launch(gather_int_backward_enabled, outi.shape, [t, outi], device=device)
    assert_np_equal(outi.numpy(), np.tile(ids + ids[::-1], (2, 1)))


def test_struct_vec_runtime_index_grad(test, device):
    t, scale, _, _ = _tables()
    x = wp.array(np.ones(29, dtype=np.float32), device=device, requires_grad=True)
    y = wp.zeros(1, dtype=float, device=device, requires_grad=True)
    tape = wp.Tape()
    with tape:
        wp.launch(weighted, 29, [t, x, y], device=device)
    tape.backward(grads={y: wp.ones(1, dtype=float, device=device)})
    assert_np_equal(x.grad.numpy(), scale, tol=1e-6)


devices = get_test_devices()


class TestStructVecIndex(unittest.TestCase):
    pass


add_function_test(TestStructVecIndex, "test_struct_vec_runtime_index", test_struct_vec_runtime_index, devices=devices)
add_function_test(
    TestStructVecIndex, "test_struct_vec_runtime_index_grad", test_struct_vec_runtime_index_grad, devices=devices
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
