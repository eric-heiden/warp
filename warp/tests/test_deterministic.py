# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for deterministic execution mode.

Validates that deterministic modes produce bit-exact reproducible results for
atomic operations across multiple runs.
"""

import re
import unittest
from pathlib import Path
from typing import Any

import numpy as np

import warp as wp
from warp._src import deterministic as wp_deterministic
from warp.tests.unittest_utils import *


def _reference_scatter_add_float32(data_np, indices_np, out_size):
    """Compute the canonical left-to-right float32 scatter reduction on CPU."""
    expected = np.zeros(out_size, dtype=np.float32)
    for value, index in zip(data_np, indices_np, strict=True):
        expected[index] = np.float32(expected[index] + value)
    return expected


# ---------------------------------------------------------------------------
# Pattern A kernels: accumulation (return value unused)
# ---------------------------------------------------------------------------


@wp.kernel
def scatter_add_kernel(
    data: wp.array(dtype=wp.float32),
    dest_indices: wp.array(dtype=wp.int32),
    output: wp.array(dtype=wp.float32),
):
    """Each thread atomically adds to output[dest_indices[tid]]."""
    tid = wp.tid()
    idx = dest_indices[tid]
    wp.atomic_add(output, idx, data[tid])


@wp.kernel
def augassign_add_kernel(
    data: wp.array(dtype=wp.float32),
    dest_indices: wp.array(dtype=wp.int32),
    output: wp.array(dtype=wp.float32),
):
    """Same as scatter_add_kernel but using += syntax."""
    tid = wp.tid()
    idx = dest_indices[tid]
    output[idx] += data[tid]


@wp.kernel
def multi_array_atomic_kernel(
    data: wp.array(dtype=wp.float32),
    dest_indices: wp.array(dtype=wp.int32),
    out_a: wp.array(dtype=wp.float32),
    out_b: wp.array(dtype=wp.float32),
    out_c: wp.array(dtype=wp.float32),
):
    """Atomic add to three different output arrays from the same kernel."""
    tid = wp.tid()
    idx = dest_indices[tid]
    val = data[tid]
    wp.atomic_add(out_a, idx, val)
    wp.atomic_add(out_b, idx, val * 2.0)
    wp.atomic_add(out_c, idx, val * 3.0)


@wp.kernel
def atomic_sub_kernel(
    data: wp.array(dtype=wp.float32),
    dest_indices: wp.array(dtype=wp.int32),
    output: wp.array(dtype=wp.float32),
):
    """Atomic sub test."""
    tid = wp.tid()
    idx = dest_indices[tid]
    wp.atomic_sub(output, idx, data[tid])


@wp.kernel
def atomic_add_2d_kernel(
    data: wp.array(dtype=wp.float32),
    row_indices: wp.array(dtype=wp.int32),
    col_indices: wp.array(dtype=wp.int32),
    output: wp.array2d(dtype=wp.float32),
):
    """Atomic add to a 2D array."""
    tid = wp.tid()
    r = row_indices[tid]
    c = col_indices[tid]
    wp.atomic_add(output, r, c, data[tid])


@wp.kernel(deterministic=True, module="unique")
def atomic_add_array_view_kernel(
    data: wp.array2d(dtype=wp.float32),
    output: wp.array2d(dtype=wp.float32),
):
    """Atomic add through a leading-index array view."""
    row, col = wp.tid()
    out_row = output[row]
    wp.atomic_add(out_row, col % 8, data[row, col])


@wp.kernel
def atomic_double_kernel(
    data: wp.array(dtype=wp.float64),
    dest_indices: wp.array(dtype=wp.int32),
    output: wp.array(dtype=wp.float64),
):
    """Atomic add with float64."""
    tid = wp.tid()
    idx = dest_indices[tid]
    wp.atomic_add(output, idx, data[tid])


@wp.kernel
def vec3_scatter_add_kernel(
    data: wp.array(dtype=wp.vec3),
    dest_indices: wp.array(dtype=wp.int32),
    output: wp.array(dtype=wp.vec3),
):
    """Atomic add with ``wp.vec3`` values."""
    tid = wp.tid()
    wp.atomic_add(output, dest_indices[tid], data[tid])


@wp.kernel
def vec3_atomic_minmax_kernel(
    points: wp.array(dtype=wp.vec3),
    out_min: wp.array(dtype=wp.vec3),
    out_max: wp.array(dtype=wp.vec3),
):
    """Component-wise deterministic min/max for bounding-box style reductions."""
    tid = wp.tid()
    p = points[tid]
    wp.atomic_min(out_min, 0, p)
    wp.atomic_max(out_max, 0, p)


@wp.kernel
def mat33_scatter_add_kernel(
    data: wp.array(dtype=wp.mat33),
    dest_indices: wp.array(dtype=wp.int32),
    output: wp.array(dtype=wp.mat33),
):
    """Atomic add with ``wp.mat33`` values."""
    tid = wp.tid()
    wp.atomic_add(output, dest_indices[tid], data[tid])


@wp.kernel(deterministic="gpu_to_gpu", deterministic_max_records=4096)
def decorator_deterministic_kernel(
    data: wp.array(dtype=wp.float32),
    output: wp.array(dtype=wp.float32),
):
    """Kernel-level GPU-to-GPU deterministic flag without module options."""
    tid = wp.tid()
    wp.atomic_add(output, tid % 8, data[tid])


@wp.func
def _det_closure_transform_a(x: wp.float32) -> wp.float32:
    return x + wp.float32(1.0)


@wp.func
def _det_closure_transform_b(x: wp.float32) -> wp.float32:
    return x + wp.float32(2.0)


@wp.func
def _det_helper_atomic_add(output: wp.array(dtype=wp.float32), index: int, value: wp.float32):
    wp.atomic_add(output, index, value)


@wp.kernel(deterministic=True, module="unique")
def helper_atomic_add_array_view_kernel(
    data: wp.array2d(dtype=wp.float32),
    output: wp.array2d(dtype=wp.float32),
):
    """Atomic add through a leading-index array view passed to a helper."""
    row, col = wp.tid()
    _det_helper_atomic_add(output[row], col % 8, data[row, col])


@wp.func
def _det_helper_counter_alloc(counter: wp.array(dtype=wp.int32)) -> int:
    return wp.atomic_add(counter, 0, 1)


@wp.func
def _det_func_scatter_add_leaf(arr: wp.array(dtype=wp.float32), idx: int, value: wp.float32):
    wp.atomic_add(arr, idx, value)


@wp.func
def _det_func_scatter_add_wrapper(dst: wp.array(dtype=wp.float32), idx: int, value: wp.float32):
    _det_func_scatter_add_leaf(dst, idx, value)


@wp.struct
class _DetStructCounterWriter:
    counter: wp.array(dtype=wp.int32)
    output: wp.array(dtype=wp.float32)


@wp.func
def _det_struct_counter_write(writer: _DetStructCounterWriter, value: wp.float32):
    slot = wp.atomic_add(writer.counter, 0, 1)
    writer.output[slot] = value


@wp.func
def _det_nested_struct_counter_write(writer: _DetStructCounterWriter, value: wp.float32):
    _det_struct_counter_write(writer, value)


@wp.func
def _det_struct_counter_write_optional_index(writer: _DetStructCounterWriter, value: wp.float32, output_index: int):
    index = output_index
    if output_index < 0:
        index = wp.atomic_add(writer.counter, 0, 1)
    writer.output[index] = value


def _make_deterministic_closure_kernel(transform_func):
    @wp.kernel(deterministic=True, module="unique")
    def _deterministic_closure_kernel(
        data: wp.array(dtype=wp.float32),
        output: wp.array(dtype=wp.float32),
    ):
        tid = wp.tid()
        wp.atomic_add(output, tid % 8, transform_func(data[tid]))

    return _deterministic_closure_kernel


def _make_deterministic_helper_scatter_kernel(transform_func):
    @wp.kernel(deterministic=True, module="unique")
    def _deterministic_helper_scatter_kernel(
        data: wp.array(dtype=wp.float32),
        output: wp.array(dtype=wp.float32),
    ):
        tid = wp.tid()
        _det_helper_atomic_add(output, tid % 8, transform_func(data[tid]))

    return _deterministic_helper_scatter_kernel


@wp.kernel(deterministic=True, module="unique")
def helper_counter_kernel(
    data: wp.array(dtype=wp.float32),
    flags: wp.array(dtype=wp.int32),
    counter: wp.array(dtype=wp.int32),
    output: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    if flags[tid] != 0:
        slot = _det_helper_counter_alloc(counter)
        output[slot] = data[tid]


@wp.kernel(deterministic=True, module="unique")
def struct_field_counter_kernel(
    data: wp.array(dtype=wp.float32),
    counts: wp.array(dtype=wp.int32),
    writer: _DetStructCounterWriter,
):
    tid = wp.tid()
    count = counts[tid]
    if count > 0:
        base = wp.atomic_add(writer.counter, 0, count)
        for i in range(count):
            writer.output[base + i] = data[tid] + wp.float32(i) * wp.float32(0.5)


@wp.kernel(deterministic=True, module="unique")
def struct_field_helper_counter_kernel(
    data: wp.array(dtype=wp.float32),
    flags: wp.array(dtype=wp.int32),
    writer: _DetStructCounterWriter,
):
    tid = wp.tid()
    if flags[tid] != 0:
        _det_struct_counter_write(writer, data[tid])


@wp.kernel(deterministic=True, module="unique")
def generic_nested_struct_field_helper_counter_kernel(
    data: wp.array(dtype=Any),
    flags: wp.array(dtype=wp.int32),
    writer: _DetStructCounterWriter,
):
    tid = wp.tid()
    if flags[tid] != 0:
        _det_nested_struct_counter_write(writer, data[tid])


@wp.kernel(deterministic=True, module="unique")
def any_struct_field_counter_kernel(
    data: wp.array(dtype=wp.float32),
    counts: wp.array(dtype=wp.int32),
    writer: Any,
):
    tid = wp.tid()
    count = counts[tid]
    if count > 0:
        base = wp.atomic_add(writer.counter, 0, count)
        for i in range(count):
            writer.output[base + i] = data[tid] + wp.float32(i) * wp.float32(0.5)


@wp.kernel(deterministic=True, module="unique")
def any_struct_field_nested_counter_kernel(
    data: wp.array(dtype=wp.float32),
    counts: wp.array(dtype=wp.int32),
    writer: Any,
):
    tid = wp.tid()
    count = counts[tid]
    if count > 0:
        base = wp.atomic_add(writer.counter, 0, count)
        for i in range(count):
            _det_struct_counter_write_optional_index(
                writer,
                data[tid] + wp.float32(i) * wp.float32(0.5),
                base + i,
            )


@wp.kernel
def func_scatter_add_kernel(
    data: wp.array(dtype=wp.float32),
    dest_indices: wp.array(dtype=wp.int32),
    output: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    _det_func_scatter_add_leaf(output, dest_indices[tid], data[tid])


@wp.kernel
def nested_func_scatter_add_kernel(
    data: wp.array(dtype=wp.float32),
    dest_indices: wp.array(dtype=wp.int32),
    accum: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    _det_func_scatter_add_wrapper(accum, dest_indices[tid], data[tid])


@wp.kernel
def triple_scatter_add_kernel(
    data: wp.array(dtype=wp.float32),
    output: wp.array(dtype=wp.float32),
):
    """Emit three deterministic scatter records per thread to the same target."""
    tid = wp.tid()
    val = data[tid]
    wp.atomic_add(output, 0, val)
    wp.atomic_add(output, 0, val * 2.0)
    wp.atomic_add(output, 0, val * 3.0)


@wp.kernel(deterministic=True, deterministic_max_records=4)
def loop_scatter_add_kernel(
    data: wp.array(dtype=wp.float32),
    counts: wp.array(dtype=wp.int32),
    output: wp.array(dtype=wp.float32),
):
    """Emit a data-dependent number of scatter records to the same target."""
    tid = wp.tid()
    val = data[tid]
    count = counts[tid]
    for _ in range(count):
        wp.atomic_add(output, 0, val)


@wp.kernel(module="unique")
def mixed_reduce_op_same_array_kernel(
    data: wp.array(dtype=wp.float32),
    output: wp.array(dtype=wp.float32),
):
    """Apply different atomic reductions to the same destination array."""
    tid = wp.tid()
    wp.atomic_add(output, 0, data[tid])
    wp.atomic_max(output, 0, 1.0)


# ---------------------------------------------------------------------------
# Pattern B kernels: counter/allocator (return value used)
# ---------------------------------------------------------------------------


@wp.kernel
def counter_kernel(
    data: wp.array(dtype=wp.float32),
    counter: wp.array(dtype=wp.int32),
    output: wp.array(dtype=wp.float32),
):
    """Allocate a slot and write data to it."""
    tid = wp.tid()
    slot = wp.atomic_add(counter, 0, 1)
    output[slot] = data[tid]


@wp.kernel
def conditional_counter_kernel(
    data: wp.array(dtype=wp.float32),
    threshold: wp.float32,
    counter: wp.array(dtype=wp.int32),
    output: wp.array(dtype=wp.float32),
):
    """Stream compaction: only emit elements above threshold."""
    tid = wp.tid()
    val = data[tid]
    if val > threshold:
        slot = wp.atomic_add(counter, 0, 1)
        output[slot] = val


@wp.kernel
def counter_side_effect_kernel(
    counter: wp.array(dtype=wp.int32),
    output: wp.array(dtype=wp.float32),
    scratch: wp.array(dtype=wp.float32),
):
    """Counter kernel with a normal array write that must not execute in phase 0."""
    tid = wp.tid()
    slot = wp.atomic_add(counter, 0, 1)
    scratch[tid] = scratch[tid] + 1.0
    output[slot] = float(tid)


# ---------------------------------------------------------------------------
# Mixed kernels: both patterns in one kernel
# ---------------------------------------------------------------------------


@wp.kernel
def mixed_pattern_kernel(
    data: wp.array(dtype=wp.float32),
    counter: wp.array(dtype=wp.int32),
    output: wp.array(dtype=wp.float32),
    accum: wp.array(dtype=wp.float32),
):
    """Counter allocation + accumulation in the same kernel."""
    tid = wp.tid()
    val = data[tid]

    # Pattern B: allocate slot
    slot = wp.atomic_add(counter, 0, 1)
    output[slot] = val

    # Pattern A: accumulate
    wp.atomic_add(accum, tid % 8, val)


# ---------------------------------------------------------------------------
# Integer atomic kernels (should pass through unchanged when return unused)
# ---------------------------------------------------------------------------


@wp.kernel
def int_atomic_add_kernel(
    dest_indices: wp.array(dtype=wp.int32),
    output: wp.array(dtype=wp.int32),
):
    """Integer atomic add (should be deterministic without transformation)."""
    tid = wp.tid()
    idx = dest_indices[tid]
    wp.atomic_add(output, idx, 1)


# ---------------------------------------------------------------------------
# Test functions
# ---------------------------------------------------------------------------


def test_scatter_add_reproducibility(test, device):
    """Verify that float atomic_add produces bit-exact identical results across runs."""
    if device.is_cpu:
        test.skipTest("CPU execution is already deterministic")

    n = 4096
    out_size = 64
    rng = np.random.default_rng(42)

    data_np = rng.random(n, dtype=np.float32)
    indices_np = rng.integers(0, out_size, size=n, dtype=np.int32)

    data = wp.array(data_np, dtype=wp.float32, device=device)
    indices = wp.array(indices_np, dtype=wp.int32, device=device)

    results = []
    for _ in range(10):
        output = wp.zeros(out_size, dtype=wp.float32, device=device)
        wp.launch(
            scatter_add_kernel,
            dim=n,
            inputs=[data, indices],
            outputs=[output],
            device=device,
        )
        results.append(output.numpy().copy())

    # All runs must produce bit-exact identical results.
    for i in range(1, len(results)):
        np.testing.assert_array_equal(
            results[0],
            results[i],
            err_msg=f"Run 0 vs run {i} differ (deterministic mode should be bit-exact)",
        )


def test_gpu_to_gpu_mode_reproducibility(test, device):
    """Verify the global ``gpu_to_gpu`` mode produces reproducible results."""
    if device.is_cpu:
        test.skipTest("CPU execution is already deterministic")

    n = 1024
    out_size = 16
    rng = np.random.default_rng(44)

    data_np = rng.random(n, dtype=np.float32)
    indices_np = rng.integers(0, out_size, size=n, dtype=np.int32)

    data = wp.array(data_np, dtype=wp.float32, device=device)
    indices = wp.array(indices_np, dtype=wp.int32, device=device)

    old_det = wp.config.deterministic
    try:
        wp.config.deterministic = "gpu_to_gpu"
        results = []
        for _ in range(3):
            output = wp.zeros(out_size, dtype=wp.float32, device=device)
            wp.launch(scatter_add_kernel, dim=n, inputs=[data, indices], outputs=[output], device=device)
            results.append(output.numpy().copy())
        for i in range(1, len(results)):
            np.testing.assert_array_equal(results[0], results[i])
    finally:
        wp.config.deterministic = old_det


def test_gpu_to_gpu_matches_canonical_float32_reference(test, device):
    """Verify ``gpu_to_gpu`` matches the canonical float32 CPU reduction order."""
    if device.is_cpu:
        test.skipTest("CPU execution is already deterministic")

    data_np = np.array(
        [
            1.0e20,
            1.0,
            -1.0e20,
            3.5,
            -2.25,
            2.0**-20,
            1.0e10,
            -1.0e10,
            7.0,
            -7.0,
            9.0,
            1.0e-7,
        ],
        dtype=np.float32,
    )
    indices_np = np.array([0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 2, 2], dtype=np.int32)
    out_size = 3

    expected = _reference_scatter_add_float32(data_np, indices_np, out_size)

    data = wp.array(data_np, dtype=wp.float32, device=device)
    indices = wp.array(indices_np, dtype=wp.int32, device=device)

    old_det = wp.config.deterministic
    try:
        wp.config.deterministic = "gpu_to_gpu"
        output = wp.zeros(out_size, dtype=wp.float32, device=device)
        wp.launch(scatter_add_kernel, dim=data_np.shape[0], inputs=[data, indices], outputs=[output], device=device)
        result = output.numpy()
    finally:
        wp.config.deterministic = old_det

    np.testing.assert_array_equal(result.view(np.uint32), expected.view(np.uint32))


def test_augassign_add_reproducibility(test, device):
    """Verify += syntax (desugars to atomic_add) is also deterministic."""
    if device.is_cpu:
        test.skipTest("CPU execution is already deterministic")

    n = 2048
    out_size = 32
    rng = np.random.default_rng(123)

    data_np = rng.random(n, dtype=np.float32)
    indices_np = rng.integers(0, out_size, size=n, dtype=np.int32)

    data = wp.array(data_np, dtype=wp.float32, device=device)
    indices = wp.array(indices_np, dtype=wp.int32, device=device)

    results = []
    for _ in range(10):
        output = wp.zeros(out_size, dtype=wp.float32, device=device)
        wp.launch(
            augassign_add_kernel,
            dim=n,
            inputs=[data, indices],
            outputs=[output],
            device=device,
        )
        results.append(output.numpy().copy())

    for i in range(1, len(results)):
        np.testing.assert_array_equal(results[0], results[i])


def test_scatter_add_correctness(test, device):
    """Compare deterministic GPU results against CPU sequential execution."""
    n = 2048
    out_size = 32
    rng = np.random.default_rng(99)

    data_np = rng.random(n, dtype=np.float32)
    indices_np = rng.integers(0, out_size, size=n, dtype=np.int32)

    # CPU sequential reference (guaranteed deterministic).
    expected = np.zeros(out_size, dtype=np.float32)
    for i in range(n):
        expected[indices_np[i]] += data_np[i]

    data = wp.array(data_np, dtype=wp.float32, device=device)
    indices = wp.array(indices_np, dtype=wp.int32, device=device)
    output = wp.zeros(out_size, dtype=wp.float32, device=device)

    wp.launch(
        scatter_add_kernel,
        dim=n,
        inputs=[data, indices],
        outputs=[output],
        device=device,
    )

    result = output.numpy()
    # Deterministic sum order may differ from Python loop order, so exact
    # match is not guaranteed.  Check within reasonable tolerance.
    np.testing.assert_allclose(result, expected, rtol=1e-4, atol=1e-5)


def test_multi_array_atomic(test, device):
    """Verify deterministic mode works with multiple target arrays."""
    if device.is_cpu:
        test.skipTest("CPU execution is already deterministic")

    n = 1024
    out_size = 16
    rng = np.random.default_rng(77)

    data_np = rng.random(n, dtype=np.float32)
    indices_np = rng.integers(0, out_size, size=n, dtype=np.int32)

    data = wp.array(data_np, dtype=wp.float32, device=device)
    indices = wp.array(indices_np, dtype=wp.int32, device=device)

    results_a, results_b, results_c = [], [], []
    for _ in range(5):
        out_a = wp.zeros(out_size, dtype=wp.float32, device=device)
        out_b = wp.zeros(out_size, dtype=wp.float32, device=device)
        out_c = wp.zeros(out_size, dtype=wp.float32, device=device)
        wp.launch(
            multi_array_atomic_kernel,
            dim=n,
            inputs=[data, indices],
            outputs=[out_a, out_b, out_c],
            device=device,
        )
        results_a.append(out_a.numpy().copy())
        results_b.append(out_b.numpy().copy())
        results_c.append(out_c.numpy().copy())

    for i in range(1, len(results_a)):
        np.testing.assert_array_equal(results_a[0], results_a[i])
        np.testing.assert_array_equal(results_b[0], results_b[i])
        np.testing.assert_array_equal(results_c[0], results_c[i])


def test_atomic_sub_deterministic(test, device):
    """Verify atomic_sub is deterministic."""
    if device.is_cpu:
        test.skipTest("CPU execution is already deterministic")

    n = 2048
    out_size = 32
    rng = np.random.default_rng(55)

    data_np = rng.random(n, dtype=np.float32)
    indices_np = rng.integers(0, out_size, size=n, dtype=np.int32)

    data = wp.array(data_np, dtype=wp.float32, device=device)
    indices = wp.array(indices_np, dtype=wp.int32, device=device)

    results = []
    for _ in range(5):
        output = wp.zeros(out_size, dtype=wp.float32, device=device)
        wp.launch(
            atomic_sub_kernel,
            dim=n,
            inputs=[data, indices],
            outputs=[output],
            device=device,
        )
        results.append(output.numpy().copy())

    for i in range(1, len(results)):
        np.testing.assert_array_equal(results[0], results[i])


def test_atomic_add_2d(test, device):
    """Verify deterministic mode with 2D array indexing."""
    if device.is_cpu:
        test.skipTest("CPU execution is already deterministic")

    n = 1024
    rows, cols = 8, 8
    rng = np.random.default_rng(88)

    data_np = rng.random(n, dtype=np.float32)
    row_np = rng.integers(0, rows, size=n, dtype=np.int32)
    col_np = rng.integers(0, cols, size=n, dtype=np.int32)

    data = wp.array(data_np, dtype=wp.float32, device=device)
    row_idx = wp.array(row_np, dtype=wp.int32, device=device)
    col_idx = wp.array(col_np, dtype=wp.int32, device=device)

    results = []
    for _ in range(5):
        output = wp.zeros(shape=(rows, cols), dtype=wp.float32, device=device)
        wp.launch(
            atomic_add_2d_kernel,
            dim=n,
            inputs=[data, row_idx, col_idx],
            outputs=[output],
            device=device,
        )
        results.append(output.numpy().copy())

    for i in range(1, len(results)):
        np.testing.assert_array_equal(results[0], results[i])


def test_atomic_add_array_view(test, device):
    """Verify deterministic scatter atomics through array views use root-array indices."""
    if device.is_cpu:
        test.skipTest("CPU execution is already deterministic")

    rows, cols, bins = 4, 128, 8
    rng = np.random.default_rng(1355)
    data_np = rng.random((rows, cols), dtype=np.float32)

    expected = np.zeros((rows, bins), dtype=np.float32)
    for row in range(rows):
        for col in range(cols):
            expected[row, col % bins] = np.float32(expected[row, col % bins] + data_np[row, col])

    data = wp.array(data_np, dtype=wp.float32, device=device)

    results = []
    for _ in range(5):
        output = wp.zeros(shape=(rows, bins), dtype=wp.float32, device=device)
        wp.launch(
            atomic_add_array_view_kernel,
            dim=(rows, cols),
            inputs=[data],
            outputs=[output],
            device=device,
        )
        results.append(output.numpy().copy())

    np.testing.assert_allclose(results[0], expected, rtol=1.0e-6, atol=1.0e-6)
    for i in range(1, len(results)):
        np.testing.assert_array_equal(results[0], results[i])


def test_helper_atomic_add_array_view(test, device):
    """Verify helper scatter atomics through array views keep caller root-array offsets."""
    if device.is_cpu:
        test.skipTest("CPU execution is already deterministic")

    rows, cols, bins = 4, 128, 8
    rng = np.random.default_rng(1356)
    data_np = rng.random((rows, cols), dtype=np.float32)

    expected = np.zeros((rows, bins), dtype=np.float32)
    for row in range(rows):
        for col in range(cols):
            expected[row, col % bins] = np.float32(expected[row, col % bins] + data_np[row, col])

    data = wp.array(data_np, dtype=wp.float32, device=device)

    results = []
    for _ in range(5):
        output = wp.zeros(shape=(rows, bins), dtype=wp.float32, device=device)
        wp.launch(
            helper_atomic_add_array_view_kernel,
            dim=(rows, cols),
            inputs=[data],
            outputs=[output],
            device=device,
        )
        results.append(output.numpy().copy())

    np.testing.assert_allclose(results[0], expected, rtol=1.0e-6, atol=1.0e-6)
    for i in range(1, len(results)):
        np.testing.assert_array_equal(results[0], results[i])


def test_atomic_double_deterministic(test, device):
    """Verify deterministic mode with float64 atomics."""
    if device.is_cpu:
        test.skipTest("CPU execution is already deterministic")

    n = 2048
    out_size = 32
    rng = np.random.default_rng(66)

    data_np = rng.random(n).astype(np.float64)
    indices_np = rng.integers(0, out_size, size=n, dtype=np.int32)

    data = wp.array(data_np, dtype=wp.float64, device=device)
    indices = wp.array(indices_np, dtype=wp.int32, device=device)

    results = []
    for _ in range(5):
        output = wp.zeros(out_size, dtype=wp.float64, device=device)
        wp.launch(
            atomic_double_kernel,
            dim=n,
            inputs=[data, indices],
            outputs=[output],
            device=device,
        )
        results.append(output.numpy().copy())

    for i in range(1, len(results)):
        np.testing.assert_array_equal(results[0], results[i])


def test_vec3_atomic_add_deterministic(test, device):
    """Verify deterministic mode for composite ``wp.vec3`` atomic adds."""
    if device.is_cpu:
        test.skipTest("CPU execution is already deterministic")

    n = 1024
    out_size = 16
    rng = np.random.default_rng(67)

    data_np = rng.standard_normal((n, 3), dtype=np.float32)
    indices_np = rng.integers(0, out_size, size=n, dtype=np.int32)

    data = wp.array(data_np, dtype=wp.vec3, device=device)
    indices = wp.array(indices_np, dtype=wp.int32, device=device)

    results = []
    for _ in range(5):
        output = wp.zeros(out_size, dtype=wp.vec3, device=device)
        wp.launch(vec3_scatter_add_kernel, dim=n, inputs=[data, indices], outputs=[output], device=device)
        results.append(output.numpy().copy())

    expected = np.zeros((out_size, 3), dtype=np.float32)
    for i in range(n):
        expected[indices_np[i]] += data_np[i]

    for result in results:
        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)
    for i in range(1, len(results)):
        np.testing.assert_array_equal(results[0], results[i])


def test_vec3_atomic_minmax_deterministic(test, device):
    """Verify deterministic component-wise ``wp.vec3`` min/max reductions."""
    if device.is_cpu:
        test.skipTest("CPU execution is already deterministic")

    n = 2048
    rng = np.random.default_rng(68)
    points_np = rng.standard_normal((n, 3), dtype=np.float32)
    points = wp.array(points_np, dtype=wp.vec3, device=device)

    mins = []
    maxs = []
    for _ in range(5):
        out_min = wp.empty(1, dtype=wp.vec3, device=device)
        out_max = wp.empty(1, dtype=wp.vec3, device=device)
        out_min.fill_(wp.vec3(np.inf, np.inf, np.inf))
        out_max.fill_(wp.vec3(-np.inf, -np.inf, -np.inf))
        wp.launch(vec3_atomic_minmax_kernel, dim=n, inputs=[points], outputs=[out_min, out_max], device=device)
        mins.append(out_min.numpy().copy())
        maxs.append(out_max.numpy().copy())

    expected_min = np.min(points_np, axis=0, keepdims=True)
    expected_max = np.max(points_np, axis=0, keepdims=True)

    for result in mins:
        np.testing.assert_allclose(result, expected_min, rtol=0.0, atol=0.0)
    for result in maxs:
        np.testing.assert_allclose(result, expected_max, rtol=0.0, atol=0.0)
    for i in range(1, len(mins)):
        np.testing.assert_array_equal(mins[0], mins[i])
        np.testing.assert_array_equal(maxs[0], maxs[i])


def test_mat33_atomic_add_deterministic(test, device):
    """Verify deterministic mode for composite ``wp.mat33`` atomic adds."""
    if device.is_cpu:
        test.skipTest("CPU execution is already deterministic")

    n = 512
    out_size = 8
    rng = np.random.default_rng(69)

    data_np = rng.standard_normal((n, 3, 3), dtype=np.float32)
    indices_np = rng.integers(0, out_size, size=n, dtype=np.int32)

    data = wp.array(data_np, dtype=wp.mat33, device=device)
    indices = wp.array(indices_np, dtype=wp.int32, device=device)

    results = []
    for _ in range(5):
        output = wp.zeros(out_size, dtype=wp.mat33, device=device)
        wp.launch(mat33_scatter_add_kernel, dim=n, inputs=[data, indices], outputs=[output], device=device)
        results.append(output.numpy().copy())

    expected = np.zeros((out_size, 3, 3), dtype=np.float32)
    for i in range(n):
        expected[indices_np[i]] += data_np[i]

    for result in results:
        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)
    for i in range(1, len(results)):
        np.testing.assert_array_equal(results[0], results[i])


def test_triple_scatter_capacity_estimate(test, device):
    """Verify kernels with >2 scatters per thread do not overflow the buffer."""
    if device.is_cpu:
        test.skipTest("CPU execution is already deterministic")

    n = 512
    rng = np.random.default_rng(12)
    data_np = rng.random(n, dtype=np.float32)
    data = wp.array(data_np, dtype=wp.float32, device=device)

    results = []
    for _ in range(3):
        output = wp.zeros(1, dtype=wp.float32, device=device)
        wp.launch(triple_scatter_add_kernel, dim=n, inputs=[data], outputs=[output], device=device)
        results.append(output.numpy().copy())

    expected = np.array([6.0 * data_np.sum()], dtype=np.float32)
    for result in results:
        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)
    for i in range(1, len(results)):
        np.testing.assert_array_equal(results[0], results[i])


def test_loop_scatter_max_records_override(test, device):
    """Verify ``deterministic_max_records`` handles dynamic loop emission counts."""
    if device.is_cpu:
        test.skipTest("CPU execution is already deterministic")

    n = 256
    rng = np.random.default_rng(71)
    data_np = rng.random(n, dtype=np.float32)
    counts_np = np.full(n, 4, dtype=np.int32)

    data = wp.array(data_np, dtype=wp.float32, device=device)
    counts = wp.array(counts_np, dtype=wp.int32, device=device)

    expected = np.array([np.dot(data_np, counts_np).astype(np.float32)], dtype=np.float32)

    results = []
    for _ in range(3):
        output = wp.zeros(1, dtype=wp.float32, device=device)
        wp.launch(loop_scatter_add_kernel, dim=n, inputs=[data, counts], outputs=[output], device=device)
        results.append(output.numpy().copy())

    for result in results:
        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)
    for i in range(1, len(results)):
        np.testing.assert_array_equal(results[0], results[i])


def test_mixed_reduce_ops_same_array(test, device):
    """Verify mixed reduction families on one array are rejected in deterministic mode."""
    if device.is_cpu:
        test.skipTest("CPU execution is already deterministic")

    data_np = np.full(4, 0.05, dtype=np.float32)
    data = wp.array(data_np, dtype=wp.float32, device=device)
    output = wp.zeros(1, dtype=wp.float32, device=device)

    with test.assertRaisesRegex(Exception, "does not support mixing"):
        wp.launch(
            mixed_reduce_op_same_array_kernel, dim=data_np.shape[0], inputs=[data], outputs=[output], device=device
        )


def test_counter_reproducibility(test, device):
    """Verify counter/allocator pattern produces deterministic slot assignments."""
    if device.is_cpu:
        test.skipTest("CPU execution is already deterministic")

    n = 1024
    rng = np.random.default_rng(33)
    data_np = rng.random(n, dtype=np.float32)
    data = wp.array(data_np, dtype=wp.float32, device=device)

    results = []
    for _ in range(10):
        counter = wp.zeros(1, dtype=wp.int32, device=device)
        output = wp.zeros(n, dtype=wp.float32, device=device)
        wp.launch(
            counter_kernel,
            dim=n,
            inputs=[data, counter],
            outputs=[output],
            device=device,
        )
        results.append(output.numpy().copy())

    for i in range(1, len(results)):
        np.testing.assert_array_equal(
            results[0],
            results[i],
            err_msg=f"Counter run 0 vs run {i} differ",
        )


def test_counter_phase0_suppresses_array_writes(test, device):
    """Verify non-counter array stores are skipped during the counting pass."""
    if device.is_cpu:
        test.skipTest("CPU execution is already deterministic")

    n = 128
    counter = wp.zeros(1, dtype=wp.int32, device=device)
    output = wp.zeros(n, dtype=wp.float32, device=device)
    scratch = wp.zeros(n, dtype=wp.float32, device=device)

    wp.launch(counter_side_effect_kernel, dim=n, inputs=[counter], outputs=[output, scratch], device=device)

    np.testing.assert_array_equal(scratch.numpy(), np.ones(n, dtype=np.float32))
    test.assertEqual(int(counter.numpy()[0]), n)


def test_counter_correctness(test, device):
    """Verify counter pattern writes all data (no lost elements)."""
    n = 512
    rng = np.random.default_rng(44)
    data_np = rng.random(n, dtype=np.float32)
    data = wp.array(data_np, dtype=wp.float32, device=device)

    counter = wp.zeros(1, dtype=wp.int32, device=device)
    output = wp.zeros(n, dtype=wp.float32, device=device)
    wp.launch(
        counter_kernel,
        dim=n,
        inputs=[data, counter],
        outputs=[output],
        device=device,
    )

    # Counter should equal n.
    count = int(counter.numpy()[0])
    test.assertEqual(count, n)

    # Output should contain a permutation of data.
    result = sorted(output.numpy().tolist())
    expected = sorted(data_np.tolist())
    np.testing.assert_allclose(result, expected, rtol=1e-6)


def test_conditional_counter(test, device):
    """Verify stream compaction pattern with conditional counter."""
    if device.is_cpu:
        test.skipTest("CPU execution is already deterministic")

    n = 2048
    threshold = 0.5
    rng = np.random.default_rng(55)
    data_np = rng.random(n, dtype=np.float32)
    expected_count = int(np.sum(data_np > threshold))

    data = wp.array(data_np, dtype=wp.float32, device=device)

    results = []
    counts = []
    for _ in range(5):
        counter = wp.zeros(1, dtype=wp.int32, device=device)
        output = wp.zeros(n, dtype=wp.float32, device=device)
        wp.launch(
            conditional_counter_kernel,
            dim=n,
            inputs=[data, threshold, counter],
            outputs=[output],
            device=device,
        )
        counts.append(int(counter.numpy()[0]))
        results.append(output.numpy()[:expected_count].copy())

    # Count should be correct.
    for c in counts:
        test.assertEqual(c, expected_count)

    # Results should be identical across runs.
    for i in range(1, len(results)):
        np.testing.assert_array_equal(results[0], results[i])


def test_mixed_pattern(test, device):
    """Verify kernel with both counter and accumulation atomics."""
    if device.is_cpu:
        test.skipTest("CPU execution is already deterministic")

    n = 512
    rng = np.random.default_rng(77)
    data_np = rng.random(n, dtype=np.float32)
    data = wp.array(data_np, dtype=wp.float32, device=device)

    results_out, results_accum = [], []
    for _ in range(5):
        counter = wp.zeros(1, dtype=wp.int32, device=device)
        output = wp.zeros(n, dtype=wp.float32, device=device)
        accum = wp.zeros(8, dtype=wp.float32, device=device)
        wp.launch(
            mixed_pattern_kernel,
            dim=n,
            inputs=[data, counter],
            outputs=[output, accum],
            device=device,
        )
        results_out.append(output.numpy().copy())
        results_accum.append(accum.numpy().copy())

    for i in range(1, len(results_out)):
        np.testing.assert_array_equal(results_out[0], results_out[i])
        np.testing.assert_array_equal(results_accum[0], results_accum[i])


def test_int_atomic_passthrough(test, device):
    """Verify integer atomics (return unused) work without overhead."""
    n = 1024
    out_size = 16
    rng = np.random.default_rng(11)
    indices_np = rng.integers(0, out_size, size=n, dtype=np.int32)
    indices = wp.array(indices_np, dtype=wp.int32, device=device)

    output = wp.zeros(out_size, dtype=wp.int32, device=device)
    wp.launch(
        int_atomic_add_kernel,
        dim=n,
        inputs=[indices],
        outputs=[output],
        device=device,
    )

    result = output.numpy()
    # Verify correctness: count of each index.
    expected = np.bincount(indices_np, minlength=out_size).astype(np.int32)
    np.testing.assert_array_equal(result, expected)


def test_module_option_override(test, device):
    """Verify per-module deterministic option works."""

    # Create a kernel with a per-module deterministic override.
    @wp.kernel(module_options={"deterministic": "gpu_to_gpu"}, module="unique")
    def per_kernel_det(
        data: wp.array(dtype=wp.float32),
        output: wp.array(dtype=wp.float32),
    ):
        tid = wp.tid()
        wp.atomic_add(output, tid % 4, data[tid])

    n = 256
    rng = np.random.default_rng(22)
    data_np = rng.random(n, dtype=np.float32)
    data = wp.array(data_np, dtype=wp.float32, device=device)

    # Ensure global config is disabled but per-kernel override still works.
    old_det = wp.config.deterministic
    try:
        wp.config.deterministic = "not_guaranteed"
        output = wp.zeros(4, dtype=wp.float32, device=device)
        wp.launch(per_kernel_det, dim=n, inputs=[data], outputs=[output], device=device)
        result = output.numpy()
        # Basic sanity: sum should be approximately correct.
        for bin_idx in range(4):
            mask = np.arange(n) % 4 == bin_idx
            expected_sum = data_np[mask].sum()
            np.testing.assert_allclose(result[bin_idx], expected_sum, rtol=1e-4)
    finally:
        wp.config.deterministic = old_det


def test_kernel_decorator_override(test, device):
    """Verify ``@wp.kernel(deterministic="gpu_to_gpu")`` works with global config off."""
    if device.is_cpu:
        test.skipTest("CPU execution is already deterministic")

    n = 512
    rng = np.random.default_rng(28)
    data_np = rng.random(n, dtype=np.float32)
    data = wp.array(data_np, dtype=wp.float32, device=device)

    old_det = wp.config.deterministic
    try:
        wp.config.deterministic = "not_guaranteed"
        results = []
        for _ in range(3):
            output = wp.zeros(8, dtype=wp.float32, device=device)
            wp.launch(decorator_deterministic_kernel, dim=n, inputs=[data], outputs=[output], device=device)
            results.append(output.numpy().copy())
        for i in range(1, len(results)):
            np.testing.assert_array_equal(results[0], results[i])
    finally:
        wp.config.deterministic = old_det


def test_deterministic_closure_kernel(test, device):
    """Verify deterministic closure kernels remain reproducible and distinct."""
    if device.is_cpu:
        test.skipTest("CPU execution is already deterministic")

    kernel_a = _make_deterministic_closure_kernel(_det_closure_transform_a)
    kernel_b = _make_deterministic_closure_kernel(_det_closure_transform_b)

    test.assertIsNot(kernel_a, kernel_b)
    test.assertNotEqual(kernel_a.module.name, kernel_b.module.name)

    n = 512
    rng = np.random.default_rng(30)
    data_np = rng.random(n, dtype=np.float32)
    data = wp.array(data_np, dtype=wp.float32, device=device)

    results_a = []
    results_b = []
    for _ in range(3):
        out_a = wp.zeros(8, dtype=wp.float32, device=device)
        out_b = wp.zeros(8, dtype=wp.float32, device=device)
        wp.launch(kernel_a, dim=n, inputs=[data], outputs=[out_a], device=device)
        wp.launch(kernel_b, dim=n, inputs=[data], outputs=[out_b], device=device)
        results_a.append(out_a.numpy().copy())
        results_b.append(out_b.numpy().copy())

    for i in range(1, len(results_a)):
        np.testing.assert_array_equal(results_a[0], results_a[i])
        np.testing.assert_array_equal(results_b[0], results_b[i])

    test.assertFalse(np.array_equal(results_a[0], results_b[0]))


def test_helper_function_scatter_atomic(test, device):
    """Verify helper-function scatter atomics stay deterministic for simple direct-array targets."""
    if device.is_cpu:
        test.skipTest("CPU execution is already deterministic")

    kernel = _make_deterministic_helper_scatter_kernel(_det_closure_transform_a)
    data_np = np.arange(32, dtype=np.float32)
    data = wp.array(data_np, dtype=wp.float32, device=device)

    expected = np.zeros(8, dtype=np.float32)
    for tid in range(data_np.shape[0]):
        bucket = tid % 8
        expected[bucket] = np.float32(expected[bucket] + (data_np[tid] + np.float32(1.0)))

    results = []
    for _ in range(3):
        output = wp.zeros(8, dtype=wp.float32, device=device)
        wp.launch(kernel, dim=32, inputs=[data], outputs=[output], device=device)
        results.append(output.numpy().copy())

    for i in range(1, len(results)):
        np.testing.assert_array_equal(results[0], results[i])
    np.testing.assert_array_equal(results[0].view(np.uint32), expected.view(np.uint32))


def test_helper_function_counter_atomic(test, device):
    """Verify helper-function counters remain deterministic for simple direct-array targets."""
    if device.is_cpu:
        test.skipTest("CPU execution is already deterministic")

    rng = np.random.default_rng(32)
    data_np = rng.random(64, dtype=np.float32)
    flags_np = (rng.random(64) > 0.4).astype(np.int32)
    data = wp.array(data_np, dtype=wp.float32, device=device)
    flags = wp.array(flags_np, dtype=wp.int32, device=device)
    expected = data_np[flags_np != 0]

    results = []
    counts = []
    for _ in range(3):
        counter = wp.zeros(1, dtype=wp.int32, device=device)
        output = wp.zeros(int(flags_np.sum()) + 4, dtype=wp.float32, device=device)
        wp.launch(helper_counter_kernel, dim=64, inputs=[data, flags], outputs=[counter, output], device=device)
        count = int(counter.numpy()[0])
        counts.append(count)
        results.append(output.numpy()[: expected.shape[0]].copy())

    np.testing.assert_array_equal(np.array(counts), np.full(3, expected.shape[0], dtype=np.int32))
    for i in range(1, len(results)):
        np.testing.assert_array_equal(results[0], results[i])
    np.testing.assert_array_equal(results[0].view(np.uint32), expected.view(np.uint32))


def test_struct_field_counter_atomic(test, device):
    """Verify deterministic counters work when the target array lives in a struct field."""
    if device.is_cpu:
        test.skipTest("CPU execution is already deterministic")

    rng = np.random.default_rng(34)
    data_np = rng.random(64, dtype=np.float32)
    counts_np = rng.integers(0, 4, size=64, dtype=np.int32)
    data = wp.array(data_np, dtype=wp.float32, device=device)
    counts_arr = wp.array(counts_np, dtype=wp.int32, device=device)
    expected = np.array(
        [
            np.float32(data_np[tid] + np.float32(i) * np.float32(0.5))
            for tid, count in enumerate(counts_np)
            for i in range(count)
        ],
        dtype=np.float32,
    )

    results = []
    counter_values = []
    for _ in range(3):
        writer = _DetStructCounterWriter()
        writer.counter = wp.zeros(1, dtype=wp.int32, device=device)
        writer.output = wp.zeros(int(counts_np.sum()) + 4, dtype=wp.float32, device=device)
        wp.launch(struct_field_counter_kernel, dim=64, inputs=[data, counts_arr, writer], device=device)
        counter_values.append(int(writer.counter.numpy()[0]))
        results.append(writer.output.numpy()[: expected.shape[0]].copy())

    np.testing.assert_array_equal(np.array(counter_values), np.full(3, expected.shape[0], dtype=np.int32))
    for i in range(1, len(results)):
        np.testing.assert_array_equal(results[0], results[i])
    np.testing.assert_array_equal(results[0].view(np.uint32), expected.view(np.uint32))


def test_struct_field_helper_counter_atomic(test, device):
    """Verify helper-function counters work when the target array lives in a struct field."""
    if device.is_cpu:
        test.skipTest("CPU execution is already deterministic")

    rng = np.random.default_rng(35)
    data_np = rng.random(64, dtype=np.float32)
    flags_np = (rng.random(64) > 0.4).astype(np.int32)
    data = wp.array(data_np, dtype=wp.float32, device=device)
    flags = wp.array(flags_np, dtype=wp.int32, device=device)
    expected = data_np[flags_np != 0]

    results = []
    counts = []
    for _ in range(3):
        writer = _DetStructCounterWriter()
        writer.counter = wp.zeros(1, dtype=wp.int32, device=device)
        writer.output = wp.zeros(int(flags_np.sum()) + 4, dtype=wp.float32, device=device)
        wp.launch(struct_field_helper_counter_kernel, dim=64, inputs=[data, flags, writer], device=device)
        counts.append(int(writer.counter.numpy()[0]))
        results.append(writer.output.numpy()[: expected.shape[0]].copy())

    np.testing.assert_array_equal(np.array(counts), np.full(3, expected.shape[0], dtype=np.int32))
    for i in range(1, len(results)):
        np.testing.assert_array_equal(results[0], results[i])
    np.testing.assert_array_equal(results[0].view(np.uint32), expected.view(np.uint32))


def test_any_struct_field_counter_atomic(test, device):
    """Verify deterministic counters work when the struct argument is typed as Any."""
    if device.is_cpu:
        test.skipTest("CPU execution is already deterministic")

    rng = np.random.default_rng(37)
    data_np = rng.random(64, dtype=np.float32)
    counts_np = rng.integers(0, 4, size=64, dtype=np.int32)
    data = wp.array(data_np, dtype=wp.float32, device=device)
    counts_arr = wp.array(counts_np, dtype=wp.int32, device=device)
    expected = np.array(
        [
            np.float32(data_np[tid] + np.float32(i) * np.float32(0.5))
            for tid, count in enumerate(counts_np)
            for i in range(count)
        ],
        dtype=np.float32,
    )

    results = []
    counter_values = []
    for _ in range(3):
        writer = _DetStructCounterWriter()
        writer.counter = wp.zeros(1, dtype=wp.int32, device=device)
        writer.output = wp.zeros(int(counts_np.sum()) + 4, dtype=wp.float32, device=device)
        wp.launch(any_struct_field_counter_kernel, dim=64, inputs=[data, counts_arr, writer], device=device)
        counter_values.append(int(writer.counter.numpy()[0]))
        results.append(writer.output.numpy()[: expected.shape[0]].copy())

    np.testing.assert_array_equal(np.array(counter_values), np.full(3, expected.shape[0], dtype=np.int32))
    for i in range(1, len(results)):
        np.testing.assert_array_equal(results[0], results[i])
    np.testing.assert_array_equal(results[0].view(np.uint32), expected.view(np.uint32))


def test_any_struct_field_nested_counter_atomic(test, device):
    """Verify nested helper counters stay correct when the helper gets an explicit output index."""
    if device.is_cpu:
        test.skipTest("CPU execution is already deterministic")

    rng = np.random.default_rng(39)
    data_np = rng.random(64, dtype=np.float32)
    counts_np = rng.integers(0, 4, size=64, dtype=np.int32)
    data = wp.array(data_np, dtype=wp.float32, device=device)
    counts_arr = wp.array(counts_np, dtype=wp.int32, device=device)
    expected = np.array(
        [
            np.float32(data_np[tid] + np.float32(i) * np.float32(0.5))
            for tid, count in enumerate(counts_np)
            for i in range(count)
        ],
        dtype=np.float32,
    )

    results = []
    counter_values = []
    for _ in range(3):
        writer = _DetStructCounterWriter()
        writer.counter = wp.zeros(1, dtype=wp.int32, device=device)
        writer.output = wp.zeros(int(counts_np.sum()) + 4, dtype=wp.float32, device=device)
        wp.launch(any_struct_field_nested_counter_kernel, dim=64, inputs=[data, counts_arr, writer], device=device)
        counter_values.append(int(writer.counter.numpy()[0]))
        results.append(writer.output.numpy()[: expected.shape[0]].copy())

    np.testing.assert_array_equal(np.array(counter_values), np.full(3, expected.shape[0], dtype=np.int32))
    for i in range(1, len(results)):
        np.testing.assert_array_equal(results[0], results[i])
    np.testing.assert_array_equal(results[0].view(np.uint32), expected.view(np.uint32))


def test_any_struct_field_counter_accumulates_across_launches(test, device):
    """Verify zero-contribution launches do not reset an existing deterministic counter."""
    if device.is_cpu:
        test.skipTest("CPU execution is already deterministic")

    data_np = np.arange(8, dtype=np.float32)
    counts_np = np.array([1, 2, 0, 1, 0, 2, 1, 0], dtype=np.int32)
    expected = np.array(
        [
            np.float32(data_np[tid] + np.float32(i) * np.float32(0.5))
            for tid, count in enumerate(counts_np)
            for i in range(count)
        ],
        dtype=np.float32,
    )

    data = wp.array(data_np, dtype=wp.float32, device=device)
    counts = wp.array(counts_np, dtype=wp.int32, device=device)
    zero_counts = wp.zeros_like(counts)

    writer = _DetStructCounterWriter()
    writer.counter = wp.zeros(1, dtype=wp.int32, device=device)
    writer.output = wp.zeros(expected.shape[0] + 4, dtype=wp.float32, device=device)

    wp.launch(any_struct_field_counter_kernel, dim=8, inputs=[data, counts, writer], device=device)
    first_count = int(writer.counter.numpy()[0])
    first_output = writer.output.numpy()[: expected.shape[0]].copy()

    wp.launch(any_struct_field_counter_kernel, dim=8, inputs=[data, zero_counts, writer], device=device)
    second_count = int(writer.counter.numpy()[0])
    second_output = writer.output.numpy()[: expected.shape[0]].copy()

    test.assertEqual(first_count, expected.shape[0])
    test.assertEqual(second_count, expected.shape[0])
    np.testing.assert_array_equal(first_output.view(np.uint32), expected.view(np.uint32))
    np.testing.assert_array_equal(second_output.view(np.uint32), expected.view(np.uint32))


def test_struct_field_counter_view_atomic(test, device):
    """Verify struct-field counters work when the field holds a 1-element array view."""
    if device.is_cpu:
        test.skipTest("CPU execution is already deterministic")

    rng = np.random.default_rng(40)
    data_np = rng.random(64, dtype=np.float32)
    counts_np = rng.integers(0, 4, size=64, dtype=np.int32)
    data = wp.array(data_np, dtype=wp.float32, device=device)
    counts_arr = wp.array(counts_np, dtype=wp.int32, device=device)
    expected = np.array(
        [
            np.float32(data_np[tid] + np.float32(i) * np.float32(0.5))
            for tid, count in enumerate(counts_np)
            for i in range(count)
        ],
        dtype=np.float32,
    )

    results = []
    counter_values = []
    packed_values = []
    for _ in range(3):
        packed_counter = wp.zeros(2, dtype=wp.int32, device=device)
        writer = _DetStructCounterWriter()
        writer.counter = packed_counter[0:1]
        writer.output = wp.zeros(int(counts_np.sum()) + 4, dtype=wp.float32, device=device)
        wp.launch(struct_field_counter_kernel, dim=64, inputs=[data, counts_arr, writer], device=device)
        packed_np = packed_counter.numpy().copy()
        packed_values.append(packed_np)
        counter_values.append(int(packed_np[0]))
        results.append(writer.output.numpy()[: expected.shape[0]].copy())

    np.testing.assert_array_equal(np.array(counter_values), np.full(3, expected.shape[0], dtype=np.int32))
    for i in range(1, len(results)):
        np.testing.assert_array_equal(results[0], results[i])
        np.testing.assert_array_equal(packed_values[0], packed_values[i])
    np.testing.assert_array_equal(results[0].view(np.uint32), expected.view(np.uint32))
    np.testing.assert_array_equal(packed_values[0], np.array([expected.shape[0], 0], dtype=np.int32))


def test_any_struct_field_counter_view_atomic(test, device):
    """Verify Any-typed struct-field counters work when the field holds a 1-element array view."""
    if device.is_cpu:
        test.skipTest("CPU execution is already deterministic")

    rng = np.random.default_rng(41)
    data_np = rng.random(64, dtype=np.float32)
    counts_np = rng.integers(0, 4, size=64, dtype=np.int32)
    data = wp.array(data_np, dtype=wp.float32, device=device)
    counts_arr = wp.array(counts_np, dtype=wp.int32, device=device)
    expected = np.array(
        [
            np.float32(data_np[tid] + np.float32(i) * np.float32(0.5))
            for tid, count in enumerate(counts_np)
            for i in range(count)
        ],
        dtype=np.float32,
    )

    results = []
    counter_values = []
    packed_values = []
    for _ in range(3):
        packed_counter = wp.zeros(2, dtype=wp.int32, device=device)
        writer = _DetStructCounterWriter()
        writer.counter = packed_counter[0:1]
        writer.output = wp.zeros(int(counts_np.sum()) + 4, dtype=wp.float32, device=device)
        wp.launch(any_struct_field_counter_kernel, dim=64, inputs=[data, counts_arr, writer], device=device)
        packed_np = packed_counter.numpy().copy()
        packed_values.append(packed_np)
        counter_values.append(int(packed_np[0]))
        results.append(writer.output.numpy()[: expected.shape[0]].copy())

    np.testing.assert_array_equal(np.array(counter_values), np.full(3, expected.shape[0], dtype=np.int32))
    for i in range(1, len(results)):
        np.testing.assert_array_equal(results[0], results[i])
        np.testing.assert_array_equal(packed_values[0], packed_values[i])
    np.testing.assert_array_equal(results[0].view(np.uint32), expected.view(np.uint32))
    np.testing.assert_array_equal(packed_values[0], np.array([expected.shape[0], 0], dtype=np.int32))


def test_generic_nested_struct_field_helper_counter_atomic(test, device):
    """Verify generic kernel overloads preserve nested helper deterministic counters."""
    if device.is_cpu:
        test.skipTest("CPU execution is already deterministic")

    rng = np.random.default_rng(36)
    data_np = rng.random(64, dtype=np.float32)
    flags_np = (rng.random(64) > 0.4).astype(np.int32)
    data = wp.array(data_np, dtype=wp.float32, device=device)
    flags = wp.array(flags_np, dtype=wp.int32, device=device)
    expected = data_np[flags_np != 0]

    results = []
    counts = []
    for _ in range(3):
        writer = _DetStructCounterWriter()
        writer.counter = wp.zeros(1, dtype=wp.int32, device=device)
        writer.output = wp.zeros(int(flags_np.sum()) + 4, dtype=wp.float32, device=device)
        wp.launch(
            generic_nested_struct_field_helper_counter_kernel, dim=64, inputs=[data, flags, writer], device=device
        )
        counts.append(int(writer.counter.numpy()[0]))
        results.append(writer.output.numpy()[: expected.shape[0]].copy())

    np.testing.assert_array_equal(np.array(counts), np.full(3, expected.shape[0], dtype=np.int32))
    for i in range(1, len(results)):
        np.testing.assert_array_equal(results[0], results[i])
    np.testing.assert_array_equal(results[0].view(np.uint32), expected.view(np.uint32))


def test_struct_field_deterministic_metadata(test, device):
    """Verify struct-field counters register deterministic metadata on helpers and kernels."""
    del device

    struct_field_counter_kernel.adj.build(None, {"deterministic": "run_to_run"})
    struct_field_helper_counter_kernel.adj.build(None, {"deterministic": "run_to_run"})
    _det_struct_counter_write.adj.build(None, {"deterministic": "run_to_run"})

    test.assertIsNotNone(_det_struct_counter_write.adj.det_meta)
    test.assertTrue(_det_struct_counter_write.adj.det_meta.has_counter)

    test.assertIsNotNone(struct_field_counter_kernel.adj.det_meta)
    test.assertTrue(struct_field_counter_kernel.adj.det_meta.has_counter)

    test.assertIsNotNone(struct_field_helper_counter_kernel.adj.det_meta)
    test.assertTrue(struct_field_helper_counter_kernel.adj.det_meta.has_counter)


def test_helper_function_deterministic_metadata(test, device):
    """Verify helper-function atomics register deterministic metadata on helpers and kernels."""
    del device

    helper_scatter_kernel = _make_deterministic_helper_scatter_kernel(_det_closure_transform_a)

    helper_scatter_kernel.adj.build(None, {"deterministic": "run_to_run"})
    helper_counter_kernel.adj.build(None, {"deterministic": "run_to_run"})
    _det_helper_atomic_add.adj.build(None, {"deterministic": "run_to_run"})
    _det_helper_counter_alloc.adj.build(None, {"deterministic": "run_to_run"})

    test.assertIsNotNone(_det_helper_atomic_add.adj.det_meta)
    test.assertTrue(_det_helper_atomic_add.adj.det_meta.has_scatter)

    test.assertIsNotNone(_det_helper_counter_alloc.adj.det_meta)
    test.assertTrue(_det_helper_counter_alloc.adj.det_meta.has_counter)

    test.assertIsNotNone(helper_scatter_kernel.adj.det_meta)
    test.assertTrue(helper_scatter_kernel.adj.det_meta.has_scatter)

    test.assertIsNotNone(helper_counter_kernel.adj.det_meta)
    test.assertTrue(helper_counter_kernel.adj.det_meta.has_counter)


def test_deterministic_func_kernel(test, device):
    """Verify deterministic atomics inside ``@wp.func`` calls remain reproducible."""
    if device.is_cpu:
        test.skipTest("CPU execution is already deterministic")

    n = 512
    out_size = 16
    rng = np.random.default_rng(74)
    data_np = rng.random(n, dtype=np.float32)
    indices_np = rng.integers(0, out_size, size=n, dtype=np.int32)

    data = wp.array(data_np, dtype=wp.float32, device=device)
    indices = wp.array(indices_np, dtype=wp.int32, device=device)
    results = []
    for _ in range(3):
        output = wp.zeros(out_size, dtype=wp.float32, device=device)
        wp.launch(func_scatter_add_kernel, dim=n, inputs=[data, indices], outputs=[output], device=device)
        results.append(output.numpy().copy())

    for result in results:
        np.testing.assert_allclose(
            result, _reference_scatter_add_float32(data_np, indices_np, out_size), rtol=1e-6, atol=1e-6
        )
    for i in range(1, len(results)):
        np.testing.assert_array_equal(results[0], results[i])


def test_nested_deterministic_func_kernel(test, device):
    """Verify deterministic helper args propagate through nested ``@wp.func`` calls."""
    if device.is_cpu:
        test.skipTest("CPU execution is already deterministic")

    n = 512
    out_size = 16
    rng = np.random.default_rng(75)
    data_np = rng.random(n, dtype=np.float32)
    indices_np = rng.integers(0, out_size, size=n, dtype=np.int32)

    data = wp.array(data_np, dtype=wp.float32, device=device)
    indices = wp.array(indices_np, dtype=wp.int32, device=device)
    results = []
    for _ in range(3):
        accum = wp.zeros(out_size, dtype=wp.float32, device=device)
        wp.launch(nested_func_scatter_add_kernel, dim=n, inputs=[data, indices], outputs=[accum], device=device)
        results.append(accum.numpy().copy())

    for result in results:
        np.testing.assert_allclose(
            result, _reference_scatter_add_float32(data_np, indices_np, out_size), rtol=1e-6, atol=1e-6
        )
    for i in range(1, len(results)):
        np.testing.assert_array_equal(results[0], results[i])


def test_record_cmd_deterministic_launch(test, device):
    """Verify ``record_cmd=True`` works for deterministic CUDA launches."""
    if device.is_cpu:
        test.skipTest("CPU execution is already deterministic")

    n = 128
    out_size = 8
    rng = np.random.default_rng(19)

    data_np = rng.random(n, dtype=np.float32)
    indices_np = rng.integers(0, out_size, size=n, dtype=np.int32)

    data = wp.array(data_np, dtype=wp.float32, device=device)
    indices = wp.array(indices_np, dtype=wp.int32, device=device)
    output = wp.zeros(out_size, dtype=wp.float32, device=device)

    cmd = wp.launch(
        scatter_add_kernel,
        dim=n,
        inputs=[data, indices],
        outputs=[output],
        device=device,
        record_cmd=True,
    )

    np.testing.assert_array_equal(output.numpy(), np.zeros(out_size, dtype=np.float32))

    cmd.launch()
    expected = output.numpy().copy()

    output_2 = wp.zeros(out_size, dtype=wp.float32, device=device)
    cmd.set_param_by_name("output", output_2)
    cmd.launch()

    np.testing.assert_array_equal(output_2.numpy(), expected)


def test_graph_capture_deterministic_launch(test, device):
    """Verify deterministic scatter launches can be captured and replayed."""
    if device.is_cpu:
        test.skipTest("Graph capture requires CUDA")

    n = 256
    rng = np.random.default_rng(29)
    data_np = rng.random(n, dtype=np.float32)
    indices_np = rng.integers(0, 8, size=n, dtype=np.int32)

    data = wp.array(data_np, dtype=wp.float32, device=device)
    indices = wp.array(indices_np, dtype=wp.int32, device=device)
    output = wp.zeros(8, dtype=wp.float32, device=device)

    wp.launch(scatter_add_kernel, dim=n, inputs=[data, indices], outputs=[output], device=device)
    output.zero_()

    with wp.ScopedCapture(device, force_module_load=False) as capture:
        wp.launch(scatter_add_kernel, dim=n, inputs=[data, indices], outputs=[output], device=device)

    wp.capture_launch(capture.graph)
    first = output.numpy().copy()

    output.zero_()
    wp.capture_launch(capture.graph)
    second = output.numpy().copy()

    np.testing.assert_array_equal(first, second)


def test_graph_capture_deterministic_closure_kernel(test, device):
    """Verify deterministic closure kernels can be captured and replayed."""
    if device.is_cpu:
        test.skipTest("Graph capture requires CUDA")

    kernel = _make_deterministic_closure_kernel(_det_closure_transform_a)

    n = 256
    rng = np.random.default_rng(31)
    data_np = rng.random(n, dtype=np.float32)
    data = wp.array(data_np, dtype=wp.float32, device=device)
    output = wp.zeros(8, dtype=wp.float32, device=device)

    wp.launch(kernel, dim=n, inputs=[data], outputs=[output], device=device)
    output.zero_()

    with wp.ScopedCapture(device, force_module_load=False) as capture:
        wp.launch(kernel, dim=n, inputs=[data], outputs=[output], device=device)

    wp.capture_launch(capture.graph)
    first = output.numpy().copy()

    output.zero_()
    wp.capture_launch(capture.graph)
    second = output.numpy().copy()

    np.testing.assert_array_equal(first, second)


def test_graph_capture_deterministic_func_kernel(test, device):
    """Verify deterministic ``@wp.func`` atomics remain capture-safe."""
    if device.is_cpu:
        test.skipTest("Graph capture requires CUDA")

    n = 256
    rng = np.random.default_rng(76)
    data_np = rng.random(n, dtype=np.float32)
    indices_np = rng.integers(0, 8, size=n, dtype=np.int32)

    data = wp.array(data_np, dtype=wp.float32, device=device)
    indices = wp.array(indices_np, dtype=wp.int32, device=device)
    output = wp.zeros(8, dtype=wp.float32, device=device)

    wp.launch(func_scatter_add_kernel, dim=n, inputs=[data, indices], outputs=[output], device=device)
    output.zero_()

    with wp.ScopedCapture(device, force_module_load=False) as capture:
        wp.launch(func_scatter_add_kernel, dim=n, inputs=[data, indices], outputs=[output], device=device)

    wp.capture_launch(capture.graph)
    first = output.numpy().copy()

    output.zero_()
    wp.capture_launch(capture.graph)
    second = output.numpy().copy()

    np.testing.assert_array_equal(first, second)


def test_graph_capture_vec3_atomic_minmax(test, device):
    """Verify composite deterministic reductions remain capture-safe."""
    if device.is_cpu:
        test.skipTest("Graph capture requires CUDA")

    n = 512
    rng = np.random.default_rng(70)
    points_np = rng.standard_normal((n, 3), dtype=np.float32)
    points = wp.array(points_np, dtype=wp.vec3, device=device)

    out_min = wp.empty(1, dtype=wp.vec3, device=device)
    out_max = wp.empty(1, dtype=wp.vec3, device=device)
    min_init = wp.vec3(np.inf, np.inf, np.inf)
    max_init = wp.vec3(-np.inf, -np.inf, -np.inf)

    out_min.fill_(min_init)
    out_max.fill_(max_init)
    wp.launch(vec3_atomic_minmax_kernel, dim=n, inputs=[points], outputs=[out_min, out_max], device=device)
    out_min.fill_(min_init)
    out_max.fill_(max_init)

    with wp.ScopedCapture(device, force_module_load=False) as capture:
        wp.launch(vec3_atomic_minmax_kernel, dim=n, inputs=[points], outputs=[out_min, out_max], device=device)

    wp.capture_launch(capture.graph)
    first_min = out_min.numpy().copy()
    first_max = out_max.numpy().copy()

    out_min.fill_(min_init)
    out_max.fill_(max_init)
    wp.capture_launch(capture.graph)
    second_min = out_min.numpy().copy()
    second_max = out_max.numpy().copy()

    np.testing.assert_array_equal(first_min, second_min)
    np.testing.assert_array_equal(first_max, second_max)


def test_deterministic_enum_parity(test, device):
    """Keep Python deterministic constants aligned with the native enums."""
    del device

    native_source = (Path(wp.__file__).resolve().parent / "native" / "deterministic.cu").read_text()

    def parse_enum(enum_name):
        match = re.search(rf"enum {enum_name} \{{(.*?)\n\}};", native_source, re.DOTALL)
        if match is None:
            raise AssertionError(f"Failed to find enum {enum_name} in deterministic.cu")

        entries = {}
        for name, value in re.findall(r"([A-Z0-9_]+)\s*=\s*([0-9]+)", match.group(1)):
            entries[name] = int(value)
        return entries

    native_reduce_ops = parse_enum("ReduceOp")
    native_deterministic_levels = parse_enum("DeterminismLevel")
    native_scalar_types = parse_enum("ScalarType")

    test.assertEqual(
        native_reduce_ops,
        {
            "REDUCE_OP_ADD": wp_deterministic.REDUCE_OP_ADD,
            "REDUCE_OP_MIN": wp_deterministic.REDUCE_OP_MIN,
            "REDUCE_OP_MAX": wp_deterministic.REDUCE_OP_MAX,
        },
    )
    test.assertEqual(
        native_deterministic_levels,
        {
            "DETERMINISTIC_NOT_GUARANTEED": wp_deterministic._DETERMINISTIC_MODE_IDS[
                wp_deterministic.DETERMINISTIC_NOT_GUARANTEED
            ],
            "DETERMINISTIC_RUN_TO_RUN": wp_deterministic._DETERMINISTIC_MODE_IDS[
                wp_deterministic.DETERMINISTIC_RUN_TO_RUN
            ],
            "DETERMINISTIC_GPU_TO_GPU": wp_deterministic._DETERMINISTIC_MODE_IDS[
                wp_deterministic.DETERMINISTIC_GPU_TO_GPU
            ],
        },
    )
    test.assertEqual(
        native_scalar_types,
        {
            "SCALAR_HALF": wp_deterministic._SCALAR_TYPE_IDS[wp.float16],
            "SCALAR_FLOAT": wp_deterministic._SCALAR_TYPE_IDS[wp.float32],
            "SCALAR_DOUBLE": wp_deterministic._SCALAR_TYPE_IDS[wp.float64],
            "SCALAR_INT": wp_deterministic._SCALAR_TYPE_IDS[wp.int32],
            "SCALAR_UINT": wp_deterministic._SCALAR_TYPE_IDS[wp.uint32],
            "SCALAR_INT64": wp_deterministic._SCALAR_TYPE_IDS[wp.int64],
            "SCALAR_UINT64": wp_deterministic._SCALAR_TYPE_IDS[wp.uint64],
        },
    )


# ---------------------------------------------------------------------------
# Test class registration
# ---------------------------------------------------------------------------

cuda_devices = get_selected_cuda_test_devices()
all_devices = get_test_devices()


class TestDeterministic(unittest.TestCase):
    """Test suite for deterministic execution mode."""

    @classmethod
    def setUpClass(cls):
        cls._old_deterministic = wp.config.deterministic
        wp.config.deterministic = "run_to_run"

    @classmethod
    def tearDownClass(cls):
        wp.config.deterministic = cls._old_deterministic


# Pattern A tests (accumulation).
add_function_test(
    TestDeterministic, "test_scatter_add_reproducibility", test_scatter_add_reproducibility, devices=cuda_devices
)
add_function_test(
    TestDeterministic,
    "test_gpu_to_gpu_mode_reproducibility",
    test_gpu_to_gpu_mode_reproducibility,
    devices=cuda_devices,
)
add_function_test(
    TestDeterministic,
    "test_gpu_to_gpu_matches_canonical_float32_reference",
    test_gpu_to_gpu_matches_canonical_float32_reference,
    devices=cuda_devices,
)
add_function_test(
    TestDeterministic, "test_augassign_add_reproducibility", test_augassign_add_reproducibility, devices=cuda_devices
)
add_function_test(TestDeterministic, "test_scatter_add_correctness", test_scatter_add_correctness, devices=all_devices)
add_function_test(TestDeterministic, "test_multi_array_atomic", test_multi_array_atomic, devices=cuda_devices)
add_function_test(
    TestDeterministic, "test_atomic_sub_deterministic", test_atomic_sub_deterministic, devices=cuda_devices
)
add_function_test(TestDeterministic, "test_atomic_add_2d", test_atomic_add_2d, devices=cuda_devices)
add_function_test(TestDeterministic, "test_atomic_add_array_view", test_atomic_add_array_view, devices=cuda_devices)
add_function_test(
    TestDeterministic,
    "test_helper_atomic_add_array_view",
    test_helper_atomic_add_array_view,
    devices=cuda_devices,
)
add_function_test(
    TestDeterministic, "test_atomic_double_deterministic", test_atomic_double_deterministic, devices=cuda_devices
)
add_function_test(
    TestDeterministic, "test_vec3_atomic_add_deterministic", test_vec3_atomic_add_deterministic, devices=cuda_devices
)
add_function_test(
    TestDeterministic,
    "test_vec3_atomic_minmax_deterministic",
    test_vec3_atomic_minmax_deterministic,
    devices=cuda_devices,
)
add_function_test(
    TestDeterministic, "test_mat33_atomic_add_deterministic", test_mat33_atomic_add_deterministic, devices=cuda_devices
)
add_function_test(
    TestDeterministic,
    "test_triple_scatter_capacity_estimate",
    test_triple_scatter_capacity_estimate,
    devices=cuda_devices,
)
add_function_test(
    TestDeterministic,
    "test_loop_scatter_max_records_override",
    test_loop_scatter_max_records_override,
    devices=cuda_devices,
)
add_function_test(
    TestDeterministic, "test_mixed_reduce_ops_same_array", test_mixed_reduce_ops_same_array, devices=cuda_devices
)
add_function_test(
    TestDeterministic, "test_deterministic_func_kernel", test_deterministic_func_kernel, devices=cuda_devices
)
add_function_test(
    TestDeterministic,
    "test_nested_deterministic_func_kernel",
    test_nested_deterministic_func_kernel,
    devices=cuda_devices,
)

# Pattern B tests (counter).
add_function_test(TestDeterministic, "test_counter_reproducibility", test_counter_reproducibility, devices=cuda_devices)
add_function_test(
    TestDeterministic,
    "test_counter_phase0_suppresses_array_writes",
    test_counter_phase0_suppresses_array_writes,
    devices=cuda_devices,
)
add_function_test(TestDeterministic, "test_counter_correctness", test_counter_correctness, devices=all_devices)
add_function_test(TestDeterministic, "test_conditional_counter", test_conditional_counter, devices=cuda_devices)

# Mixed pattern tests.
add_function_test(TestDeterministic, "test_mixed_pattern", test_mixed_pattern, devices=cuda_devices)

# Passthrough / override tests.
add_function_test(TestDeterministic, "test_int_atomic_passthrough", test_int_atomic_passthrough, devices=all_devices)
add_function_test(TestDeterministic, "test_module_option_override", test_module_option_override, devices=all_devices)
add_function_test(
    TestDeterministic, "test_kernel_decorator_override", test_kernel_decorator_override, devices=cuda_devices
)
add_function_test(
    TestDeterministic, "test_deterministic_closure_kernel", test_deterministic_closure_kernel, devices=cuda_devices
)
add_function_test(
    TestDeterministic,
    "test_helper_function_scatter_atomic",
    test_helper_function_scatter_atomic,
    devices=cuda_devices,
)
add_function_test(
    TestDeterministic,
    "test_helper_function_counter_atomic",
    test_helper_function_counter_atomic,
    devices=cuda_devices,
)
add_function_test(
    TestDeterministic,
    "test_struct_field_counter_atomic",
    test_struct_field_counter_atomic,
    devices=cuda_devices,
)
add_function_test(
    TestDeterministic,
    "test_struct_field_helper_counter_atomic",
    test_struct_field_helper_counter_atomic,
    devices=cuda_devices,
)
add_function_test(
    TestDeterministic,
    "test_any_struct_field_counter_atomic",
    test_any_struct_field_counter_atomic,
    devices=cuda_devices,
)
add_function_test(
    TestDeterministic,
    "test_any_struct_field_nested_counter_atomic",
    test_any_struct_field_nested_counter_atomic,
    devices=cuda_devices,
)
add_function_test(
    TestDeterministic,
    "test_any_struct_field_counter_accumulates_across_launches",
    test_any_struct_field_counter_accumulates_across_launches,
    devices=cuda_devices,
)
add_function_test(
    TestDeterministic,
    "test_struct_field_counter_view_atomic",
    test_struct_field_counter_view_atomic,
    devices=cuda_devices,
)
add_function_test(
    TestDeterministic,
    "test_any_struct_field_counter_view_atomic",
    test_any_struct_field_counter_view_atomic,
    devices=cuda_devices,
)
add_function_test(
    TestDeterministic,
    "test_generic_nested_struct_field_helper_counter_atomic",
    test_generic_nested_struct_field_helper_counter_atomic,
    devices=cuda_devices,
)
add_function_test(
    TestDeterministic,
    "test_struct_field_deterministic_metadata",
    test_struct_field_deterministic_metadata,
    devices=all_devices,
)
add_function_test(
    TestDeterministic,
    "test_helper_function_deterministic_metadata",
    test_helper_function_deterministic_metadata,
    devices=all_devices,
)
add_function_test(
    TestDeterministic,
    "test_record_cmd_deterministic_launch",
    test_record_cmd_deterministic_launch,
    devices=cuda_devices,
)
add_function_test(
    TestDeterministic,
    "test_graph_capture_deterministic_launch",
    test_graph_capture_deterministic_launch,
    devices=cuda_devices,
)
add_function_test(
    TestDeterministic,
    "test_graph_capture_deterministic_closure_kernel",
    test_graph_capture_deterministic_closure_kernel,
    devices=cuda_devices,
)
add_function_test(
    TestDeterministic,
    "test_graph_capture_deterministic_func_kernel",
    test_graph_capture_deterministic_func_kernel,
    devices=cuda_devices,
)
add_function_test(
    TestDeterministic,
    "test_graph_capture_vec3_atomic_minmax",
    test_graph_capture_vec3_atomic_minmax,
    devices=cuda_devices,
)
add_function_test(
    TestDeterministic, "test_deterministic_enum_parity", test_deterministic_enum_parity, devices=all_devices
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
