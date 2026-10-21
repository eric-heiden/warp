# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import threading
import unittest
from typing import Any

import numpy as np

import warp as wp
from warp.tests.unittest_utils import *


def random_transforms(rng, n):
    q = rng.normal(size=(n, 4)).astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return np.ascontiguousarray(np.concatenate((rng.normal(size=(n, 3)).astype(np.float32), q), axis=1))


@wp.kernel
def transform_chain(
    a: wp.array[wp.transform],
    b: wp.array[wp.transform],
    c: wp.array[wp.transform],
    ab: wp.array[wp.transform],
    abc: wp.array[wp.transform],
):
    i = wp.tid()
    ab[i] = wp.transform_multiply(a[i], b[i])
    abc[i] = wp.transform_multiply(ab[i], c[i])


@wp.kernel
def transform_rows(x: wp.transform, src: wp.array[wp.transform], out: wp.array[wp.transform]):
    i = wp.tid()
    out[i] = wp.transform_multiply(x, src[i])


@wp.kernel
def broadcast_scale(
    a: wp.array2d[wp.transform], b: wp.array2d[wp.transform], scale: float, out: wp.array2d[wp.transform]
):
    i, j = wp.tid()
    t = wp.transform_multiply(a[wp.min(i, a.shape[0] - 1), j], b[i, wp.min(j, b.shape[1] - 1)])
    out[i, j] = wp.transform(wp.transform_get_translation(t) * scale, wp.transform_get_rotation(t))


@wp.kernel
def scalar_params(i32: wp.int32, u64: wp.uint64, f64: wp.float64, f16: wp.float16, flag: bool, out: wp.array[float]):
    out[0] = float(i32)
    out[1] = float(u64)
    out[2] = float(f64)
    out[3] = float(f16)
    out[4] = wp.where(flag, 1.0, 0.0)


@wp.struct
class Offsets:
    shift: float
    values: wp.array[float]


@wp.kernel
def struct_param(params: Offsets, out: wp.array[float]):
    i = wp.tid()
    out[i] = params.values[i] + params.shift


@wp.kernel
def indexed_param(a: wp.indexedarray[float], out: wp.array[float]):
    i = wp.tid()
    out[i] = a[i] * 2.0


@wp.kernel
def generic_scale(a: wp.array[Any], s: Any):
    i = wp.tid()
    a[i] = a[i] * s


def launch_reference(kernel, dim, args):
    """Run wp.launch on copies of ``args`` (NumPy arrays become Warp arrays) and return them as NumPy."""
    copies = []
    inputs = []
    for value, param in zip(args, kernel.adj.args, strict=True):
        if isinstance(value, np.ndarray) and wp._src.types.is_array(param.type):
            copy = wp.array(value, dtype=param.type.dtype, device="cpu")
            copies.append(copy)
            inputs.append(copy)
        else:
            copies.append(None)
            inputs.append(value)
    wp.launch(kernel, dim=dim, inputs=inputs, device="cpu")
    return [None if copy is None else copy.numpy() for copy in copies]


class TestCpuLaunch(unittest.TestCase):
    def test_matches_launch_numpy_and_warp_arrays(self):
        """Verify that NumPy and Warp array arguments must produce the same results as wp.launch."""
        rng = np.random.default_rng(0)
        launcher = wp.CpuLauncher(transform_chain)
        for n in (1, 80, 1000):
            with self.subTest(n=n):
                args = [random_transforms(rng, n) for _ in range(5)]
                expected = launch_reference(transform_chain, n, args)

                numpy_args = [value.copy() for value in args]
                launcher(n, *numpy_args)
                for result, reference in zip(numpy_args[3:], expected[3:], strict=True):
                    assert_np_equal(result, reference.reshape(result.shape))

                warp_args = [wp.array(value, dtype=wp.transform, device="cpu") for value in args]
                launcher(n, *warp_args)
                for result, reference in zip(warp_args[3:], expected[3:], strict=True):
                    assert_np_equal(result.numpy(), reference)

    def test_numpy_views(self):
        """Verify that read-only and outer-strided NumPy arrays must be passed without copies."""
        rng = np.random.default_rng(1)
        a = random_transforms(rng, 80)
        a.setflags(write=False)
        b = random_transforms(rng, 160)[::2]
        c = random_transforms(rng, 80)
        expected = launch_reference(transform_chain, 80, [a, np.ascontiguousarray(b), c, c, c])
        ab = np.empty((80, 7), dtype=np.float32)
        abc = np.empty((80, 7), dtype=np.float32)
        wp.CpuLauncher(transform_chain)(80, a, b, c, ab, abc)
        assert_np_equal(ab, expected[3])
        assert_np_equal(abc, expected[4])

    def test_value_params(self):
        """Verify that transform values must be accepted as Warp values, equivalent classes, and NumPy rows."""
        rng = np.random.default_rng(2)
        row = random_transforms(rng, 1)[0]
        src = random_transforms(rng, 50)
        expected = launch_reference(transform_rows, 50, [wp.transform(*row), src, src])[2]
        launcher = wp.CpuLauncher(transform_rows)
        equivalent_class = wp._src.types.transformation(wp.float32)
        for label, value in (
            ("wp.transform", wp.transform(*row)),
            ("NumPy row", row),
            ("equivalent class", equivalent_class(*row)),
        ):
            with self.subTest(value=label):
                out = np.empty_like(src)
                launcher(50, value, src, out)
                assert_np_equal(out, expected)

        # Values wp.launch() cannot convert are rejected the same way.
        out = np.empty_like(src)
        with self.assertRaisesRegex(ValueError, "Failed to convert argument for param x"):
            wp.launch(transform_rows, dim=50, inputs=[tuple(row.tolist()), src, out], device="cpu")
        with self.assertRaisesRegex(ValueError, "Failed to convert argument for param x"):
            launcher(50, tuple(row.tolist()), src, out)

    def test_broadcast_2d_and_scalar(self):
        """Verify that 2D launches with tuple dimensions and a scalar parameter must match wp.launch."""
        rng = np.random.default_rng(3)
        a = random_transforms(rng, 6).reshape(1, 6, 7)
        b = random_transforms(rng, 5).reshape(5, 1, 7)
        launcher = wp.CpuLauncher(broadcast_scale)
        for scale in (2.5, 2, np.float32(2.5), wp.float32(2.5)):
            with self.subTest(scale=type(scale).__name__):
                out = np.empty((5, 6, 7), dtype=np.float32)
                launcher((5, 6), a, b, scale, out)
                expected = launch_reference(broadcast_scale, (5, 6), [a, b, scale, out])[3]
                assert_np_equal(out, expected)

    def test_scalar_params(self):
        """Verify that scalar parameters of each width must be packed like wp.launch packs them."""
        launcher = wp.CpuLauncher(scalar_params)
        for values in ((-7, 2**40, 0.25, 1.5, True), (np.int32(3), np.uint64(5), np.float64(-2.0), 0.5, False)):
            with self.subTest(values=values):
                out = np.zeros(5, dtype=np.float32)
                launcher(1, *values, out)
                expected = launch_reference(scalar_params, 1, [*values, out])[5]
                assert_np_equal(out, expected)

    def test_struct_and_indexed_params(self):
        """Verify that struct and indexed-array parameters must be converted as in wp.launch."""
        params = Offsets()
        params.shift = 1.5
        params.values = wp.array(np.arange(8, dtype=np.float32), device="cpu")
        out = np.zeros(8, dtype=np.float32)
        wp.CpuLauncher(struct_param)(8, params, out)
        assert_np_equal(out, np.arange(8, dtype=np.float32) + 1.5)

        source = wp.array(np.arange(10, dtype=np.float32), device="cpu")
        indices = wp.array(np.array([3, 1, 4, 1, 5], dtype=np.int32), device="cpu")
        out = np.zeros(5, dtype=np.float32)
        wp.CpuLauncher(indexed_param)(5, wp.indexedarray(source, [indices]), out)
        assert_np_equal(out, np.array([6.0, 2.0, 8.0, 2.0, 10.0], dtype=np.float32))

    def test_generic_kernel_overload(self):
        """Verify that generic kernels must be rejected and their concrete overloads launched."""
        with self.assertRaisesRegex(ValueError, "generic"):
            wp.CpuLauncher(generic_scale)
        overload = wp.overload(generic_scale, [wp.array[wp.float64], wp.float64])
        values = np.arange(4, dtype=np.float64)
        wp.CpuLauncher(overload)(4, values, 3.0)
        assert_np_equal(values, np.arange(4, dtype=np.float64) * 3.0)

    def test_rejected_arguments(self):
        """Verify that arrays with a mismatched dtype, layout, or dimension count must raise."""
        launcher = wp.CpuLauncher(transform_rows)
        out = np.empty((4, 7), dtype=np.float32)
        x = wp.transform_identity()
        for label, src in (
            ("inner shape", np.zeros((4, 6), dtype=np.float32)),
            ("dtype", np.zeros((4, 7), dtype=np.float64)),
            ("dimension count", np.zeros((1, 4, 7), dtype=np.float32)),
            ("Warp dtype", wp.zeros(4, dtype=wp.vec3, device="cpu")),
        ):
            with self.subTest(case=label):
                with self.assertRaises(RuntimeError):
                    launcher(4, x, src, out)
        with self.assertRaises(TypeError):
            launcher(4, x, out)

    def test_launch_dimensions(self):
        """Verify that empty launches must be no-ops and negative dimensions must raise."""
        launcher = wp.CpuLauncher(transform_rows)
        src = random_transforms(np.random.default_rng(4), 3)
        out = np.zeros_like(src)
        launcher(0, wp.transform_identity(), src, out)
        assert_np_equal(out, np.zeros_like(src))
        with self.assertRaises(ValueError):
            launcher(-1, wp.transform_identity(), src, out)

    def test_threads(self):
        """Verify that concurrent launches from several threads must not share argument buffers."""
        rng = np.random.default_rng(5)
        launcher = wp.CpuLauncher(transform_rows)
        n = 1000
        jobs = [(wp.transform(*random_transforms(rng, 1)[0]), random_transforms(rng, n)) for _ in range(8)]
        expected = [launch_reference(transform_rows, n, [x, src, src])[2] for x, src in jobs]
        failures = []

        def worker(job, reference):
            for _ in range(200):
                out = np.empty((n, 7), dtype=np.float32)
                launcher(n, job[0], job[1], out)
                if not np.array_equal(out, reference):
                    failures.append(job)
                    return

        threads = [threading.Thread(target=worker, args=pair) for pair in zip(jobs, expected, strict=True)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(failures, [])

    def test_module_reload(self):
        """Verify that a launcher must pick up a module that was modified and reloaded."""
        launcher = wp.CpuLauncher(transform_rows)
        src = random_transforms(np.random.default_rng(6), 4)
        out = np.empty_like(src)
        launcher(4, wp.transform_identity(), src, out)
        enable_backward = transform_rows.module.options["enable_backward"]
        try:
            wp.set_module_options({"enable_backward": not enable_backward}, module=transform_rows.module)
            launcher(4, wp.transform_identity(), src, out)
            device = wp.get_device("cpu")
            self.assertIs(launcher._module_exec, transform_rows.module.execs[(device.context, 1)])
        finally:
            wp.set_module_options({"enable_backward": enable_backward}, module=transform_rows.module)
        assert_np_equal(out, src)

    def test_tape_and_capture_fall_back(self):
        """Verify that launches must be recorded on an active Tape and in a CPU graph capture."""
        rng = np.random.default_rng(7)
        x = wp.transform(*random_transforms(rng, 1)[0])
        src = wp.array(random_transforms(rng, 16), dtype=wp.transform, device="cpu")
        expected = launch_reference(transform_rows, 16, [x, src.numpy(), src.numpy()])[2]
        launcher = wp.CpuLauncher(transform_rows)

        tape = wp.Tape()
        out = wp.zeros(16, dtype=wp.transform, device="cpu")
        with tape:
            launcher(16, x, src, out)
        self.assertEqual(len(tape.launches), 1)
        assert_np_equal(out.numpy(), expected)

        out.zero_()
        with wp.ScopedCapture(device="cpu") as capture:
            launcher(16, x, src, out)
        out.zero_()
        wp.capture_launch(capture.graph)
        assert_np_equal(out.numpy(), expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
