# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for :func:`warp.gemm` (runtime-loaded cuBLASLt, and the built-in Warp kernel used without it)."""

import os
import tempfile
import unittest

import numpy as np

import warp as wp
from warp.tests.unittest_utils import *


def _gelu_tanh(z):
    return 0.5 * z * (1.0 + np.tanh(0.7978845608028654 * (z + 0.044715 * z**3)))


def _gelu_tanh_grad(z):
    u = 0.7978845608028654 * (z + 0.044715 * z**3)
    t = np.tanh(u)
    return 0.5 * (1.0 + t) + 0.5 * z * (1.0 - t * t) * 0.7978845608028654 * (1.0 + 3.0 * 0.044715 * z * z)


def test_gemm_basic(test, device):
    rng = np.random.default_rng(0)
    for M, N, K, ta, tb in [
        (64, 48, 32, False, False),
        (65, 33, 17, True, False),
        (40, 24, 16, False, True),
        (16, 8, 64, True, True),
    ]:
        a = rng.standard_normal((K, M) if ta else (M, K)).astype(np.float32)
        b = rng.standard_normal((N, K) if tb else (K, N)).astype(np.float32)
        ref = (a.T if ta else a) @ (b.T if tb else b)
        out = wp.gemm(wp.array(a, device=device), wp.array(b, device=device), trans_a=ta, trans_b=tb)
        assert_np_equal(out.numpy(), ref, tol=1e-4)


def test_gemm_alpha_beta_bias_activation(test, device):
    rng = np.random.default_rng(1)
    M, N, K = 128, 64, 32
    a, b = rng.standard_normal((M, K)).astype(np.float32), rng.standard_normal((K, N)).astype(np.float32)
    c0 = rng.standard_normal((M, N)).astype(np.float32)
    bias = rng.standard_normal(N).astype(np.float32)
    wa, wb = wp.array(a, device=device), wp.array(b, device=device)
    wc = wp.array(c0, device=device)
    wp.gemm(wa, wb, wc, alpha=0.5, beta=2.0)
    assert_np_equal(wc.numpy(), 0.5 * a @ b + 2.0 * c0, tol=1e-4)
    out = wp.gemm(wa, wb, bias=wp.array(bias, device=device), activation="relu")
    assert_np_equal(out.numpy(), np.maximum(a @ b + bias, 0.0), tol=1e-4)
    aux = wp.zeros((M, N), dtype=float, device=device)
    out = wp.gemm(wa, wb, bias=wp.array(bias, device=device), activation="gelu", aux=aux)
    assert_np_equal(aux.numpy(), a @ b + bias, tol=1e-4)
    assert_np_equal(out.numpy(), _gelu_tanh(a @ b + bias), tol=1e-3)


def test_gemm_backward_epilogues(test, device):
    rng = np.random.default_rng(2)
    B, D, H = 256, 32, 48
    z = rng.standard_normal((B, H)).astype(np.float32)  # pre-activation of the layer below
    dy = rng.standard_normal((B, D)).astype(np.float32)
    w = rng.standard_normal((H, D)).astype(np.float32)
    # dZ = (dY @ W^T) * gelu'(Z) and the bias gradient sum_rows(dZ)
    dz = wp.zeros((B, H), dtype=float, device=device)
    gb = wp.zeros(H, dtype=float, device=device)
    wp.gemm(
        wp.array(dy, device=device),
        wp.array(w, device=device),
        dz,
        trans_b=True,
        activation="gelu",
        grad_aux=wp.array(z, device=device),
        bias_grad=gb,
    )
    ref = (dy @ w.T) * _gelu_tanh_grad(z)
    assert_np_equal(dz.numpy(), ref, tol=1e-3)
    assert_np_equal(gb.numpy(), ref.sum(axis=0), tol=1e-2)
    # weight gradient with fused bias gradient (BGRADA): gW = X^T @ dY, gb = sum_rows(dY)
    x = rng.standard_normal((B, H)).astype(np.float32)
    gw = wp.zeros((H, D), dtype=float, device=device)
    gb2 = wp.zeros(D, dtype=float, device=device)
    wp.gemm(wp.array(x, device=device), wp.array(dy, device=device), gw, trans_a=True, bias_grad=gb2)
    assert_np_equal(gw.numpy(), x.T @ dy, tol=1e-3)
    assert_np_equal(gb2.numpy(), dy.sum(axis=0), tol=1e-3)


def test_gemm_batched_and_misaligned(test, device):
    rng = np.random.default_rng(3)
    a = rng.standard_normal((3, 20, 12)).astype(np.float32)
    b = rng.standard_normal((3, 12, 9)).astype(np.float32)
    out = wp.gemm(wp.array(a, device=device), wp.array(b, device=device))
    assert_np_equal(out.numpy(), a @ b, tol=1e-4)
    # operands at 4-byte offsets inside a larger buffer must select alignment-compatible algorithms (TF32 too)
    flat = wp.array(rng.standard_normal(1 + 64 * 32 + 32 * 16).astype(np.float32), device=device)
    wa = wp.array(ptr=flat.ptr + 4, shape=(64, 32), dtype=float, device=device)
    wb = wp.array(ptr=flat.ptr + 4 * (1 + 64 * 32), shape=(32, 16), dtype=float, device=device)
    for tf32 in (False, True):
        out = wp.gemm(wa, wb, tf32=tf32)
        assert_np_equal(out.numpy(), wa.numpy() @ wb.numpy(), tol=1e-1 if tf32 else 1e-4)


def test_gemm_graph_capture(test, device):
    rng = np.random.default_rng(4)
    a, b = rng.standard_normal((96, 64)).astype(np.float32), rng.standard_normal((64, 80)).astype(np.float32)
    wa, wb = wp.array(a, device=device), wp.array(b, device=device)
    out = wp.zeros((96, 80), dtype=float, device=device)
    wp.gemm(wa, wb, out)  # create the plan (and autotune) outside the capture
    out.zero_()
    with wp.ScopedCapture(device) as cap:
        wp.gemm(wa, wb, out, beta=1.0)
    for _ in range(3):
        wp.capture_launch(cap.graph)
    assert_np_equal(out.numpy(), 3.0 * (a @ b), tol=1e-3)


def test_gemm_tape_backward(test, device):
    rng = np.random.default_rng(5)
    M, N, K = 33, 20, 17
    for ta, tb in [(False, False), (True, False), (False, True), (True, True)]:
        an = rng.standard_normal((K, M) if ta else (M, K)).astype(np.float32)
        bn = rng.standard_normal((N, K) if tb else (K, N)).astype(np.float32)
        biasn = rng.standard_normal(N).astype(np.float32)
        gn = rng.standard_normal((M, N)).astype(np.float32)
        a = wp.array(an, device=device, requires_grad=True)
        b = wp.array(bn, device=device, requires_grad=True)
        bias = wp.array(biasn, device=device, requires_grad=True)
        tape = wp.Tape()
        with tape:
            out = wp.gemm(a, b, bias=bias, trans_a=ta, trans_b=tb, alpha=0.5)
        out.grad = wp.array(gn, device=device)
        tape.backward()
        opa, opb = (an.T if ta else an), (bn.T if tb else bn)
        assert_np_equal(out.numpy(), 0.5 * opa @ opb + biasn, tol=1e-4)
        ga, gb = 0.5 * gn @ opb.T, 0.5 * opa.T @ gn
        assert_np_equal(a.grad.numpy(), ga.T if ta else ga, tol=1e-4)
        assert_np_equal(b.grad.numpy(), gb.T if tb else gb, tol=1e-4)
        assert_np_equal(bias.grad.numpy(), gn.sum(axis=0), tol=1e-4)


def test_gemm_record_tape_false(test, device):
    rng = np.random.default_rng(3)
    a = wp.array(rng.standard_normal((8, 5)).astype(np.float32), device=device, requires_grad=True)
    b = wp.array(rng.standard_normal((5, 6)).astype(np.float32), device=device, requires_grad=True)
    tape = wp.Tape()
    with tape:
        out = wp.gemm(a, b, record_tape=False)
    # nothing is recorded, and the output allocated by the call does not require gradients
    test.assertEqual(len(tape.launches), 0)
    test.assertFalse(out.requires_grad)
    np.testing.assert_allclose(out.numpy(), a.numpy() @ b.numpy(), rtol=1e-5, atol=1e-5)


def test_gemm_float64_batched(test, device):
    rng = np.random.default_rng(6)
    a = rng.standard_normal((3, 10, 6))
    b = rng.standard_normal((3, 5, 6))
    c = rng.standard_normal((3, 10, 5))
    out = wp.array(c, dtype=wp.float64, device=device)
    wp.gemm(
        wp.array(a, dtype=wp.float64, device=device),
        wp.array(b, dtype=wp.float64, device=device),
        out,
        trans_b=True,
        alpha=0.25,
        beta=-1.5,
    )
    assert_np_equal(out.numpy(), 0.25 * a @ b.transpose(0, 2, 1) - 1.5 * c, tol=1e-12)


def test_gemm_mixed_precision(test, device):
    """float16 / bfloat16 operands with float32 outputs (and bias)."""
    rng = np.random.default_rng(8)
    for dtype, np_dtype in ((wp.float16, np.float16), (wp.bfloat16, None)):
        a = rng.standard_normal((48, 32)).astype(np.float32)
        b = rng.standard_normal((40, 32)).astype(np.float32)
        bias = rng.standard_normal(40).astype(np.float32)
        wa, wb = wp.array(a, dtype=dtype, device=device), wp.array(b, dtype=dtype, device=device)
        # reference on the operands rounded to their precision
        ar, br = wa.numpy().astype(np.float32), wb.numpy().astype(np.float32)
        out = wp.zeros((48, 40), dtype=wp.float32, device=device)
        wp.gemm(wa, wb, out, trans_b=True, bias=wp.array(bias, device=device))
        assert_np_equal(out.numpy(), ar @ br.T + bias, tol=1e-3)
    with test.assertRaisesRegex(TypeError, "output dtype"):
        wp.gemm(
            wp.zeros((4, 4), dtype=wp.float32, device=device),
            wp.zeros((4, 4), dtype=wp.float32, device=device),
            wp.zeros((4, 4), dtype=wp.float16, device=device),
        )


def test_gemm_argument_checks(test, device):
    a = wp.zeros((4, 3), dtype=float, device=device)
    b = wp.zeros((3, 2), dtype=float, device=device)
    with test.assertRaisesRegex(TypeError, "dtype"):
        wp.gemm(a, b, wp.zeros((4, 2), dtype=wp.float64, device=device))
    with test.assertRaisesRegex(ValueError, "aux requires activation='gelu'"):
        wp.gemm(a, b, activation="relu", aux=wp.zeros((4, 2), dtype=float, device=device))
    with test.assertRaisesRegex(ValueError, "shape"):
        wp.gemm(a, a)
    with test.assertRaisesRegex(ValueError, "bias must be a 1D array"):
        wp.gemm(a, b, bias=wp.zeros(3, dtype=float, device=device))
    other = "cuda:0" if device.is_cpu and wp.is_cuda_available() else "cpu"
    with test.assertRaisesRegex(ValueError, "is on device"):
        wp.gemm(a, wp.zeros((3, 2), dtype=float, device=other))


def test_gemm_api_capture_save_load(test, device):
    """A GEMM inside an API capture uses the built-in kernel, so saved graphs contain it."""
    rng = np.random.default_rng(7)
    an, bn = rng.standard_normal((24, 16)).astype(np.float32), rng.standard_normal((16, 8)).astype(np.float32)
    biasn = rng.standard_normal(8).astype(np.float32)
    a, b = wp.array(an, device=device), wp.array(bn, device=device)
    bias = wp.array(biasn, device=device)
    out = wp.zeros((24, 8), dtype=float, device=device)
    wp.gemm(a, b, out, bias=bias, activation="relu")  # load the modules outside of the capture
    out.zero_()
    with wp.ScopedCapture(device=device, apic=True) as capture:
        wp.gemm(a, b, out, bias=bias, activation="relu")
    expected = np.maximum(an @ bn + biasn, 0.0)
    wp.capture_launch(capture.graph)
    assert_np_equal(out.numpy(), expected, tol=1e-4)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "gemm_graph")
        wp.capture_save(capture.graph, path, inputs={"a": a, "b": b, "bias": bias}, outputs={"out": out})
        loaded = wp.capture_load(path, device=device)
        wp.capture_launch(loaded)
        result = wp.zeros((24, 8), dtype=float, device=device)
        loaded.get_param("out", result)
        assert_np_equal(result.numpy(), expected, tol=1e-4)
        del loaded


def test_gemm_is_cublaslt_available(test, device):
    available = wp.is_cublaslt_available()
    test.assertIsInstance(available, bool)
    if not wp.is_cuda_available():
        test.assertFalse(available)


def _with_builtin_kernel(fn):
    """Run a test with the built-in Warp kernel on CUDA devices too."""

    def wrapped(test, device):
        import warp._src.gemm as gemm_module  # noqa: PLC0415

        gemm_module._force_fallback = True
        try:
            fn(test, device)
        finally:
            gemm_module._force_fallback = False

    return wrapped


class TestGemm(unittest.TestCase):
    pass


devices = get_test_devices(mode="basic")
cuda_devices = get_selected_cuda_test_devices(mode="basic")

_tests = [
    ("test_gemm_basic", test_gemm_basic),
    ("test_gemm_alpha_beta_bias_activation", test_gemm_alpha_beta_bias_activation),
    ("test_gemm_backward_epilogues", test_gemm_backward_epilogues),
    ("test_gemm_batched_and_misaligned", test_gemm_batched_and_misaligned),
    ("test_gemm_graph_capture", test_gemm_graph_capture),
    ("test_gemm_tape_backward", test_gemm_tape_backward),
    ("test_gemm_record_tape_false", test_gemm_record_tape_false),
    ("test_gemm_float64_batched", test_gemm_float64_batched),
    ("test_gemm_mixed_precision", test_gemm_mixed_precision),
]
for _name, _fn in _tests:
    add_function_test(TestGemm, _name, _fn, devices=devices, check_output=False)
    # the built-in kernel on CUDA devices (used without cuBLASLt and in API captures)
    add_function_test(
        TestGemm, _name + "_builtin_kernel", _with_builtin_kernel(_fn), devices=cuda_devices, check_output=False
    )
add_function_test(TestGemm, "test_gemm_argument_checks", test_gemm_argument_checks, devices=devices, check_output=False)
add_function_test(
    TestGemm, "test_gemm_api_capture_save_load", test_gemm_api_capture_save_load, devices=devices, check_output=False
)
add_function_test(
    TestGemm, "test_gemm_is_cublaslt_available", test_gemm_is_cublaslt_available, devices=devices, check_output=False
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
