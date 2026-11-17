# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for :func:`warp.gemm` (runtime-loaded cuBLASLt, and the built-in Warp kernel used without it)."""

import os
import tempfile
import unittest
from typing import Any

import numpy as np

import warp as wp
from warp.tests.unittest_utils import *


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
    out = wp.gemm(wa, wb, activation="relu")
    assert_np_equal(out.numpy(), np.maximum(a @ b, 0.0), tol=1e-4)


def test_gemm_bias_grad(test, device):
    rng = np.random.default_rng(2)
    B, D, H = 256, 32, 48
    dy = rng.standard_normal((B, D)).astype(np.float32)
    # weight gradient with fused bias gradient (BGRADA): gW = X^T @ dY, gb = sum_rows(dY)
    x = rng.standard_normal((B, H)).astype(np.float32)
    gw = wp.zeros((H, D), dtype=float, device=device)
    gb2 = wp.zeros(D, dtype=float, device=device)
    wp.gemm(wp.array(x, device=device), wp.array(dy, device=device), gw, trans_a=True, bias_grad=gb2)
    assert_np_equal(gw.numpy(), x.T @ dy, tol=1e-3)
    assert_np_equal(gb2.numpy(), dy.sum(axis=0), tol=1e-3)
    # batched: per-batch weight gradients, one bias gradient summed over the batch as well
    xb = rng.standard_normal((3, B, H)).astype(np.float32)
    dyb = rng.standard_normal((3, B, D)).astype(np.float32)
    gwb = wp.zeros((3, H, D), dtype=float, device=device)
    gbb = wp.zeros(D, dtype=float, device=device)
    wp.gemm(wp.array(xb, device=device), wp.array(dyb, device=device), gwb, trans_a=True, bias_grad=gbb)
    assert_np_equal(gwb.numpy(), np.swapaxes(xb, 1, 2) @ dyb, tol=1e-3)
    assert_np_equal(gbb.numpy(), dyb.sum(axis=(0, 1)), tol=1e-3)


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


def _to_float32(x):
    v = x.numpy()
    if x.dtype == wp.bfloat16 and v.dtype == np.uint16:  # raw bits unless ml_dtypes is installed
        return (v.astype(np.uint32) << 16).view(np.float32)
    return v.astype(np.float32)


def test_gemm_mixed_precision(test, device):
    """float16 / bfloat16 operands with float32 outputs (and bias)."""
    rng = np.random.default_rng(8)
    for dtype in (wp.float16, wp.bfloat16):
        a = rng.standard_normal((48, 32)).astype(np.float32)
        b = rng.standard_normal((40, 32)).astype(np.float32)
        bias = rng.standard_normal(40).astype(np.float32)
        wa, wb = wp.array(a, dtype=dtype, device=device), wp.array(b, dtype=dtype, device=device)
        # reference on the operands rounded to their precision
        ar, br = _to_float32(wa), _to_float32(wb)
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
    with test.assertRaisesRegex(ValueError, "unsupported activation"):
        wp.gemm(a, b, activation="gelu")
    with test.assertRaisesRegex(ValueError, "bias_grad requires trans_a=True"):
        wp.gemm(a, b, bias_grad=wp.zeros(2, dtype=float, device=device))
    with test.assertRaisesRegex(ValueError, "shape"):
        wp.gemm(a, a)
    with test.assertRaisesRegex(ValueError, "bias must be a 1D array"):
        wp.gemm(a, b, bias=wp.zeros(3, dtype=float, device=device))
    other = "cuda:0" if device.is_cpu and wp.is_cuda_available() else "cpu"
    with test.assertRaisesRegex(ValueError, "is on device"):
        wp.gemm(a, wp.zeros((3, 2), dtype=float, device=other))


def test_gemm_api_capture_save_load(test, device):
    """A GEMM inside an API capture is recorded as a GEMM operation that saved graphs replay natively."""
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
        test.assertTrue(wp._src.context.runtime.core.wp_apic_graph_has_op(loaded._native_graph, 21))  # APIC_OP_GEMM
        wp.capture_launch(loaded)
        result = wp.zeros((24, 8), dtype=float, device=device)
        loaded.get_param("out", result)
        assert_np_equal(result.numpy(), expected, tol=1e-4)
        del loaded


def test_gemm_api_capture_transposed_and_bias_grad(test, device):
    """Saved graphs replay transposed, batched and bias-gradient GEMMs."""
    rng = np.random.default_rng(9)
    xn, gn = rng.standard_normal((40, 12)).astype(np.float32), rng.standard_normal((40, 7)).astype(np.float32)
    an, bn = rng.standard_normal((3, 5, 6)).astype(np.float32), rng.standard_normal((3, 4, 6)).astype(np.float32)
    x, g = wp.array(xn, device=device), wp.array(gn, device=device)
    a, b = wp.array(an, device=device), wp.array(bn, device=device)
    dw = wp.zeros((12, 7), dtype=float, device=device)
    db = wp.zeros(7, dtype=float, device=device)
    c = wp.zeros((3, 5, 4), dtype=float, device=device)
    wp.gemm(x, g, dw, trans_a=True, bias_grad=db)
    wp.gemm(a, b, c, trans_b=True, alpha=0.5)
    with wp.ScopedCapture(device=device, apic=True) as capture:
        wp.gemm(x, g, dw, trans_a=True, bias_grad=db)
        wp.gemm(a, b, c, trans_b=True, alpha=0.5)
    expected_c = 0.5 * an @ np.swapaxes(bn, 1, 2)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "gemm_graph")
        wp.capture_save(capture.graph, path, inputs={"x": x, "g": g}, outputs={"dw": dw, "db": db, "c": c})
        loaded = wp.capture_load(path, device=device)
        wp.capture_launch(loaded)
        results = {
            "dw": wp.zeros((12, 7), dtype=float, device=device),
            "db": wp.zeros(7, dtype=float, device=device),
            "c": wp.zeros((3, 5, 4), dtype=float, device=device),
        }
        for name, arr in results.items():
            loaded.get_param(name, arr)
        assert_np_equal(results["dw"].numpy(), xn.T @ gn, tol=1e-4)
        assert_np_equal(results["db"].numpy(), gn.sum(axis=0), tol=1e-4)
        assert_np_equal(results["c"].numpy(), expected_c, tol=1e-4)
        del loaded


@wp.func
def _silu(x: Any):
    return x / (x.dtype(1.0) + wp.exp(-x))


@wp.func
def _odd_columns_tanh(x: float, i: int, j: int):
    if j % 2 == 1:
        return wp.tanh(x)
    return x + float(i)


def _silu_np(x):
    return x / (1.0 + np.exp(-x))


def test_gemm_func_activation(test, device):
    """A warp.func activation is applied elementwise after the GEMM and its bias."""
    rng = np.random.default_rng(10)
    M, N, K = 37, 20, 13
    an, bn = rng.standard_normal((M, K)), rng.standard_normal((K, N))
    biasn = rng.standard_normal(N)
    for dtype, tol in ((wp.float32, 1e-4), (wp.float64, 1e-10)):
        np_t = np.float32 if dtype == wp.float32 else np.float64
        a, b = wp.array(an.astype(np_t), device=device), wp.array(bn.astype(np_t), device=device)
        bias = wp.array(biasn.astype(np_t), device=device)
        out = wp.gemm(a, b, bias=bias, alpha=0.5, activation=_silu)
        assert_np_equal(out.numpy(), _silu_np(0.5 * an @ bn + biasn), tol=tol)
    a32, b32 = wp.array(an.astype(np.float32), device=device), wp.array(bn.astype(np.float32), device=device)
    # f(x, i, j) sees the row and column of the element
    pre = an @ bn
    expected = np.where(np.arange(N) % 2 == 1, np.tanh(pre), pre + np.arange(M)[:, None])
    out = wp.gemm(a32, b32, activation=_odd_columns_tanh)
    assert_np_equal(out.numpy(), expected, tol=1e-4)
    # batched, transposed operand
    ab, bb = rng.standard_normal((3, K, M)).astype(np.float32), rng.standard_normal((3, K, N)).astype(np.float32)
    out = wp.gemm(wp.array(ab, device=device), wp.array(bb, device=device), trans_a=True, activation=_silu)
    assert_np_equal(out.numpy(), _silu_np(np.swapaxes(ab, 1, 2) @ bb), tol=1e-4)
    # batched with f(x, i, j): i and j index the rows and columns of each matrix
    out = wp.gemm(wp.array(ab, device=device), wp.array(bb, device=device), trans_a=True, activation=_odd_columns_tanh)
    pre_b = np.swapaxes(ab, 1, 2) @ bb
    expected_b = np.where(np.arange(N) % 2 == 1, np.tanh(pre_b), pre_b + np.arange(M)[None, :, None])
    assert_np_equal(out.numpy(), expected_b, tol=1e-4)
    # strided output (a column block of a wider array) takes the indexed kernel instead of the flat one
    wide = wp.zeros((M, 2 * N), dtype=float, device=device)
    view = wide[:, N:]
    test.assertFalse(view.is_contiguous)
    wp.gemm(a32, b32, view, activation=_odd_columns_tanh)
    assert_np_equal(wide.numpy()[:, N:], expected, tol=1e-4)
    assert_np_equal(wide.numpy()[:, :N], np.zeros((M, N)), tol=0.0)
    # half-precision operands and output
    a16, b16 = wp.array(an.astype(np.float16), device=device), wp.array(bn.astype(np.float16), device=device)
    out = wp.gemm(a16, b16, activation=_silu)
    ref = _silu_np(an.astype(np.float16).astype(np.float32) @ bn.astype(np.float16).astype(np.float32))
    assert_np_equal(out.numpy().astype(np.float32), ref, tol=2e-2)
    with test.assertRaisesRegex(TypeError, "must take"):

        @wp.func
        def two_args(x: float, i: int):
            return x

        wp.gemm(a32, b32, activation=two_args)


def test_gemm_activation_tape_backward(test, device):
    """Fused ReLU and warp.func activations are recorded on a tape; the activation adjoint is Warp's."""
    rng = np.random.default_rng(11)
    M, N, K = 29, 18, 11
    for ta, tb in [(False, False), (True, False), (False, True)]:
        an = rng.standard_normal((K, M) if ta else (M, K)).astype(np.float32)
        bn = rng.standard_normal((N, K) if tb else (K, N)).astype(np.float32)
        biasn = rng.standard_normal(N).astype(np.float32)
        gn = rng.standard_normal((M, N)).astype(np.float32)
        opa, opb = (an.T if ta else an), (bn.T if tb else bn)
        pre = 0.5 * opa @ opb + biasn
        sig = 1.0 / (1.0 + np.exp(-pre))
        for activation, y, dpre in (
            ("relu", np.maximum(pre, 0.0), gn * (pre > 0.0)),
            (_silu, pre * sig, gn * sig * (1.0 + pre * (1.0 - sig))),
        ):
            a = wp.array(an, device=device, requires_grad=True)
            b = wp.array(bn, device=device, requires_grad=True)
            bias = wp.array(biasn, device=device, requires_grad=True)
            tape = wp.Tape()
            with tape:
                out = wp.gemm(a, b, bias=bias, trans_a=ta, trans_b=tb, alpha=0.5, activation=activation)
            out.grad = wp.array(gn, device=device)
            tape.backward()
            assert_np_equal(out.numpy(), y, tol=1e-4)
            ga, gb = 0.5 * dpre @ opb.T, 0.5 * opa.T @ dpre
            assert_np_equal(a.grad.numpy(), ga.T if ta else ga, tol=1e-3)
            assert_np_equal(b.grad.numpy(), gb.T if tb else gb, tol=1e-3)
            assert_np_equal(bias.grad.numpy(), dpre.sum(axis=0), tol=1e-3)


def test_gemm_func_activation_api_capture(test, device):
    """A GEMM with a warp.func activation is saved as a GEMM operation followed by the activation kernel."""
    rng = np.random.default_rng(12)
    an, bn = rng.standard_normal((24, 16)).astype(np.float32), rng.standard_normal((16, 8)).astype(np.float32)
    biasn = rng.standard_normal(8).astype(np.float32)
    a, b, bias = wp.array(an, device=device), wp.array(bn, device=device), wp.array(biasn, device=device)
    out = wp.zeros((24, 8), dtype=float, device=device)
    wp.gemm(a, b, out, bias=bias, activation=_silu)  # load the modules outside of the capture
    out.zero_()
    with wp.ScopedCapture(device=device, apic=True) as capture:
        wp.gemm(a, b, out, bias=bias, activation=_silu)
    expected = _silu_np(an @ bn + biasn)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "gemm_graph")
        wp.capture_save(capture.graph, path, inputs={"a": a, "b": b, "bias": bias}, outputs={"out": out})
        loaded = wp.capture_load(path, device=device)
        test.assertTrue(wp._src.context.runtime.core.wp_apic_graph_has_op(loaded._native_graph, 21))  # APIC_OP_GEMM
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


def test_gemm_cublaslt_search_order(test, device):
    """Installed cuBLASLt libraries are ranked newest first, without loading them."""
    import json  # noqa: PLC0415

    import warp._src.gemm as gemm_module  # noqa: PLC0415

    def touch(path, text=""):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(text)

    with tempfile.TemporaryDirectory() as tmp:
        site = os.path.join(tmp, "site-packages")
        wheel12 = os.path.join(site, "nvidia", "cublas", "lib", "libcublasLt.so.12")
        wheel13 = os.path.join(site, "nvidia", "cu13", "lib", "libcublasLt.so.13")
        touch(wheel12)
        touch(wheel13)
        touch(
            os.path.join(site, "nvidia_cublas_cu12-12.8.4.1.dist-info", "RECORD"),
            "nvidia/cublas/lib/libcublasLt.so.12,,1\n",
        )
        touch(
            os.path.join(site, "nvidia_cublas-13.1.0.3.dist-info", "RECORD"), "nvidia/cu13/lib/libcublasLt.so.13,,1\n"
        )
        # a toolkit with version.json, and one known only by its directory name
        tk129 = os.path.join(tmp, "toolkit")
        touch(os.path.join(tk129, "version.json"), json.dumps({"libcublas": {"version": "12.9.1.4"}}))
        touch(os.path.join(tk129, "lib64", "libcublasLt.so.12"))
        tk126 = os.path.join(tmp, "cuda-12.6")
        touch(os.path.join(tk126, "lib64", "libcublasLt.so.12"))
        system = os.path.join(tmp, "system")
        touch(os.path.join(system, "libcublasLt.so.12"))

        found = gemm_module._find_cublaslt([site], [tk126, tk129], [system])
        test.assertEqual([v for v, _ in found], [(13, 1, 0, 3), (12, 9, 1, 4), (12, 8, 4, 1), (12, 6), (12,)])
        test.assertEqual(os.path.realpath(found[0][1]), os.path.realpath(wheel13))
        test.assertEqual(os.path.realpath(found[2][1]), os.path.realpath(wheel12))
        test.assertEqual(os.path.dirname(found[4][1]), system)

    test.assertEqual(gemm_module._parse_version("12.9.1.4.dist-info"), (12, 9, 1, 4))
    test.assertEqual(gemm_module._parse_version("13.1.0.3"), (13, 1, 0, 3))
    test.assertEqual(gemm_module._parse_version("v12"), ())


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
    ("test_gemm_bias_grad", test_gemm_bias_grad),
    ("test_gemm_batched_and_misaligned", test_gemm_batched_and_misaligned),
    ("test_gemm_graph_capture", test_gemm_graph_capture),
    ("test_gemm_tape_backward", test_gemm_tape_backward),
    ("test_gemm_record_tape_false", test_gemm_record_tape_false),
    ("test_gemm_float64_batched", test_gemm_float64_batched),
    ("test_gemm_mixed_precision", test_gemm_mixed_precision),
    ("test_gemm_func_activation", test_gemm_func_activation),
    ("test_gemm_activation_tape_backward", test_gemm_activation_tape_backward),
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
    TestGemm,
    "test_gemm_api_capture_transposed_and_bias_grad",
    test_gemm_api_capture_transposed_and_bias_grad,
    devices=devices,
    check_output=False,
)
add_function_test(
    TestGemm,
    "test_gemm_func_activation_api_capture",
    test_gemm_func_activation_api_capture,
    devices=devices,
    check_output=False,
)
add_function_test(
    TestGemm, "test_gemm_is_cublaslt_available", test_gemm_is_cublaslt_available, devices=devices, check_output=False
)
add_function_test(
    TestGemm, "test_gemm_cublaslt_search_order", test_gemm_cublaslt_search_order, devices=None, check_output=False
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
