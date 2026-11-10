# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for :func:`warp.gemm` (runtime-loaded cuBLASLt)."""

import unittest

import numpy as np

import warp as wp
from warp.tests.unittest_utils import *


def _cublaslt_available():
    try:
        from warp._src.gemm import _ensure_loaded  # noqa: PLC0415

        _ensure_loaded()
        return True
    except Exception:
        return False


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


devices = get_selected_cuda_test_devices(mode="basic") if _cublaslt_available() else []


class TestGemm(unittest.TestCase):
    pass


for _name, _fn in [
    ("test_gemm_basic", test_gemm_basic),
    ("test_gemm_alpha_beta_bias_activation", test_gemm_alpha_beta_bias_activation),
    ("test_gemm_backward_epilogues", test_gemm_backward_epilogues),
    ("test_gemm_batched_and_misaligned", test_gemm_batched_and_misaligned),
    ("test_gemm_graph_capture", test_gemm_graph_capture),
]:
    add_function_test(TestGemm, _name, _fn, devices=devices, check_output=False)


if __name__ == "__main__":
    unittest.main(verbosity=2)
