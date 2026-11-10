# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""cuBLASLt-backed GEMM for Warp arrays.

The wrapper loads cuBLASLt lazily at runtime (it is not linked into ``warp.so``), caches one heuristic plan
per problem shape and issues a single ``cublasLtMatmul()`` per call. Plans are created on the first call
(do a warm-up call before graph capture to keep plan creation outside of the captured region), and the
launch itself is safe for CUDA graph capture.
"""

from __future__ import annotations

import atexit
import ctypes
import glob
import os
import sys

import warp._src.context as _ctx
import warp._src.types as _types

# cudaDataType_t
_CUDA_R_32F, _CUDA_R_64F, _CUDA_R_16F, _CUDA_R_16BF = 0, 1, 2, 14
# cublasComputeType_t
_COMPUTE_32F, _COMPUTE_32F_TF32, _COMPUTE_64F, _COMPUTE_32F_16F = 68, 77, 70, 74
# cublasLtEpilogue_t
_EP = {
    (None, False): 0,
    (None, True): 4,
    ("relu", False): 2,
    ("relu", True): 6,
    ("gelu", False): 32,
    ("gelu", True): 36,
}
_EP_RELU_AUX_BIAS, _EP_RELU_AUX, _EP_GELU_AUX_BIAS, _EP_GELU_AUX = 134, 130, 164, 160
_EP_DRELU_BGRAD, _EP_DRELU, _EP_DGELU_BGRAD, _EP_DGELU = 152, 136, 208, 192
_EP_BGRADA = 256

_DTYPES = {
    _types.float32: (_CUDA_R_32F, 4),
    _types.float64: (_CUDA_R_64F, 8),
    _types.float16: (_CUDA_R_16F, 2),
    _types.bfloat16: (_CUDA_R_16BF, 2),
}

_state = {"loaded": None, "version": 0}
_plans: dict = {}


def _lib_candidates():
    """Candidate cuBLASLt libraries, newest major version first.

    Newer cuBLASLt releases carry native kernels for newer GPU architectures (older ones fall back to
    forward-compatible PTX kernels), so CUDA 13 libraries are preferred over CUDA 12 ones, and pip wheels
    (``nvidia-cublas``) or the configured CUDA toolkit are preferred over whatever the loader finds first.
    """
    env = os.environ.get("WARP_CUBLASLT_PATH")
    if env:
        yield env

    def found(patterns):
        for base in sys.path:
            for p in patterns:
                yield from sorted(glob.glob(os.path.join(base, p)), reverse=True)
        for var in ("WARP_CUDA_PATH", "CUDA_HOME", "CUDA_PATH"):
            root = os.environ.get(var)
            if root:
                for p in patterns:
                    name = os.path.basename(p)
                    yield from sorted(glob.glob(os.path.join(root, "lib*", name)), reverse=True)
                    yield from sorted(glob.glob(os.path.join(root, "targets", "*", "lib", name)), reverse=True)

    for major in ("13", "12"):
        yield from found(
            (f"nvidia/cu{major}/lib/libcublasLt.so.{major}*", f"nvidia/cublas/lib/libcublasLt.so.{major}*")
        )
        yield f"libcublasLt.so.{major}"
        yield f"cublasLt64_{major}.dll"
    yield "libcublasLt.so"


def _setup_signatures(core):
    core.wp_cublaslt_init.argtypes = [ctypes.c_char_p]
    core.wp_cublaslt_init.restype = ctypes.c_int
    # NB: argument order must match the native signature (see wp_cublaslt_plan_create)
    core.wp_cublaslt_plan_create.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,  # a,b,c types, compute, scale
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,  # ta, tb, M, N, K
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,  # lda ldb ldc
        ctypes.c_int,
        ctypes.c_longlong,
        ctypes.c_longlong,
        ctypes.c_longlong,  # batch, strides
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,  # epilogue, bias_type, aux_ld, aux_type, tune, align
    ]
    core.wp_cublaslt_plan_create.restype = ctypes.c_void_p
    core.wp_cublaslt_plan_run.argtypes = [ctypes.c_void_p] * 7 + [ctypes.c_float, ctypes.c_float]
    core.wp_cublaslt_plan_run.restype = ctypes.c_bool
    core.wp_cublaslt_plan_destroy.argtypes = [ctypes.c_void_p]
    core.wp_cublaslt_plan_destroy.restype = None


def _ensure_loaded():
    if _state["loaded"] is not None:
        if not _state["loaded"]:
            raise RuntimeError("cuBLASLt could not be loaded; set WARP_CUBLASLT_PATH to libcublasLt")
        return
    if _ctx.runtime is None:
        _ctx.init()
    core = _ctx.runtime.core
    _setup_signatures(core)
    for cand in _lib_candidates():
        v = core.wp_cublaslt_init(cand.encode())
        if v:
            _state["loaded"], _state["version"] = True, v
            return
    _state["loaded"] = False
    raise RuntimeError("cuBLASLt could not be loaded; set WARP_CUBLASLT_PATH to libcublasLt")


def _destroy_plans():
    if _state["loaded"]:
        for p in _plans.values():
            _ctx.runtime.core.wp_cublaslt_plan_destroy(p)
    _plans.clear()


atexit.register(_destroy_plans)


def _mat_info(a, trans):
    """Return (rows, cols, ld, batch, batch_stride) of op(a) given a 2D/3D row-major array."""
    if a.ndim == 2:
        batch, bs = 1, 0
        r, c = a.shape
        ld = a.strides[0] // a.strides[1]
        inner = a.strides[1]
    elif a.ndim == 3:
        batch, bs = a.shape[0], a.strides[0] // a.strides[2]
        r, c = a.shape[1], a.shape[2]
        ld = a.strides[1] // a.strides[2]
        inner = a.strides[2]
    else:
        raise ValueError("gemm() expects 2D or 3D (batched) arrays")
    if inner != _DTYPES[a.dtype][1] or ld < c:
        raise ValueError("gemm() requires arrays with contiguous rows")
    if trans:
        r, c = c, r
    return r, c, ld, batch, bs


def gemm(
    a,
    b,
    out=None,
    *,
    alpha: float = 1.0,
    beta: float = 0.0,
    trans_a: bool = False,
    trans_b: bool = False,
    bias=None,
    activation: str | None = None,
    aux=None,
    grad_aux=None,
    bias_grad=None,
    tf32: bool = False,
    autotune: int = 8,
    stream=None,
):
    """Compute ``out = activation(alpha * op(a) @ op(b) + bias) + beta * out`` with cuBLASLt.

    All arrays are row-major with contiguous rows, either 2D ``(rows, cols)`` or 3D ``(batch, rows, cols)``.
    The call enqueues a single kernel launch on the device's current stream and can be captured in a CUDA graph.

    Args:
        a: Left operand, shape ``(M, K)`` (or ``(K, M)`` if ``trans_a``).
        b: Right operand, shape ``(K, N)`` (or ``(N, K)`` if ``trans_b``).
        out: Output array of shape ``(M, N)``. Allocated if ``None``.
        alpha: Scale of the matrix product.
        beta: Scale of the existing contents of ``out`` (0 overwrites). Not combined with ``activation``.
        trans_a: Use the transpose of ``a``.
        trans_b: Use the transpose of ``b``.
        bias: Optional 1D array of length ``N`` added to each output row (fused epilogue).
        activation: Optional fused activation: ``"relu"`` or ``"gelu"`` (tanh approximation).
        aux: Optional output buffer ``(M, N)`` storing the pre-activation data needed for a fused activation backward pass.
        grad_aux: If given, runs the backward epilogue ``out = (op(a) @ op(b)) * gelu'(grad_aux)`` where ``grad_aux`` is
            the pre-activation buffer saved by a forward call with ``aux``. Requires ``activation="gelu"``.
        bias_grad: Optional 1D output array of length ``N``. With ``grad_aux`` it receives the column sums of the result.
            Without ``grad_aux`` (weight-gradient GEMM ``a.T @ b`` with ``trans_a=True``) it receives the column sums
            of ``b`` over the reduction dimension, i.e. the bias gradient of a linear layer.
        tf32: Allow TF32 tensor cores for float32 inputs.
        autotune: Number of cuBLASLt heuristic candidates to time on the first (non-captured) call with ``beta == 0``;
            the fastest is kept. Use 0 or 1 to disable.
        stream: CUDA stream to launch on, defaults to the current stream of the device.

    Returns:
        The output array.
    """
    _ensure_loaded()
    if a.dtype != b.dtype:
        raise TypeError("gemm() operands must have the same dtype")
    if a.dtype not in _DTYPES:
        raise TypeError(f"gemm() unsupported dtype {a.dtype}")
    M, K, lda, batch, sa = _mat_info(a, trans_a)
    K2, N, ldb, batch_b, sb = _mat_info(b, trans_b)
    if K != K2 or batch != batch_b:
        raise ValueError(f"gemm() shape mismatch: op(a) is ({M},{K}), op(b) is ({K2},{N})")
    if out is None:
        out = _ctx.empty(((batch, M, N) if a.ndim == 3 else (M, N)), dtype=a.dtype, device=a.device)
    M_c, N_c, ldc, batch_c, sc = _mat_info(out, False)
    if (M_c, N_c, batch_c) != (M, N, batch):
        raise ValueError(
            f"gemm() output shape ({batch_c}, {M_c}, {N_c}) does not match the product ({batch}, {M}, {N})"
        )
    dt, _ = _DTYPES[a.dtype]
    if a.dtype == _types.float64:
        compute, scale = _COMPUTE_64F, _CUDA_R_64F
    else:
        compute = _COMPUTE_32F_TF32 if (tf32 and a.dtype == _types.float32) else _COMPUTE_32F
        scale = _CUDA_R_32F
    if beta != 0.0 and activation is not None:
        raise ValueError("gemm() does not support beta != 0 together with a fused activation")
    ep = _EP[(activation, bias is not None)]
    aux_ld = 0
    if aux is not None:
        if activation is None:
            raise ValueError("aux requires an activation")
        ep = {
            ("relu", True): _EP_RELU_AUX_BIAS,
            ("relu", False): _EP_RELU_AUX,
            ("gelu", True): _EP_GELU_AUX_BIAS,
            ("gelu", False): _EP_GELU_AUX,
        }[(activation, bias is not None)]
    if grad_aux is not None:
        if activation != "gelu" or aux is not None:
            raise ValueError("grad_aux requires activation='gelu' and no aux output")
        ep = _EP_DGELU_BGRAD if bias_grad is not None else _EP_DGELU
        aux_ld = N
    elif aux is not None:
        aux_ld = N
    elif bias_grad is not None:
        # weight-gradient GEMM out = op(a) @ op(b): fuse the column sums of op(b) over K (the batch) into bias_grad
        if activation is not None or bias is not None or not trans_a or trans_b:
            raise ValueError("bias_grad without grad_aux requires trans_a=True, trans_b=False and no bias/activation")
        ep = _EP_BGRADA
    # pointer alignment is part of the plan: cuBLASLt algorithms are only valid for the alignment they were chosen for
    min_align = 256
    for arr in (a, b, out, bias, aux, grad_aux, bias_grad):
        if arr is not None and arr.ptr:
            min_align = min(min_align, arr.ptr & -arr.ptr)
    for ld_elems in (lda, ldb, ldc):
        ld_bytes = ld_elems * _DTYPES[a.dtype][1]
        min_align = min(min_align, ld_bytes & -ld_bytes)
    key = (
        a.device.context,
        min_align,
        dt,
        compute,
        trans_a,
        trans_b,
        M,
        N,
        K,
        lda,
        ldb,
        ldc,
        batch,
        sa,
        sb,
        sc,
        ep,
        aux_ld,
    )
    plan = _plans.get(key)
    if plan is None:
        plan = _ctx.runtime.core.wp_cublaslt_plan_create(
            a.device.context,
            dt,
            dt,
            dt,
            compute,
            scale,
            int(trans_a),
            int(trans_b),
            M,
            N,
            K,
            lda,
            ldb,
            ldc,
            batch,
            sa,
            sb,
            sc,
            ep,
            dt,
            aux_ld,
            dt,
            autotune,
            min_align,
        )
        if not plan:
            raise RuntimeError(
                f"gemm() failed to create a cuBLASLt plan: {_ctx.runtime.core.wp_get_error_string().decode()}"
            )
        _plans[key] = plan
    if stream is None:
        stream_ptr = a.device.stream.cuda_stream
    else:
        stream_ptr = stream.cuda_stream
    ok = _ctx.runtime.core.wp_cublaslt_plan_run(
        plan,
        stream_ptr,
        a.ptr,
        b.ptr,
        out.ptr,
        (bias_grad.ptr if bias_grad is not None else (bias.ptr if bias is not None else None)),
        (grad_aux.ptr if grad_aux is not None else (aux.ptr if aux is not None else None)),
        float(alpha),
        float(beta),
    )
    if not ok:
        raise RuntimeError(f"gemm() launch failed: {_ctx.runtime.core.wp_get_error_string().decode()}")
    return out
