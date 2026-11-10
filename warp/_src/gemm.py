# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GEMM for Warp arrays, backed by cuBLASLt on CUDA devices.

The wrapper loads cuBLASLt lazily at runtime (it is not linked into ``warp.so``), caches one heuristic plan
per problem shape and issues a single ``cublasLtMatmul()`` per call. Plans are created on the first call
(do a warm-up call before graph capture to keep plan creation outside of the captured region), and the
launch itself is safe for CUDA graph capture.

CPU arrays, CUDA devices without a loadable cuBLASLt, and CUDA API captures (``apic=True``, whose operation
stream cannot represent cuBLASLt calls) use a built-in Warp kernel instead: it supports the same arguments,
is recorded like any other kernel launch, and is much slower than cuBLASLt for large problems.
"""

from __future__ import annotations

import atexit
import ctypes
import glob
import os
import sys
from typing import Any

import warp._src.context as _ctx
import warp._src.types as _types
from warp._src.logger import log_warning

_INSTALL_HINT = (
    "install the cuBLAS wheel matching Warp's CUDA version (pip install warp-lang[cublas-cu12] or "
    "warp-lang[cublas-cu13]), or set WARP_CUBLASLT_PATH to libcublasLt"
)

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
            (
                f"nvidia/cu{major}/lib/libcublasLt.so.{major}*",
                f"nvidia/cublas/lib/libcublasLt.so.{major}*",
                f"nvidia/cu{major}/bin/cublasLt64_{major}.dll",
                f"nvidia/cu{major}/bin/*/cublasLt64_{major}.dll",
                f"nvidia/cublas/bin/cublasLt64_{major}.dll",
            )
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
    core.wp_cublaslt_plan_run.argtypes = [ctypes.c_void_p] * 7 + [ctypes.c_double, ctypes.c_double]
    core.wp_cublaslt_plan_run.restype = ctypes.c_bool
    core.wp_cublaslt_plan_destroy.argtypes = [ctypes.c_void_p]
    core.wp_cublaslt_plan_destroy.restype = None


def _ensure_loaded():
    if _state["loaded"] is not None:
        if not _state["loaded"]:
            raise RuntimeError(f"cuBLASLt could not be loaded; {_INSTALL_HINT}")
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
    raise RuntimeError(f"cuBLASLt could not be loaded; {_INSTALL_HINT}")


def is_cublaslt_available() -> bool:
    """Return whether :func:`warp.gemm` can use cuBLASLt on CUDA devices.

    cuBLASLt is loaded at runtime, on the first call of this function or of :func:`warp.gemm`, from the
    ``nvidia-cublas`` wheels (``pip install warp-lang[cublas-cu12]`` or ``warp-lang[cublas-cu13]``), the CUDA
    toolkit or the library search path; ``WARP_CUBLASLT_PATH`` selects a library explicitly. Without it,
    :func:`warp.gemm` uses a much slower built-in Warp kernel.
    """
    if _ctx.runtime is None:
        _ctx.init()
    if not _ctx.is_cuda_available():
        return False
    try:
        _ensure_loaded()
    except RuntimeError:
        return False
    return True


_force_fallback = False  # tests: use the built-in kernel on CUDA devices too


def _select_backend(device) -> str:
    """``"cublaslt"`` or ``"warp"`` (the built-in kernel) for a call on ``device``."""
    if device.is_cpu or _force_fallback:
        return "warp"
    if _ctx._get_apic_capture_for_device(device) is not None:
        log_warning(
            "wp.gemm() inside an API capture (apic=True) uses the built-in Warp GEMM kernel, since the captured "
            "operation stream cannot represent cuBLASLt calls; it is much slower than cuBLASLt for large problems.",
            once=True,
        )
        return "warp"
    if not is_cublaslt_available():
        log_warning(
            f"wp.gemm() uses the built-in Warp GEMM kernel, which is much slower than cuBLASLt: {_INSTALL_HINT}.",
            once=True,
        )
        return "warp"
    return "cublaslt"


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
    record_tape: bool = True,
):
    """Compute ``out = activation(alpha * op(a) @ op(b) + bias) + beta * out``.

    All arrays are row-major with contiguous rows, either 2D ``(rows, cols)`` or 3D ``(batch, rows, cols)``, and
    on the same device. On CUDA devices, the call enqueues a single cuBLASLt kernel launch on the device's current
    stream and can be captured in a CUDA graph. cuBLASLt is loaded at runtime (see
    :func:`warp.is_cublaslt_available`); CPU arrays, CUDA devices without cuBLASLt and API captures
    (``apic=True``) use a built-in Warp kernel instead, which is recorded in CPU graphs and saved graphs like any
    kernel launch but is much slower than cuBLASLt for large problems.

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
        aux: Optional output buffer ``(M, N)`` storing the pre-activation values needed by the backward epilogue
            (``grad_aux``). Requires ``activation="gelu"``.
        grad_aux: If given, runs the backward epilogue ``out = (op(a) @ op(b)) * gelu'(grad_aux)`` where ``grad_aux`` is
            the pre-activation buffer saved by a forward call with ``aux``. Requires ``activation="gelu"``.
        bias_grad: Optional 1D output array of length ``N``. With ``grad_aux`` it receives the column sums of the result.
            Without ``grad_aux`` (weight-gradient GEMM ``a.T @ b`` with ``trans_a=True``) it receives the column sums
            of ``b`` over the reduction dimension, i.e. the bias gradient of a linear layer.
        tf32: Allow TF32 tensor cores for float32 inputs (cuBLASLt only).
        autotune: Number of cuBLASLt heuristic candidates to time on the first (non-captured) call with ``beta == 0``;
            the fastest is kept. Use 0 or 1 to disable.
        stream: CUDA stream to launch on, defaults to the current stream of the device.
        record_tape: Whether to record the adjoint on the active :class:`warp.Tape` (if any). Pass ``False`` when
            the caller records its own backward pass (e.g. with :meth:`warp.Tape.record_func`), like
            ``record_tape`` of :func:`warp.launch`.

    When a :class:`warp.Tape` is recording and ``a``, ``b`` or ``bias`` require gradients, the adjoint GEMMs (and
    the bias column sums) are recorded on the tape, so ``tape.backward()`` accumulates into their ``grad`` arrays.
    This requires ``beta=0`` and no fused activation or backward epilogue.

    Returns:
        The output array.
    """
    if _ctx.runtime is None:
        _ctx.init()
    if a.dtype != b.dtype:
        raise TypeError("gemm() operands must have the same dtype")
    if a.dtype not in _DTYPES:
        raise TypeError(f"gemm() unsupported dtype {a.dtype}")
    for name, x in (
        ("b", b),
        ("out", out),
        ("bias", bias),
        ("aux", aux),
        ("grad_aux", grad_aux),
        ("bias_grad", bias_grad),
    ):
        if x is not None:
            if x.device != a.device:
                raise ValueError(
                    f"gemm() array '{name}' is on device {x.device}, expected {a.device} (the device of 'a')"
                )
            if x.dtype != a.dtype:
                raise TypeError(f"gemm() array '{name}' has dtype {x.dtype}, expected {a.dtype}")
    M, K, lda, batch, sa = _mat_info(a, trans_a)
    K2, N, ldb, batch_b, sb = _mat_info(b, trans_b)
    if K != K2 or batch != batch_b:
        raise ValueError(f"gemm() shape mismatch: op(a) is ({M},{K}), op(b) is ({K2},{N})")
    if out is None:
        recording = (
            record_tape
            and _ctx.runtime.tape is not None
            and any(x is not None and x.requires_grad for x in (a, b, bias))
        )
        out = _ctx.empty(
            ((batch, M, N) if a.ndim == 3 else (M, N)), dtype=a.dtype, device=a.device, requires_grad=recording
        )
    M_c, N_c, _ldc, batch_c, _sc = _mat_info(out, False)
    if (M_c, N_c, batch_c) != (M, N, batch):
        raise ValueError(
            f"gemm() output shape ({batch_c}, {M_c}, {N_c}) does not match the product ({batch}, {M}, {N})"
        )
    if beta != 0.0 and activation is not None:
        raise ValueError("gemm() does not support beta != 0 together with a fused activation")
    if activation not in (None, "relu", "gelu"):
        raise ValueError(f"gemm() unsupported activation {activation!r} (expected None, 'relu' or 'gelu')")
    if aux is not None and activation != "gelu":
        raise ValueError("gemm() aux requires activation='gelu'")
    if grad_aux is not None and (activation != "gelu" or aux is not None):
        raise ValueError("gemm() grad_aux requires activation='gelu' and no aux output")
    if (
        grad_aux is None
        and bias_grad is not None
        and (activation is not None or bias is not None or not trans_a or trans_b)
    ):
        raise ValueError(
            "gemm() bias_grad without grad_aux requires trans_a=True, trans_b=False and no bias/activation"
        )
    for name, x in (("aux", aux), ("grad_aux", grad_aux)):
        if x is not None and tuple(x.shape) != tuple(out.shape):
            raise ValueError(f"gemm() {name} shape {tuple(x.shape)} does not match the output shape {tuple(out.shape)}")
    for name, x in (("bias", bias), ("bias_grad", bias_grad)):
        if x is not None and (x.ndim != 1 or x.shape[0] != N):
            raise ValueError(f"gemm() {name} must be a 1D array of length {N}")
    tape = _ctx.runtime.tape if record_tape else None
    recorded = tape is not None and any(x is not None and x.requires_grad for x in (a, b, bias))
    if recorded and (activation is not None or grad_aux is not None or bias_grad is not None or beta != 0.0):
        raise RuntimeError(
            "gemm() can only be recorded on a wp.Tape without fused activations or backward epilogues and with beta=0"
        )
    if _select_backend(a.device) == "warp":
        _fallback_gemm(a, b, out, alpha, beta, trans_a, trans_b, bias, activation, aux, grad_aux, bias_grad, K)
    else:
        _ensure_loaded()
        _cublaslt_gemm(
            a,
            b,
            out,
            alpha,
            beta,
            trans_a,
            trans_b,
            bias,
            activation,
            aux,
            grad_aux,
            bias_grad,
            tf32,
            autotune,
            stream,
            M,
            N,
            K,
            lda,
            ldb,
            batch,
            sa,
            sb,
        )

    if recorded:
        _record_backward(tape, a, b, out, bias, alpha, trans_a, trans_b, tf32, stream)
    return out


def _cublaslt_gemm(
    a,
    b,
    out,
    alpha,
    beta,
    trans_a,
    trans_b,
    bias,
    activation,
    aux,
    grad_aux,
    bias_grad,
    tf32,
    autotune,
    stream,
    M,
    N,
    K,
    lda,
    ldb,
    batch,
    sa,
    sb,
):
    """Launch the GEMM with cuBLASLt (arguments validated by :func:`gemm`)."""
    _, _, ldc, _, sc = _mat_info(out, False)
    dt, _ = _DTYPES[a.dtype]
    if a.dtype == _types.float64:
        compute, scale = _COMPUTE_64F, _CUDA_R_64F
    else:
        compute = _COMPUTE_32F_TF32 if (tf32 and a.dtype == _types.float32) else _COMPUTE_32F
        scale = _CUDA_R_32F
    ep = _EP[(activation, bias is not None)]
    aux_ld = 0
    if aux is not None:
        ep = {
            ("relu", True): _EP_RELU_AUX_BIAS,
            ("relu", False): _EP_RELU_AUX,
            ("gelu", True): _EP_GELU_AUX_BIAS,
            ("gelu", False): _EP_GELU_AUX,
        }[(activation, bias is not None)]
    if grad_aux is not None:
        ep = _EP_DGELU_BGRAD if bias_grad is not None else _EP_DGELU
        aux_ld = N
    elif aux is not None:
        aux_ld = N
    elif bias_grad is not None:
        # weight-gradient GEMM out = op(a) @ op(b): fuse the column sums of op(b) over K (the batch) into bias_grad
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


# ---------------------------------------------------------------------------------------------------------------------
# Built-in Warp kernel (CPU arrays, CUDA without cuBLASLt, CUDA API captures)

_fallback_cache: dict = {}
_GELU_C0 = 0.7978845608028654  # sqrt(2 / pi)
_GELU_C1 = 0.044715
_GELU_C3 = 3.0 * _GELU_C1


def _fallback_kernels(dtype, ndim: int):
    """GEMM and column-sum kernels for ``dtype`` and 2D or 3D (batched) arrays, created on first use."""
    key = (dtype, ndim)
    if key in _fallback_cache:
        return _fallback_cache[key]
    import warp as wp  # noqa: PLC0415

    ACC = wp.float64 if dtype == _types.float64 else wp.float32  # accumulation type

    @wp.func
    def gelu(z: Any):
        u = z.dtype(_GELU_C0) * (z + z.dtype(_GELU_C1) * z * z * z)
        return z.dtype(0.5) * z * (z.dtype(1.0) + wp.tanh(u))

    @wp.func
    def gelu_grad(z: Any):
        u = z.dtype(_GELU_C0) * (z + z.dtype(_GELU_C1) * z * z * z)
        t = wp.tanh(u)
        return z.dtype(0.5) * (z.dtype(1.0) + t) + z.dtype(0.5) * z * (z.dtype(1.0) - t * t) * z.dtype(_GELU_C0) * (
            z.dtype(1.0) + z.dtype(_GELU_C3) * z * z
        )

    @wp.func
    def epilogue(
        v: Any, bias_v: Any, has_bias: int, z: Any, aux_out: int, act: int, grad_mode: int, beta: Any, old: Any
    ):
        # returns (output, pre-activation)
        if grad_mode != 0:
            return v * gelu_grad(z), v
        if has_bias != 0:
            v = v + bias_v
        pre = v
        if act == 1:
            v = wp.max(v, v.dtype(0.0))
        elif act == 2:
            v = gelu(v)
        if beta != v.dtype(0.0):
            v = v + beta * old
        return v, pre

    if ndim == 2:

        @wp.kernel(enable_backward=False, module="unique")
        def gemm_kernel(
            a: wp.array2d[Any],
            b: wp.array2d[Any],
            out: wp.array2d[Any],
            bias: wp.array1d[Any],
            aux: wp.array2d[Any],  # pre-activation output, or the GELU input of the backward epilogue
            K: int,
            trans_a: int,
            trans_b: int,
            alpha: float,
            beta: float,
            act: int,
            grad_mode: int,
        ):
            i, j = wp.tid()
            s = ACC(0.0)
            for k in range(K):
                av = ACC(0.0)
                bv = ACC(0.0)
                if trans_a != 0:
                    av = ACC(a[k, i])
                else:
                    av = ACC(a[i, k])
                if trans_b != 0:
                    bv = ACC(b[j, k])
                else:
                    bv = ACC(b[k, j])
                s += av * bv
            has_bias = int(bias.shape[0] > 0)
            has_aux = int(aux.shape[0] > 0)
            bias_v = ACC(0.0)
            if has_bias != 0:
                bias_v = ACC(bias[j])
            z = ACC(0.0)
            if grad_mode != 0:
                z = ACC(aux[i, j])
            old = ACC(0.0)
            if beta != 0.0:
                old = ACC(out[i, j])
            r, pre = epilogue(s * ACC(alpha), bias_v, has_bias, z, has_aux, act, grad_mode, ACC(beta), old)
            if has_aux != 0 and grad_mode == 0:
                aux[i, j] = aux.dtype(pre)
            out[i, j] = out.dtype(r)

        @wp.kernel(enable_backward=False, module="unique")
        def colsum_kernel(x: wp.array2d[Any], g: wp.array1d[Any]):
            j = wp.tid()
            s = ACC(0.0)
            for r in range(x.shape[0]):
                s += ACC(x[r, j])
            g[j] = g.dtype(s)

    else:

        @wp.kernel(enable_backward=False, module="unique")
        def gemm_kernel(
            a: wp.array3d[Any],
            b: wp.array3d[Any],
            out: wp.array3d[Any],
            bias: wp.array1d[Any],
            aux: wp.array3d[Any],
            K: int,
            trans_a: int,
            trans_b: int,
            alpha: float,
            beta: float,
            act: int,
            grad_mode: int,
        ):
            n, i, j = wp.tid()
            s = ACC(0.0)
            for k in range(K):
                av = ACC(0.0)
                bv = ACC(0.0)
                if trans_a != 0:
                    av = ACC(a[n, k, i])
                else:
                    av = ACC(a[n, i, k])
                if trans_b != 0:
                    bv = ACC(b[n, j, k])
                else:
                    bv = ACC(b[n, k, j])
                s += av * bv
            has_bias = int(bias.shape[0] > 0)
            has_aux = int(aux.shape[0] > 0)
            bias_v = ACC(0.0)
            if has_bias != 0:
                bias_v = ACC(bias[j])
            z = ACC(0.0)
            if grad_mode != 0:
                z = ACC(aux[n, i, j])
            old = ACC(0.0)
            if beta != 0.0:
                old = ACC(out[n, i, j])
            r, pre = epilogue(s * ACC(alpha), bias_v, has_bias, z, has_aux, act, grad_mode, ACC(beta), old)
            if has_aux != 0 and grad_mode == 0:
                aux[n, i, j] = aux.dtype(pre)
            out[n, i, j] = out.dtype(r)

        @wp.kernel(enable_backward=False, module="unique")
        def colsum_kernel(x: wp.array3d[Any], g: wp.array1d[Any]):
            j = wp.tid()
            s = ACC(0.0)
            for n in range(x.shape[0]):
                for r in range(x.shape[1]):
                    s += ACC(x[n, r, j])
            g[j] = g.dtype(s)

    _fallback_cache[key] = (gemm_kernel, colsum_kernel)
    return _fallback_cache[key]


def _placeholder(dtype, ndim: int, device):
    """Empty array standing in for an absent optional argument of the built-in kernels."""
    key = ("placeholder", dtype, ndim, str(device))
    if key not in _fallback_cache:
        _fallback_cache[key] = _ctx.empty((0,) * ndim, dtype=dtype, device=device)
    return _fallback_cache[key]


def _fallback_gemm(a, b, out, alpha, beta, trans_a, trans_b, bias, activation, aux, grad_aux, bias_grad, K):
    """Launch the GEMM with the built-in Warp kernels (arguments validated by :func:`gemm`)."""
    gemm_kernel, colsum_kernel = _fallback_kernels(a.dtype, a.ndim)
    device = a.device
    act = {None: 0, "relu": 1, "gelu": 2}[activation]
    side = grad_aux if grad_aux is not None else aux
    _ctx.launch(
        gemm_kernel,
        dim=out.shape,
        inputs=[
            a,
            b,
            out,
            bias if bias is not None else _placeholder(a.dtype, 1, device),
            side if side is not None else _placeholder(a.dtype, a.ndim, device),
            K,
            int(trans_a),
            int(trans_b),
            float(alpha),
            float(beta),
            act,
            int(grad_aux is not None),
        ],
        device=device,
        record_tape=False,
    )
    if bias_grad is not None:
        # column sums of the result (backward epilogue) or of b over the reduction dimension (weight gradient)
        source = out if grad_aux is not None else b
        _ctx.launch(colsum_kernel, dim=bias_grad.shape[0], inputs=[source, bias_grad], device=device, record_tape=False)


_colsum_kernel = None


def _colsum(dout, grad):
    """grad[j] += sum over all rows (and batches) of dout[..., j]."""
    global _colsum_kernel
    import warp as wp  # noqa: PLC0415

    if _colsum_kernel is None:
        rows_per_thread = 64

        @wp.kernel(enable_backward=False, module="unique")
        def colsum(d: wp.array2d[Any], g: wp.array1d[Any]):
            r, j = wp.tid()
            s = d.dtype(0.0)
            r0 = r * rows_per_thread
            for k in range(rows_per_thread):
                i = r0 + k
                if i < d.shape[0]:
                    s += d[i, j]
            wp.atomic_add(g, j, s)

        _colsum_kernel = (colsum, rows_per_thread)
    kernel, rows_per_thread = _colsum_kernel
    d2 = dout if dout.ndim == 2 else dout.reshape((-1, dout.shape[-1]))
    nrb = (d2.shape[0] + rows_per_thread - 1) // rows_per_thread
    wp.launch(kernel, dim=(nrb, d2.shape[1]), inputs=[d2, grad.flatten()], device=d2.device)


def _record_backward(tape, a, b, out, bias, alpha, trans_a, trans_b, tf32, stream):
    """Record the adjoint of ``out = alpha * op(a) @ op(b) + bias`` on ``tape``.

    With ``G = d(loss)/d(out)``: ``d op(a) = alpha * G @ op(b)^T``, ``d op(b) = alpha * op(a)^T @ G`` and
    ``d bias = column sums of G``; gradients are accumulated into the ``grad`` arrays.
    """

    def backward():
        g = out.grad
        if g is None:
            return
        kw = {"tf32": tf32, "stream": stream, "alpha": alpha, "beta": 1.0}
        if a.requires_grad and a.grad is not None:
            if not trans_a:
                gemm(g, b, a.grad, trans_b=not trans_b, **kw)  # (M,N) @ op(b)^T
            else:
                gemm(b, g, a.grad, trans_a=trans_b, trans_b=True, **kw)  # a.grad (K,M) += op(b) @ G^T
        if b.requires_grad and b.grad is not None:
            if not trans_b:
                gemm(a, g, b.grad, trans_a=not trans_a, **kw)  # (K,N) = op(a)^T @ G
            else:
                gemm(g, a, b.grad, trans_a=True, trans_b=trans_a, **kw)  # b.grad (N,K) += G^T @ op(a)
        if bias is not None and bias.requires_grad and bias.grad is not None:
            _colsum(g, bias.grad)

    tape.record_func(backward, [x for x in (a, b, out, bias) if x is not None and x.requires_grad])
