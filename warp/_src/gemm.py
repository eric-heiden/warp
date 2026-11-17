# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GEMM for Warp arrays, backed by cuBLASLt on CUDA devices.

The wrapper loads cuBLASLt lazily at runtime (it is not linked into ``warp.so``), caches one heuristic plan
per problem shape and issues a single ``cublasLtMatmul()`` per call. Plans are created on the first call
(do a warm-up call before graph capture to keep plan creation outside of the captured region), and the
launch itself is safe for CUDA graph capture.

Every call goes through one native entry point (``wp_gemm_device`` / ``wp_gemm_host``). Inside an API capture
(``apic=True``, and CPU captures) it records a semantic GEMM operation, so a saved graph re-plans the GEMM with
the cuBLASLt of the process that loads it. CPU arrays and CUDA devices without a loadable cuBLASLt use built-in
GEMMs compiled into ``warp.so``.
"""

from __future__ import annotations

import ctypes
import glob
import os
import sys
from typing import Any

import warp._src.context as _ctx
import warp._src.types as _types
from warp._src.logger import log_debug, log_warning

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
}
_EP_BGRADA = 256

_DTYPES = {
    _types.float32: (_CUDA_R_32F, 4),
    _types.float64: (_CUDA_R_64F, 8),
    _types.float16: (_CUDA_R_16F, 2),
    _types.bfloat16: (_CUDA_R_16BF, 2),
}

_state = {"loaded": None, "version": 0, "path": None}

# cuBLASLt file names per CUDA major version: (Linux soname, Windows DLL)
_LIB_NAMES = {13: ("libcublasLt.so.13", "cublasLt64_13.dll"), 12: ("libcublasLt.so.12", "cublasLt64_12.dll")}
# Lowest compute capability each cuBLASLt major version supports (CUDA 13 dropped Maxwell, Pascal and Volta).
_MIN_ARCH = {13: 75, 12: 50}


def _parse_version(text):
    """Leading dotted version number in ``text`` as a tuple of ints, e.g. ``"12.9.1.4"`` -> ``(12, 9, 1, 4)``."""
    out = []
    for part in text.split("."):
        digits = ""
        for ch in part:
            if not ch.isdigit():
                break
            digits += ch
        if not digits:
            break
        out.append(int(digits))
        if len(digits) != len(part):
            break
    return tuple(out)


def _wheel_versions(site_dir):
    """Map library paths shipped by ``nvidia-cublas*`` wheels installed in ``site_dir`` to the wheel version."""
    versions = {}
    for dist in glob.glob(os.path.join(site_dir, "nvidia_cublas*.dist-info")):
        # nvidia_cublas_cu12-12.9.1.4.dist-info, nvidia_cublas-13.1.0.3.dist-info
        version = _parse_version(os.path.basename(dist).split("-", 1)[-1])
        try:
            with open(os.path.join(dist, "RECORD"), encoding="utf-8") as f:
                for line in f:
                    rel = line.split(",", 1)[0]
                    if "ublasLt" in rel:
                        versions[os.path.normcase(os.path.realpath(os.path.join(site_dir, rel)))] = version
        except OSError:
            continue
    return versions


def _toolkit_version(root):
    """cuBLAS version of the CUDA toolkit at ``root`` from its ``version.json``, or the CUDA version in its name."""
    try:
        import json  # noqa: PLC0415

        with open(os.path.join(root, "version.json"), encoding="utf-8") as f:
            info = json.load(f)
        return _parse_version(info.get("libcublas", info.get("cuda", {})).get("version", ""))
    except (OSError, ValueError, AttributeError):
        pass
    name = os.path.basename(os.path.normpath(root)).lower()
    for prefix in ("cuda-", "v"):
        if name.startswith(prefix):
            return _parse_version(name[len(prefix) :])
    return ()


def _find_cublaslt(site_dirs, toolkit_roots, system_dirs):
    """Return ``[(version, path), ...]`` for every cuBLASLt library found, newest first, without loading any.

    Versions come from the wheel metadata (pip ``nvidia-cublas`` wheels in ``site_dirs``), the versioned file
    name the library resolves to (``libcublasLt.so.12.9.1.4``), or the toolkit's ``version.json``. A library
    whose version cannot be determined ranks by its major version alone, below every versioned library of that
    major version.
    """
    found = {}

    def add(path, version):
        real = os.path.normcase(os.path.realpath(path))
        if os.path.isfile(real):
            version = max(version, _parse_version(os.path.basename(real).partition(".so.")[2]))
            if version > found.get(real, (None, ()))[1]:
                found[real] = (path, version)

    for major, (so, dll) in _LIB_NAMES.items():
        for base in site_dirs:
            wheel = _wheel_versions(base)
            for pattern in (
                f"nvidia/cu{major}/lib/{so}",
                f"nvidia/cublas/lib/{so}",
                f"nvidia/cu{major}/bin/{dll}",
                f"nvidia/cu{major}/bin/*/{dll}",
                f"nvidia/cublas/bin/{dll}",
            ):
                for path in glob.glob(os.path.join(base, pattern)):
                    real = os.path.normcase(os.path.realpath(path))
                    add(path, wheel.get(real, (major,)))
        for root in toolkit_roots:
            version = _toolkit_version(root)
            if version[:1] != (major,):
                version = (major,)
            for sub in ("lib64", "lib", os.path.join("targets", "*", "lib"), "bin", os.path.join("bin", "x64")):
                for name in (so, dll):
                    for path in glob.glob(os.path.join(root, sub, name)):
                        add(path, version)
        for d in system_dirs:
            for path in glob.glob(os.path.join(d, so)):
                add(path, (major,))
    return sorted(((v, p) for p, v in found.values()), key=lambda vp: vp[0], reverse=True)


def _default_search_dirs():
    """Site directories, CUDA toolkit roots and system library directories to search for cuBLASLt."""
    site_dirs = [p for p in sys.path if p and os.path.isdir(p)]
    roots = [os.environ[v] for v in ("WARP_CUDA_PATH", "CUDA_HOME", "CUDA_PATH") if os.environ.get(v)]
    system_dirs = []
    if sys.platform == "win32":
        pf = os.environ.get("ProgramFiles", r"C:\Program Files")
        roots += glob.glob(os.path.join(pf, "NVIDIA GPU Computing Toolkit", "CUDA", "v*"))
    elif sys.platform.startswith("linux"):
        roots += glob.glob("/usr/local/cuda*") + glob.glob("/opt/cuda*")
        system_dirs += [*glob.glob("/usr/lib/*-linux-gnu"), "/usr/lib64", "/usr/lib"]
    return site_dirs, roots, system_dirs


def _supported_majors():
    """cuBLASLt major versions the CUDA driver and every CUDA device of this process can use."""
    runtime = _ctx.runtime
    driver = runtime.driver_version if runtime is not None else None
    archs = [d.arch for d in runtime.cuda_devices] if runtime is not None else []
    return [m for m in _LIB_NAMES if (driver is None or driver[0] >= m) and all(a >= _MIN_ARCH[m] for a in archs if a)]


def _lib_candidates():
    """cuBLASLt libraries to try loading, in order.

    ``WARP_CUBLASLT_PATH`` comes first. Then every cuBLASLt found in pip ``nvidia-cublas`` wheels on
    ``sys.path``, CUDA toolkits (``WARP_CUDA_PATH``, ``CUDA_HOME``, ``CUDA_PATH`` and the default install
    locations) and the system library directories, newest version first: newer cuBLASLt releases carry kernels
    tuned for newer GPUs, while older ones fall back to slower generic kernels. Major versions that the CUDA
    driver or one of the GPUs cannot use are skipped. The plain library names, resolved by the system loader,
    come last.
    """
    env = os.environ.get("WARP_CUBLASLT_PATH")
    if env:
        yield env
    majors = _supported_majors()
    for version, path in _find_cublaslt(*_default_search_dirs()):
        if version[0] in majors:
            yield path
    for major in majors:
        yield from _LIB_NAMES[major]
    yield "libcublasLt.so"


class _GemmDesc(ctypes.Structure):
    """Mirror of ``wp_gemm_desc`` in ``warp/native/gemm.h``."""

    _fields_ = [
        ("ab_dtype", ctypes.c_int32),
        ("c_dtype", ctypes.c_int32),
        ("trans_a", ctypes.c_int32),
        ("trans_b", ctypes.c_int32),
        ("m", ctypes.c_int32),
        ("n", ctypes.c_int32),
        ("k", ctypes.c_int32),
        ("lda", ctypes.c_int32),
        ("ldb", ctypes.c_int32),
        ("ldc", ctypes.c_int32),
        ("batch", ctypes.c_int32),
        ("epilogue", ctypes.c_int32),
        ("stride_a", ctypes.c_int64),
        ("stride_b", ctypes.c_int64),
        ("stride_c", ctypes.c_int64),
        ("alpha", ctypes.c_double),
        ("beta", ctypes.c_double),
        ("tf32", ctypes.c_int32),
        ("autotune", ctypes.c_int32),
    ]


# APICType values (warp/native/apic_types.h)
_APIC_TYPES = {_types.float32: 5, _types.float64: 6, _types.float16: 7, _types.bfloat16: 8}
# APICGemmEpilogue values
_APIC_EP = {(None, False): 0, (None, True): 1, ("relu", False): 2, ("relu", True): 3}
_APIC_EP_BGRADA = 4

_native_ready = False


def _setup_native(core):
    global _native_ready
    if _native_ready:
        return
    core.wp_gemm_host.argtypes = [ctypes.POINTER(_GemmDesc)] + [ctypes.c_void_p] * 4
    core.wp_gemm_host.restype = ctypes.c_bool
    core.wp_gemm_device.argtypes = (
        [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(_GemmDesc)] + [ctypes.c_void_p] * 4 + [ctypes.c_int]
    )
    core.wp_gemm_device.restype = ctypes.c_bool
    core.wp_gemm_set_host_blas.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    core.wp_gemm_set_host_blas.restype = None
    _register_host_blas(core)
    _native_ready = True


_use_host_blas = os.environ.get("WARP_GEMM_HOST_BLAS", "1") != "0"  # "0" forces the built-in host GEMM
_host_blas_info = None


def _numpy_blas():
    """Return ``(sgemm, dgemm, ilp64, path)`` for the CBLAS library NumPy links against, or ``None``.

    NumPy wheels bundle OpenBLAS (``scipy_openblas64`` with prefixed ILP64 symbols in NumPy 2, ``openblas64_``
    in NumPy 1) or use Accelerate on macOS. Loading it again with ctypes returns the library NumPy already loaded.
    """
    import numpy as np  # noqa: PLC0415

    base = os.path.dirname(os.path.dirname(np.__file__))
    candidates = []
    for d in (os.path.join(base, "numpy.libs"), os.path.join(os.path.dirname(np.__file__), ".dylibs"), base):
        candidates += sorted(glob.glob(os.path.join(d, "*openblas*")))
    if sys.platform == "darwin":
        candidates.append("/System/Library/Frameworks/Accelerate.framework/Accelerate")
    symbols = (
        ("scipy_cblas_sgemm64_", "scipy_cblas_dgemm64_", True),
        ("cblas_sgemm64_", "cblas_dgemm64_", True),
        ("scipy_cblas_sgemm", "scipy_cblas_dgemm", False),
        ("cblas_sgemm", "cblas_dgemm", False),
    )
    for path in candidates:
        try:
            lib = ctypes.CDLL(path)
        except OSError:
            continue
        for s_name, d_name, ilp64 in symbols:
            try:
                s_fn, d_fn = getattr(lib, s_name), getattr(lib, d_name)
            except AttributeError:
                continue
            return (
                ctypes.cast(s_fn, ctypes.c_void_p).value,
                ctypes.cast(d_fn, ctypes.c_void_p).value,
                ilp64,
                path,
            )
    return None


def _register_host_blas(core):
    global _host_blas_info
    if _host_blas_info is None:
        _host_blas_info = _numpy_blas() or (None, None, False, None)
    s_fn, d_fn, ilp64, _ = _host_blas_info
    core.wp_gemm_set_host_blas(s_fn, d_fn, int(ilp64), int(_use_host_blas))


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
    if os.environ.get("WARP_CUBLASLT_PATH", "").lower() in ("none", "0", "off"):
        _state["loaded"] = False  # cuBLASLt disabled explicitly: use the built-in GEMMs
        raise RuntimeError("cuBLASLt is disabled by WARP_CUBLASLT_PATH")
    for cand in _lib_candidates():
        v = core.wp_cublaslt_init(cand.encode())
        if v:
            _state["loaded"], _state["version"], _state["path"] = True, v, cand
            log_debug(f"Loaded cuBLASLt {v} from {cand}")
            return
    _state["loaded"] = False
    raise RuntimeError(f"cuBLASLt could not be loaded; {_INSTALL_HINT}")


def is_cublaslt_available() -> bool:
    """Return whether :func:`warp.gemm` can use cuBLASLt on CUDA devices.

    cuBLASLt is loaded at runtime, on the first call of this function or of :func:`warp.gemm`, from the
    ``nvidia-cublas`` wheels (``pip install warp-lang[cublas-cu12]`` or ``warp-lang[cublas-cu13]``), the CUDA
    toolkit or the library search path. When several are installed, the newest one that the CUDA driver and
    GPUs support is used; ``WARP_CUBLASLT_PATH`` selects a library explicitly. Without it,
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


_force_fallback = False  # tests: use the built-in GEMM on CUDA devices too
# Built-in GEMM implementation: "native" (compiled into warp.so, recorded as an APIC GEMM operation) or "warp"
# (the generic Warp kernels below, recorded as kernel launches). Kept selectable for benchmarking.
_builtin_impl = "native"


def _select_backend(device) -> str:
    """``"cublaslt"`` or ``"builtin"`` for a call on ``device``."""
    if device.is_cpu or _force_fallback:
        return "builtin"
    if not is_cublaslt_available():
        log_warning(
            f"wp.gemm() uses the built-in Warp GEMM kernel, which is much slower than cuBLASLt: {_INSTALL_HINT}.",
            once=True,
        )
        return "builtin"
    return "cublaslt"


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
    bias_grad=None,
    tf32: bool = False,
    autotune: int = 8,
    stream=None,
    record_tape: bool = True,
):
    """Compute ``out = activation(alpha * op(a) @ op(b) + bias) + beta * out``.

    All arrays are row-major with contiguous rows, either 2D ``(rows, cols)`` or 3D ``(batch, rows, cols)``, and
    on the same device. ``a`` and ``b`` have the same dtype (``float16``, ``bfloat16``, ``float32`` or ``float64``);
    the other arrays have the dtype of ``out``: the dtype of the operands or, for mixed precision, ``float32``
    with ``float16`` or ``bfloat16`` operands (the products are accumulated in ``float32`` in all cases but
    ``float64``). On CUDA devices, the call enqueues a single cuBLASLt kernel launch on the device's current
    stream and can be captured in a CUDA graph. cuBLASLt is loaded at runtime (see
    :func:`warp.is_cublaslt_available`). CUDA devices without cuBLASLt use a built-in tiled GEMM compiled into Warp,
    and CPU arrays use the BLAS library NumPy ships with, or a built-in blocked GEMM. Inside API captures
    (``apic=True``) and CPU captures the call is recorded as a GEMM operation: saved graphs re-plan it with the
    cuBLASLt available where they are loaded, or fall back to the built-in GEMMs.

    Args:
        a: Left operand, shape ``(M, K)`` (or ``(K, M)`` if ``trans_a``).
        b: Right operand, shape ``(K, N)`` (or ``(N, K)`` if ``trans_b``).
        out: Output array of shape ``(M, N)``. Allocated if ``None``.
        alpha: Scale of the matrix product.
        beta: Scale of the existing contents of ``out`` (0 overwrites). Not combined with ``activation``.
        trans_a: Use the transpose of ``a``.
        trans_b: Use the transpose of ``b``.
        bias: Optional 1D array of length ``N`` added to each output row (fused epilogue).
        activation: Optional fused activation: ``"relu"``. Its backward pass needs only the output (``out > 0``).
        bias_grad: Optional 1D output array of length ``N`` receiving, for a weight-gradient GEMM ``a.T @ b``
            (``trans_a=True``, ``trans_b=False``, no bias or activation), the column sums of ``b`` over the reduction
            dimension, i.e. the bias gradient of a linear layer. With 3D (batched) arrays the sums also run over the
            batch.
        tf32: Allow TF32 tensor cores for float32 inputs (cuBLASLt only).
        autotune: Number of cuBLASLt heuristic candidates to time on the first (non-captured) call with ``beta == 0``;
            the fastest is kept. Use 0 or 1 to disable.
        stream: CUDA stream to launch on, defaults to the current stream of the device.
        record_tape: Whether to record the adjoint on the active :class:`warp.Tape` (if any). Pass ``False`` when
            the caller records its own backward pass (e.g. with :meth:`warp.Tape.record_func`), like
            ``record_tape`` of :func:`warp.launch`.

    When a :class:`warp.Tape` is recording and ``a``, ``b`` or ``bias`` require gradients, the adjoint GEMMs (and
    the bias column sums) are recorded on the tape, so ``tape.backward()`` accumulates into their ``grad`` arrays.
    This requires ``beta=0``, no fused activation and no ``bias_grad``.

    Returns:
        The output array.
    """
    if _ctx.runtime is None:
        _ctx.init()
    if a.dtype != b.dtype:
        raise TypeError("gemm() operands must have the same dtype")
    if a.dtype not in _DTYPES:
        raise TypeError(f"gemm() unsupported dtype {a.dtype}")
    out_dtype = out.dtype if out is not None else a.dtype
    if out_dtype != a.dtype and not (a.dtype in (_types.float16, _types.bfloat16) and out_dtype == _types.float32):
        raise TypeError(
            f"gemm() output dtype {out_dtype} is not supported with {a.dtype} operands (expected {a.dtype}, or "
            "float32 for float16 and bfloat16 operands)"
        )
    for name, x in (
        ("b", b),
        ("out", out),
        ("bias", bias),
        ("bias_grad", bias_grad),
    ):
        if x is not None:
            if x.device != a.device:
                raise ValueError(
                    f"gemm() array '{name}' is on device {x.device}, expected {a.device} (the device of 'a')"
                )
            expected = a.dtype if name == "b" else out_dtype
            if x.dtype != expected:
                raise TypeError(f"gemm() array '{name}' has dtype {x.dtype}, expected {expected}")
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
    if activation not in (None, "relu"):
        raise ValueError(f"gemm() unsupported activation {activation!r} (expected None or 'relu')")
    if bias_grad is not None and (activation is not None or bias is not None or not trans_a or trans_b):
        raise ValueError("gemm() bias_grad requires trans_a=True, trans_b=False and no bias or activation")
    for name, x in (("bias", bias), ("bias_grad", bias_grad)):
        if x is not None and (x.ndim != 1 or x.shape[0] != N):
            raise ValueError(f"gemm() {name} must be a 1D array of length {N}")
    tape = _ctx.runtime.tape if record_tape else None
    recorded = tape is not None and any(x is not None and x.requires_grad for x in (a, b, bias))
    if recorded and (activation is not None or bias_grad is not None or beta != 0.0):
        raise RuntimeError(
            "gemm() can only be recorded on a wp.Tape without a fused activation or bias_grad and with beta=0"
        )
    if recorded and out.dtype != a.dtype:
        raise RuntimeError("gemm() with mixed operand and output dtypes cannot be recorded on a wp.Tape")
    backend = _select_backend(a.device)
    if backend == "builtin" and _builtin_impl == "warp":
        _fallback_gemm(a, b, out, alpha, beta, trans_a, trans_b, bias, activation, bias_grad, K)
    else:
        _native_gemm(
            a,
            b,
            out,
            alpha,
            beta,
            trans_a,
            trans_b,
            bias,
            activation,
            bias_grad,
            tf32,
            autotune,
            stream,
            backend,
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


def _native_gemm(
    a,
    b,
    out,
    alpha,
    beta,
    trans_a,
    trans_b,
    bias,
    activation,
    bias_grad,
    tf32,
    autotune,
    stream,
    backend,
    M,
    N,
    K,
    lda,
    ldb,
    batch,
    sa,
    sb,
):
    """Launch the GEMM through ``wp_gemm_device`` / ``wp_gemm_host`` (arguments validated by :func:`gemm`)."""
    core = _ctx.runtime.core
    _setup_native(core)
    _, _, ldc, _, sc = _mat_info(out, False)
    ep = _APIC_EP_BGRADA if bias_grad is not None else _APIC_EP[(activation, bias is not None)]
    desc = _GemmDesc(
        _APIC_TYPES[a.dtype],
        _APIC_TYPES[out.dtype],
        int(trans_a),
        int(trans_b),
        M,
        N,
        K,
        lda,
        ldb,
        ldc,
        batch,
        ep,
        sa,
        sb,
        sc,
        float(alpha),
        float(beta),
        int(bool(tf32)),
        int(autotune),
    )
    vec = bias_grad if bias_grad is not None else bias
    device = a.device
    capture = _ctx._get_apic_capture_for_device(device)
    if capture is not None:
        for x in (a, b, out, vec):
            if x is not None:
                capture.track_array(x)
    if device.is_cpu:
        ok = core.wp_gemm_host(ctypes.byref(desc), a.ptr, b.ptr, out.ptr, vec.ptr if vec is not None else None)
    else:
        stream_ptr = device.stream.cuda_stream if stream is None else stream.cuda_stream
        ok = core.wp_gemm_device(
            device.context,
            stream_ptr,
            ctypes.byref(desc),
            a.ptr,
            b.ptr,
            out.ptr,
            vec.ptr if vec is not None else None,
            1 if backend == "cublaslt" else 2,
        )
    if not ok:
        raise RuntimeError(f"gemm() failed: {core.wp_get_error_string().decode()}")


# ---------------------------------------------------------------------------------------------------------------------
# Generic Warp kernels (the previous built-in GEMM, selectable with ``_builtin_impl = "warp"`` for benchmarks)

_fallback_cache: dict = {}


def _fallback_kernels(dtype, ndim: int):
    """GEMM and column-sum kernels for ``dtype`` and 2D or 3D (batched) arrays, created on first use."""
    key = (dtype, ndim)
    if key in _fallback_cache:
        return _fallback_cache[key]
    import warp as wp  # noqa: PLC0415

    ACC = wp.float64 if dtype == _types.float64 else wp.float32  # accumulation type

    @wp.func
    def epilogue(v: Any, bias_v: Any, has_bias: int, relu: int, beta: Any, old: Any):
        if has_bias != 0:
            v = v + bias_v
        if relu != 0:
            v = wp.max(v, v.dtype(0.0))
        if beta != v.dtype(0.0):
            v = v + beta * old
        return v

    if ndim == 2:

        @wp.kernel(enable_backward=False, module="unique")
        def gemm_kernel(
            a: wp.array2d[Any],
            b: wp.array2d[Any],
            out: wp.array2d[Any],
            bias: wp.array1d[Any],
            K: int,
            trans_a: int,
            trans_b: int,
            alpha: float,
            beta: float,
            relu: int,
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
            bias_v = ACC(0.0)
            if has_bias != 0:
                bias_v = ACC(bias[j])
            old = ACC(0.0)
            if beta != 0.0:
                old = ACC(out[i, j])
            out[i, j] = out.dtype(epilogue(s * ACC(alpha), bias_v, has_bias, relu, ACC(beta), old))

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
            K: int,
            trans_a: int,
            trans_b: int,
            alpha: float,
            beta: float,
            relu: int,
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
            bias_v = ACC(0.0)
            if has_bias != 0:
                bias_v = ACC(bias[j])
            old = ACC(0.0)
            if beta != 0.0:
                old = ACC(out[n, i, j])
            out[n, i, j] = out.dtype(epilogue(s * ACC(alpha), bias_v, has_bias, relu, ACC(beta), old))

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


def _fallback_gemm(a, b, out, alpha, beta, trans_a, trans_b, bias, activation, bias_grad, K):
    """Launch the GEMM with the built-in Warp kernels (arguments validated by :func:`gemm`)."""
    gemm_kernel, colsum_kernel = _fallback_kernels(a.dtype, a.ndim)
    device = a.device
    _ctx.launch(
        gemm_kernel,
        dim=out.shape,
        inputs=[
            a,
            b,
            out,
            bias if bias is not None else _placeholder(out.dtype, 1, device),
            K,
            int(trans_a),
            int(trans_b),
            float(alpha),
            float(beta),
            int(activation == "relu"),
        ],
        device=device,
        record_tape=False,
    )
    if bias_grad is not None:
        # weight gradient a.T @ b: column sums of b over the reduction dimension
        _ctx.launch(colsum_kernel, dim=bias_grad.shape[0], inputs=[b, bias_grad], device=device, record_tape=False)


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
