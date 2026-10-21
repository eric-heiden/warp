# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Low-overhead repeated launches of one kernel on the CPU.

:class:`CpuLauncher` resolves everything :func:`warp.launch` resolves per call (module
executable, kernel hooks, argument struct layout) once, and generates a launch function
for the kernel's signature. Per call, the generated function checks and converts each
argument with unrolled code, writes the launch bounds and the whole argument struct with
one :meth:`struct.Struct.pack_into` call, and calls the kernel's CPU entry point.
"""

from __future__ import annotations

import ctypes
import struct
import threading
from typing import Any

import numpy as np

import warp
import warp.config
from warp._src import context as _context
from warp._src import types as _types
from warp._src.context import DeviceLike, Kernel

_SCALAR_FORMATS = {
    ctypes.c_int8: "b",
    ctypes.c_uint8: "B",
    ctypes.c_int16: "h",
    ctypes.c_uint16: "H",
    ctypes.c_int32: "i",
    ctypes.c_uint32: "I",
    ctypes.c_int64: "q",
    ctypes.c_uint64: "Q",
    ctypes.c_float: "f",
    ctypes.c_double: "d",
    ctypes.c_bool: "?",
}
"""Struct format codes for scalar parameters packed straight from Python numbers."""

_ARRAY_T_OFFSETS = {
    name: getattr(_types.array_t, name).offset for name in ("data", "grad", "shape", "strides", "ndim", "flags")
}


def _array_t_layout(ndim: int) -> list[tuple[int, str]]:
    """Offsets and formats of the ``array_t`` fields a regular ``ndim``-dimensional array sets."""
    offsets = _ARRAY_T_OFFSETS
    layout = [(offsets["data"], "Q"), (offsets["grad"], "Q")]
    layout += [(offsets["shape"] + 4 * i, "i") for i in range(ndim)]
    layout += [(offsets["strides"] + 4 * i, "i") for i in range(ndim)]
    layout += [(offsets["ndim"], "H"), (offsets["flags"], "H")]
    return layout


class CpuLauncher:
    """Launch one kernel on the CPU repeatedly with minimal Python overhead.

    A :func:`warp.launch` on the CPU spends several microseconds per call resolving the
    module, building launch bounds, and converting arguments, which dominates kernels
    over tens to thousands of elements. A ``CpuLauncher`` does that work once per kernel
    and packs each call's arguments with a single native-layout write, so a launch costs
    about 1 µs plus a fraction of a microsecond per argument.

    The launcher accepts the same arguments as :func:`warp.launch` and converts them
    with the same rules. These are packed without per-call conversion overhead:

    - regular :class:`warp.array` arguments of the parameter's dtype and dimension count;
    - NumPy arrays for regular array parameters, without copying, when their dtype is the
      parameter's scalar type and their trailing dimensions match a vector or matrix dtype;
    - values of the parameter's vector, matrix, quaternion, or transform type, or NumPy
      arrays of the same shape and scalar type;
    - Python and NumPy numbers for scalar parameters.

    Other arguments, such as indexed arrays and structs, are converted as in
    :func:`warp.launch`.

    Launches run synchronously in the calling thread, like CPU launches through
    :func:`warp.launch`. Each thread uses its own argument buffer, so a launcher can be
    shared between threads. Calls are forwarded to :func:`warp.launch` while a
    :class:`warp.Tape` or a CPU graph capture is active, while
    :attr:`warp.config.print_launches` is set, or while
    :attr:`warp.config.launch_array_access_mode` is not ``RELAXED``. The kernel runs with
    ``block_dim=1``, and only forward launches are supported.

    Args:
        kernel: The kernel to launch. Generic kernels must be passed as a concrete
            overload, for example from :func:`warp.overload`.
        device: The CPU device to launch on.

    Raises:
        ValueError: If ``device`` is not a CPU device or ``kernel`` is generic.
        RuntimeError: If the kernel's module cannot be loaded on ``device``.

    Example:
        .. code-block:: python

            @wp.kernel
            def scale(x: wp.array[float], s: float):
                i = wp.tid()
                x[i] = x[i] * s


            launch = wp.CpuLauncher(scale)
            values = np.ones(80, dtype=np.float32)
            launch(len(values), values, 2.0)  # values is updated in place
    """

    def __init__(self, kernel: Kernel, device: DeviceLike = "cpu"):
        _context.init()
        self.device = _context.runtime.get_device(device)
        """The CPU device the kernel is launched on."""
        if not self.device.is_cpu:
            raise ValueError(f"CpuLauncher requires a CPU device, got '{self.device}'")
        if not isinstance(kernel, Kernel):
            raise TypeError("CpuLauncher requires a function decorated with @wp.kernel")
        if kernel.is_generic:
            raise ValueError(
                f"Kernel '{kernel.key}' is generic; pass a concrete overload, for example from wp.overload()"
            )
        if not kernel.uses_warp_entry_point_abi:
            raise ValueError(
                f"Kernel '{kernel.key}' uses entry_point_abi='{kernel.options['entry_point_abi']}' "
                "and cannot be launched by CpuLauncher."
            )

        self.kernel = kernel
        """The launched kernel."""
        self._arg_info = [(arg.label, arg.type) for arg in kernel.adj.args]
        self._local = threading.local()
        self._value_types: dict[int, set[type]] = {}
        self._bind_module()

        # Parameter defaults built like Launch() builds them; their ctypes define the struct layout.
        default = _context.Launch(kernel, self.device, hooks=self._hooks, block_dim=1)
        self._default_params = default.params[1 : 1 + len(self._arg_info)]
        self._args_cls = type(
            f"{kernel.key}_args",
            (ctypes.Structure,),
            {"_fields_": [(name, type(p)) for (name, _), p in zip(self._arg_info, self._default_params, strict=True)]},
        )
        self._bounds_cls = _types._launch_bounds_classes[kernel.adj.kernel_dim]
        self._launch = self._generate_launch()

    def __call__(self, dim: int | tuple[int, ...], *args: Any) -> None:
        """Launch the kernel over ``dim`` threads with ``args`` as its arguments, in order.

        Args:
            dim: The launch dimensions, as for :func:`warp.launch`.
            args: One value per kernel parameter.
        """
        self._launch(dim, *args)

    # -- module binding -------------------------------------------------------------

    def _bind_module(self) -> None:
        module = self.kernel.module
        module_exec = module.load(self.device, 1)
        if not module_exec:
            raise RuntimeError(f"Failed to load module {module.name} on device {self.device}")
        hooks = module_exec.get_kernel_hooks(self.kernel)
        if hooks.forward is None:
            raise RuntimeError(f"Failed to find forward kernel '{self.kernel.key}' in module '{module.name}'")
        self._module_exec = module_exec
        self._hooks = hooks
        self._exec_key = (self.device.context, 1)
        # Module.mark_modified() replaces the hashers dict, and a reload replaces the execs entry.
        self._hashers = module.hashers

    # -- per-thread argument buffer -------------------------------------------------------

    def _new_buffer(self):
        # Launch bounds at offset 0 and the argument struct at self._args_offset, in one buffer.
        buffer = ctypes.create_string_buffer(self._args_offset + ctypes.sizeof(self._args_cls))
        state = (buffer, ctypes.byref(buffer, 0), ctypes.byref(buffer, self._args_offset))
        self._local.state = state
        return state

    # -- code generation --------------------------------------------------------------------

    def _generate_launch(self):
        """Generate the launch function for this kernel's signature."""
        kernel_dim = self.kernel.adj.kernel_dim
        bounds_cls, args_cls = self._bounds_cls, self._args_cls
        self._args_offset = (ctypes.sizeof(bounds_cls) + 15) & ~15

        # (byte offset in the buffer, struct format, Python expression of the value)
        fields: list[tuple[int, str, str]] = [
            (bounds_cls.shape.offset + 4 * i, "I", f"d{i}") for i in range(kernel_dim)
        ]
        fields += [(bounds_cls.size.offset, "Q", "size"), (bounds_cls.coord_mult.offset, "Q", "coord_mult")]

        names = {
            "_ndarray": np.ndarray,
            "_warp_array": warp.array,
            "_addressof": ctypes.addressof,
            "_from_buffer": ctypes.c_char.from_buffer,
            "_types_equal": _types.types_equal,
            "_context": _context,
            "_config": warp.config,
            "_RELAXED": warp.config.LaunchArrayAccessMode.RELAXED,
            "_launcher": self,
            "_local": self._local,
            "_tid_limit": self.kernel.adj.tid_extent_limit_candidate,
            "_module": self.kernel.module,
            "_exec_key": self._exec_key,
            "_struct_error": struct.error,
        }
        params = [f"a{i}" for i in range(len(self._arg_info))]
        signature = ", ".join(["dim", *params])
        lines = [f"def launch({signature}):"]
        if kernel_dim == 1:
            lines += [
                "    if type(dim) is int:",
                "        d0 = size = dim",
                "        coord_mult = 1",
                "    else:",
                "        (d0,), size, coord_mult = _launcher._resolve_dim(dim)",
            ]
        else:
            dims = ", ".join(f"d{i}" for i in range(kernel_dim))
            lines.append(f"    ({dims}), size, coord_mult = _launcher._resolve_dim(dim)")
        lines += [
            "    if size <= 0:",
            "        if size < 0:",
            "            raise ValueError('Launch dimensions must be non-negative')",
            "        return",
            "    if size > _tid_limit:",
            "        _launcher._check_tid_extents(dim, size)",
            "    runtime = _context.runtime",
            "    if (",
            "        runtime.tape is not None",
            "        or runtime._apic_capture is not None",
            "        or _config.print_launches",
            "        or _config.launch_array_access_mode is not _RELAXED",
            "    ):",
            f"        return _launcher._fallback(dim, [{', '.join(params)}])",
            "    if (",
            "        _module.hashers is not _launcher._hashers",
            "        or _module.has_unresolved_static_expressions",
            "        or _module.execs.get(_exec_key) is not _launcher._module_exec",
            "    ):",
            "        _launcher._bind_module()",
        ]

        for index, (name, arg_type) in enumerate(self._arg_info):
            a = f"a{index}"
            param_type = type(self._default_params[index])
            base = self._args_offset + getattr(args_cls, name).offset
            if param_type is _types.array_t:
                lines += self._array_arg_lines(index, a, arg_type, names)
                ndim = arg_type.ndim
                values = [f"p{index}", f"g{index}"]
                values += [f"s{index}_{i}" for i in range(ndim)] + [f"z{index}_{i}" for i in range(ndim)]
                values += [str(ndim), f"f{index}"]
                fields += [
                    (base + offset, fmt, value)
                    for (offset, fmt), value in zip(_array_t_layout(ndim), values, strict=True)
                ]
            elif param_type in _SCALAR_FORMATS and arg_type not in (warp.float16, warp.bfloat16):
                # float16/bfloat16 store bit patterns in an integer ctype and go through pack_arg below.
                code = _SCALAR_FORMATS[param_type]
                accepted = "t is float or t is int" if code in "fd" else ("t is bool" if code == "?" else "t is int")
                lines += [
                    f"    t = type({a})",
                    f"    if not ({accepted}):",
                    f"        {a} = _launcher._scalar_value({index}, {a})",
                ]
                fields.append((base, code, a))
            elif issubclass(param_type, ctypes.Array) and _types.type_is_composite(arg_type):
                names[f"_value_types{index}"] = self._value_types.setdefault(index, {param_type})
                names[f"_np_dtype{index}"] = np.dtype(_types.warp_type_to_np_dtype[arg_type._wp_scalar_type_])
                names[f"_shape{index}"] = tuple(arg_type._shape_)
                lines += [
                    f"    t = type({a})",
                    f"    if t in _value_types{index}:",
                    f"        b{index} = bytes({a})",
                    # tobytes() always emits C order, so any view of the right shape works.
                    f"    elif t is _ndarray and {a}.dtype == _np_dtype{index} and {a}.shape == _shape{index}:",
                    f"        b{index} = {a}.tobytes()",
                    "    else:",
                    f"        b{index} = _launcher._value_bytes({index}, {a})",
                ]
                fields.append((base, f"{ctypes.sizeof(param_type)}s", f"b{index}"))
            else:
                # Structs, indexed arrays, textures, float16, ...: Warp's packing rules, copied as bytes.
                lines.append(f"    b{index} = _launcher._packed_bytes({index}, {a})")
                fields.append((base, f"{ctypes.sizeof(param_type)}s", f"b{index}"))

        packer, values = self._make_packer(fields)
        names["_pack_into"] = packer.pack_into
        lines += [
            "    try:",
            "        state = _local.state",
            "    except AttributeError:",
            "        state = _launcher._new_buffer()",
            "    try:",
            f"        _pack_into(state[0], 0, {', '.join(values)})",
            "    except _struct_error:",
            "        # Out-of-range values: let wp.launch() convert them or report the error.",
            f"        return _launcher._fallback(dim, [{', '.join(params)}])",
            "    _launcher._hooks.forward(state[1], state[2])",
        ]
        self._source = "\n".join(lines)
        exec(compile(self._source, f"<wp.CpuLauncher {self.kernel.key}>", "exec"), names)
        return names["launch"]

    def _array_arg_lines(self, index: int, a: str, arg_type, names: dict) -> list[str]:
        """Generated code that sets p (data), g (grad), f (flags), s*/z* (shape/strides) for one array."""
        ndim = arg_type.ndim
        dtype = arg_type.dtype
        names[f"_dtype{index}"] = dtype
        layout = _types._ndarray_layout(dtype)
        shape_vars = ", ".join(f"s{index}_{i}" for i in range(ndim))
        stride_vars = ", ".join(f"z{index}_{i}" for i in range(ndim))
        shape_values = ", ".join(f"s[{i}]" for i in range(ndim))
        stride_values = ", ".join(f"z[{i}]" for i in range(ndim))
        lines = [f"    t = type({a})"]
        if layout is not None:
            np_dtype, inner_shape, inner_strides, _ = layout
            names[f"_np_dtype{index}"] = np_dtype
            inner_checks = "".join(
                f" or s[{ndim + i}] != {inner_shape[i]} or z[{ndim + i}] != {inner_strides[i]}"
                for i in range(len(inner_shape))
            )
            lines += [
                "    if t is _ndarray:",
                f"        s = {a}.shape",
                f"        z = {a}.strides",
                f"        if {a}.dtype != _np_dtype{index} or len(s) != {ndim + len(inner_shape)}{inner_checks}:",
                f"            _launcher._reject_array({index}, {a})",
                "        try:",
                f"            p{index} = _addressof(_from_buffer({a}))",
                "        except (TypeError, ValueError, BufferError):",
                f"            p{index} = {a}.ctypes.data",
                f"        g{index} = 0",
                f"        f{index} = 0",
                "    elif t is _warp_array:",
            ]
        else:
            lines.append("    if t is _warp_array:")
        lines += [
            f"        if ({a}.dtype is not _dtype{index} and not _types_equal({a}.dtype, _dtype{index})) or {a}.ndim != {ndim}:",
            f"            _launcher._reject_array({index}, {a})",
            f"        p{index} = {a}.ptr or 0",
            f"        grad = {a}._grad",
            f"        g{index} = 0 if grad is None else (grad.ptr or 0)",
            f"        f{index} = 1 if {a}._retain_grad else 0",
            f"        s = {a}.shape",
            f"        z = {a}.strides",
            "    else:",
            f"        p{index}, g{index}, s, z, f{index} = _launcher._array_fields({index}, {a})",
            f"    {shape_vars} = {shape_values}",
            f"    {stride_vars} = {stride_values}",
        ]
        return lines

    @staticmethod
    def _make_packer(fields: list[tuple[int, str, str]]) -> tuple[struct.Struct, list[str]]:
        """One native-byte-order struct format, with explicit padding, writing ``fields`` in offset order."""
        fmt = "="
        values = []
        position = 0
        for offset, code, value in sorted(fields):
            if offset < position:
                raise RuntimeError("Overlapping launch argument fields")
            if offset > position:
                fmt += f"{offset - position}x"
            fmt += code
            position = offset + struct.calcsize("=" + code)
            values.append(value)
        return struct.Struct(fmt), values

    # -- conversions off the hot path ---------------------------------------------------------

    def _resolve_dim(self, dim) -> tuple[tuple[int, ...], int, int]:
        dim, size = _context._normalize_launch_dim(dim)
        kernel_dim = self.kernel.adj.kernel_dim
        padded = dim + (1,) * max(0, kernel_dim - len(dim))
        # Extra trailing dimensions fold into coord_mult, as in wp.launch().
        coord_mult = 1
        for extent in padded[kernel_dim:]:
            coord_mult *= extent
        return padded[:kernel_dim], size, coord_mult

    def _check_tid_extents(self, dim, size: int) -> None:
        dim, _ = _context._normalize_launch_dim(dim)
        _context._validate_kernel_tid_extents(self.kernel, dim, size, 1)

    def _pack_arg(self, index: int, value):
        name, arg_type = self._arg_info[index]
        return _context.pack_arg(self.kernel, arg_type, name, value, self.device)

    def _reject_array(self, index: int, value) -> None:
        # Raises with wp.launch()'s error message for this argument.
        self._pack_arg(index, value)
        name, arg_type = self._arg_info[index]
        raise RuntimeError(
            f"Error launching kernel '{self.kernel.key}', argument '{name}' expects an array of type "
            f"{_types.type_repr(arg_type)}, but passed value has type {type(value).__name__}."
        )

    def _array_fields(self, index: int, value):
        packed = self._pack_arg(index, value)
        return packed.data, packed.grad, tuple(packed.shape), tuple(packed.strides), packed.flags

    def _scalar_value(self, index: int, value):
        return self._pack_arg(index, value).value

    def _value_bytes(self, index: int, value) -> bytes:
        _, arg_type = self._arg_info[index]
        value_type = type(value)
        if isinstance(value, ctypes.Array) and _types.types_equal(value_type, arg_type):
            # Equivalent value classes (e.g. wp.types.vector(3, float) vs. wp.vec3) take the fast path next time.
            self._value_types[index].add(value_type)
            return bytes(value)
        return bytes(self._pack_arg(index, value))

    def _packed_bytes(self, index: int, value) -> bytes:
        return bytes(self._pack_arg(index, value))

    def _fallback(self, dim, args: list) -> None:
        # Graph capture records Warp arrays, so view NumPy arguments as Warp arrays.
        for index, (value, (_, arg_type)) in enumerate(zip(args, self._arg_info, strict=True)):
            if type(value) is np.ndarray and _types.is_array(arg_type):
                args[index] = warp.array(value, dtype=arg_type.dtype, device=self.device, copy=False)
        warp.launch(self.kernel, dim=dim, inputs=args, device=self.device, block_dim=1)
