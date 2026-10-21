# Low-Overhead CPU Launches

**Status**: Proposed

**Issue**: none yet

## Motivation

Host-side code increasingly uses CPU kernels for small batches: Newton's `ModelBuilder` composes
per-copy transforms while merging builders (one copy transform applied to ~80 body poses, free-root
joint coordinates, twist rotation), and its USD importer computes rest-shape geometry for triangles,
tetrahedra, and bending edges. Kernels are attractive there because they reuse Warp's builtins
(`wp.transform_multiply`, `wp.quat_rotate`, ...) instead of hand-mirrored NumPy, and stay
bit-identical to the builtins.

At these sizes the kernel is cheap and the launch is not. `wp.launch` on the CPU costs 8 µs with
one existing `wp.array` argument and grows by about 1 µs per Warp array and 4 µs per NumPy array. With
`wp.array(ndarray, copy=False)` per call it is 20–60 µs. A NumPy expression over 80 rows costs 30–70 µs,
so the kernel only wins after fusing several NumPy calls. Measured in Newton, the launch path decided
whether a kernel version paid off at all.

Where the time goes for one `wp.launch` with 3 existing Warp arrays and 80 elements (11 µs in total,
x86-64, Python 3.12):

| step | µs |
| --- | --- |
| `Module.load()` hash check | 0.8 |
| `get_kernel_hooks()` | 0.35 |
| launch bounds (new ctypes struct) | 1.0 |
| `init`, `get_device`, block_dim, dim normalization, tid-extent check | 0.7 |
| `pack_arg` per Warp array | 0.85 × 3 |
| `invoke`: new `ArgsStruct`, one `setattr` per field, native call | 2.75 (call ≈ 1.0) |
| rest of `wp.launch` (argument lists, deterministic/APIC/tape checks) | ≈ 2.5 |

Per NumPy argument, `pack_arg` spends about 4 µs: `__array_interface__` builds a dict (1.0 µs),
`array_ctype_from_interface` re-derives the dtype (2.0 µs), and the `array_t` constructor adds 0.7 µs.

The floor in pure Python is the ctypes call itself (≈ 0.55 µs for an empty function) plus getting a
NumPy data pointer (0.36 µs through `ctypes.c_char.from_buffer`, 1 µs through `ndarray.ctypes.data`).

## Requirements

| ID | Requirement | Priority | Notes |
| --- | --- | --- | --- |
| R1 | A small CPU launch costs a few µs, not tens of µs | Must | 1–1000 elements |
| R2 | NumPy arrays are accepted without copies | Must | host-side callers hold NumPy buffers |
| R3 | Results and argument conversions match `wp.launch` | Must | same rules, same errors |
| R4 | Safe to call from several threads | Must | |
| R5 | Launches are still recorded by `wp.Tape` and CPU graph capture | Must | fall back to `wp.launch` |
| R6 | Existing `wp.launch` callers get faster without code changes | Should | |
| R7 | Value parameters (`wp.transform`, ...) accept NumPy rows | Could | `wp.launch` rejects them today |

**Non-goals**: CUDA launches (driver overhead dominates differently), adjoint launches, and
cooperative CPU blocks (`block_dim > 1`).

## Design

### Approach

Two independent changes:

1. **Cheaper argument packing inside `wp.launch` (no API change).** `pack_arg` gets fast paths
   for an exact `wp.array` or host `np.ndarray` passed to a regular array parameter. The NumPy path
   caches the expected dtype layout per Warp dtype (`_ndarray_layout`), reads the data pointer through
   the buffer protocol, and returns `None` for anything unusual, so the existing conversion still
   handles and reports those cases. The cached `invoke` paths build the argument struct with one
   positional constructor call.

2. **`wp.CpuLauncher(kernel)`, an opt-in launcher for repeated launches of one kernel.** It
   resolves the module executable, kernel hooks, and argument struct layout once. It then generates a
   Python launch function for the kernel's signature, in which each argument's type dispatch and
   checks are unrolled. That function collects every field value into local variables and writes the
   launch bounds and the whole argument struct with one `struct.Struct.pack_into` call into a
   per-thread buffer. The struct format is derived from the ctypes field offsets, with explicit
   padding. One ctypes call then runs the kernel.

Measured as interleaved A/B runs against `main`, µs per launch, with transform arrays:

| path | 1 arg, n=1 | 1 arg, n=80 | 3 args, n=1 | 3 args, n=80 | 5 args, n=1 | 5 args, n=80 |
| --- | --- | --- | --- | --- | --- | --- |
| `wp.launch`, Warp arrays (main) | 7.81 | 8.18 | 10.21 | 10.92 | 12.75 | 14.07 |
| `wp.launch`, Warp arrays (this change) | 7.52 | 7.96 | 9.89 | 10.44 | 12.21 | 13.43 |
| `wp.launch`, NumPy (main) | 12.82 | 13.04 | 21.75 | 22.65 | 31.37 | 32.82 |
| `wp.launch`, NumPy (this change) | 10.57 | 10.57 | 16.31 | 17.10 | 22.58 | 23.80 |
| `Launch.launch()` replay (main) | 1.75 | 2.19 | 2.22 | 2.89 | 2.61 | 3.92 |
| `CpuLauncher`, Warp arrays | 1.25 | 1.67 | 1.62 | 2.34 | 2.29 | 3.59 |
| `CpuLauncher`, NumPy | 1.78 | 2.18 | 2.90 | 3.61 | 4.32 | 5.65 |

At 100,000 elements all paths are within 2 % of each other.

In Newton prototypes on Warp 1.18, launcher-based kernels replaced NumPy transform helpers bit-exactly.
Building 1.57M boxes with rotated copies took 38.6–39.3 s with NumPy and 27.4–29.0 s with kernels.
USD volume import with authored rest shapes (10,368 tets) went from 569 ms to 399 ms.

### Alternatives Considered

- **A per-call fast path that duplicates `wp.launch` (cache module exec, hooks and packers per
  kernel).** A prototype reached 3.4 µs with 1 Warp array and 9 µs with 3 NumPy arrays. It
  duplicates the decision logic of `wp.launch` (tape, APIC, deterministic mode, access modes,
  generic kernels) on a second code path, which is easy to let diverge. This change keeps
  `wp.launch` on one path and moves the large win into an explicit opt-in object.
- **Rebinding recorded `Launch` objects by raw pointer.** `Launch.launch()` replay is already
  1.7–2.7 µs, but rebinding an array costs `wp.array(ptr=...)` (3.8 µs) plus `set_param_at_index`
  (6.5 µs). `set_param_at_index_from_ctype` with a private `array_t` costs about 2.5 µs per argument
  and exposes internals. A persistent struct inside `Launch` would make replays cheaper, but callers
  still pay per-argument rebinding, and `Launch` objects are single-threaded.
- **A lighter `wp.array` constructor.** `wp.array(ptr=...)` and `wp.array(ndarray, copy=False)`
  spend 3.8–5 µs in Python bookkeeping. Even halving that leaves `pack_arg` and the fixed `wp.launch`
  cost.
- **Field-by-field writes into a persistent ctypes struct (first launcher prototype).** At about
  1.3 µs per NumPy argument it was 2× slower than one `pack_into`, because each ctypes field write
  costs 70–200 ns, and reading `struct.field.shape` allocates a new view each time.
- **Native packing.** Warp's native library is loaded through ctypes and does not link libpython, so
  it cannot read Python objects or the buffer protocol. A small CPython extension could bring a
  launch down to roughly the ctypes-call floor, but it adds a compiled extension to the package.

### Key Implementation Details

- **Generated code.** The launch function is Python source compiled with `exec`. Its body depends
  only on the kernel signature. It is kept as `CpuLauncher._source` for debugging, and the
  `compile()` filename names the kernel, so tracebacks are readable.
- **Conversions match `wp.launch`.** Arguments the fast paths do not cover go through `pack_arg`
  and are copied as bytes: indexed arrays, structs, textures, float16/bfloat16 scalars, and values
  of other types. Rejected arrays re-run `pack_arg` to raise `wp.launch`'s error. One deliberate
  difference: a NumPy array whose dimension count does not match the parameter is rejected, whereas
  `array_ctype_from_interface` accepts it.
- **Value classes.** Warp can create distinct but equivalent value classes, for example
  `wp.types.vector(3, wp.float32)` and `wp.vec3f`. The launcher learns such classes on first use,
  so they take the fast path afterwards. Without this, each value went through `pack_arg` (≈ 2 µs).
- **Staleness.** Before each launch the launcher checks `module.hashers` identity (replaced by
  `Module.mark_modified()`), `module.execs[key]` identity (replaced on reload), and
  `module.has_unresolved_static_expressions`, which forces the rebind path every call. A module
  generation counter in `Module` would make this check explicit.
- **Thread safety.** ctypes `CFUNCTYPE` calls release the GIL while the kernel runs, so the
  argument buffer is per thread (`threading.local`). Out-of-range values raise `struct.error`, which
  is caught, and the call is retried through `wp.launch` so its conversion rules apply.
- **NumPy lifetimes.** CPU launches are synchronous, so the launcher keeps no reference after a
  call. Recorded `Launch` objects keep their NumPy arguments alive through `array_t._ref`, as the
  interface path already did.

## Testing Strategy

- `warp/tests/test_launch.py`: NumPy arguments to CPU array parameters, including outer-strided,
  read-only, and vector/matrix dtypes, a recorded launch, and rejection of mismatched inner shape,
  inner strides, and dtypes. These tests pass on `main` and with the change, which pins the
  unchanged behavior.
- `warp/tests/test_cpu_launch.py` checks `CpuLauncher` against `wp.launch` for:
  - Warp and NumPy arrays, views, 2D broadcast launches with tuple dimensions, and scalars of
    every width including float16;
  - transform values as Warp values, equivalent classes, and NumPy rows;
  - structs, indexed arrays, and concrete overloads of generic kernels;
  - rejected arguments, empty and negative dimensions, eight concurrent threads, module reload,
    and `wp.Tape` and CPU graph-capture fallbacks.
- `asv/benchmarks/api/launch.py::CpuLaunchOverhead` tracks the CPU launch paths.
