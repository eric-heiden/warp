// Copyright 2024 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "engine_collision_convex.h"  // mjx/cuda

#include <driver_types.h>  // cuda
#include <pybind11/pybind11.h>
#include <xla/ffi/api/c_api.h>
#include <xla/ffi/api/ffi.h>

namespace mujoco::mjx::cuda {

namespace ffi = xla::ffi;

static const auto *kGjkEpa =
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // geom_pair,
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // geom_xpos,
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // geom_xmat,
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // geom_size,
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // geom_dataid,
        .Arg<ffi::Buffer<ffi::DataType::F32>>()  // convex_vertex_array,
        .Arg<ffi::Buffer<ffi::DataType::U32>>()  // convex_vertex_offset,
        .Attr<uint>("ngeom")
        .Attr<uint>("npair")
        .Attr<uint>("ncon")
        .Attr<uint>("geom_type0")
        .Attr<uint>("geom_type1")
        .Attr<float>("depth_extension")
        .Attr<uint>("gjk_iteration_count")
        .Attr<uint>("epa_iteration_count")
        .Attr<uint>("epa_best_count")
        .Attr<uint>("multi_polygon_count")
        .Attr<float>("multi_tilt_angle")
        .Ret<ffi::Buffer<ffi::DataType::F32>>()  // dist,
        .Ret<ffi::Buffer<ffi::DataType::F32>>()  // pos,
        .Ret<ffi::Buffer<ffi::DataType::F32>>()  // normal,
        .Ret<ffi::Buffer<ffi::DataType::F32>>()  // simplex,
        .To(LaunchKernel_Gjk_Epa)
        .release();

XLA_FFI_Error *gjk_epa(XLA_FFI_CallFrame *call_frame) {
  return kGjkEpa->Call(call_frame);
}

namespace {

namespace py = pybind11;

template <typename T>
py::capsule EncapsulateFfiCall(T *fn) {
  static_assert(std::is_invocable_r_v<XLA_FFI_Error *, T, XLA_FFI_CallFrame *>,
                "Encapsulated function must be an XLA FFI handler");
  return py::capsule(reinterpret_cast<void *>(fn), "xla._CUSTOM_CALL_TARGET");
}

PYBIND11_MODULE(_engine_collision_convex, m) {
  m.def("gjk_epa", []() { return EncapsulateFfiCall(gjk_epa); });
}

}  // namespace

}  // namespace mujoco::mjx::cuda
