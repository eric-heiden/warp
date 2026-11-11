// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Warp-level reductions (wp.warp_sum, wp.warp_max): butterfly reductions with __shfl_xor_sync over the 32 lanes
// of a warp, every lane receives the result. On the CPU each thread is its own warp.

namespace wp {

#if defined(__CUDA_ARCH__)
#define WP_WARP_FULL_MASK 0xffffffffu

template <typename T> inline CUDA_CALLABLE_DEVICE T warp_butterfly_sum(T x)
{
    WP_PRAGMA_UNROLL
    for (int m = 16; m > 0; m >>= 1)
        x += __shfl_xor_sync(WP_WARP_FULL_MASK, x, m);
    return x;
}

template <typename T> inline CUDA_CALLABLE_DEVICE T warp_butterfly_max(T x)
{
    WP_PRAGMA_UNROLL
    for (int m = 16; m > 0; m >>= 1) {
        T y = __shfl_xor_sync(WP_WARP_FULL_MASK, x, m);
        x = y > x ? y : x;
    }
    return x;
}
#endif

template <typename T> inline CUDA_CALLABLE T warp_sum(T x)
{
#if defined(__CUDA_ARCH__)
    return warp_butterfly_sum(x);
#else
    return x;
#endif
}

template <typename T> inline CUDA_CALLABLE T warp_max(T x)
{
#if defined(__CUDA_ARCH__)
    return warp_butterfly_max(x);
#else
    return x;
#endif
}

// 16-bit floats are reduced in float32
inline CUDA_CALLABLE half warp_sum(half x) { return half(warp_sum(float(x))); }
inline CUDA_CALLABLE half warp_max(half x) { return half(warp_max(float(x))); }
#ifndef WP_NO_BFLOAT16
inline CUDA_CALLABLE bfloat16 warp_sum(bfloat16 x) { return bfloat16(warp_sum(float(x))); }
inline CUDA_CALLABLE bfloat16 warp_max(bfloat16 x) { return bfloat16(warp_max(float(x))); }
#endif

// every lane's output depends on every lane's input: the adjoint of the sum is the sum of the output adjoints
template <typename T> inline CUDA_CALLABLE void adj_warp_sum(T x, T& adj_x, T adj_ret) { adj_x += warp_sum(adj_ret); }

// the summed output adjoints flow to the lane(s) holding the maximum
template <typename T> inline CUDA_CALLABLE void adj_warp_max(T x, T ret, T& adj_x, T adj_ret)
{
    T s = warp_sum(adj_ret);
    if (x == ret)
        adj_x += s;
}

}  // namespace wp
