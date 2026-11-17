// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Built-in device GEMM used by wp.gemm() when cuBLASLt cannot be loaded, and by
// API-capture (APIC) graph reconstruction on a guest without cuBLASLt.
//
// Shared-memory tiled SIMT kernels: each block computes a BM x BN output tile,
// each thread a TM x TN register sub-tile, staged through BK-wide shared slices
// with the next slice prefetched into registers. Skinny problems (a few rows)
// use a split-K kernel instead. The kernels are compiled into warp.so for all of
// Warp's CUDA targets, so they need no runtime compilation, no extra library and
// no Python. They do not use tensor cores, so they trail cuBLASLt for large
// FP16/BF16 problems.
//
// Kernels are templated on the operand type only; the output (and bias) type,
// which is either the operand type or float32 for mixed precision, is a runtime
// flag to keep the number of instantiations (and the size of warp.so) small.

#include "warp.h"

#include "apic_types.h"
#include "cuda_util.h"
#include "error.h"
#include "gemm.h"

#include <type_traits>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace {

template <typename T> struct acc_of {
    using type = float;
};
template <> struct acc_of<double> {
    using type = double;
};

template <typename A, typename T> __device__ __forceinline__ A to_acc(T v) { return A(v); }
template <> __device__ __forceinline__ float to_acc<float, __half>(__half v) { return __half2float(v); }
template <> __device__ __forceinline__ float to_acc<float, __nv_bfloat16>(__nv_bfloat16 v)
{
    return __bfloat162float(v);
}

template <typename T, typename A> __device__ __forceinline__ T from_acc(A v) { return T(v); }
template <> __device__ __forceinline__ __half from_acc<__half, float>(float v) { return __float2half_rn(v); }
template <> __device__ __forceinline__ __nv_bfloat16 from_acc<__nv_bfloat16, float>(float v)
{
    return __float2bfloat16_rn(v);
}

struct GemmArgs {
    const void* a;
    const void* b;
    void* c;
    const void* bias;
    int m, n, k, lda, ldb, ldc;
    long long sa, sb, sc;
    int trans_a, trans_b;
    int has_bias, relu;
    int c_f32;  // output and bias are float32 (mixed precision) rather than the operand type
    double alpha, beta;
};

// Output/bias element i, of the operand type or float32.
template <typename TAB, typename ACC> __device__ __forceinline__ ACC load_c(const void* base, long long i, int c_f32)
{
    return c_f32 ? ACC(static_cast<const float*>(base)[i]) : to_acc<ACC>(static_cast<const TAB*>(base)[i]);
}

template <typename TAB, typename ACC> __device__ __forceinline__ void store_c(void* base, long long i, int c_f32, ACC v)
{
    if (c_f32)
        static_cast<float*>(base)[i] = float(v);
    else
        static_cast<TAB*>(base)[i] = from_acc<TAB>(v);
}

// out[i] = relu?(alpha * acc + bias[j]) + beta * out[i]; alpha and beta are converted to ACC once per thread
// (double conversions and compares are slow on GPUs with reduced FP64 throughput).
template <typename TAB, typename ACC>
__device__ __forceinline__ void epilogue(const GemmArgs& p, ACC alpha, ACC beta, long long i, int j, ACC acc)
{
    ACC v = alpha * acc;
    if (p.has_bias)
        v += load_c<TAB, ACC>(p.bias, j, p.c_f32);
    if (p.relu)
        v = v > ACC(0) ? v : ACC(0);
    if (beta != ACC(0))
        v += beta * load_c<TAB, ACC>(p.c, i, p.c_f32);
    store_c<TAB, ACC>(p.c, i, p.c_f32, v);
}

template <typename TAB, int BM, int BN, int BK, int TM, int TN>
__global__ void __launch_bounds__((BM / TM) * (BN / TN)) gemm_tiled(GemmArgs p)
{
    using ACC = typename acc_of<TAB>::type;
    constexpr int NT = (BM / TM) * (BN / TN);
    constexpr int TX = BN / TN;  // threads along N

    __shared__ ACC As[BK][BM + 1];
    __shared__ ACC Bs[BK][BN + 1];

    const int z = blockIdx.z;
    const TAB* A = static_cast<const TAB*>(p.a) + z * p.sa;
    const TAB* B = static_cast<const TAB*>(p.b) + z * p.sb;

    const int m0 = blockIdx.y * BM;
    const int n0 = blockIdx.x * BN;
    const int tid = threadIdx.x;
    const int tx = tid % TX;
    const int ty = tid / TX;

    ACC acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j)
            acc[i][j] = ACC(0);

    // Each thread stages EA elements of the op(A) tile and EB of the op(B) tile per K-slice; consecutive threads
    // walk the contiguous dimension of each operand. The next slice is prefetched into registers before computing
    // on the current one, so global latency overlaps the FMAs.
    constexpr int EA = (BM * BK + NT - 1) / NT;
    constexpr int EB = (BK * BN + NT - 1) / NT;
    ACC ra_pf[EA], rb_pf[EB];

    auto load_slice = [&](int k0) {
#pragma unroll
        for (int t = 0; t < EA; ++t) {
            const int e = tid + t * NT;
            const int i = p.trans_a ? e % BM : e / BK;  // A is (K, M) if transposed, else (M, K)
            const int kk = p.trans_a ? e / BM : e % BK;
            const int gi = m0 + i, gk = k0 + kk;
            ACC v = ACC(0);
            if (e < BM * BK && gi < p.m && gk < p.k)
                v = to_acc<ACC>(p.trans_a ? A[(long long)gk * p.lda + gi] : A[(long long)gi * p.lda + gk]);
            ra_pf[t] = v;
        }
#pragma unroll
        for (int t = 0; t < EB; ++t) {
            const int e = tid + t * NT;
            const int j = p.trans_b ? e / BK : e % BN;  // B is (N, K) if transposed, else (K, N)
            const int kk = p.trans_b ? e % BK : e / BN;
            const int gj = n0 + j, gk = k0 + kk;
            ACC v = ACC(0);
            if (e < BK * BN && gj < p.n && gk < p.k)
                v = to_acc<ACC>(p.trans_b ? B[(long long)gj * p.ldb + gk] : B[(long long)gk * p.ldb + gj]);
            rb_pf[t] = v;
        }
    };
    auto store_slice = [&]() {
#pragma unroll
        for (int t = 0; t < EA; ++t) {
            const int e = tid + t * NT;
            if (e < BM * BK) {
                if (p.trans_a)
                    As[e / BM][e % BM] = ra_pf[t];
                else
                    As[e % BK][e / BK] = ra_pf[t];
            }
        }
#pragma unroll
        for (int t = 0; t < EB; ++t) {
            const int e = tid + t * NT;
            if (e < BK * BN) {
                if (p.trans_b)
                    Bs[e % BK][e / BK] = rb_pf[t];
                else
                    Bs[e / BN][e % BN] = rb_pf[t];
            }
        }
    };

    load_slice(0);
    for (int k0 = 0; k0 < p.k; k0 += BK) {
        store_slice();
        __syncthreads();
        if (k0 + BK < p.k)
            load_slice(k0 + BK);  // in flight while computing on the current slice

#pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            ACC ra[TM], rb[TN];
#pragma unroll
            for (int i = 0; i < TM; ++i)
                ra[i] = As[kk][ty + i * (BM / TM)];
#pragma unroll
            for (int j = 0; j < TN; ++j)
                rb[j] = Bs[kk][tx + j * TX];
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j)
                    acc[i][j] += ra[i] * rb[j];
        }
        __syncthreads();
    }

    const ACC alpha = ACC(p.alpha), beta = ACC(p.beta);
#pragma unroll
    for (int i = 0; i < TM; ++i) {
        const int gi = m0 + ty + i * (BM / TM);
        if (gi >= p.m)
            continue;
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int gj = n0 + tx + j * TX;
            if (gj < p.n)
                epilogue<TAB, ACC>(p, alpha, beta, z * p.sc + (long long)gi * p.ldc + gj, gj, acc[i][j]);
        }
    }
}

// Skinny GEMM (M <= MAXM rows, e.g. single-sample inference): a block computes 32 output columns for all rows,
// splitting K over 8 thread rows and reducing through shared memory, so even a handful of blocks keeps many loads
// in flight instead of walking K serially per tile.
template <typename TAB, int MAXM> __global__ void __launch_bounds__(256) gemm_small_m(GemmArgs p)
{
    using ACC = typename acc_of<TAB>::type;
    __shared__ ACC part[8][MAXM][33];
    const int z = blockIdx.z;
    const TAB* A = static_cast<const TAB*>(p.a) + z * p.sa;
    const TAB* B = static_cast<const TAB*>(p.b) + z * p.sb;
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int j = blockIdx.x * 32 + tx;

    ACC acc[MAXM];
#pragma unroll
    for (int i = 0; i < MAXM; ++i)
        acc[i] = ACC(0);
    if (j < p.n) {
#pragma unroll 4
        for (int k = ty; k < p.k; k += 8) {
            const ACC bv = to_acc<ACC>(p.trans_b ? B[(long long)j * p.ldb + k] : B[(long long)k * p.ldb + j]);
#pragma unroll
            for (int i = 0; i < MAXM; ++i)
                if (i < p.m)
                    acc[i] += to_acc<ACC>(p.trans_a ? A[(long long)k * p.lda + i] : A[(long long)i * p.lda + k]) * bv;
        }
    }
#pragma unroll
    for (int i = 0; i < MAXM; ++i)
        part[ty][i][tx] = acc[i];
    __syncthreads();
    if (j >= p.n)
        return;
    // thread row ty finalizes output rows ty, ty + 8, ...
    const ACC alpha = ACC(p.alpha), beta = ACC(p.beta);
    for (int i = ty; i < p.m && i < MAXM; i += 8) {
        ACC v = ACC(0);
#pragma unroll
        for (int y = 0; y < 8; ++y)
            v += part[y][i][tx];
        epilogue<TAB, ACC>(p, alpha, beta, z * p.sc + (long long)i * p.ldc + j, j, v);
    }
}

// bias_grad[j] = sum over batches and k of op(b)[k, j] (op(b) = b for BGRADA, see gemm()).
template <typename TAB> __global__ void gemm_colsum(GemmArgs p, int batch, void* g)
{
    using ACC = typename acc_of<TAB>::type;
    __shared__ ACC part[8][33];
    const TAB* b = static_cast<const TAB*>(p.b);
    const int j = blockIdx.x * 32 + threadIdx.x;
    ACC s = ACC(0);
    if (j < p.n)
        for (int z = 0; z < batch; ++z)
            for (int r = threadIdx.y; r < p.k; r += 8)
                s += to_acc<ACC>(b[z * p.sb + (long long)r * p.ldb + j]);
    part[threadIdx.y][threadIdx.x] = s;
    __syncthreads();
    if (threadIdx.y == 0 && j < p.n) {
        for (int y = 1; y < 8; ++y)
            s += part[y][threadIdx.x];
        store_c<TAB, ACC>(g, j, p.c_f32, s);
    }
}

template <typename TAB, int BM, int BN, int BK, int TM, int TN>
void launch_tiled(const GemmArgs& p, int batch, cudaStream_t stream)
{
    dim3 grid((p.n + BN - 1) / BN, (p.m + BM - 1) / BM, batch);
    gemm_tiled<TAB, BM, BN, BK, TM, TN><<<grid, (BM / TM) * (BN / TN), 0, stream>>>(p);
}

template <typename TAB>
bool run(const wp_gemm_desc& d, const void* a, const void* b, void* c, void* bias, cudaStream_t stream)
{
    GemmArgs p;
    p.a = a;
    p.b = b;
    p.c = c;
    p.bias = (d.epilogue == APIC_GEMM_EPILOGUE_BIAS || d.epilogue == APIC_GEMM_EPILOGUE_RELU_BIAS) ? bias : nullptr;
    p.m = d.m;
    p.n = d.n;
    p.k = d.k;
    p.lda = d.lda;
    p.ldb = d.ldb;
    p.ldc = d.ldc;
    p.sa = d.stride_a;
    p.sb = d.stride_b;
    p.sc = d.stride_c;
    p.trans_a = d.trans_a;
    p.trans_b = d.trans_b;
    p.has_bias = p.bias != nullptr;
    p.relu = d.epilogue == APIC_GEMM_EPILOGUE_RELU || d.epilogue == APIC_GEMM_EPILOGUE_RELU_BIAS;
    p.c_f32 = d.c_dtype == APIC_TYPE_FLOAT32 && d.ab_dtype != APIC_TYPE_FLOAT32;
    p.alpha = d.alpha;
    p.beta = d.beta;

    // Skinny problems split K across threads; otherwise the largest tile that still yields enough blocks to fill
    // the GPU: 128x128 (8x8 per thread, not for float64 to limit code size), 64x64 (4x4) or 32x32 (2x2).
    const long long tiles128 = (long long)((d.m + 127) / 128) * ((d.n + 127) / 128) * d.batch;
    const long long tiles64 = (long long)((d.m + 63) / 64) * ((d.n + 63) / 64) * d.batch;
    if (d.m <= 8) {
        dim3 grid((d.n + 31) / 32, 1, d.batch);
        gemm_small_m<TAB, 8><<<grid, dim3(32, 8), 0, stream>>>(p);
    } else if (!std::is_same<TAB, double>::value && tiles128 >= 128)
        launch_tiled<
            TAB, std::is_same<TAB, double>::value ? 64 : 128, std::is_same<TAB, double>::value ? 64 : 128, 8,
            std::is_same<TAB, double>::value ? 4 : 8, std::is_same<TAB, double>::value ? 4 : 8>(p, d.batch, stream);
    else if (tiles64 >= 96)
        launch_tiled<TAB, 64, 64, 16, 4, 4>(p, d.batch, stream);
    else
        launch_tiled<TAB, 32, 32, 16, 2, 2>(p, d.batch, stream);

    if (d.epilogue == APIC_GEMM_EPILOGUE_BGRADA && bias)
        gemm_colsum<TAB><<<dim3((d.n + 31) / 32), dim3(32, 8), 0, stream>>>(p, d.batch, bias);
    return check_cuda(cudaGetLastError());
}

template <typename TAB> bool run_colsum(const wp_gemm_desc& d, const void* b, void* bias, cudaStream_t stream)
{
    GemmArgs p = {};
    p.b = b;
    p.n = d.n;
    p.k = d.k;
    p.ldb = d.ldb;
    p.sb = d.stride_b;
    p.c_f32 = d.c_dtype == APIC_TYPE_FLOAT32 && d.ab_dtype != APIC_TYPE_FLOAT32;
    gemm_colsum<TAB><<<dim3((d.n + 31) / 32), dim3(32, 8), 0, stream>>>(p, d.batch, bias);
    return check_cuda(cudaGetLastError());
}

}  // namespace

bool wp_gemm_builtin_device(
    void* context, void* stream, const wp_gemm_desc& d, const void* a, const void* b, void* c, void* bias
)
{
    if (d.m <= 0 || d.n <= 0 || d.batch <= 0)
        return true;
    ContextGuard guard(context);
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    switch (d.ab_dtype) {
    case APIC_TYPE_FLOAT32:
        return run<float>(d, a, b, c, bias, s);
    case APIC_TYPE_FLOAT64:
        return run<double>(d, a, b, c, bias, s);
    case APIC_TYPE_FLOAT16:
        return run<__half>(d, a, b, c, bias, s);
    case APIC_TYPE_BFLOAT16:
        return run<__nv_bfloat16>(d, a, b, c, bias, s);
    default:
        wp::set_error_string("Warp GEMM error: unsupported dtype %d", d.ab_dtype);
        return false;
    }
}

bool wp_gemm_colsum_device(void* context, void* stream, const wp_gemm_desc& d, const void* b, void* bias)
{
    if (d.n <= 0 || !bias)
        return true;
    ContextGuard guard(context);
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    switch (d.ab_dtype) {
    case APIC_TYPE_FLOAT32:
        return run_colsum<float>(d, b, bias, s);
    case APIC_TYPE_FLOAT64:
        return run_colsum<double>(d, b, bias, s);
    case APIC_TYPE_FLOAT16:
        return run_colsum<__half>(d, b, bias, s);
    case APIC_TYPE_BFLOAT16:
        return run_colsum<__nv_bfloat16>(d, b, bias, s);
    default:
        wp::set_error_string("Warp GEMM error: unsupported dtype %d", d.ab_dtype);
        return false;
    }
}
