// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// wp.gemm() native entry points.
//
// wp_gemm_device() and wp_gemm_host() are the single native boundary for
// wp.gemm(): they record an APIC_OP_GEMM operation when an API capture is
// active and dispatch to cuBLASLt or to the built-in GEMMs. Recording the
// semantic operation instead of the cuBLASLt kernel lets a saved graph
// re-plan the GEMM with the cuBLASLt of the process that loads it, or fall
// back to the built-in device GEMM there.

#include "warp.h"

#include "apic.h"
#include "apic_internal.h"
#include "apic_types.h"
#include "error.h"
#include "gemm.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <type_traits>
#include <vector>

#if !defined(_WIN32)
#include <dlfcn.h>
#endif

bool g_gemm_host_use_blas = true;  // tests/benchmarks: force the built-in host GEMM

namespace {

// ---------------------------------------------------------------------------------------------------------------------
// Scalar conversions (host)

inline float half_to_float(uint16_t h)
{
    uint32_t sign = uint32_t(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1fu;
    uint32_t mant = h & 0x3ffu;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {  // subnormal: normalize
            exp = 127 - 15 + 1;
            while ((mant & 0x400u) == 0) {
                mant <<= 1;
                --exp;
            }
            mant &= 0x3ffu;
            bits = sign | (exp << 23) | (mant << 13);
        }
    } else if (exp == 0x1f) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

inline uint16_t float_to_half(float f)
{
    uint32_t x;
    memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000u;
    int32_t exp = int32_t((x >> 23) & 0xffu) - 127 + 15;
    uint32_t mant = x & 0x7fffffu;
    if (((x >> 23) & 0xffu) == 0xffu)  // inf / nan
        return uint16_t(sign | 0x7c00u | (mant ? 0x200u : 0u));
    if (exp >= 0x1f)
        return uint16_t(sign | 0x7c00u);
    if (exp <= 0) {
        if (exp < -10)
            return uint16_t(sign);
        mant |= 0x800000u;
        uint32_t shift = uint32_t(14 - exp);
        uint32_t h = mant >> shift;
        uint32_t rem = mant & ((1u << shift) - 1u);
        uint32_t half = 1u << (shift - 1);
        if (rem > half || (rem == half && (h & 1u)))
            ++h;
        return uint16_t(sign | h);
    }
    uint32_t h = sign | (uint32_t(exp) << 10) | (mant >> 13);
    uint32_t rem = mant & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (h & 1u)))
        ++h;  // may carry into the exponent, which is the correct rounding
    return uint16_t(h);
}

inline float bf16_to_float(uint16_t h)
{
    uint32_t bits = uint32_t(h) << 16;
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

inline uint16_t float_to_bf16(float f)
{
    uint32_t x;
    memcpy(&x, &f, 4);
    if ((x & 0x7fffffffu) > 0x7f800000u)
        return uint16_t((x >> 16) | 0x40u);  // quiet nan
    x += 0x7fffu + ((x >> 16) & 1u);
    return uint16_t(x >> 16);
}

template <typename ACC> inline ACC load_elem(const void* p, int64_t i, int dtype)
{
    switch (dtype) {
    case APIC_TYPE_FLOAT32:
        return ACC(static_cast<const float*>(p)[i]);
    case APIC_TYPE_FLOAT64:
        return ACC(static_cast<const double*>(p)[i]);
    case APIC_TYPE_FLOAT16:
        return ACC(half_to_float(static_cast<const uint16_t*>(p)[i]));
    default:
        return ACC(bf16_to_float(static_cast<const uint16_t*>(p)[i]));
    }
}

template <typename ACC> inline void store_elem(void* p, int64_t i, int dtype, ACC v)
{
    switch (dtype) {
    case APIC_TYPE_FLOAT32:
        static_cast<float*>(p)[i] = float(v);
        break;
    case APIC_TYPE_FLOAT64:
        static_cast<double*>(p)[i] = double(v);
        break;
    case APIC_TYPE_FLOAT16:
        static_cast<uint16_t*>(p)[i] = float_to_half(float(v));
        break;
    default:
        static_cast<uint16_t*>(p)[i] = float_to_bf16(float(v));
        break;
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Host BLAS provider
//
// CBLAS sgemm/dgemm function pointers. Python registers the OpenBLAS (or Accelerate) library NumPy ships with, which
// every Warp installation has; standalone C++ loaders try the system BLAS. ILP64 libraries take 64-bit integers.

enum { CBLAS_ROW_MAJOR = 101, CBLAS_NO_TRANS = 111, CBLAS_TRANS = 112 };
typedef void (*cblas_sgemm32_t)(
    int, int, int, int, int, int, float, const float*, int, const float*, int, float, float*, int
);
typedef void (*cblas_dgemm32_t)(
    int, int, int, int, int, int, double, const double*, int, const double*, int, double, double*, int
);
typedef void (*cblas_sgemm64_t)(
    int,
    int,
    int,
    int64_t,
    int64_t,
    int64_t,
    float,
    const float*,
    int64_t,
    const float*,
    int64_t,
    float,
    float*,
    int64_t
);
typedef void (*cblas_dgemm64_t)(
    int,
    int,
    int,
    int64_t,
    int64_t,
    int64_t,
    double,
    const double*,
    int64_t,
    const double*,
    int64_t,
    double,
    double*,
    int64_t
);

struct HostBlas {
    void* sgemm = nullptr;
    void* dgemm = nullptr;
    bool ilp64 = false;
    bool tried_default = false;
} g_host_blas;

void try_default_host_blas()
{
    if (g_host_blas.tried_default || g_host_blas.sgemm)
        return;
    g_host_blas.tried_default = true;
#if defined(__APPLE__)
    const char* libs[] = { "/System/Library/Frameworks/Accelerate.framework/Accelerate" };
#elif defined(_WIN32)
    const char* libs[] = { nullptr };
#else
    const char* libs[] = { "libopenblas.so.0", "libopenblas.so", "libblis.so.4", "libblas.so.3" };
#endif
#if !defined(_WIN32)
    for (const char* name : libs) {
        void* h = dlopen(name, RTLD_NOW | RTLD_LOCAL);
        if (!h)
            continue;
        void* s = dlsym(h, "cblas_sgemm");
        void* dd = dlsym(h, "cblas_dgemm");
        if (s && dd) {
            g_host_blas.sgemm = s;
            g_host_blas.dgemm = dd;
            g_host_blas.ilp64 = false;
            return;
        }
        dlclose(h);
    }
#else
    (void)libs;
#endif
}

// Run c = alpha * op(a) @ op(b) + beta * c with the host BLAS for each batch. Returns false if no BLAS is available.
template <typename T> bool blas_gemm(const wp_gemm_desc& d, const T* a, const T* b, T* c, double beta)
{
    try_default_host_blas();
    void* fn = sizeof(T) == 4 ? g_host_blas.sgemm : g_host_blas.dgemm;
    if (!fn)
        return false;
    const int ta = d.trans_a ? CBLAS_TRANS : CBLAS_NO_TRANS;
    const int tb = d.trans_b ? CBLAS_TRANS : CBLAS_NO_TRANS;
    for (int z = 0; z < d.batch; ++z) {
        const T* A = a + int64_t(z) * d.stride_a;
        const T* B = b + int64_t(z) * d.stride_b;
        T* C = c + int64_t(z) * d.stride_c;
        if (g_host_blas.ilp64) {
            if (sizeof(T) == 4)
                ((cblas_sgemm64_t)fn)(
                    CBLAS_ROW_MAJOR, ta, tb, d.m, d.n, d.k, float(d.alpha), (const float*)A, d.lda, (const float*)B,
                    d.ldb, float(beta), (float*)C, d.ldc
                );
            else
                ((cblas_dgemm64_t)fn)(
                    CBLAS_ROW_MAJOR, ta, tb, d.m, d.n, d.k, d.alpha, (const double*)A, d.lda, (const double*)B, d.ldb,
                    beta, (double*)C, d.ldc
                );
        } else {
            if (sizeof(T) == 4)
                ((cblas_sgemm32_t)fn)(
                    CBLAS_ROW_MAJOR, ta, tb, d.m, d.n, d.k, float(d.alpha), (const float*)A, d.lda, (const float*)B,
                    d.ldb, float(beta), (float*)C, d.ldc
                );
            else
                ((cblas_dgemm32_t)fn)(
                    CBLAS_ROW_MAJOR, ta, tb, d.m, d.n, d.k, d.alpha, (const double*)A, d.lda, (const double*)B, d.ldb,
                    beta, (double*)C, d.ldc
                );
        }
    }
    return true;
}

// ---------------------------------------------------------------------------------------------------------------------
// Built-in host GEMM
//
// GOTO-style blocking: op(b) is packed into NR-wide column panels and op(a) into MR-high row panels (converted to
// the accumulation type), and an MR x NR register-blocked micro-kernel accumulates each k-panel into a packed
// accumulator. The micro-kernel is compiled for AVX-512, AVX2+FMA and the baseline ISA and selected at run time
// from the CPU's features.

constexpr int KC = 256;  // k-panel depth
constexpr int NC = 512;  // n-panel width

#if defined(__GNUC__)
// Explicit vector registers: MR rows x 2 vectors of W = VB / sizeof(ACC) lanes (NR = 2 W), updated with broadcast
// FMAs. Accumulators are named variables rather than an array: plain loops let GCC's SLP vectorizer pick a
// transposed layout, and an array of vectors is kept in memory across iterations.
template <typename ACC, int MR, int NR, int VB>
__attribute__((always_inline)) inline void
micro_impl(const ACC* __restrict ap, const ACC* __restrict bp, int kc, ACC* __restrict out)
{
    typedef ACC V __attribute__((vector_size(VB)));
    typedef ACC VU __attribute__((vector_size(VB), aligned(sizeof(ACC))));  // unaligned loads/stores
    constexpr int W = VB / int(sizeof(ACC));
    static_assert(NR == 2 * W && MR <= 8, "micro-kernel expects two vectors per row and at most 8 rows");
    const V one = V {} + ACC(1);
#define WP_LD(p) (*reinterpret_cast<const VU*>(p))
#define WP_ST(p, v) (*reinterpret_cast<VU*>(p) = (v))
    V c00 = {}, c01 = {}, c10 = {}, c11 = {}, c20 = {}, c21 = {}, c30 = {}, c31 = {};
    V c40 = {}, c41 = {}, c50 = {}, c51 = {}, c60 = {}, c61 = {}, c70 = {}, c71 = {};
#define WP_ROW(r, X) \
    if (MR > r) {    \
        X(r)         \
    }
#define WP_LOAD(r) \
    c##r##0 = WP_LD(out + r * NR); \
    c##r##1 = WP_LD(out + r * NR + W);
#define WP_FMA(r) \
    { \
        const V av = a[r] * one; \
        c##r##0 += av * b0; \
        c##r##1 += av * b1; \
    }
#define WP_STORE(r) \
    WP_ST(out + r * NR, c##r##0); \
    WP_ST(out + r * NR + W, c##r##1);
#define WP_ALL_ROWS(X) WP_ROW(0, X) WP_ROW(1, X) WP_ROW(2, X) WP_ROW(3, X) WP_ROW(4, X) WP_ROW(5, X) WP_ROW(6, X) WP_ROW(7, X)
    WP_ALL_ROWS(WP_LOAD)
    for (int k = 0; k < kc; ++k) {
        const V b0 = WP_LD(bp + k * NR);
        const V b1 = WP_LD(bp + k * NR + W);
        const ACC* __restrict a = ap + k * MR;
        WP_ALL_ROWS(WP_FMA)
    }
    WP_ALL_ROWS(WP_STORE)
#undef WP_ALL_ROWS
#undef WP_STORE
#undef WP_FMA
#undef WP_LOAD
#undef WP_ROW
#undef WP_ST
#undef WP_LD
}
#else
template <typename ACC, int MR, int NR, int VB>
inline void micro_impl(const ACC* __restrict ap, const ACC* __restrict bp, int kc, ACC* __restrict out)
{
    ACC acc[MR][NR];
    for (int r = 0; r < MR; ++r)
        for (int j = 0; j < NR; ++j)
            acc[r][j] = out[r * NR + j];
    for (int k = 0; k < kc; ++k)
        for (int r = 0; r < MR; ++r)
            for (int j = 0; j < NR; ++j)
                acc[r][j] += ap[k * MR + r] * bp[k * NR + j];
    for (int r = 0; r < MR; ++r)
        for (int j = 0; j < NR; ++j)
            out[r * NR + j] = acc[r][j];
}
#endif

template <typename ACC> using micro_fn = void (*)(const ACC*, const ACC*, int, ACC*);

#define WP_GEMM_MICRO(name, ACC, MR, NR, VB, TARGET) \
    TARGET void name(const ACC* ap, const ACC* bp, int kc, ACC* out) { micro_impl<ACC, MR, NR, VB>(ap, bp, kc, out); }

#if (defined(__x86_64__) || defined(_M_X64)) && defined(__GNUC__)
#define WP_GEMM_X86 1
// AVX-512: 8 x 2 zmm accumulators; AVX2: 6 x 2 ymm; baseline: 4 x 2 xmm (SSE2 on x86-64, NEON on aarch64)
WP_GEMM_MICRO(
    micro_f32_avx512, float, 8, 32, 64, __attribute__((target("avx512f,avx512vl,avx2,fma,prefer-vector-width=512")))
)
WP_GEMM_MICRO(
    micro_f64_avx512, double, 8, 16, 64, __attribute__((target("avx512f,avx512vl,avx2,fma,prefer-vector-width=512")))
)
WP_GEMM_MICRO(micro_f32_avx2, float, 6, 16, 32, __attribute__((target("avx2,fma"))))
WP_GEMM_MICRO(micro_f64_avx2, double, 6, 8, 32, __attribute__((target("avx2,fma"))))
#endif
WP_GEMM_MICRO(micro_f32_base, float, 4, 8, 16, )
WP_GEMM_MICRO(micro_f64_base, double, 4, 4, 16, )

template <typename ACC, int MR, int NR>
void gemm_host_blocked(const wp_gemm_desc& d, const void* a, const void* b, void* c, void* bias, micro_fn<ACC> micro)
{
    const int M = d.m, N = d.n, K = d.k;
    const bool has_bias = d.epilogue == APIC_GEMM_EPILOGUE_BIAS || d.epilogue == APIC_GEMM_EPILOGUE_RELU_BIAS;
    const bool relu = d.epilogue == APIC_GEMM_EPILOGUE_RELU || d.epilogue == APIC_GEMM_EPILOGUE_RELU_BIAS;
    const ACC alpha = ACC(d.alpha), beta = ACC(d.beta);
    const uint32_t ab_size = apic_type_size(uint8_t(d.ab_dtype));
    const uint32_t c_size = apic_type_size(uint8_t(d.c_dtype));

    const int m_panels = (M + MR - 1) / MR;
    const int nc_max = N < NC ? N : NC;
    const int n_panels_max = (nc_max + NR - 1) / NR;
    std::vector<ACC> bp(size_t(KC) * n_panels_max * NR);
    std::vector<ACC> ap(size_t(KC) * m_panels * MR);
    // accumulators for an n-panel, laid out [m_panel][n_panel][MR][NR]
    std::vector<ACC> accbuf(size_t(m_panels) * n_panels_max * MR * NR);

    for (int z = 0; z < d.batch; ++z) {
        const char* A = static_cast<const char*>(a) + int64_t(z) * d.stride_a * ab_size;
        const char* B = static_cast<const char*>(b) + int64_t(z) * d.stride_b * ab_size;
        char* C = static_cast<char*>(c) + int64_t(z) * d.stride_c * c_size;

        for (int n0 = 0; n0 < N; n0 += NC) {
            const int nc = (N - n0 < NC) ? (N - n0) : NC;
            const int n_panels = (nc + NR - 1) / NR;
            std::fill(accbuf.begin(), accbuf.begin() + size_t(m_panels) * n_panels * MR * NR, ACC(0));
            for (int k0 = 0; k0 < K; k0 += KC) {
                const int kc = (K - k0 < KC) ? (K - k0) : KC;
                // pack op(b)[k0:k0+kc, n0:n0+nc] as [n_panel][k][NR], zero padded
                for (int jp = 0; jp < n_panels; ++jp)
                    for (int kk = 0; kk < kc; ++kk)
                        for (int jj = 0; jj < NR; ++jj) {
                            const int j = jp * NR + jj;
                            ACC v = ACC(0);
                            if (j < nc) {
                                const int64_t idx = d.trans_b ? int64_t(n0 + j) * d.ldb + (k0 + kk)
                                                              : int64_t(k0 + kk) * d.ldb + (n0 + j);
                                v = load_elem<ACC>(B, idx, d.ab_dtype);
                            }
                            bp[(size_t(jp) * kc + kk) * NR + jj] = v;
                        }
                // pack op(a)[:, k0:k0+kc] as [m_panel][k][MR], zero padded
                for (int ip = 0; ip < m_panels; ++ip)
                    for (int kk = 0; kk < kc; ++kk)
                        for (int ii = 0; ii < MR; ++ii) {
                            const int i = ip * MR + ii;
                            ACC v = ACC(0);
                            if (i < M) {
                                const int64_t idx
                                    = d.trans_a ? int64_t(k0 + kk) * d.lda + i : int64_t(i) * d.lda + (k0 + kk);
                                v = load_elem<ACC>(A, idx, d.ab_dtype);
                            }
                            ap[(size_t(ip) * kc + kk) * MR + ii] = v;
                        }
                for (int ip = 0; ip < m_panels; ++ip)
                    for (int jp = 0; jp < n_panels; ++jp)
                        micro(
                            ap.data() + size_t(ip) * kc * MR, bp.data() + size_t(jp) * kc * NR, kc,
                            accbuf.data() + (size_t(ip) * n_panels + jp) * MR * NR
                        );
            }
            // epilogue
            for (int i = 0; i < M; ++i) {
                const int ip = i / MR, ii = i % MR;
                for (int j = 0; j < nc; ++j) {
                    const int jp = j / NR, jj = j % NR;
                    ACC v = alpha * accbuf[((size_t(ip) * n_panels + jp) * MR + ii) * NR + jj];
                    if (has_bias)
                        v += load_elem<ACC>(bias, n0 + j, d.c_dtype);
                    if (relu)
                        v = v > ACC(0) ? v : ACC(0);
                    const int64_t ci = int64_t(i) * d.ldc + (n0 + j);
                    if (beta != ACC(0))
                        v += beta * load_elem<ACC>(C, ci, d.c_dtype);
                    store_elem<ACC>(C, ci, d.c_dtype, v);
                }
            }
        }
    }
}

int host_isa()
{
    static int isa = -1;
    if (isa < 0) {
        isa = 0;
        // WARP_GEMM_HOST_ISA caps the micro-kernel ISA (0 = baseline, 1 = AVX2, 2 = AVX-512), for testing
        const char* cap_env = std::getenv("WARP_GEMM_HOST_ISA");
        const int cap = cap_env ? std::atoi(cap_env) : 2;
#if defined(WP_GEMM_X86)
        __builtin_cpu_init();
        if (__builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512vl"))
            isa = 2;
        else if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma"))
            isa = 1;
#endif
        isa = std::min(isa, cap);
    }
    return isa;
}

// Skinny host GEMM (few rows, e.g. single-sample inference): no packing, each output row accumulates the rows of
// op(b) (or dot products over contiguous k when b is transposed) directly.
template <typename ACC> void gemm_host_skinny(const wp_gemm_desc& d, const void* a, const void* b, void* c, void* bias)
{
    const bool has_bias = d.epilogue == APIC_GEMM_EPILOGUE_BIAS || d.epilogue == APIC_GEMM_EPILOGUE_RELU_BIAS;
    const bool relu = d.epilogue == APIC_GEMM_EPILOGUE_RELU || d.epilogue == APIC_GEMM_EPILOGUE_RELU_BIAS;
    const ACC alpha = ACC(d.alpha), beta = ACC(d.beta);
    const uint32_t ab_size = apic_type_size(uint8_t(d.ab_dtype));
    const uint32_t c_size = apic_type_size(uint8_t(d.c_dtype));
    const bool native_b = (std::is_same<ACC, float>::value && d.ab_dtype == APIC_TYPE_FLOAT32)
        || (std::is_same<ACC, double>::value && d.ab_dtype == APIC_TYPE_FLOAT64);
    std::vector<ACC> row(size_t(d.n));
    for (int z = 0; z < d.batch; ++z) {
        const char* A = static_cast<const char*>(a) + int64_t(z) * d.stride_a * ab_size;
        const char* B = static_cast<const char*>(b) + int64_t(z) * d.stride_b * ab_size;
        char* C = static_cast<char*>(c) + int64_t(z) * d.stride_c * c_size;
        for (int i = 0; i < d.m; ++i) {
            std::fill(row.begin(), row.end(), ACC(0));
            for (int k = 0; k < d.k; ++k) {
                const ACC av
                    = load_elem<ACC>(A, d.trans_a ? int64_t(k) * d.lda + i : int64_t(i) * d.lda + k, d.ab_dtype);
                if (d.trans_b) {
                    for (int j = 0; j < d.n; ++j)
                        row[j] += av * load_elem<ACC>(B, int64_t(j) * d.ldb + k, d.ab_dtype);
                } else if (native_b) {
                    const ACC* __restrict br = reinterpret_cast<const ACC*>(B) + int64_t(k) * d.ldb;
                    ACC* __restrict rr = row.data();
                    for (int j = 0; j < d.n; ++j)
                        rr[j] += av * br[j];
                } else {
                    for (int j = 0; j < d.n; ++j)
                        row[j] += av * load_elem<ACC>(B, int64_t(k) * d.ldb + j, d.ab_dtype);
                }
            }
            for (int j = 0; j < d.n; ++j) {
                ACC v = alpha * row[j];
                if (has_bias)
                    v += load_elem<ACC>(bias, j, d.c_dtype);
                if (relu)
                    v = v > ACC(0) ? v : ACC(0);
                const int64_t ci = int64_t(i) * d.ldc + j;
                if (beta != ACC(0))
                    v += beta * load_elem<ACC>(C, ci, d.c_dtype);
                store_elem<ACC>(C, ci, d.c_dtype, v);
            }
        }
    }
}

template <typename ACC>
void gemm_builtin_typed(const wp_gemm_desc& d, const void* a, const void* b, void* c, void* bias)
{
    if (d.m <= 4)
        return gemm_host_skinny<ACC>(d, a, b, c, bias);
    const int isa = host_isa();
#if defined(WP_GEMM_X86)
    if (std::is_same<ACC, float>::value) {
        if (isa == 2)
            return gemm_host_blocked<float, 8, 32>(d, a, b, c, bias, (micro_fn<float>)micro_f32_avx512);
        if (isa == 1)
            return gemm_host_blocked<float, 6, 16>(d, a, b, c, bias, (micro_fn<float>)micro_f32_avx2);
        return gemm_host_blocked<float, 4, 8>(d, a, b, c, bias, (micro_fn<float>)micro_f32_base);
    } else {
        if (isa == 2)
            return gemm_host_blocked<double, 8, 16>(d, a, b, c, bias, (micro_fn<double>)micro_f64_avx512);
        if (isa == 1)
            return gemm_host_blocked<double, 6, 8>(d, a, b, c, bias, (micro_fn<double>)micro_f64_avx2);
        return gemm_host_blocked<double, 4, 4>(d, a, b, c, bias, (micro_fn<double>)micro_f64_base);
    }
#else
    (void)isa;
    if (std::is_same<ACC, float>::value)
        return gemm_host_blocked<float, 4, 8>(d, a, b, c, bias, (micro_fn<float>)micro_f32_base);
    return gemm_host_blocked<double, 4, 4>(d, a, b, c, bias, (micro_fn<double>)micro_f64_base);
#endif
}

// bias_grad[j] = sum over batches and k of b[k, j] (trans_b is false for BGRADA)
template <typename ACC> void host_colsum(const wp_gemm_desc& d, const void* b, void* bias)
{
    const uint32_t ab_size = apic_type_size(uint8_t(d.ab_dtype));
    std::vector<ACC> s(size_t(d.n), ACC(0));
    for (int z = 0; z < d.batch; ++z) {
        const char* B = static_cast<const char*>(b) + int64_t(z) * d.stride_b * ab_size;
        for (int kk = 0; kk < d.k; ++kk)
            for (int j = 0; j < d.n; ++j)
                s[j] += load_elem<ACC>(B, int64_t(kk) * d.ldb + j, d.ab_dtype);
    }
    for (int j = 0; j < d.n; ++j)
        store_elem<ACC>(bias, j, d.c_dtype, s[j]);
}

// Host GEMM: the host BLAS for float32/float64, the built-in blocked GEMM otherwise (or without a BLAS).
template <typename ACC> void gemm_host_typed(const wp_gemm_desc& d, const void* a, const void* b, void* c, void* bias)
{
    const bool has_bias = d.epilogue == APIC_GEMM_EPILOGUE_BIAS || d.epilogue == APIC_GEMM_EPILOGUE_RELU_BIAS;
    const bool relu = d.epilogue == APIC_GEMM_EPILOGUE_RELU || d.epilogue == APIC_GEMM_EPILOGUE_RELU_BIAS;
    bool done = false;
    if (d.ab_dtype == d.c_dtype && (d.ab_dtype == APIC_TYPE_FLOAT32 || d.ab_dtype == APIC_TYPE_FLOAT64)
        && g_gemm_host_use_blas && d.k > 0) {
        done = blas_gemm<ACC>(d, static_cast<const ACC*>(a), static_cast<const ACC*>(b), static_cast<ACC*>(c), d.beta);
        if (done && (has_bias || relu)) {
            // epilogue pass: c = relu(c + bias) (beta != 0 never combines with relu, see wp.gemm)
            const ACC* bv = static_cast<const ACC*>(bias);
            for (int z = 0; z < d.batch; ++z) {
                ACC* C = static_cast<ACC*>(c) + int64_t(z) * d.stride_c;
                for (int i = 0; i < d.m; ++i) {
                    ACC* row = C + int64_t(i) * d.ldc;
                    for (int j = 0; j < d.n; ++j) {
                        ACC v = row[j] + (has_bias ? bv[j] : ACC(0));
                        row[j] = relu && v < ACC(0) ? ACC(0) : v;
                    }
                }
            }
        }
    }
    if (!done)
        gemm_builtin_typed<ACC>(d, a, b, c, bias);
    if (d.epilogue == APIC_GEMM_EPILOGUE_BGRADA && bias)
        host_colsum<ACC>(d, b, bias);
}

bool desc_valid(const wp_gemm_desc& d)
{
    const bool ab_ok = d.ab_dtype == APIC_TYPE_FLOAT16 || d.ab_dtype == APIC_TYPE_BFLOAT16
        || d.ab_dtype == APIC_TYPE_FLOAT32 || d.ab_dtype == APIC_TYPE_FLOAT64;
    const bool c_ok = d.c_dtype == d.ab_dtype
        || (d.c_dtype == APIC_TYPE_FLOAT32 && (d.ab_dtype == APIC_TYPE_FLOAT16 || d.ab_dtype == APIC_TYPE_BFLOAT16));
    return ab_ok && c_ok && d.m >= 0 && d.n >= 0 && d.k >= 0 && d.batch >= 1 && d.epilogue >= 0
        && d.epilogue <= APIC_GEMM_EPILOGUE_BGRADA && d.lda >= 0 && d.ldb >= 0 && d.ldc >= 0 && d.stride_a >= 0
        && d.stride_b >= 0 && d.stride_c >= 0;
}

// Record ``d`` into the active APIC state. Returns false if a pointer could not be resolved.
bool record(APICState* state, const wp_gemm_desc& d, const void* a, const void* b, void* c, void* bias)
{
    const uint32_t ab = apic_type_size(uint8_t(d.ab_dtype));
    const uint32_t cs = apic_type_size(uint8_t(d.c_dtype));
    uint64_t a_bytes, b_bytes, c_bytes, bias_bytes;
    wp_gemm_spans(d, ab, cs, &a_bytes, &b_bytes, &c_bytes, &bias_bytes);
    APICGemmRecord rec = {};
    rec.header.op_type = APIC_OP_GEMM;
    rec.header.total_size = sizeof(rec);
    const bool host = state->is_cpu;
    auto resolve = [&](const void* p, uint64_t bytes) {
        return host ? apic_resolve_host_ptr(state, uint64_t(p), bytes)
                    : apic_resolve_live_ptr(state, uint64_t(p), bytes);
    };
    APICAddress aa = resolve(a, a_bytes);
    APICAddress ba = resolve(b, b_bytes);
    APICAddress ca = resolve(c, c_bytes);
    rec.a_region_id = aa.region_id;
    rec.a_offset = aa.offset;
    rec.b_region_id = ba.region_id;
    rec.b_offset = ba.offset;
    rec.c_region_id = ca.region_id;
    rec.c_offset = ca.offset;
    rec.bias_region_id = -1;
    if (bias && bias_bytes) {
        APICAddress bb = resolve(bias, bias_bytes);
        rec.bias_region_id = bb.region_id;
        rec.bias_offset = bb.offset;
    }
    rec.stride_a = d.stride_a;
    rec.stride_b = d.stride_b;
    rec.stride_c = d.stride_c;
    rec.alpha = d.alpha;
    rec.beta = d.beta;
    rec.m = d.m;
    rec.n = d.n;
    rec.k = d.k;
    rec.lda = d.lda;
    rec.ldb = d.ldb;
    rec.ldc = d.ldc;
    rec.batch = d.batch;
    rec.ab_dtype = uint8_t(d.ab_dtype);
    rec.c_dtype = uint8_t(d.c_dtype);
    rec.trans_a = uint8_t(d.trans_a != 0);
    rec.trans_b = uint8_t(d.trans_b != 0);
    rec.epilogue = uint8_t(d.epilogue);
    rec.tf32 = uint8_t(d.tf32 != 0);
    state->append_bytes(&rec, sizeof(rec));
    state->operation_count++;
    return true;
}

}  // namespace

bool wp_gemm_builtin_host(const wp_gemm_desc& d, const void* a, const void* b, void* c, void* bias)
{
    if (d.m == 0 || d.n == 0)
        return true;
    if (d.ab_dtype == APIC_TYPE_FLOAT64)
        gemm_host_typed<double>(d, a, b, c, bias);
    else
        gemm_host_typed<float>(d, a, b, c, bias);
    return true;
}

wp_gemm_desc wp_gemm_desc_from_record(const APICGemmRecord& r)
{
    wp_gemm_desc d = {};
    d.ab_dtype = r.ab_dtype;
    d.c_dtype = r.c_dtype;
    d.trans_a = r.trans_a;
    d.trans_b = r.trans_b;
    d.m = r.m;
    d.n = r.n;
    d.k = r.k;
    d.lda = r.lda;
    d.ldb = r.ldb;
    d.ldc = r.ldc;
    d.batch = r.batch;
    d.epilogue = r.epilogue;
    d.stride_a = r.stride_a;
    d.stride_b = r.stride_b;
    d.stride_c = r.stride_c;
    d.alpha = r.alpha;
    d.beta = r.beta;
    d.tf32 = r.tf32;
    d.autotune = 0;
    return d;
}

bool wp_gemm_record_valid(const APICGemmRecord& r) { return desc_valid(wp_gemm_desc_from_record(r)); }

extern "C" {

// Register CBLAS sgemm/dgemm entry points for host GEMMs (``ilp64``: 64-bit integer arguments). Passing null
// pointers reverts to the built-in host GEMM. ``use_blas`` = 0 forces the built-in host GEMM (benchmarks, tests).
WP_API void wp_gemm_set_host_blas(void* sgemm, void* dgemm, int ilp64, int use_blas)
{
    g_host_blas.sgemm = sgemm;
    g_host_blas.dgemm = dgemm;
    g_host_blas.ilp64 = ilp64 != 0;
    g_host_blas.tried_default = sgemm != nullptr;
    g_gemm_host_use_blas = use_blas != 0;
}

// Host GEMM. During a CPU capture the call is recorded and not executed.
WP_API bool wp_gemm_host(const wp_gemm_desc* d, void* a, void* b, void* c, void* bias)
{
    if (!d || !desc_valid(*d)) {
        wp::set_error_string("Warp GEMM error: invalid GEMM descriptor");
        return false;
    }
    if (APICState* state = wp_apic_get_recording_state())
        return record(state, *d, a, b, c, bias);
    return wp_gemm_builtin_host(*d, a, b, c, bias);
}

// Device GEMM on ``stream``. ``backend``: 0 = cuBLASLt if loaded, else built-in; 1 = cuBLASLt; 2 = built-in.
// During a CUDA API capture the call is also recorded (and still enqueued, so the driver captures it).
WP_API bool
wp_gemm_device(void* context, void* stream, const wp_gemm_desc* d, void* a, void* b, void* c, void* bias, int backend)
{
    if (!d || !desc_valid(*d)) {
        wp::set_error_string("Warp GEMM error: invalid GEMM descriptor");
        return false;
    }
#if WP_ENABLE_CUDA
    if (APICState* state = wp_apic_get_cuda_recording_state())
        record(state, *d, a, b, c, bias);
    if (backend == 2 || (backend == 0 && !wp_gemm_cublaslt_loaded()))
        return wp_gemm_builtin_device(context, stream, *d, a, b, c, bias);
    return wp_gemm_cublaslt(context, stream, *d, a, b, c, bias);
#else
    (void)context;
    (void)stream;
    (void)a;
    (void)b;
    (void)c;
    (void)bias;
    (void)backend;
    wp::set_error_string("Warp GEMM error: Warp was built without CUDA support");
    return false;
#endif
}

}  // extern "C"
