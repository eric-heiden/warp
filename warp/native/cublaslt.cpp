// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Runtime-loaded cuBLASLt GEMM plans for Warp.
//
// cuBLASLt is loaded with dlopen()/LoadLibrary() so that Warp does not gain a link-time dependency on it.
// A "plan" captures the matmul descriptor, matrix layouts and the heuristically selected algorithm for a
// fixed problem shape. Plans are created eagerly (outside of graph capture, ideally) and executing a plan is
// just a single cublasLtMatmul() launch on the given stream, which makes it safe for CUDA graph capture.
//
// All matrices are row-major from the caller's point of view; the wrapper maps this to column-major cuBLASLt
// calls by swapping the operands: C^T = op(B)^T * op(A)^T.

#include "warp.h"
#include "error.h"

#if WP_ENABLE_CUDA

#include "cuda_util.h"

#include <cublasLt.h>
#include <cuda_runtime.h>

#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace {

#define LT_FUNCS(X) \
    X(cublasLtCreate) \
    X(cublasLtDestroy) \
    X(cublasLtMatmulDescCreate) \
    X(cublasLtMatmulDescDestroy) \
    X(cublasLtMatmulDescSetAttribute) \
    X(cublasLtMatrixLayoutCreate) \
    X(cublasLtMatrixLayoutDestroy) \
    X(cublasLtMatrixLayoutSetAttribute) \
    X(cublasLtMatmulPreferenceCreate) \
    X(cublasLtMatmulPreferenceDestroy) \
    X(cublasLtMatmulPreferenceSetAttribute) \
    X(cublasLtMatmulAlgoGetHeuristic) \
    X(cublasLtMatmul) \
    X(cublasLtGetVersion)

#define DECL(name) decltype(&name) pfn_##name = nullptr;
LT_FUNCS(DECL)
#undef DECL

void* g_lib = nullptr;
std::mutex g_mutex;
constexpr size_t WORKSPACE_BYTES = 32u << 20;

struct DeviceState {
    cublasLtHandle_t handle = nullptr;
    void* workspace = nullptr;
};
std::map<void*, DeviceState> g_devices;

struct Plan {
    void* context = nullptr;
    cublasLtMatmulDesc_t desc = nullptr;
    cublasLtMatrixLayout_t la = nullptr, lb = nullptr, lc = nullptr, ld = nullptr;
    cublasLtMatmulAlgo_t algo;
    cublasLtHandle_t handle = nullptr;
    void* workspace = nullptr;
    size_t workspace_size = 0;
    bool has_bias = false;
    bool has_aux = false;
    bool has_bgrad = false;
    size_t ws_needed = 0;
};

bool lt_ok(cublasStatus_t s, const char* what)
{
    if (s == CUBLAS_STATUS_SUCCESS)
        return true;
    wp::set_error_string("cuBLASLt error %d in %s", int(s), what);
    return false;
}

bool load_library(const char* path)
{
    if (g_lib)
        return true;
#if defined(_WIN32)
    g_lib = (void*)LoadLibraryA(path);
#else
    g_lib = dlopen(path, RTLD_NOW | RTLD_LOCAL);
#endif
    if (!g_lib)
        return false;
    bool ok = true;
#if defined(_WIN32)
#define LOAD(name) \
    pfn_##name = (decltype(&name))GetProcAddress((HMODULE)g_lib, #name); \
    ok = ok && pfn_##name;
#else
#define LOAD(name) \
    pfn_##name = (decltype(&name))dlsym(g_lib, #name); \
    ok = ok && pfn_##name;
#endif
    LT_FUNCS(LOAD)
#undef LOAD
    if (!ok) {
#if !defined(_WIN32)
        dlclose(g_lib);
#endif
        g_lib = nullptr;
    }
    return ok;
}

DeviceState* get_device_state(void* context)
{
    std::lock_guard<std::mutex> lock(g_mutex);
    DeviceState& st = g_devices[context];
    if (!st.handle) {
        if (!lt_ok(pfn_cublasLtCreate(&st.handle), "cublasLtCreate"))
            return nullptr;
    }
    if (!st.workspace) {
        st.workspace = wp_alloc_device(context, WORKSPACE_BYTES, "cublaslt_workspace");
        if (!st.workspace)
            return nullptr;
    }
    return &st;
}

}  // namespace

extern "C" {

// Try to load cuBLASLt from ``path``. Returns the cuBLASLt version (e.g. 120902) or 0 on failure.
WP_API int wp_cublaslt_init(const char* path)
{
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!load_library(path))
        return 0;
    return int(pfn_cublasLtGetVersion());
}

// Create a plan for C[M,N] = alpha * op(A)[M,K] @ op(B)[K,N] + beta * C, all row-major.
//   a_type/b_type/c_type: cudaDataType_t, compute: cublasComputeType_t, scale_type: cudaDataType_t
//   epilogue: cublasLtEpilogue_t (raw value), bias_type: cudaDataType_t of the bias vector (or aux for dgrad)
//   aux_ld / aux_type: leading dimension (row-major, elements per row) of the epilogue auxiliary buffer
WP_API void* wp_cublaslt_plan_create(
    void* context,
    int a_type, int b_type, int c_type, int compute, int scale_type,
    int trans_a, int trans_b, int M, int N, int K,
    int lda, int ldb, int ldc,
    int batch, long long stride_a, long long stride_b, long long stride_c,
    int epilogue, int bias_type, int aux_ld, int aux_type
)
{
    if (!g_lib) {
        wp::set_error_string("cuBLASLt is not loaded");
        return nullptr;
    }
    ContextGuard guard(context);
    DeviceState* dev = get_device_state(context);
    if (!dev)
        return nullptr;

    std::unique_ptr<Plan> p(new Plan());
    p->context = context;
    p->handle = dev->handle;
    p->workspace = dev->workspace;
    p->workspace_size = WORKSPACE_BYTES;

    cublasOperation_t opa = trans_b ? CUBLAS_OP_T : CUBLAS_OP_N;  // Lt "A" is our B
    cublasOperation_t opb = trans_a ? CUBLAS_OP_T : CUBLAS_OP_N;  // Lt "B" is our A

    bool ok = lt_ok(pfn_cublasLtMatmulDescCreate(&p->desc, (cublasComputeType_t)compute, (cudaDataType_t)scale_type), "MatmulDescCreate");
    ok = ok && lt_ok(pfn_cublasLtMatmulDescSetAttribute(p->desc, CUBLASLT_MATMUL_DESC_TRANSA, &opa, sizeof(opa)), "TRANSA");
    ok = ok && lt_ok(pfn_cublasLtMatmulDescSetAttribute(p->desc, CUBLASLT_MATMUL_DESC_TRANSB, &opb, sizeof(opb)), "TRANSB");
    if (!ok)
        return nullptr;

    if (epilogue != 0) {
        cublasLtEpilogue_t ep = (cublasLtEpilogue_t)epilogue;
        ok = lt_ok(pfn_cublasLtMatmulDescSetAttribute(p->desc, CUBLASLT_MATMUL_DESC_EPILOGUE, &ep, sizeof(ep)), "EPILOGUE");
        // Epilogue enum bit 2 (value 4) means bias, bit 3 (8) relu/gelu... query by known values below.
        switch (ep) {
        case CUBLASLT_EPILOGUE_BIAS:
        case CUBLASLT_EPILOGUE_RELU_BIAS:
        case CUBLASLT_EPILOGUE_GELU_BIAS:
        case CUBLASLT_EPILOGUE_RELU_AUX_BIAS:
        case CUBLASLT_EPILOGUE_GELU_AUX_BIAS:
            p->has_bias = true;
            break;
        case CUBLASLT_EPILOGUE_DRELU_BGRAD:
        case CUBLASLT_EPILOGUE_DGELU_BGRAD:
            p->has_bgrad = true;
            break;
        default:
            break;
        }
        switch (ep) {
        case CUBLASLT_EPILOGUE_RELU_AUX:
        case CUBLASLT_EPILOGUE_RELU_AUX_BIAS:
        case CUBLASLT_EPILOGUE_GELU_AUX:
        case CUBLASLT_EPILOGUE_GELU_AUX_BIAS:
        case CUBLASLT_EPILOGUE_DRELU:
        case CUBLASLT_EPILOGUE_DRELU_BGRAD:
        case CUBLASLT_EPILOGUE_DGELU:
        case CUBLASLT_EPILOGUE_DGELU_BGRAD:
            p->has_aux = true;
            break;
        default:
            break;
        }
        if (ok && (p->has_bias || p->has_bgrad)) {
            cudaDataType_t bt = (cudaDataType_t)bias_type;
            ok = lt_ok(pfn_cublasLtMatmulDescSetAttribute(p->desc, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &bt, sizeof(bt)), "BIAS_DATA_TYPE");
        }
        if (ok && p->has_aux) {
            int64_t ald = aux_ld;
            ok = lt_ok(pfn_cublasLtMatmulDescSetAttribute(p->desc, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD, &ald, sizeof(ald)), "AUX_LD");
        }
        if (!ok)
            return nullptr;
    }

    // layouts (column-major view of our row-major buffers)
    auto make_layout = [&](cublasLtMatrixLayout_t* l, int type, uint64_t rows, uint64_t cols, int64_t ld, int64_t stride) {
        if (!lt_ok(pfn_cublasLtMatrixLayoutCreate(l, (cudaDataType_t)type, rows, cols, ld), "LayoutCreate"))
            return false;
        if (batch > 1) {
            int32_t b = batch;
            if (!lt_ok(pfn_cublasLtMatrixLayoutSetAttribute(*l, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &b, sizeof(b)), "BATCH_COUNT"))
                return false;
            if (!lt_ok(pfn_cublasLtMatrixLayoutSetAttribute(*l, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride, sizeof(stride)), "STRIDE"))
                return false;
        }
        return true;
    };
    // Lt A = our B: shape (N x K) if !trans_b else (K x N) in column-major
    ok = make_layout(&p->la, b_type, trans_b ? K : N, trans_b ? N : K, ldb, stride_b);
    // Lt B = our A: (K x M) if !trans_a else (M x K)
    ok = ok && make_layout(&p->lb, a_type, trans_a ? M : K, trans_a ? K : M, lda, stride_a);
    ok = ok && make_layout(&p->lc, c_type, N, M, ldc, stride_c);
    ok = ok && make_layout(&p->ld, c_type, N, M, ldc, stride_c);
    if (!ok)
        return nullptr;

    cublasLtMatmulPreference_t pref = nullptr;
    if (!lt_ok(pfn_cublasLtMatmulPreferenceCreate(&pref), "PreferenceCreate"))
        return nullptr;
    size_t ws = WORKSPACE_BYTES;
    pfn_cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws));
    cublasLtMatmulHeuristicResult_t res[1];
    int found = 0;
    cublasStatus_t st = pfn_cublasLtMatmulAlgoGetHeuristic(dev->handle, p->desc, p->la, p->lb, p->lc, p->ld, pref, 1, res, &found);
    pfn_cublasLtMatmulPreferenceDestroy(pref);
    if (!lt_ok(st, "AlgoGetHeuristic"))
        return nullptr;
    if (found < 1) {
        wp::set_error_string("cuBLASLt found no algorithm for M=%d N=%d K=%d", M, N, K);
        return nullptr;
    }
    p->algo = res[0].algo;
    p->ws_needed = res[0].workspaceSize;
    return p.release();
}

// bias: bias vector for *_BIAS epilogues (length N); for *_BGRAD epilogues this is the output bias gradient.
// aux: epilogue auxiliary buffer (activation input / mask), or null.
WP_API bool wp_cublaslt_plan_run(
    void* plan, void* stream, const void* A, const void* B, void* C, const void* bias, void* aux, float alpha, float beta
)
{
    Plan* p = static_cast<Plan*>(plan);
    ContextGuard guard(p->context);
    if (p->has_bias || p->has_bgrad) {
        const void* b = bias;
        if (!lt_ok(pfn_cublasLtMatmulDescSetAttribute(p->desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &b, sizeof(b)), "BIAS_POINTER"))
            return false;
    }
    if (p->has_aux) {
        void* a = aux;
        if (!lt_ok(pfn_cublasLtMatmulDescSetAttribute(p->desc, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER, &a, sizeof(a)), "AUX_POINTER"))
            return false;
    }
    // Lt A = our B, Lt B = our A
    return lt_ok(
        pfn_cublasLtMatmul(
            p->handle, p->desc, &alpha, B, p->la, A, p->lb, &beta, C, p->lc, C, p->ld, &p->algo, p->workspace,
            p->workspace_size, (cudaStream_t)stream
        ),
        "cublasLtMatmul"
    );
}

WP_API void wp_cublaslt_plan_destroy(void* plan)
{
    Plan* p = static_cast<Plan*>(plan);
    if (!p)
        return;
    if (p->desc) pfn_cublasLtMatmulDescDestroy(p->desc);
    if (p->la) pfn_cublasLtMatrixLayoutDestroy(p->la);
    if (p->lb) pfn_cublasLtMatrixLayoutDestroy(p->lb);
    if (p->lc) pfn_cublasLtMatrixLayoutDestroy(p->lc);
    if (p->ld) pfn_cublasLtMatrixLayoutDestroy(p->ld);
    delete p;
}

}  // extern "C"

#else  // !WP_ENABLE_CUDA

extern "C" {
WP_API int wp_cublaslt_init(const char*) { return 0; }
WP_API void* wp_cublaslt_plan_create(void*, int, int, int, int, int, int, int, int, int, int, int, int, int, int, long long, long long, long long, int, int, int, int) { return nullptr; }
WP_API bool wp_cublaslt_plan_run(void*, void*, const void*, const void*, void*, const void*, void*, float, float) { return false; }
WP_API void wp_cublaslt_plan_destroy(void*) {}
}

#endif
