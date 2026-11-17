// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Semantic GEMM descriptor shared by wp.gemm(), its API-capture record and
// the built-in host/device GEMM implementations.
//
//   c = epilogue(alpha * op(a) @ op(b) + bias) + beta * c
//
// All matrices are row-major with contiguous rows; leading dimensions and
// batch strides are in elements. Element types use the APICType values
// (APIC_TYPE_FLOAT16/BFLOAT16/FLOAT32/FLOAT64) and epilogues the
// APICGemmEpilogue values, so a descriptor maps 1:1 to an APICGemmRecord.

#pragma once

#include <stdint.h>

struct wp_gemm_desc {
    int32_t ab_dtype;  // APICType
    int32_t c_dtype;  // APICType
    int32_t trans_a;
    int32_t trans_b;
    int32_t m;
    int32_t n;
    int32_t k;
    int32_t lda;
    int32_t ldb;
    int32_t ldc;
    int32_t batch;
    int32_t epilogue;  // APICGemmEpilogue
    int64_t stride_a;
    int64_t stride_b;
    int64_t stride_c;
    double alpha;
    double beta;
    int32_t tf32;
    int32_t autotune;  // cuBLASLt heuristic candidates to time on the first uncaptured call
};

// Number of bytes spanned by a row-major (batched) matrix with ``rows`` x ``cols`` elements.
inline uint64_t wp_gemm_span_bytes(int rows, int cols, int ld, int batch, int64_t stride, uint32_t elem_size)
{
    if (rows <= 0 || cols <= 0 || batch <= 0)
        return 0;
    uint64_t last = uint64_t(batch - 1) * uint64_t(stride) + uint64_t(rows - 1) * uint64_t(ld) + uint64_t(cols);
    return last * elem_size;
}

// Byte spans of the a, b, c and bias buffers a GEMM accesses (bias is 0 when the epilogue does not use it).
inline void wp_gemm_spans(
    const wp_gemm_desc& d, uint32_t ab_size, uint32_t c_size, uint64_t* a, uint64_t* b, uint64_t* c, uint64_t* bias
)
{
    *a = wp_gemm_span_bytes(d.trans_a ? d.k : d.m, d.trans_a ? d.m : d.k, d.lda, d.batch, d.stride_a, ab_size);
    *b = wp_gemm_span_bytes(d.trans_b ? d.n : d.k, d.trans_b ? d.k : d.n, d.ldb, d.batch, d.stride_b, ab_size);
    *c = wp_gemm_span_bytes(d.m, d.n, d.ldc, d.batch, d.stride_c, c_size);
    *bias = (d.epilogue == 1 || d.epilogue == 3 || d.epilogue == 4) ? uint64_t(d.n > 0 ? d.n : 0) * c_size : 0;
}

// Built-in implementations (no cuBLASLt). Return false and set the error string on failure.
bool wp_gemm_builtin_host(const wp_gemm_desc& d, const void* a, const void* b, void* c, void* bias);
bool wp_gemm_builtin_device(
    void* context, void* stream, const wp_gemm_desc& d, const void* a, const void* b, void* c, void* bias
);

// cuBLASLt implementation (cublaslt.cpp). Returns false if cuBLASLt is not loaded or the call fails.
bool wp_gemm_cublaslt(
    void* context, void* stream, const wp_gemm_desc& d, const void* a, const void* b, void* c, void* bias
);
// Create the cuBLASLt handle and workspace for ``context`` (must not run during stream capture).
bool wp_gemm_cublaslt_prepare(void* context);
bool wp_gemm_cublaslt_loaded();
extern "C" int wp_cublaslt_init_default();

// APIC record conversion (gemm.cpp).
struct APICGemmRecord;
wp_gemm_desc wp_gemm_desc_from_record(const APICGemmRecord& r);
bool wp_gemm_record_valid(const APICGemmRecord& r);
