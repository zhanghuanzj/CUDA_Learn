#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <type_traits>

template <typename value_t>
__device__ void
mma_m16n8k16_f32_accum(
    float &d1, float &d2, float &d3, float &d4,
    uint32_t const &a1, uint32_t const &a2,
    uint32_t const &a3, uint32_t const &a4,
    uint32_t const &b1, uint32_t const &b2,
    float const &c1, float const &c2,
    float const &c3, float const &c4
) {
    static_assert(std::is_same_v<value_t, half> ||
                      std::is_same_v<value_t, __nv_bfloat16>,
                  "value_t must be either half or __nv_bfloat16");

    if constexpr (std::is_same_v<value_t, __nv_bfloat16>) {
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                     "{%0,%1,%2,%3},"
                     "{%4,%5,%6,%7},"
                     "{%8,%9},"
                     "{%10,%11,%12,%13};"
                     : "=f"(d1), "=f"(d2), "=f"(d3), "=f"(d4)
                     : "r"(a1), "r"(a2), "r"(a3), "r"(a4),
                       "r"(b1), "r"(b2),
                       "f"(c1), "f"(c2), "f"(c3), "f"(c4));
    } else {
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                     "{%0,%1,%2,%3},"
                     "{%4,%5,%6,%7},"
                     "{%8,%9},"
                     "{%10,%11,%12,%13};"
                     : "=f"(d1), "=f"(d2), "=f"(d3), "=f"(d4)
                     : "r"(a1), "r"(a2), "r"(a3), "r"(a4),
                       "r"(b1), "r"(b2),
                       "f"(c1), "f"(c2), "f"(c3), "f"(c4));
    }
}

// 显式实例化（必须写出完整签名）
template __device__ void mma_m16n8k16_f32_accum<half>(
    float&, float&, float&, float&,
    const uint32_t&, const uint32_t&, const uint32_t&, const uint32_t&,
    const uint32_t&, const uint32_t&,
    const float&, const float&, const float&, const float&
);

template __device__ void mma_m16n8k16_f32_accum<__nv_bfloat16>(
    float&, float&, float&, float&,
    const uint32_t&, const uint32_t&, const uint32_t&, const uint32_t&,
    const uint32_t&, const uint32_t&,
    const float&, const float&, const float&, const float&
);

__device__ void cp_async_commit() { 
    asm volatile("cp.async.commit_group;"); 
}

template <int ngroups>
__device__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;" ::"n"(ngroups));
}

// Transfer: GMEM → SMEM using cp.async (16B per thread,  bytes warp-wide)
template <int size, typename T>
__device__ void cp_async(T *smem_to, const T *gmem_from) {
    static_assert(size == 16);
    uint32_t smem_ptr = __cvta_generic_to_shared(smem_to);
    asm volatile("cp.async.cg.shared.global [%0], [%1], %2;"
                 :
                 : "r"(smem_ptr), "l"(gmem_from), "n"(size));
}

/**
 * reinterpret_cast<uint4*>(GMEM[dst])[0] = reinterpret_cast<uint4*>(SMEM[src])[0];
*/