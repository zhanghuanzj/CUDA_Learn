#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>   // 定义 uint32_t, int32_t 等
#include <type_traits>      // 👈 提供 std::is_same_v


template <typename value_t>
__device__ void
mma_m16n8k16_f32_accum(
    float &d1, float &d2, float &d3, float &d4,
    uint32_t const &a1, uint32_t const &a2,
    uint32_t const &a3, uint32_t const &a4,
    uint32_t const &b1, uint32_t const &b2,
    float const &c1, float const &c2,
    float const &c3, float const &c4
);

__global__ void test_mma_kernel(float* matrix_a, float* matrix_b, float* matrix_d);

// 封装 kernel launch 的函数（只能在 .cu 中定义！）
void launch_mma_test_kernel(float* d_result);

