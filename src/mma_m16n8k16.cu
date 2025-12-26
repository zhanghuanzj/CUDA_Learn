#include "mma_m16n8k16.h"
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


__global__ void test_mma_kernel(float* matrix_a, float* matrix_b, float* matrix_d) {

    int tidx = threadIdx.x;
    int row_a = tidx / 4;
    int col_a = (tidx % 4) * 2;
    __half a_vals[8] = {
        __float2half(matrix_a[row_a * 16 + col_a]), __float2half(matrix_a[row_a * 16 + col_a + 1]),
        __float2half(matrix_a[(row_a + 8)*16 + col_a]), __float2half(matrix_a[(row_a + 8)*16 + col_a + 1]),
        __float2half(matrix_a[row_a * 16 + col_a + 8]), __float2half(matrix_a[row_a * 16 + col_a + 9]),
        __float2half(matrix_a[(row_a + 8)*16 + col_a + 8]), __float2half(matrix_a[(row_a + 8)*16 + col_a + 9])
    };

    int row_b = (tidx % 4) * 2;
    int col_b = tidx / 4;
    __half b_vals[4] = {
        __float2half(matrix_b[row_b*8 + col_b]), __float2half(matrix_b[(row_b+1)*8 + col_b]),
        __float2half(matrix_b[(row_b+8)*8 + col_b]), __float2half(matrix_b[(row_b+9)*8 + col_b])
    };

    uint32_t a1 = reinterpret_cast<uint32_t&>(a_vals[0]);
    uint32_t a2 = reinterpret_cast<uint32_t&>(a_vals[2]);
    uint32_t a3 = reinterpret_cast<uint32_t&>(a_vals[4]);
    uint32_t a4 = reinterpret_cast<uint32_t&>(a_vals[6]);

    uint32_t b1 = reinterpret_cast<uint32_t&>(b_vals[0]);
    uint32_t b2 = reinterpret_cast<uint32_t&>(b_vals[2]);

    float c1 = 0.0f, c2 = 0.0f, c3 = 0.0f, c4 = 0.0f;
    float d1, d2, d3, d4;

    mma_m16n8k16_f32_accum<half>(
        d1, d2, d3, d4,
        a1, a2, a3, a4,
        b1, b2,
        c1, c2, c3, c4
    );

    int row_d = tidx / 4;
    int col_d = (tidx % 4) * 2;

    matrix_d[row_d*8 + col_d] = d1;
    matrix_d[row_d*8 + col_d + 1] = d2;
    matrix_d[(row_d + 8)*8 + col_d] = d3;
    matrix_d[(row_d + 8)*8 + col_d + 1] = d4;
}

// 封装 kernel launch 的函数（只能在 .cu 中定义！）
void launch_mma_test_kernel(float* matrix_a, float* matrix_b, float* matrix_d) {
    test_mma_kernel<<<1, 32>>>(matrix_a, matrix_b, matrix_d);
    cudaDeviceSynchronize();  // 等待 kernel 完成
}

