#include "ptx_functions.cuh"

__global__ void test_cp_async_kernel(const float* __restrict__ gmem_input, 
                                     float* __restrict__ gmem_output, 
                                     int num_elements) {
    extern __shared__ float smem[];
    
    int tid = threadIdx.x;
    int total_threads = blockDim.x;
    
    // 每次拷贝 16 字节 = 4 floats
    constexpr int COPY_FLOATS = 4;
    constexpr int COPY_BYTES = 16;
    
    // 计算每个线程负责的传输数据块数量
    int total_data_blocks = (num_elements + COPY_FLOATS - 1) / COPY_FLOATS;
    int blocks_per_thread = total_data_blocks / total_threads;
    
    for (int b = 0; b < blocks_per_thread; ++b) {
        int block_idx = tid * blocks_per_thread + b;
        int global_offset = block_idx * COPY_FLOATS;
        
        if (global_offset >= num_elements) break;
        
        // 共享内存目标地址（每个线程有自己的区域）
        float* smem_dst = &smem[tid * COPY_FLOATS];
        
        // 发起异步拷贝（16 字节）
        cp_async<COPY_BYTES>(&smem_dst[0], &gmem_input[global_offset]);
        cp_async_commit();
        
        // 等待拷贝完成
        cp_async_wait<0>();
        
        // 将 shared memory 数据写回全局内存供主机读取
        for (int i = 0; i < COPY_FLOATS; ++i) {
            gmem_output[global_offset + i] = smem_dst[i];
        }
    }
}

// C++ 包装函数，用于从 .cc 文件调用
void launch_test_cp_async_kernel(const float* d_input, float* d_output, int num_elements) {
    constexpr int BLOCK_SIZE = 32;
    size_t smem_size = num_elements * sizeof(float);
    test_cp_async_kernel<<<1, BLOCK_SIZE, smem_size>>>(d_input, d_output, num_elements);
}

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

