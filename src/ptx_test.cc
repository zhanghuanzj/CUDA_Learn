#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <vector>

// 声明包装函数
void launch_test_cp_async_kernel(const float* d_input, float* d_output, int num_elements);
void launch_mma_test_kernel(float* matrix_a, float* matrix_b, float* matrix_d);

void print_matrix(float *matrix, int M, int N) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            std::cout << matrix[i * N + j] << ",";
        }
        std::cout << std::endl;
    }
}

TEST(MmaTest, BasicHalf) {
    constexpr int M = 16;
    constexpr int N = 8;
    constexpr int K = 16;

    float h_matrix_a[M*K];
    float* d_matrix_a;
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < K; ++j) {
            h_matrix_a[i * K + j] = i + j;
        }
    }
    print_matrix(h_matrix_a, M, K);
    cudaMalloc(&d_matrix_a, M * K * sizeof(float));
    cudaMemcpy(d_matrix_a, h_matrix_a, M * K * sizeof(float), cudaMemcpyHostToDevice);

    float h_matrix_b[M*N];
    float* d_matrix_b;
    for (size_t i = 0; i < M*N; ++i) {
        h_matrix_b[i] = 1.0f;
    }
    print_matrix(h_matrix_b, M, N);
    cudaMalloc(&d_matrix_b, M * N * sizeof(float));
    cudaMemcpy(d_matrix_b, h_matrix_b, M * N * sizeof(float), cudaMemcpyHostToDevice);

    float h_matrix_d[M*N];
    float* d_matrix_d;
    cudaMalloc(&d_matrix_d, M * N * sizeof(float));

    launch_mma_test_kernel(d_matrix_a, d_matrix_b,d_matrix_d);
    cudaDeviceSynchronize();
    cudaMemcpy(h_matrix_d, d_matrix_d, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // 填充
    for (int i = 0; i < M; ++i) {
        float val = 120.0f + i * 16.0f;
        for (int j = 0; j < N; ++j) {
            EXPECT_EQ(h_matrix_d[i * N + j], val);
            std::cout<<h_matrix_d[i * N + j]<<",";
        }
        std::cout<<std::endl;
    }

    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    
    cudaFree(d_matrix_b);
    cudaFree(d_matrix_a);
    cudaFree(d_matrix_d);
}


// ========== 单测 ==========
TEST(PtxKernelTest, BasicCopy) {
    // 跳过非 Ampere+ 架构
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    if (prop.major < 8) {
        GTEST_SKIP() << "cp.async requires SM 8.0+ (Ampere or newer)";
    }

    const int N = 256; // 必须是 4 的倍数（16字节对齐）
    std::vector<float> h_input(N);
    std::vector<float> h_output(N);

    // 初始化输入数据
    for (int i = 0; i < N; ++i) {
        h_input[i] = static_cast<float>(i + 1);
    }

    // 分配设备内存
    float *d_input, *d_output;
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, N * sizeof(float));

    // 拷贝输入到设备
    cudaMemcpy(d_input, h_input.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    // 启动 kernel
    launch_test_cp_async_kernel(d_input, d_output, N);

    // 同步并检查错误
    cudaDeviceSynchronize();
    ASSERT_EQ(cudaSuccess, cudaGetLastError());

    // 拷贝结果回主机
    cudaMemcpy(h_output.data(), d_output, N * sizeof(float), cudaMemcpyDeviceToHost);

    // 验证结果
    for (int i = 0; i < N; ++i) {
        EXPECT_FLOAT_EQ(h_input[i], h_output[i]) 
            << "Mismatch at index " << i;
    }

    // 清理
    cudaFree(d_input);
    cudaFree(d_output);
}