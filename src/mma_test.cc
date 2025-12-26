#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include "mma_m16n8k16.h"

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
