
# PTX kernel 测试 (.cu -> .o)
genrule(
    name = "ptx_kernel_o",
    srcs = ["ptx_kernel_test.cu", "ptx_functions.cuh"],
    outs = ["ptx_kernel_o.cu.o"],
    cmd = "nvcc -std=c++17  -Xcompiler -fPIC --expt-relaxed-constexpr -gencode=arch=compute_86,code=sm_86 -c $(location ptx_kernel_test.cu) -o $@",
)

test_deps = [
    "@com_google_googletest//:gtest",
    "@com_google_googletest//:gtest_main",
]

# 单测二进制
cc_test(
    name = "ptx_test",
    srcs = [
        "ptx_test.cc",
        ":ptx_kernel_o",  # 链接 ptx kernel 的 .o
    ],
    deps = [
        "@local_cuda//:cuda_runtime",
    ] + test_deps,
)

