# 主 MMA 库 (.cu -> .o)
genrule(
    name = "mma_cu_o",
    srcs = ["mma_m16n8k16.cu", "mma_m16n8k16.h"],
    outs = ["mma_m16n8k16.cu.o"],
    cmd = "nvcc -std=c++17  -Xcompiler -fPIC --expt-relaxed-constexpr -gencode=arch=compute_86,code=sm_86 -c $(location mma_m16n8k16.cu) -o $@",
)

cc_library(
    name = "mma_lib",
    srcs = [":mma_cu_o"],
    hdrs = ["mma_m16n8k16.h"],
    alwayslink = True,
)

test_deps = [
    "@com_google_googletest//:gtest",
    "@com_google_googletest//:gtest_main",
]


# 单测二进制
cc_test(
    name = "mma_test",
    srcs = [
        "mma_test.cc",
        ":mma_cu_o",  # 链接 kernel 的 .o
    ],
    deps = [
        ":mma_lib",
        "@local_cuda//:cuda_runtime",
    ] + test_deps,
)
