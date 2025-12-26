# WORKSPACE
workspace(name = "CUDA_Learn")

# 引入 Google Test
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
  name = "com_google_googletest",
  urls = ["https://github.com/google/googletest/archive/f8d7d77c06936315286eb55f8de22cd23c188571.zip"],
  strip_prefix = "googletest-f8d7d77c06936315286eb55f8de22cd23c188571",
)

# WORKSPACE
new_local_repository(
    name = "local_cuda",
    path = "/usr/local/cuda",
    build_file_content = """
cc_library(
    name = "cuda_runtime",
    hdrs = glob([
        "include/**/*.h",
        "include/**/*.hpp",
        "include/**/*.inc",
        "include/nv/**/*", 
        "include/thrust/**/*", 
        "include/crt/**/*",      # ← 关键：包含 crt 目录
    ]),
    includes = ["include"],
    linkopts = ["-L/usr/local/cuda/lib64", "-lcudart"],
    visibility = ["//visibility:public"],
)
""",
)

