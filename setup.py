from setuptools import setup, find_packages
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os

src_dir = os.path.join(os.path.dirname(__file__), "src")
csrc_dir = os.path.join(os.path.dirname(__file__), "csrc")

sources = [
    os.path.join(csrc_dir, "bindings.cpp"),
    os.path.join(src_dir, "dense_matvec.cu"),
    os.path.join(src_dir, "silu_threshold.cu"),
    os.path.join(src_dir, "sparse_matvec.cu"),
    os.path.join(src_dir, "dense_ffn.cu"),
    os.path.join(src_dir, "sparse_ffn.cu"),
    os.path.join(src_dir, "sparse_ffn_batched.cu"),
]

nvcc_flags = [
    "-O3",
    "--use_fast_math",
    "-std=c++17",
    "-gencode=arch=compute_80,code=sm_80",
    "-gencode=arch=compute_86,code=sm_86",
    "-gencode=arch=compute_89,code=sm_89",
    "-gencode=arch=compute_90,code=sm_90",
]

setup(
    name="sparse_ffn",
    version="0.1.0",
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name="sparse_ffn._C",
            sources=sources,
            include_dirs=[src_dir, csrc_dir],
            extra_compile_args={
                "cxx": ["-std=c++17", "-O3"],
                "nvcc": nvcc_flags,
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
