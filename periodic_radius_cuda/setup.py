from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# Get the path to the PyTorch libraries
torch_lib_path = os.path.join(os.path.dirname(os.__file__), 'site-packages/torch/lib')

setup(
    name='custom_radius_cuda',
    ext_modules=[
        CUDAExtension(
            'custom_radius_cuda',
            sources=['periodic_radius_cuda.cu'],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3', '-arch=sm_60'],
            },
            extra_link_args=[f'-Wl,-rpath,{torch_lib_path}']  # Add rpath
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)