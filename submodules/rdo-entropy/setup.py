from pathlib import Path
from setuptools import setup, Extension
import torch
root = Path(__file__).resolve().parent
include = str(Path(torch.__file__).parent/'include')
setup(name='smolgs-rdo-entropy', version='0.1', packages=['smolgs_rdo_entropy'], ext_modules=[
    Extension('smolgs_rdo_entropy.rans', [str(root/'cpp_exts/rans/rans_interface.cpp')],
              include_dirs=[include, str(root/'cpp_exts/ryg_rans'), str(root/'cpp_exts/rans')],
              language='c++', extra_compile_args=['-O3', '-std=c++14']),
    Extension('smolgs_rdo_entropy._CXX', [str(root/'cpp_exts/ops/ops.cpp')],
              include_dirs=[include], language='c++', extra_compile_args=['-O3', '-std=c++14'])])
