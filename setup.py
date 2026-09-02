import os
import sys
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

# Path to your CPLEX Studio installation
CPLEX_DIR = "/Users/vojtech/Applications/CPLEX_Studio222"

# IBM ILOG CPLEX include directories
include_dirs = [
    os.path.join(CPLEX_DIR, "cplex", "include"),
    os.path.join(CPLEX_DIR, "cpoptimizer", "include"),
    os.path.join(CPLEX_DIR, "concert", "include"),
]

# Path to libraries. We need to detect architecture since M-series macs might have 'arm64_osx' or run under 'x86-64_osx'
cplex_lib_base = os.path.join(CPLEX_DIR, "cplex", "lib")
cpo_lib_base = os.path.join(CPLEX_DIR, "cpoptimizer", "lib")
concert_lib_base = os.path.join(CPLEX_DIR, "concert", "lib")

arch_dir = "x86-64_osx"
if os.path.exists(os.path.join(cplex_lib_base, "arm64_osx")):
    arch_dir = "arm64_osx"

# Usually IBM uses 'static_pic' or 'static_pic' for macOS libraries
if os.path.exists(os.path.join(cplex_lib_base, arch_dir, "static_pic")):
    format_dir = "static_pic"
elif os.path.exists(os.path.join(cplex_lib_base, arch_dir, "static_pic")): # Just in case it's named identically but checking logic
    format_dir = "static_pic"
else:
    format_dir = "static_pic" # default guess

cplex_lib = os.path.join(cplex_lib_base, arch_dir, format_dir)
cpo_lib = os.path.join(cpo_lib_base, arch_dir, format_dir)
concert_lib = os.path.join(concert_lib_base, arch_dir, format_dir)

library_dirs = [cplex_lib, cpo_lib, concert_lib]
libraries = ["cp", "ilocplex", "concert", "m", "pthread"]

class get_pybind_include(object):
    """Helper class to determine the pybind11 include path.
    The purpose of this class is to postpone importing pybind11
    until it is actually installed, so that the ``get_include()``
    method can be invoked."""
    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        import pybind11
        return pybind11.get_include(self.user)

ext_modules = [
    Extension(
        "cp_lns_core",
        ["cp_lns_core.cpp"],
        include_dirs=[get_pybind_include(), get_pybind_include(user=True)] + include_dirs,
        library_dirs=library_dirs,
        libraries=libraries,
        language="c++",
        extra_compile_args=["-std=c++17", "-O3", "-fPIC", "-Wall"],
        extra_link_args=["-framework", "CoreFoundation", "-framework", "IOKit"]
    ),
]

setup(
    name="cp_lns_core",
    version="1.0.0",
    description="CP Optimizer LNS Backend with Pybind11",
    ext_modules=ext_modules,
    install_requires=["pybind11>=2.5.0"],
)
