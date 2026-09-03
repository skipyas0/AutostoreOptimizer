import os
import sys
import subprocess
from setuptools import setup, Extension, Command
from setuptools.command.build_ext import build_ext

# Determine target architecture from CLI arguments, environment, or platform
if "ubuntu" in sys.argv:
    target_arch = "ubuntu"
elif "mac" in sys.argv:
    target_arch = "mac"
elif "--arch=ubuntu" in sys.argv or "--target=ubuntu" in sys.argv:
    target_arch = "ubuntu"
elif "--arch=mac" in sys.argv or "--target=mac" in sys.argv:
    target_arch = "mac"
elif "CPLEX_ARCH" in os.environ:
    target_arch = os.environ["CPLEX_ARCH"].lower()
else:
    target_arch = "mac" if sys.platform == "darwin" else "ubuntu"


# ---------------------------------------------------------------------------
# Architecture-specific CPLEX configuration
# ---------------------------------------------------------------------------
if target_arch == "ubuntu":
    CPLEX_DIR = os.environ.get("CPLEX_DIR", "/home/kloudvoj/ibm/ILOG/CPLEX_Studio222")
    arch_dir = "x86-64_linux"
    format_dir = "static_pic"
    extra_compile_args = ["-std=c++17", "-O3", "-fPIC", "-Wall", "-DIL_STD"]
    extra_link_args = ["-pthread", "-ldl"]
    libraries = ["cp", "ilocplex", "concert", "m", "pthread", "dl"]
else:  # mac
    CPLEX_DIR = os.environ.get("CPLEX_DIR", "/Users/vojtech/Applications/CPLEX_Studio222")
    cplex_lib_base = os.path.join(CPLEX_DIR, "cplex", "lib")
    arch_dir = "arm64_osx" if os.path.exists(os.path.join(cplex_lib_base, "arm64_osx")) else "x86-64_osx"
    format_dir = "static_pic"
    extra_compile_args = ["-std=c++17", "-O3", "-fPIC", "-Wall", "-DIL_STD"]
    extra_link_args = ["-framework", "CoreFoundation", "-framework", "IOKit"]
    libraries = ["cp", "ilocplex", "concert", "m", "pthread"]

include_dirs = [
    os.path.join(CPLEX_DIR, "cplex", "include"),
    os.path.join(CPLEX_DIR, "cpoptimizer", "include"),
    os.path.join(CPLEX_DIR, "concert", "include"),
]

cplex_lib_base = os.path.join(CPLEX_DIR, "cplex", "lib")
cpo_lib_base = os.path.join(CPLEX_DIR, "cpoptimizer", "lib")
concert_lib_base = os.path.join(CPLEX_DIR, "concert", "lib")


def get_lib_dir(base, arch, fmt):
    p = os.path.join(base, arch, fmt)
    if os.path.exists(p):
        return p
    p_arch = os.path.join(base, arch)
    if os.path.exists(p_arch):
        return p_arch
    return p


library_dirs = [
    get_lib_dir(cplex_lib_base, arch_dir, format_dir),
    get_lib_dir(cpo_lib_base, arch_dir, format_dir),
    get_lib_dir(concert_lib_base, arch_dir, format_dir),
]


# ---------------------------------------------------------------------------
# Helpers for Distrobox detection and execution
# ---------------------------------------------------------------------------
def is_running_in_container():
    if "--direct" in sys.argv:
        return True
    if os.environ.get("DISTROBOX_ENTERED") == "1":
        return True
    if os.path.exists("/run/.containerenv") or os.path.exists("/.dockerenv"):
        return True
    if os.path.exists("/etc/os-release"):
        try:
            with open("/etc/os-release", "r") as f:
                content = f.read()
                # If on Ubuntu and not on NixOS host
                if "ID=ubuntu" in content and not os.path.exists("/etc/NIXOS"):
                    return True
        except Exception:
            pass
    return False


def get_distrobox_container(requested=None):
    if requested:
        return requested
    if "DISTROBOX_CONTAINER" in os.environ:
        return os.environ["DISTROBOX_CONTAINER"]

    try:
        res = subprocess.run(
            ["distrobox", "list", "--no-color"],
            capture_output=True,
            text=True,
            check=False,
        )
        if res.returncode == 0:
            lines = res.stdout.strip().splitlines()
            containers = []
            for line in lines[1:]:
                parts = [p.strip() for p in line.split("|")]
                if len(parts) >= 2:
                    c_name = parts[1]
                    c_image = parts[3] if len(parts) >= 4 else ""
                    containers.append((c_name, c_image))

            # Prioritize ubuntu 22.04
            for c_name, c_image in containers:
                if "22.04" in c_name or "22.04" in c_image or "22-04" in c_name:
                    return c_name
            for c_name, c_image in containers:
                if "ubuntu" in c_name.lower() or "ubuntu" in c_image.lower():
                    return c_name
            if containers:
                return containers[0][0]
    except Exception:
        pass

    return "ubuntu"


# ---------------------------------------------------------------------------
# Custom Setuptools Commands: 'mac' and 'ubuntu'
# ---------------------------------------------------------------------------
class MacCommand(Command):
    description = "Compile the C++ extension for macOS (inplace)"
    user_options = [
        ("inplace", "i", "Build extension in-place"),
    ]

    def initialize_options(self):
        self.inplace = True

    def finalize_options(self):
        pass

    def run(self):
        print(">>> Building C++ extension for macOS...")
        build_ext_cmd = self.reinitialize_command("build_ext")
        build_ext_cmd.inplace = 1
        self.run_command("build_ext")


class UbuntuCommand(Command):
    description = "Compile the C++ extension for Ubuntu (runs inside distrobox on NixOS)"
    user_options = [
        ("direct", None, "Directly run build_ext without wrapping in distrobox"),
        ("container=", None, "Name of distrobox container to use"),
        ("inplace", "i", "Build extension in-place"),
    ]

    def initialize_options(self):
        self.direct = False
        self.container = None
        self.inplace = True

    def finalize_options(self):
        pass

    def run(self):
        if is_running_in_container() or self.direct:
            print(">>> Building C++ extension for Ubuntu (x86-64 Linux)...")
            build_ext_cmd = self.reinitialize_command("build_ext")
            build_ext_cmd.inplace = 1
            self.run_command("build_ext")
        else:
            container = get_distrobox_container(self.container)
            project_dir = os.path.dirname(os.path.abspath(__file__))
            venv_python = os.path.join(project_dir, ".venv", "bin", "python")
            python_bin = venv_python if os.path.exists(venv_python) else sys.executable
            setup_script = os.path.abspath(__file__)

            cmd = [
                "distrobox",
                "enter",
                container,
                "--",
                python_bin,
                setup_script,
                "ubuntu",
                "--direct",
            ]
            print(f">>> Entering distrobox container '{container}' to compile C++ extension...")
            print("Command:", " ".join(cmd))
            try:
                subprocess.run(cmd, check=True)
                print(">>> Compilation inside distrobox succeeded.")
            except FileNotFoundError:
                print("Error: 'distrobox' command not found. Are you on the NixOS server?")
                sys.exit(1)
            except subprocess.CalledProcessError as e:
                print(f"Build failed inside distrobox with exit code {e.returncode}")
                sys.exit(e.returncode)


# ---------------------------------------------------------------------------
# Pybind11 include helper & Extension definition
# ---------------------------------------------------------------------------
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
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
]

setup(
    name="cp_lns_core",
    version="1.0.0",
    description="CP Optimizer LNS Backend with Pybind11",
    ext_modules=ext_modules,
    cmdclass={
        "mac": MacCommand,
        "ubuntu": UbuntuCommand,
    },
    install_requires=["pybind11>=2.5.0"],
)
