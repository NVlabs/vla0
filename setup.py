# Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the CC BY-NC 4.0 license [see LICENSE for details].

import os
import subprocess
import sys

from setuptools import find_packages, setup
from setuptools.command.develop import develop
from setuptools.command.install import install

try:
    from setuptools.command.editable_wheel import editable_wheel

    HAS_EDITABLE_WHEEL = True
except ImportError:
    HAS_EDITABLE_WHEEL = False
    editable_wheel = None

requirements = [
    "black==23.3.0",
    "flake8==7.1.0",
    "isort==5.12.0",
    "yacs==0.1.8",
    "tqdm",
    "tensorboardx",
    "einops",
    "pre-commit",
    "tbparse",
]

# --- Optional Requirements ---
# Define different sets of optional dependencies
# Users can install these using pip install .[extra_name]
extras_require = {
    # 'qwen': Dependencies specifically for Qwen functionality
    "qwen": [
        "peft",
        "qwen-vl-utils",
        "bitsandbytes",
        "hf_xet",  # for faster downloading of models from huggingface
    ],
    "libero": [""],
}

# --- Create an 'all' extra ---
# Convenience group to install all optional dependencies
all_extras = []
for extra_deps in extras_require.values():
    all_extras.extend(extra_deps)
# Use set to remove duplicates if a package is in multiple extras
extras_require["all"] = list(set(all_extras))

# At the top, after imports
print(f"Environment: PIP_REQ_EXTRAS={os.environ.get('PIP_REQ_EXTRAS', 'not set')}")


def edit_cmake_version(cmake_file_path):
    """Edit CMakeLists.txt to set cmake_minimum_required to version 3.5"""
    import re

    if not os.path.exists(cmake_file_path):
        print(f"Warning: CMakeLists.txt not found at {cmake_file_path}")
        return

    print(f"Editing {cmake_file_path} to set cmake_minimum_required to 3.5")

    # Read the file
    with open(cmake_file_path, "r") as f:
        content = f.read()

    # Replace cmake_minimum_required line using regex
    # This will match any version number
    modified_content = re.sub(
        r"cmake_minimum_required\s*\(\s*VERSION\s+[\d.]+\s*\)",
        "cmake_minimum_required(VERSION 3.5)",
        content,
        flags=re.IGNORECASE,
    )

    # Write back to file
    with open(cmake_file_path, "w") as f:
        f.write(modified_content)

    print("Successfully updated cmake_minimum_required to VERSION 3.5")


def patch_libero_torch_load(libero_path):
    """
    Patch LIBERO's benchmark/__init__.py to add weights_only=False to torch.load

    Issue: PyTorch 2.6+ changed torch.load default from weights_only=False to True.
    LIBERO's code uses torch.load() without the parameter, causing warnings or failures.
    This function patches the code to explicitly set weights_only=False.
    """
    benchmark_init = os.path.join(
        libero_path, "libero", "libero", "benchmark", "__init__.py"
    )

    if not os.path.exists(benchmark_init):
        print(f"Warning: Could not find {benchmark_init} to patch")
        return

    print(f"Patching {benchmark_init} to add weights_only=False to torch.load")

    # Read the file
    with open(benchmark_init, "r") as f:
        content = f.read()

    # Apply the patch - replace torch.load( with torch.load(..., weights_only=False)
    # Handle both single line and multiline calls
    import re

    # Pattern to match torch.load(init_states_path) and replace with weights_only=False
    # This handles the specific case in the file
    modified_content = re.sub(
        r"torch\.load\(init_states_path\)",
        r"torch.load(init_states_path, weights_only=False)",
        content,
    )

    # Write back if changes were made
    if modified_content != content:
        with open(benchmark_init, "w") as f:
            f.write(modified_content)
        print(f"Successfully patched torch.load in {benchmark_init}")
    else:
        print(
            f"No changes needed in {benchmark_init} (already patched or pattern not found)"
        )


# --- Shared Post-Install Tasks ---
# This function is called by CustomInstallCommand, CustomDevelopCommand, and CustomEditableWheelCommand
def _run_post_install_tasks():
    """
    Shared installation logic for all custom install commands.

    Handles:
    - Installing pre-commit hooks (if available)
    - Installing optional extras based on PIP_REQ_EXTRAS environment variable
    - Installing libero dependencies
    """
    print("\n" + "=" * 80)
    print("Running custom post-install tasks...")
    print("=" * 80 + "\n")

    # Try to install pre-commit hooks (may not be available during wheel build)
    try:
        subprocess.check_call(["pre-commit", "install"])
        print("✓ pre-commit hooks installed")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"⚠ Skipping pre-commit install (not available yet): {e}")
        print(
            "  You can manually run 'pre-commit install' after installation completes."
        )

    # Use environment variable approach for extras
    requested_extras = []

    # Check environment variable
    if os.environ.get("PIP_REQ_EXTRAS"):
        extras = os.environ.get("PIP_REQ_EXTRAS").split(",")
        requested_extras.extend(extras)
        print(
            f"Found PIP_REQ_EXTRAS environment variable: {os.environ.get('PIP_REQ_EXTRAS')}"
        )
    else:
        print("PIP_REQ_EXTRAS environment variable not found")

    # Debug info
    print(f"Debug: sys.argv = {sys.argv}")
    print(f"Debug: cwd = {os.getcwd()}")
    print(f"Debug: __file__ = {os.path.abspath(__file__)}")
    print(f"Requested extras: {list(set(requested_extras))}")  # Deduplicate

    if "libero" in requested_extras or "all" in requested_extras:
        print("\n" + "-" * 80)
        print("Installing libero extras...")
        print("-" * 80 + "\n")

        # Calculate absolute paths relative to workspace root
        setup_dir = os.path.dirname(os.path.abspath(__file__))
        workspace_root = setup_dir  # setup.py is at workspace root
        libero_path = os.path.join(workspace_root, "libs", "LIBERO")

        print(f"Debug: Setup dir = {setup_dir}")
        print(f"Debug: Workspace root = {workspace_root}")
        print(f"Debug: LIBERO path = {libero_path}")

        if not os.path.exists(libero_path):
            # git clone LIBERO
            subprocess.check_call(
                [
                    "git",
                    "clone",
                    "https://github.com/Lifelong-Robot-Learning/LIBERO.git",
                    libero_path,
                ]
            )

        # FIX: Create missing __init__.py in libero/ directory
        # Issue: LIBERO's package structure has libero/configs/, libero/libero/, libero/lifelong/
        # but the top-level libero/ directory is missing __init__.py. This causes setuptools'
        # find_packages() to skip the libero/ directory entirely, resulting in no packages
        # being discovered. The symptoms are:
        #   - Empty top_level.txt in package metadata
        #   - ModuleNotFoundError when trying to import libero
        # Solution: Create the missing __init__.py so find_packages() recognizes libero/ as a package.
        # This file can be empty - it just needs to exist for Python to treat the directory as a package.
        libero_init_file = os.path.join(libero_path, "libero", "__init__.py")
        if not os.path.exists(libero_init_file):
            print(f"Creating missing {libero_init_file} to fix package discovery")
            open(libero_init_file, "a").close()  # equivalent to 'touch'

        # FIX: Patch torch.load to add weights_only=False for PyTorch 2.6+ compatibility
        patch_libero_torch_load(libero_path)

        egl_probe_path = os.path.join(workspace_root, "libs", "egl_probe")
        print(f"Debug: egl_probe path = {egl_probe_path}")

        if not os.path.exists(egl_probe_path):
            # git clone egl_probe
            subprocess.check_call(
                [
                    "git",
                    "clone",
                    "https://github.com/StanfordVL/egl_probe.git",
                    egl_probe_path,
                ]
            )

        # Edit CMakeLists.txt before installation
        cmake_file = os.path.join(egl_probe_path, "egl_probe", "CMakeLists.txt")
        edit_cmake_version(cmake_file)
        # install cmake via conda as it is required by egl_probe
        subprocess.check_call(["conda", "install", "-y", "-c", "conda-forge", "cmake"])
        # install egl_probe
        print(f"Installing egl_probe from {egl_probe_path}")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-v", "-e", egl_probe_path]
        )

        # install LIBERO
        print(f"Installing LIBERO from {libero_path}")
        subprocess.check_call(
            [
                "pip",
                "install",
                "-v",
                "-r",
                os.path.join(libero_path, "requirements.txt"),
            ]
        )
        subprocess.check_call(["pip", "install", "-v", "-e", libero_path])

        # Force upgrade numpy to meet our requirements
        # LIBERO donwgrades numpy to 1.22.0, which is not compatible with our requirements of save dataset stats
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-v", "--upgrade", "numpy==1.26.4"]
        )

    print("\n" + "=" * 80)
    print("Custom post-install tasks completed!")
    print("=" * 80 + "\n")


# --- Custom Install Commands ---
# These commands hook into pip's installation process to run custom post-install tasks


class CustomInstallCommand(install):
    """Custom installation command to run pre-commit install and handle local dependencies"""

    def run(self):
        install.run(self)
        _run_post_install_tasks()


class CustomDevelopCommand(develop):
    """Custom develop command to handle local dependencies for development mode"""

    def run(self):
        develop.run(self)
        _run_post_install_tasks()


if HAS_EDITABLE_WHEEL:

    class CustomEditableWheelCommand(editable_wheel):
        """Custom editable_wheel command for modern pip (PEP 660)"""

        def run(self):
            editable_wheel.run(self)
            _run_post_install_tasks()


# Set up command class dictionary
cmdclass_dict = {
    "install": CustomInstallCommand,
    "develop": CustomDevelopCommand,
}

# Add editable_wheel command if available (for modern pip versions)
if HAS_EDITABLE_WHEEL:
    cmdclass_dict["editable_wheel"] = CustomEditableWheelCommand

setup(
    name="rv_train",
    packages=find_packages(),
    cmdclass=cmdclass_dict,
    install_requires=requirements,
    extras_require=extras_require,
)
