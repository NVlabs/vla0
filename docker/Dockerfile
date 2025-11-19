# Base image: NVIDIA CUDA 12.4.1 with development tools on Ubuntu 20.04
FROM nvidia/cuda:12.4.1-devel-ubuntu20.04

# Set DEBIAN_FRONTEND to noninteractive to avoid prompts during apt-get install
ENV DEBIAN_FRONTEND=noninteractive

# === Log Step 2: Install system dependencies ===
# This command first updates the list of available packages from the configured software repositories.
# If the update is successful, it then proceeds to install a series of software packages.
# - git: version control
# - wget: network downloader
# - curl: data transfer utility
# - vim: text editor
# - cmake: build system generator
# - libglib2.0-0: a core GLib library, often a dependency for software like OpenCV/CV2
# - libgl1-mesa-glx: Mesa OpenGL/GLX runtime library, for graphics, often required by MuJoCo
# - libosmesa6-dev: Mesa off-screen rendering development libraries, for MuJoCo headless rendering
# - libglew-dev: OpenGL Extension Wrangler Library development files; resolves <GL/glew.h> errors for mujoco_py
# - libegl1-mesa: Mesa EGL runtime library; resolves libEGL.so.1 errors for mujoco_py
# - libopengl0: Vendor-neutral GLdispatch library for OpenGL; resolves libOpenGL.so.0 errors for mujoco_py
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    vim \
    cmake \
    libglib2.0-0 \
    libgl1-mesa-glx \
    libosmesa6-dev \
    libglew-dev \
    libegl1-mesa \
    libopengl0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# === Log Step 3: Install Miniconda ===
# This step downloads the Miniconda3 installer, executes it in batch mode to install
# Miniconda into /opt/conda, and then removes the installer script.
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -afy

# === Log Step 4: Configure Environment Variables ===
# MUJOCO_GL: Instructs MuJoCo to use EGL for rendering.
ENV MUJOCO_GL=egl

# PATH: Add Conda and CUDA to PATH.
# This ensures /usr/local/cuda/bin (from base image) and /opt/conda/bin are prioritized.
ENV PATH /usr/local/cuda/bin:/opt/conda/bin:$PATH

# LD_LIBRARY_PATH: Add MuJoCo to LD_LIBRARY_PATH.
ENV LD_LIBRARY_PATH /root/.mujoco/mujoco210/bin${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

# === Log Step 5: Create and Activate Conda Environment ===
# Create a Conda environment named 'test' with Python 3.10.
# The '-y' flag auto-confirms.
RUN conda create -n test python=3.10 -y && conda clean -afy

# The order of conda init bash and echo 'conda activate test' >> /root/.bashrc is important. It ensures that the 'test' environment is activated when the user runs a bash shell.
# conda init bash is required for conda to work in the container whenever the user runs a bash shell. It adds the conda init script to the .bashrc file.
# echo 'conda activate test' >> /root/.bashrc is required for conda to activate the 'test' environment when the user runs a bash shell.
RUN conda init bash
RUN echo 'conda activate test' >> /root/.bashrc

# Set the SHELL to use the 'test' conda environment for subsequent RUN commands.
# This effectively "activates" the environment while we are installing the project dependencies. The above conda activate test takes care of the when when we run the container.
SHELL ["conda", "run", "-n", "test", "/bin/bash", "-c"]

# === Log Step 6 & 7: Project Setup and Installation ===
# Argument for the project directory name. This name will be used for the
# project's directory inside ${APP_DIR} (e.g., /app/RV-train if ARG is RV-train).
ARG PROJECT_DIR_NAME=RV-train
# Base directory for applications within the container.
ENV APP_DIR /app

# Copy the project code into the container.
# The build context (specified by "." in `docker build -f docker/Dockerfile .`)
# is assumed to be the root of the "RV-train" project.
# This command copies the entire content of the build context (current directory ".")
# into ${APP_DIR}/${PROJECT_DIR_NAME} in the container (e.g. /app/RV-train).
COPY . ${APP_DIR}/${PROJECT_DIR_NAME}

# Set working directory to the project root inside the container.
# This path (e.g., /app/RV-train) now contains the project files.
WORKDIR ${APP_DIR}/${PROJECT_DIR_NAME}

# Log Step 6 (continued): Install the main project package.
# This installs the Python package located in the current directory (project root)
# in "editable" mode (-e), with all optional dependencies ([all]).
RUN echo "Current directory: $(pwd)" && \
    PIP_REQ_EXTRAS=all pip install -e ".[all]"

# Log Step 7: Install the RoboVerse library from within the project.
# This installs the RoboVerse library, located in ./libs/RoboVerse/ relative to
# the project root, in editable mode, with all optional dependencies.
RUN echo "Current directory: $(pwd)" && \
    PIP_REQ_EXTRAS=all pip install -e "./libs/RoboVerse[all]"

# Remove .git directory to save space
RUN rm -rf .git
RUN rm -rf ./libs/RoboVerse/.git

# === Log Step 8: Install specific bitsandbytes version ===
# This command installs a specific pre-compiled wheel of bitsandbytes.
# This was to resolve a GLIBC version incompatibility issue, as the
# manylinux_2_24 wheel is compatible with older GLIBC versions like 2.31 (on Ubuntu 20.04).
RUN pip install --force-reinstall https://github.com/bitsandbytes-foundation/bitsandbytes/releases/download/continuous-release_main/bitsandbytes-1.33.7.preview-py3-none-manylinux_2_24_x86_64.whl

# === Log Step 10: Configure EGL for NVIDIA ===
# This step manually creates and configures a JSON file for NVIDIA's EGL driver.
# This is often a critical fix for mujoco_py when it fails to initialize OpenGL
# on systems with NVIDIA GPUs, especially when MUJOCO_GL=egl is set.
# Such failures can manifest as "RuntimeError: Failed to initialize OpenGL".
RUN echo '{ \
    "file_format_version" : "1.0.0", \
    "ICD" : { \
        "library_path" : "libEGL_nvidia.so.0" \
    } \
}' > /usr/share/glvnd/egl_vendor.d/10_nvidia.json && \
    chmod 644 /usr/share/glvnd/egl_vendor.d/10_nvidia.json