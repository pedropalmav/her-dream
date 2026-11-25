FROM ubuntu:24.04

# The torch wheels pinned in uv.lock bundle their own CUDA/cuDNN libraries, so a plain
# Ubuntu base is enough: the NVIDIA Container Toolkit injects the driver at run time.
ENV PYTHONUNBUFFERED=1 \
    NUMBA_CACHE_DIR=/tmp \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
    UV_LINK_MODE=copy \
    UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH=/opt/venv/bin:$PATH

# Install necessary packages
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    ca-certificates \
    libgl1 \
    libegl1 \
    cmake \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Create an ICD config file to register the NVIDIA EGL implementation with glvnd.
RUN mkdir -p /usr/share/glvnd/egl_vendor.d && \
    echo '{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}' \
    > /usr/share/glvnd/egl_vendor.d/10_nvidia.json

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# Set the working directory
WORKDIR /workspace

# Install the locked dependencies first, in their own layer, so that code changes do
# not invalidate it. The venv and the uv-managed Python live under /opt, outside
# /workspace, so mounting the repo over /workspace at run time does not hide them.
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-install-project

# Install the her_dream package itself (editable, pointing at /workspace/her_dream, so a
# repo mounted over /workspace is what gets imported).
COPY . .
RUN uv sync --frozen

# Set the default MuJoCo rendering backend to EGL, which is compatible with headless
# environments and does not require a display server.
ENV MUJOCO_GL=egl
