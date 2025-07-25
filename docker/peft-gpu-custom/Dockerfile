
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu20.04

# Environments for python workspace
ENV HOME=/root
ENV WORKSPACE=/workspace
ENV PYTHON_VERSION=3.10.4

# Install essential packages
RUN apt-get update -y \
    && apt-get install -y --no-install-recommends \
        build-essential \
        libssl-dev \
        zlib1g-dev \
        libbz2-dev \
        libreadline-dev \
        libsqlite3-dev \
        wget \
        curl \
        git \
        libncursesw5-dev \
        xz-utils \
        libxml2-dev \
        tk-dev \
        libxml2-dev \
        libxmlsec1-dev \
        libffi-dev \
        liblzma-dev \
        ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
 
# install uv
ENV UV_LINK_MODE=copy
ENV UV_COMPILE_BYTECODE=1
ENV UV_PYTHON=python${PYTHON_VERSION}
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR $WORKSPACE

RUN set -ex \
    && uv venv .venv -p $PYTHON_VERSION
ENV PATH="/workspace/.venv/bin:$PATH"

ENV PYTHONPATH=src
RUN --mount=type=cache,target=/root/.cache \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    set -ex \
    && uv sync --frozen --no-install-project --no-group dev \
    && uv sync --frozen --no-install-project --group dev --no-build-isolation
