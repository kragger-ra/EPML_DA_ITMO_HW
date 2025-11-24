# Dockerfile for Data Science project with pixi
FROM ubuntu:22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PIXI_VERSION=0.59.0 \
    PYTHONUNBUFFERED=1 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install pixi
RUN curl -fsSL https://pixi.sh/install.sh | bash && \
    echo 'export PATH="$HOME/.pixi/bin:$PATH"' >> ~/.bashrc

ENV PATH="/root/.pixi/bin:$PATH"

# Set working directory
WORKDIR /workspace

# Copy project files
COPY pixi.toml pixi.lock pyproject.toml ./
COPY src ./src
COPY tests ./tests
COPY notebooks ./notebooks

# Install dependencies using pixi
RUN pixi install

# Create data directories
RUN mkdir -p data/raw data/processed data/interim models reports/figures

# Expose Jupyter port
EXPOSE 8888

# Default command
CMD ["/bin/bash"]
