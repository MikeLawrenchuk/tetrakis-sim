FROM python:3.11-slim-bookworm AS build

# Pull Debian security updates at build time (reduces OS-package CVEs).
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Workdir & non-root user
ARG USER=app
RUN useradd -m ${USER}
WORKDIR /home/${USER}

# Requirements first (leverages layer caching)
COPY requirements.txt .
RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install --no-cache-dir -r requirements.txt

# Application code: copy all relevant directories and files
COPY lattice_sim.py .
COPY README.md .
COPY tetrakis_sim/ ./tetrakis_sim/
COPY scripts/ ./scripts/
# COPY notebooks/ ./notebooks/   # Uncomment if you want notebooks in the image

# Switch to non-root
USER ${USER}

# Default: start a bash shell (so you can run Python, scripts, or open a shell)
CMD ["bash"]

