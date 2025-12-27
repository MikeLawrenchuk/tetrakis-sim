FROM python:3.11-slim-bookworm AS build

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Workdir & non-root user
ARG USER=app
RUN useradd -m ${USER}
WORKDIR /home/${USER}

# Install the package (so console scripts exist) using pyproject.toml
# Include plot extras so PNG output works inside the container.
COPY pyproject.toml README.md LICENSE ./
COPY tetrakis_sim/ ./tetrakis_sim/
COPY scripts/ ./scripts/
COPY lattice_sim.py ./

RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install --no-cache-dir ".[plot]"

# Switch to non-root
USER ${USER}

# Default shell
CMD ["bash"]
