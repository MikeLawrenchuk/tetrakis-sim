FROM python:3.11.9-slim AS build

# Workdir & non-root user
ARG USER=app
RUN useradd -m ${USER}
WORKDIR /home/${USER}

# Requirements first (leverages layer caching)
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Application code: copy all relevant directories and files
COPY lattice_sim.py .
COPY README.md .
COPY tetrakis_sim/ ./tetrakis_sim/
COPY scripts/ ./scripts/
# COPY notebooks/ ./notebooks/   # Uncomment if you want notebooks in the image

# Switch to non-root
USER ${USER}

# Expose port for Jupyter (optional, only if you want to run a notebook server)
# EXPOSE 8888

# Default: start a bash shell (so you can run Python, scripts, or open a shell)
CMD ["bash"]
