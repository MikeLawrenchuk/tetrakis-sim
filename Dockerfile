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
# COPY notebooks/ ./notebooks/

# Switch to non-root
USER ${USER}

# Expose port for Jupyter (optional, needed only if running notebook in container)
EXPOSE 8888

# Default: start a bash shell
CMD ["bash"]
