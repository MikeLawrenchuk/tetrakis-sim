# ---------- build stage ----------
FROM python:3.12-slim AS build

# Workdir & non-root user
ARG USER=app
RUN useradd -m ${USER}
WORKDIR /home/${USER}

# Requirements first (leverages layer caching)
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Application code
COPY lattice_sim.py .

# Switch to non-root
USER ${USER}

# ---------- runtime stage ----------
# (still slim, identical image here; separate stage ready for
#  multi-stage builds if the project grows)
CMD ["python", "lattice_sim.py"]
