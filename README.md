
---

# Tetrakis‑Sim: Degree‑19 Spacetime Sandbox

This repository contains a **modular discrete‑geometry playground** based on the tetrakis‑square tiling, using **row/column** uniqueness constraints only (degree 19 per vertex).

You can:

* Build finite rectangular patches of the lattice (2D and 3D)
* Inject local curvature by removing a 45° wedge (one intra‑cell edge)
* Explore geodesic bending visually via included Jupyter notebooks
* Run everything reproducibly inside a slim Docker container
* Use a modern CLI for automation, scripting, and experiments

---

## 🚀 Quick Start

### 1. Clone and Build

```bash
git clone https://github.com/MikeLawrenchuk/tetrakis-sim.git
cd tetrakis-sim
docker build -t tetrakis-sim .
```

---

### 2. Run CLI (inside Docker)

```bash
# Basic run (15x15, wedge defect)
docker run --rm -e PYTHONPATH=/home/app tetrakis-sim python scripts/run_sim.py --size 15 --defect wedge

# Flat sheet (no defect)
docker run --rm -e PYTHONPATH=/home/app tetrakis-sim python scripts/run_sim.py --size 15 --defect none
```

> **Note:**
> The Docker image is optimized for CLI/scripts only.
> **Jupyter notebooks are NOT included in the Docker image.**
> To use Jupyter, follow the local instructions below.

---

### 3. Run Locally (with venv) — Recommended for Notebooks

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt -r requirements-dev.txt
python scripts/run_sim.py --size 15 --defect wedge
jupyter notebook
```

* **requirements.txt** — core dependencies (for running code/scripts)
* **requirements-dev.txt** — notebook & developer-only dependencies

---

## 📓 Notebooks & Interactive Demos

See notebooks in [`notebooks/`](./notebooks/):

* `03-tetrakis-3d-demo.ipynb` — Modern 3D lattice construction and visualization
* `explore_tetrakis.txt` — Play with lattice construction and graph properties
* `tetrakis_geodesic_demo.txt` — Modular, up-to-date geodesic demo

Open these in Jupyter or VS Code for interactive exploration.

*Notebooks are versioned for documentation and reproducibility, but **not included in Docker images**.*

---

## 📁 File Layout

| File / Folder          | Role                                                          |
| ---------------------- | ------------------------------------------------------------- |
| `Dockerfile`           | Slim Python 3.11 image, sets up non‑root user `app`           |
| `requirements.txt`     | Pinned runtime deps (`networkx`, `matplotlib`)                |
| `requirements-dev.txt` | Dev/notebook-only deps (`jupyter`, `ipywidgets`)              |
| `tetrakis_sim/`        | **Modular package**: lattice, defects, CLI, physics, plotting |
| `scripts/`             | Entrypoints for CLI (calls `main()` from modular package)     |
| `notebooks/`           | Interactive Jupyter notebooks                                 |
| `tests/`               | Unit tests (pytest)                                           |
| `.gitignore`           | Keeps caches/scratch files out of Git                         |

---

## 🛣️ Project Roadmap

* **2D Lattice and Defect Modeling**
* **3D Lattice (“Floors”)**
* **Wave and FFT Simulations**
* **Advanced Defects (e.g., Black Holes)**
* **Extensible Physics & User Customization**
* **Visualization & Interactive Exploration**

See [issues](./issues) and [project board](./projects) for ideas and progress.

---

## 📄 License

Distributed under the MIT License; see `LICENSE` for details.

---

## 💬 Interactive Demos

Explore the latest example notebooks in [`notebooks/`](./notebooks/).
To try them, use Jupyter locally as described above.

---

**For questions, bug reports, or ideas, open an [issue](./issues) or start a [discussion](./discussions)!**

---


