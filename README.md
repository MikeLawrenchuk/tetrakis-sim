



# Tetrakis‑Sim: Degree‑19 Spacetime Sandbox

This repository contains a **modular discrete‑geometry playground** based on
the tetrakis‑square tiling, using **row/column** uniqueness constraints only
(degree 19 per vertex). You can:

* Build finite rectangular patches of the lattice (2D for now)
* Inject local curvature by removing a 45° wedge (one intra‑cell edge)
* Explore geodesic bending visually via included Jupyter notebooks
* Run everything reproducibly inside a slim Docker container
* Use a modern CLI for automation, scripting, and experiments

---

## Quick Start

### Clone and Build

```bash
git clone https://github.com/MikeLawrenchuk/tetrakis-sim.git
cd tetrakis-sim
docker build -t tetrakis-sim .
````

### Run CLI (inside Docker)

```bash
# Basic run (15x15, wedge defect)
docker run --rm -e PYTHONPATH=/home/app tetrakis-sim python scripts/run_sim.py --size 15 --defect wedge

# Flat sheet (no defect)
docker run --rm -e PYTHONPATH=/home/app tetrakis-sim python scripts/run_sim.py --size 15 --defect none
```

### Run Jupyter Notebooks (inside Docker)

```bash
docker run --rm -p 8888:8888 -e PYTHONPATH=/home/app tetrakis-sim \
    jupyter notebook --notebook-dir=/home/app/notebooks --ip=0.0.0.0 --allow-root
```

Then open [http://localhost:8888](http://localhost:8888) in your browser.

### Run Locally (with venv)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python scripts/run_sim.py --size 15 --defect wedge
jupyter notebook
```

---

## Notebooks & Interactive Demos

See notebooks in [`notebooks/`](./notebooks/):

* `explore_tetrakis.ipynb` — Play with lattice construction and graph properties
* `geodesic_bending_demo.ipynb` — (Legacy) Minimal geodesic bending demo
* `tetrakis_geodesic_demo.ipynb` — **Recommended shareable demo**: modular, up-to-date

Open these in Jupyter or VS Code for interactive exploration.

---

## File Layout

| File / Folder        | Role                                                              |
| -------------------- | ----------------------------------------------------------------- |
| `Dockerfile`         | Slim Python 3.12 image, sets up non‑root user `app`.              |
| `requirements.txt`   | Pinned runtime deps (`networkx`, `matplotlib`, `jupyter`).        |
| `lattice_sim.py`     | (Legacy) Original script for building/printing degree/edge count. |
| `tetrakis_sim/`      | **Modular package**: lattice, defects, CLI, physics, plotting     |
| `scripts/run_sim.py` | Entrypoint for CLI (calls `main()` from modular package)          |
| `notebooks/`         | Interactive Jupyter notebooks                                     |
| `.gitignore`         | Keep caches/scratch files out of Git                              |

---

## License

Distributed under the MIT License; see `LICENSE` for full text.

---

## Project Roadmap

This project will evolve in the following stages:

1. **2D Lattice and Defect Modeling**
2. **3D Lattice (“Floors”)**
3. **Wave and FFT Simulations**
4. **Advanced Defects (e.g., Black Holes)**
5. **Extensible Physics & User Customization**
6. **Visualization & Interactive Exploration**

**Follow the [issues](./issues) and [project board](./projects) for current work and future ideas!**

---

## Interactive Demos

See the latest example notebooks in [`notebooks/`](./notebooks/).
To try them, use Jupyter or Docker as described above.

---

```

---


