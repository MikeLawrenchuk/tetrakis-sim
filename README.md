# Tetrakis‑Sim: Degree‑19 Spacetime Sandbox

This repository contains a *minimal* discrete‑geometry playground based on
the tetrakis‑square tiling, using **row/column** uniqueness constraints only
(degree 19 per vertex).  It lets you:

* build finite rectangular patches of the lattice;
* inject local curvature by removing a 45° wedge (one intra‑cell edge);
* explore geodesic bending visually (see `geodesic_bending_demo.ipynb`);
* run everything reproducibly inside a slim Docker container.

## Quick start

```bash
# clone & enter
git clone https://github.com/MikeLawrenchuk/tetrakis-sim.git
cd tetrakis-sim

# build and run the container
docker build -t tetrakis-sim .
docker run --rm tetrakis-sim                  # with curvature defect
docker run --rm tetrakis-sim python lattice_sim.py --none   # flat sheet
```

## Demo notebook

Open **`geodesic_bending_demo.ipynb`** in JupyterLab / VS Code to reproduce
the coloured wave‑front plot that shows how geodesics deviate around the
+45° wedge placed at the centre of a 30×30 window.

## File layout

| file | role |
|------|------|
| `Dockerfile` | Slim Python 3.12 image, sets up non‑root user `app`. |
| `lattice_sim.py` | CLI driver to build the graph and print degree / edge count. |
| `requirements.txt` | Pinned runtime deps (`networkx`, `matplotlib`). |
| `geodesic_bending_demo.ipynb` | Notebook created automatically; visualises geodesic bending. |
| `.gitignore`, `.dockerignore` | Keep caches and scratch files out of Git / Docker context. |

## License

Distributed under the MIT License; see `LICENSE` for full text.
