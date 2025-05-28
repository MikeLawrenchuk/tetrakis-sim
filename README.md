# Tetrakis-Sim: Discrete Geometry & Wave Physics

**Author:** Mike Lawrenchuk
**Project:** [tetrakis-sim](https://github.com/MikeLawrenchuk/tetrakis-sim)

---

## Overview

*Tetrakis-Sim* is a research-grade Python toolkit for discrete geometry, wave simulation, and spectral analysis on 2D/3D tetrakis-square lattices with advanced defects (black holes, wedges, event horizons).

**Features:**

* Modular 2D/3D lattice and defect construction
* Physics engine: wave propagation (FDTD) on arbitrary graphs
* Black hole/event horizon and wedge defect modeling
* Batch CLI for automated parameter sweeps
* FFT, frequency mapping, mean spectrum, and advanced visualization
* Professional workflow: Docker, reproducibility, batch logs, metadata
* Interactive analysis notebooks and publication-ready plots

---

## Quick Start

### 1. **Install (dev mode for notebooks/CLI)**

```bash
pip install -e .
pip install -r requirements-dev.txt  # for pandas, jupyter, etc.
```

### 2. **Run a Simulation with the CLI**

```bash
python scripts/run_batch.py --size 11 --radius 2.5 --layers 7 --steps 50
```

**CLI arguments:**

* `--size`: Lattice width/height (NxN)
* `--radius`: Black hole radius
* `--layers`: Number of 3D floors
* `--steps`: Time steps
* `--c`, `--dt`, `--damping`: Physics options
* `--defect_type`: `blackhole`, `wedge`, `none`
* `--outdir`: Output directory (default: `batch_cli_output`)
* `--prefix`: Custom prefix for all output files

See all options:

```bash
python scripts/run_batch.py --help
```

---

### 3. **Batch Parameter Sweeps (bash/zsh)**

```bash
for size in 7 9 11; do
  for radius in 1.5 2.5 3.5; do
    python scripts/run_batch.py --size $size --radius $radius --layers 5 --steps 40
  done
done
```

---

### 4. **Output Files**

Saved to `batch_cli_output/` (or as specified):

* `*_fft_node(NODE).png`: FFT spectrum/time-series plot
* `*_spectrum.csv`: FFT data (frequency, amplitude)
* `*_metadata.json`: Metadata (parameters, node, dominant frequency, etc.)

---

### 5. **Analyze Batch Results in a Notebook**

**See:**

* `notebooks/analyze_batch_outputs.ipynb`

```python
import glob, pandas as pd, numpy as np

csv_files = glob.glob('batch_cli_output/*_spectrum.csv')
summary = []
for f in csv_files:
    import re
    m = re.search(r'size(\d+)_radius([0-9.]+)', f)
    size = int(m.group(1)); radius = float(m.group(2))
    data = np.loadtxt(f, delimiter=',', skiprows=1)
    freq, spectrum = data[:,0], data[:,1]
    dominant_freq = freq[np.argmax(spectrum)]
    summary.append({"file": f, "size": size, "radius": radius, "dominant_freq": dominant_freq})
df = pd.DataFrame(summary).sort_values(["size","radius"])
df.to_csv('batch_cli_output/batch_fft_summary.csv', index=False)
print(df)
```

Plot dominant frequency vs. radius:

```python
for size in df['size'].unique():
    subset = df[df['size']==size]
    plt.plot(subset['radius'], subset['dominant_freq'], marker='o', label=f"size={size}")
plt.xlabel("Black hole radius"); plt.ylabel("Dominant frequency"); plt.legend(); plt.show()
```

---

## Example: Advanced Run with Metadata

```bash
python scripts/run_batch.py --size 13 --radius 3.5 --layers 7 --steps 100 --c 1.5 --dt 0.15 --damping 0.02 --prefix "my_experiment"
```

---

## Features

* **Reproducible:** Every run saves all parameters and dominant frequencies as metadata.
* **Flexible:** Parameter sweeps, custom physics, and batch runs are easy and automated.
* **Publication-ready:** All output is compatible with pandas, matplotlib, and can be included in scientific papers, reports, or slides.
* **Professional CLI:** Friendly help, error-checking, and clear outputs.
* **Docker-ready:** Build a reproducible environment for any machine with a single command.

---

## Documentation

* **Source code:** See `tetrakis_sim/`
* **Notebooks:** See `notebooks/analyze_batch_outputs.ipynb` for batch data analysis.
* **CLI script:** See `scripts/run_batch.py --help`

---

## Contributing / Collaboration

Pull requests, issues, and collaborations are welcome!
If you’re interested in research, teaching, or publishing with this code, please open an issue or contact [Mike Lawrenchuk](mailto:your-email@example.com).

---

## License

Distributed under the MIT License; see `LICENSE` for details.
---

*For questions, collaboration, or scientific consulting, open an issue or contact the author!*

---
