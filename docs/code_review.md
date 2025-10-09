# Tetrakis-Sim – Code Review

## Overview
The project provides a lattice generator, basic wave simulation utilities, and a batch CLI pipeline for running experiments on tetrakis-square graphs. The architecture is modular, but several core components are incomplete or fragile, which makes end-to-end runs unreliable.

## High-Priority Findings
- **Incomplete physics API.** `tetrakis_sim.physics` exposes helpers (`apply_defect`, `plot_lattice`, `plot_fft`) that are unimplemented, so any consumer expecting the documented workflow will fail at runtime. 【F:tetrakis_sim/physics.py†L59-L85】
- **CLI ignores `--defect_type`.** The batch script always constructs a 3D lattice and applies a black hole defect regardless of the user’s selection. Passing `--defect_type=wedge` or `--defect_type=none` has no effect, which makes the advertised interface misleading. 【F:scripts/run_batch.py†L22-L55】
- **FFT frequency scaling is wrong.** `run_fft` returns raw `np.fft.fftfreq` output, which assumes a unit sampling interval. Because simulations use a configurable `dt`, the reported frequencies are off by a factor of `dt`, and downstream spectrum files/plots misreport dominant frequencies. 【F:tetrakis_sim/physics.py†L66-L78】

## Additional Issues & Risks
- **Unstable time-stepping defaults.** The explicit FDTD update does not enforce any stability criterion. On dense tetrakis graphs the effective degree is large, so the default `c=1.0`, `dt=0.2` can violate the CFL bound and explode numerically without warning. Adding validation (e.g., limit `c*dt` based on max degree) or adaptive damping would make runs predictable. 【F:tetrakis_sim/physics.py†L24-L55】
- **3D wedge defect is unsafe.** `apply_wedge_defect` assumes 3-tuples when unpacking nodes; running it on a 3D lattice (4-tuples) raises `ValueError`. The helper should branch on dimensionality like the black-hole defect does. 【F:tetrakis_sim/defects.py†L7-L21】
- **Metadata omits actual defect footprint.** Although `apply_blackhole_defect` returns the removed nodes, the CLI throws that information away, so downstream analysis cannot reconstruct the cavity shape or horizon. Persisting this list (or at least its count) would improve reproducibility. 【F:scripts/run_batch.py†L42-L79】【F:tetrakis_sim/defects.py†L23-L47】

## Suggestions
1. Finish the plotting/defect stubs in `physics.py`, or drop the unused exports to avoid signaling non-existent functionality.
2. Route CLI defect handling through `tetrakis_sim.defects.apply_defect`, passing `defect_type`, and add validation for unsupported 3D wedge operations.
3. Accept `dt` in `run_fft` (or store it in the simulation history) and pass `d=dt` to `np.fft.fftfreq` so spectrum peaks reflect physical units.
4. Compute the maximum node degree once per graph and constrain `c*dt` accordingly; emit a warning or auto-scale `dt` if the user’s parameters are unsafe.
5. Store removed nodes (or counts) alongside metadata for reproducible post-processing.

## Strengths
- Modular lattice/defect separation makes it easy to plug in new geometries. 【F:tetrakis_sim/lattice.py†L4-L55】【F:tetrakis_sim/defects.py†L23-L73】
- The batch CLI already saves plots, CSV spectra, and run metadata, giving a solid foundation for reproducible sweeps. 【F:scripts/run_batch.py†L56-L79】
