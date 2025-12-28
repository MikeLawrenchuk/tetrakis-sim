"""examples/ulam_diagonal_helix_lastdigit_demo.py

Fixed, robust demo:
- Builds a prime-helix where radii come from Ulam-diagonal primes.
- Colors points by the *last digit of the prime radius* (prime_radius % 10).
- Avoids empty-bucket crashes.
- Optionally auto-extends N_RINGS until a 9-ending prime appears (if present).
- Optionally rescales x/y for plotting only (keeps prime_radius intact).

Run:
  python examples/ulam_diagonal_helix_lastdigit_demo.py
"""

from __future__ import annotations

import math
from collections import Counter

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import networkx as nx

from tetrakis_sim.prime_helix import add_prime_helix
from tetrakis_sim.ulam import primes_on_ulam_diagonals_with_offsets


def main() -> None:
    # --- knobs ---
    ULAM_N = 500_000
    N_RINGS = 1_000

    # If True, automatically increase the number of rings to include
    # the first Ulam-diagonal prime ending in 9 (if one exists in the pool).
    ENSURE_LAST_DIGIT_9 = True

    DTHETA_DEG = 6.0
    PITCH = 1.8
    NODE_SIZE = 12

    # Plot scaling only (does not change prime values / last digits):
    RESCALE_XY_FOR_PLOT = True
    TARGET_XY_MAX = 450.0  # used only if RESCALE_XY_FOR_PLOT
    # -----------

    KMAX = 2
    radii_all = primes_on_ulam_diagonals_with_offsets(ULAM_N, kmax=KMAX)

    if len(radii_all) == 0:
        raise RuntimeError(f"No Ulam-diagonal primes found up to {ULAM_N}.")

    idx9 = next((i for i, p in enumerate(radii_all) if p % 10 == 9), None)
    print("First index with last digit 9:", idx9)

    n_rings = N_RINGS
    if ENSURE_LAST_DIGIT_9 and idx9 is not None:
        n_rings = max(n_rings, idx9 + 1)

    if n_rings > len(radii_all):
        n_rings = len(radii_all)
        print(f"Note: only {n_rings} diagonal primes available up to ULAM_N; using all of them.")

    print(f"ULAM_N={ULAM_N}  KMAX={KMAX}  pool={len(radii_all)}  using_rings={n_rings}")


    radii = radii_all[:n_rings]

    # Build graph
    G = nx.Graph()
    add_prime_helix(
        G,
        n_rings=n_rings,
        radii=radii,
        dtheta=math.radians(DTHETA_DEG),
        pitch=PITCH,
    )

    # Count rings per last digit (count each ring once: j==0)
    ring_last_digit = Counter()
    for n in G:
        if isinstance(n, tuple) and len(n) == 2 and n[1] == 0:
            ring_last_digit[G.nodes[n]["prime_radius"] % 10] += 1

    print("Rings by last digit:", dict(sorted(ring_last_digit.items())))

    # Determine which digit classes are actually present (avoid empty classes)
    present_digits = [d for d in (1, 3, 7, 9) if ring_last_digit.get(d, 0) > 0]
    if not present_digits:
        # Fallback: whatever is present
        present_digits = sorted(ring_last_digit.keys())

    # Prepare plot
    cmap = plt.get_cmap("tab10", 10)
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    # Optional x/y scaling for plotting
    if RESCALE_XY_FOR_PLOT:
        # Find max |x| or |y| among all nodes
        xy_max = 0.0
        for n in G:
            x, y, _z = G.nodes[n]["pos"]
            xy_max = max(xy_max, abs(float(x)), abs(float(y)))
        scale = (TARGET_XY_MAX / xy_max) if xy_max > 0 else 1.0
    else:
        scale = 1.0

    # Plot each last-digit class
    for d in present_digits:
        nodes_d = [n for n in G if (G.nodes[n]["prime_radius"] % 10) == d]
        if not nodes_d:
            continue

        positions = [G.nodes[n]["pos"] for n in nodes_d]
        xs, ys, zs = zip(*positions)

        # Apply plot-only scaling to x/y
        xs = [float(x) * scale for x in xs]
        ys = [float(y) * scale for y in ys]
        zs = [float(z) for z in zs]

        label = f"{d} (rings={ring_last_digit.get(d, 0)})"
        ax.scatter(xs, ys, zs, s=NODE_SIZE, color=cmap(d), depthshade=True, label=label)

    ax.set_title("Prime-Helix (colored by last digit of prime radius)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend(title="prime_radius % 10", loc="upper left", bbox_to_anchor=(1.02, 1.0))

    fig.savefig("ulam_diagonal_helix_lastdigit.png", dpi=180, bbox_inches="tight")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

