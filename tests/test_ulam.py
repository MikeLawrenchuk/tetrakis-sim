"""Unit tests for tetrakis_sim.ulam.

These tests are designed to be:
- fast (small N),
- deterministic,
- robust to changes in the exact prime distribution on diagonals.
"""

from __future__ import annotations

import numpy as np

from tetrakis_sim.ulam import (
    primes_on_ulam_diagonals,
    primes_on_ulam_diagonals_with_offsets,
    primes_up_to,
    sieve_primes,
    ulam_spiral_coords,
)


def test_sieve_primes_basic() -> None:
    is_prime = sieve_primes(30)

    # Known primes
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29):
        assert is_prime[p]

    # Known composites / non-primes
    for n in (0, 1, 4, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20, 21, 22, 24, 25, 26, 27, 28, 30):
        assert not is_prime[n]


def test_ulam_coords_center_and_lengths() -> None:
    x, y = ulam_spiral_coords(200)
    assert len(x) == 201
    assert len(y) == 201

    # Convention: value 1 is at the origin
    assert x[1] == 0
    assert y[1] == 0


def test_primes_up_to_matches_sieve() -> None:
    n = 500
    plist = primes_up_to(n)
    is_prime = sieve_primes(n)

    assert np.array_equal(plist, np.flatnonzero(is_prime))


def test_offsets_kmax0_matches_main_diagonals() -> None:
    """kmax=0 should reproduce the strict main-diagonal rule."""
    n = 10_000
    a = primes_on_ulam_diagonals(n)
    b = primes_on_ulam_diagonals_with_offsets(n, kmax=0)
    assert np.array_equal(a, b)


def test_offsets_are_superset_and_all_prime() -> None:
    n = 20_000
    base = primes_on_ulam_diagonals_with_offsets(n, kmax=0)
    wider = primes_on_ulam_diagonals_with_offsets(n, kmax=2)

    # Wider selection should not be smaller
    assert len(wider) >= len(base)

    # All returned values must be prime
    is_prime = sieve_primes(n)
    assert all(is_prime[int(p)] for p in wider)

    # Wider selection should typically add something for moderate n
    assert len(wider) > len(base)
