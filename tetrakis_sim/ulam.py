from __future__ import annotations

import numpy as np


def sieve_primes(limit: int) -> np.ndarray:
    """Return boolean array is_prime[0..limit] using a simple sieve."""
    if limit < 2:
        return np.zeros(limit + 1, dtype=bool)

    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[:2] = False

    # Strike out evens > 2
    is_prime[4::2] = False

    p = 3
    while p * p <= limit:
        if is_prime[p]:
            is_prime[p * p :: 2 * p] = False
        p += 2

    return is_prime


def ulam_spiral_coords(n: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Return x,y coordinates for integer values 0..n with 1 at (0,0).

    Standard walk: right, up, left, down with step lengths 1,1,2,2,3,3,...
    """
    x = np.zeros(n + 1, dtype=int)
    y = np.zeros(n + 1, dtype=int)

    if n < 1:
        return x, y

    cx, cy = 0, 0
    x[1], y[1] = 0, 0

    step_len = 1
    direction = 0  # 0=R, 1=U, 2=L, 3=D
    steps_in_leg = 0

    for value in range(2, n + 1):
        if direction == 0:
            cx += 1
        elif direction == 1:
            cy += 1
        elif direction == 2:
            cx -= 1
        else:
            cy -= 1

        x[value], y[value] = cx, cy
        steps_in_leg += 1

        if steps_in_leg == step_len:
            steps_in_leg = 0
            direction = (direction + 1) % 4
            # Increase step length after completing two legs
            if direction in (0, 2):
                step_len += 1

    return x, y


def primes_up_to(n: int) -> np.ndarray:
    """Return a 1-D array of all primes <= n."""
    return np.flatnonzero(sieve_primes(n))


def primes_on_ulam_diagonals(n: int) -> np.ndarray:
    """
    Return primes <= n that lie on the two main Ulam diagonals (x=y or x=-y).
    """
    is_prime = sieve_primes(n)
    x, y = ulam_spiral_coords(n)

    values = np.arange(1, n + 1)
    prime_vals = values[is_prime[1:]]  # aligns is_prime[1] with value=1

    diag_mask = (x[prime_vals] == y[prime_vals]) | (x[prime_vals] == -y[prime_vals])
    return prime_vals[diag_mask]
