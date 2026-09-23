"""
2D Ising model on an L x L square lattice with periodic boundary conditions.
 
    H = -J * sum_<ij> s_i s_j - h * sum_i s_i,   s_i in {-1, +1}
 
Algorithms
----------
- Checkerboard Metropolis (vectorised NumPy): local updates, fine away from T_c.
- Wolff single-cluster (Numba if available): beats critical slowing down near T_c.
 
Observables (per spin): <e>, <|m|>, specific heat C, susceptibility chi,
Binder cumulant U4 = 1 - <m^4> / (3 <m^2>^2). Errors via binning.
 
Exact reference: T_c = 2 / ln(1 + sqrt 2) ~ 2.269185 (J = k_B = 1),
Onsager magnetisation m(T) = (1 - sinh(2/T)^-4)^(1/8) for T < T_c.
"""
 
import numpy as np
 
try:
    from numba import njit
except ImportError:  # graceful fallback: pure Python (slow for Wolff)
    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        return lambda f: f
 
T_C = 2.0 / np.log(1.0 + np.sqrt(2.0))
 
 
# ----------------------------------------------------------------------------
# Lattice utilities
# ----------------------------------------------------------------------------
def neighbour_sum(s):
    """Sum of the 4 nearest neighbours of each site (periodic BCs)."""
    return (np.roll(s, 1, 0) + np.roll(s, -1, 0) +
            np.roll(s, 1, 1) + np.roll(s, -1, 1))
 
 
def energy(s, J=1.0, h=0.0):
    """Total energy; each bond counted once via right and down neighbours."""
    bonds = s * (np.roll(s, -1, 0) + np.roll(s, -1, 1))
    return -J * bonds.sum() - h * s.sum()
 
 
def init_lattice(L, state="random", rng=None):
    rng = np.random.default_rng() if rng is None else rng
    if state == "up":
        return np.ones((L, L), dtype=np.int8)
    return rng.choice(np.array([-1, 1], dtype=np.int8), size=(L, L))


# ----------------------------------------------------------------------------
# Checkerboard Metropolis
# ----------------------------------------------------------------------------
def metropolis_sweep(s, beta, J, h, masks, rng):
    """One sweep = update black sublattice, then white. Sites on the same
    sublattice share no bonds, so updating them simultaneously is exact."""
    for mask in masks:
        dE = 2.0 * s * (J * neighbour_sum(s) + h)
        accept = mask & (rng.random(s.shape) < np.exp(-beta * dE))
        s[accept] *= -1
    return s
 
 
def checkerboard_masks(L):
    ii, jj = np.indices((L, L))
    black = (ii + jj) % 2 == 0
    return black, ~black  # L must be even for a proper bipartition
