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


# ----------------------------------------------------------------------------
# Wolff cluster algorithm (h = 0 only)
# ----------------------------------------------------------------------------
@njit(cache=True)
def wolff_step(s, p_add, seed_i, seed_j, rands):
    """Grow and flip one cluster. `rands` is a pre-drawn uniform buffer
    (length >= 4 L^2) so the kernel needs no RNG state. Returns cluster size."""
    L = s.shape[0]
    stack_i = np.empty(L * L, dtype=np.int64)
    stack_j = np.empty(L * L, dtype=np.int64)
    s0 = s[seed_i, seed_j]
    s[seed_i, seed_j] = -s0          # flip on insertion = visited marker
    stack_i[0], stack_j[0] = seed_i, seed_j
    top, size, k = 1, 1, 0
    di = (1, -1, 0, 0)
    dj = (0, 0, 1, -1)
    while top > 0:
        top -= 1
        i, j = stack_i[top], stack_j[top]
        for d in range(4):
            ni = (i + di[d]) % L
            nj = (j + dj[d]) % L
            if s[ni, nj] == s0:
                if rands[k] < p_add:
                    s[ni, nj] = -s0
                    stack_i[top], stack_j[top] = ni, nj
                    top += 1
                    size += 1
                k += 1
    return size
 
 
def wolff_sweep(s, beta, J, rng):
    """Flip clusters until ~L^2 spins have been flipped (≈ one sweep)."""
    L = s.shape[0]
    p_add = 1.0 - np.exp(-2.0 * beta * J)
    flipped = 0
    while flipped < L * L:
        i, j = rng.integers(0, L, size=2)
        flipped += wolff_step(s, p_add, i, j, rng.random(4 * L * L))
    return s


# ----------------------------------------------------------------------------
# Simulation driver
# ----------------------------------------------------------------------------
def binned_error(x, n_bins=20):
    """Standard error from n_bins block means (handles autocorrelation)."""
    n = len(x) // n_bins
    means = x[: n * n_bins].reshape(n_bins, n).mean(axis=1)
    return means.std(ddof=1) / np.sqrt(n_bins)
 
 
def simulate(L, T, n_therm=2000, n_meas=10000, algo="auto",
             J=1.0, h=0.0, seed=None, s=None):
    """Run one temperature. Returns a dict of observables and the final lattice."""
    rng = np.random.default_rng(seed)
    beta = 1.0 / T
    N = L * L
    if algo == "auto":
        algo = "wolff" if (h == 0.0 and abs(T - T_C) < 0.5) else "metropolis"
    if algo == "wolff" and h != 0.0:
        raise ValueError("Wolff as implemented requires h = 0")
    if s is None:
        s = init_lattice(L, "up" if T < T_C else "random", rng)
 
    masks = checkerboard_masks(L)
    step = (lambda: wolff_sweep(s, beta, J, rng)) if algo == "wolff" else \
           (lambda: metropolis_sweep(s, beta, J, h, masks, rng))
 
    for _ in range(n_therm):
        step()
 
    E = np.empty(n_meas)
    M = np.empty(n_meas)
    for t in range(n_meas):
        step()
        E[t] = energy(s, J, h)
        M[t] = s.sum()
 
    e, m = E / N, np.abs(M) / N
    m2, m4 = (M / N) ** 2, (M / N) ** 4
    out = {
        "L": L, "T": T, "algo": algo,
        "e": e.mean(), "e_err": binned_error(e),
        "m": m.mean(), "m_err": binned_error(m),
        "C": beta**2 * (np.mean(E**2) - np.mean(E) ** 2) / N,
        "chi": beta * N * (np.mean(m2) - np.mean(m) ** 2),
        "U4": 1.0 - np.mean(m4) / (3.0 * np.mean(m2) ** 2),
    }
    return out, s
 
 
def onsager_m(T):
    T = np.asarray(T, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        m = (1.0 - np.sinh(2.0 / T) ** -4) ** 0.125
    return np.where(T < T_C, m, 0.0)
 
 
def temperature_scan(L, temps, **kw):
    """Scan temperatures low -> high, reusing the lattice as a warm start."""
    results, s = [], None
    for T in sorted(temps):
        r, s = simulate(L, T, s=s, **kw)
        results.append(r)
        print(f"L={L:3d} T={T:.3f} [{r['algo']:10s}] "
              f"e={r['e']:+.4f}±{r['e_err']:.4f}  |m|={r['m']:.4f}±{r['m_err']:.4f}  "
              f"C={r['C']:.3f}  chi={r['chi']:8.3f}  U4={r['U4']:.4f}")
    return results
