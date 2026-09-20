"""
Phase portraits of three one-dimensional flows, with the vector field on the
line marking stability.
 
    (1)  Ndot = -aN ln(bN)
    (2)  xdot = x - x^3
    (3)  xdot = 1 + (1/2) cos x
    (4)  xdot = e^x - cos x
 
Top row:    f(x) vs x, with the phase line underneath. Arrows point right
            where f > 0, left where f < 0. Filled circle = stable fixed point
            (f'(x*) < 0), open circle = unstable (f'(x*) > 0).
Bottom row: the direction field in the (t, x) plane with sample trajectories.
 
Run it directly to open the figure:
 
    python phase_portraits.py
 
or import the pieces and reuse them on your own systems:
 
    from phase_portraits import find_roots, phase_portrait, flow_arrows
    phase_portrait(lambda x: x**2 - 1, lambda x: 2*x, (-3, 3))
"""
 
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.integrate import solve_ivp
 
# ----------------------------------------------------------------------
# systems: (name, f, f', x-window, y-window)
# ----------------------------------------------------------------------
SYSTEMS = [
    (r"$\dot{x} = -x * \ln(x)$",
     lambda x: -x * np.log(x),
     lambda x: -np.log(x) - 1,
     (0.01, 10), (-5, 0)),
    (r"$\dot{x} = x - x^3$",
     lambda x: x - x**3,
     lambda x: 1 - 3 * x**2,
     (-2.0, 2.0), (-2.2, 2.2)),
    (r"$\dot{x} = 1 + \frac{1}{2}\cos x$",
     lambda x: 1 + 0.5 * np.cos(x),
     lambda x: -0.5 * np.sin(x),
     (-2 * np.pi, 2 * np.pi), (-0.4, 2.0)),
    (r"$\dot{x} = e^{x} - \cos x$",
     lambda x: np.exp(x) - np.cos(x),
     lambda x: np.exp(x) + np.sin(x),
     (-12.0, 0.8), (-2.0, 2.0)),
]
 
 
def find_roots(f, lo, hi, n=200_000):
    """Bracket-and-bisect every sign change on a fine grid."""
    g = np.linspace(lo, hi, n)
    v = f(g)
    idx = np.where(np.sign(v[:-1]) != np.sign(v[1:]))[0]
    return np.array([brentq(f, g[i], g[i + 1]) for i in idx])
 
 
def flow_arrows(ax, f, lo, hi, y=0.0, n=26, length=None):
    """Arrows on the phase line: direction = sign(f), so they point toward
    stable fixed points and away from unstable ones."""
    xs = np.linspace(lo, hi, n)
    if length is None:
        length = 0.45 * (hi - lo) / n
    for x in xs:
        s = np.sign(f(x))
        if s == 0:
            continue
        ax.annotate("", xy=(x + s * length, y), xytext=(x - s * length, y),
                    arrowprops=dict(arrowstyle="-|>", lw=1.3,
                                    color="C0" if s > 0 else "C3"))
 
 
def draw_phase_line(ax, f, fp, xlim, ylim, title=""):
    """Top-row panel: f(x), shaded by sign, plus the phase line."""
    xlo, xhi = xlim
    roots = find_roots(f, xlo, xhi)
    x = np.linspace(xlo, xhi, 3000)
    y = f(x)
 
    ax.fill_between(x, 0, y, where=y > 0, color="C0", alpha=0.12)
    ax.fill_between(x, 0, y, where=y < 0, color="C3", alpha=0.12)
    ax.plot(x, y, "k-", lw=2)
    ax.axhline(0, color="0.35", lw=1.0)
 
    flow_arrows(ax, f, xlo + 0.03 * (xhi - xlo), xhi - 0.03 * (xhi - xlo))
 
    for r in roots:
        ax.plot(r, 0, "o", ms=10, zorder=5,
                mfc="k" if fp(r) < 0 else "w", mec="k", mew=1.8)
 
    ax.set(xlim=xlim, ylim=ylim, xlabel="$x$",
           ylabel=r"$\dot{x} = f(x)$", title=title)
    ax.grid(alpha=0.15)
    return roots
 
 
def draw_direction_field(ax, f, fp, xlim, T=6.0, n_traj=11, n_arrows=22):
    """Bottom-row panel: slope field in (t, x) with sample trajectories."""
    xlo, xhi = xlim
    tt, xx = np.meshgrid(np.linspace(0, T, n_arrows),
                         np.linspace(xlo, xhi, n_arrows))
    U, V = np.ones_like(tt), f(xx)
    N = np.hypot(U, V)
    ax.quiver(tt, xx, U / N, V / N, color="0.55", angles="xy",
              width=0.0035, scale=28)
 
    for x0 in np.linspace(xlo, xhi, n_traj):
        sol = solve_ivp(lambda t, s: f(s), (0, T), [x0],
                        rtol=1e-8, atol=1e-10, max_step=0.05)
        ax.plot(sol.t, sol.y[0], lw=1.8, alpha=0.85)
 
    for r in find_roots(f, xlo, xhi):
        ax.axhline(r, color="k", ls="-" if fp(r) < 0 else "--",
                   lw=1.6, alpha=0.8)
 
    ax.set(xlim=(0, T), ylim=xlim, xlabel="$t$", ylabel="$x$",
           title="direction field and trajectories")
    ax.grid(alpha=0.15)
 
 
def phase_portrait(f, fp, xlim, ylim=None, title="", show=True):
    """Two-panel portrait for a single system f, f' on the window xlim."""
    if ylim is None:
        x = np.linspace(*xlim, 2000) # type: ignore
        pad = 0.15 * np.ptp(f(x))
        ylim = (f(x).min() - pad, f(x).max() + pad)
    fig, (a0, a1) = plt.subplots(2, 1, figsize=(6, 8))
    draw_phase_line(a0, f, fp, xlim, ylim, title)
    draw_direction_field(a1, f, fp, xlim)
    fig.tight_layout()
    if show:
        plt.show()
    return fig
 
 
def main(save="phase_portraits.png", show=True):
    fig, axes = plt.subplots(2, len(SYSTEMS), figsize=(15.5, 8.4))
    for col, (name, f, fp, xlim, ylim) in enumerate(SYSTEMS):
        roots = draw_phase_line(axes[0, col], f, fp, xlim, ylim, name)
        draw_direction_field(axes[1, col], f, fp, xlim)

        print(f"\n{name}  on [{xlim[0]:.3g}, {xlim[1]:.3g}]")
        if roots.size == 0:
            print("   no fixed points; flow is one-way")
        for r in roots:
            d = fp(r)
            kind = "stable" if d < 0 else ("unstable" if d > 0 else "degenerate")
            print(f"   x* = {r: .6f}   f'(x*) = {d: .4f}   {kind}")

    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150)
    if show:
        plt.show()          # opens the interactive window
    return fig
 
 
if __name__ == "__main__":
    main()
