import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm


# Linear system z' = A z, exact solution z(t) = e^{At} z(0)
lam = -1.0
c = 1.0
A = np.array([[lam, c],
              [0.0, lam]])

# Vector field on a grid
X, Y = np.meshgrid(np.linspace(-3, 3, 30), np.linspace(-3, 3, 30))
U = A[0, 0] * X + A[0, 1] * Y
V = A[1, 0] * X + A[1, 1] * Y
plt.streamplot(X, Y, U, V, color="lightgray", density=1.2)

# Trajectories from initial points on a circle, using the matrix exponential
t = np.linspace(0, 5, 300)
for th in np.linspace(0, 2 * np.pi, 12, endpoint=False):
    z0 = 2.5 * np.array([np.cos(th), np.sin(th)])
    z = np.array([expm(A * ti) @ z0 for ti in t])
    plt.plot(z[:, 0], z[:, 1])
    plt.plot(*z0, "ko", ms=3)

plt.xlim(-3, 3)
plt.ylim(-3, 3)
plt.gca().set_aspect("equal")
plt.xlabel("x")
plt.ylabel("y")
plt.title(f"Linear flow, lambda = {lam}, c = {c}")
plt.show()
