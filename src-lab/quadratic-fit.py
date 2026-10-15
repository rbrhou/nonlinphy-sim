import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

data = np.array([
    (2.2, 6.38), (1.1, -1.21), (-0.11, -4.18), (-1.1, -3.63), (-2.2, 1.65),
], dtype=float)


def model_function(x, a_0, a_1, a_2):
    """
    y = a_0 + a_1 * x + a_2 * x^2
    """
    return a_0 + a_1 * x + a_2 * x**2

def load_data(data):
    x = data[:, 0]
    y = data[:, 1]
    return x, y

def sum_of_squares(model, popt, x, y_measured):
    return np.sum((y_measured - model(x, *popt)) ** 2)

def plot_fit(x, y, popt):
    plt.plot(x, y, 'o', label="data")
    plt.plot(x, model_function(x, *popt), label=r"$a_0 + a_1 x + a_2 x^2$")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Parabola fit vs. data")
    plt.legend()
    plt.show()

def plot_residuals(x, y, popt):
    residuals = y - model_function(x, *popt)
    plt.plot(x, residuals, 'o')
    plt.axhline(0, color="black")
    plt.xlabel("x")
    plt.ylabel("Residual")
    plt.title("Residuals plot")
    plt.show()


x, y = load_data(data)

popt, pcov = curve_fit(model_function, x, y)
pstd = np.sqrt(np.diag(pcov))

ss_res = sum_of_squares(model_function, popt, x, y)

print(f"a_0 = {popt[0]:.4f} ± {pstd[0]:.4f}")
print(f"a_1 = {popt[1]:.4f} ± {pstd[1]:.4f}")
print(f"a_2 = {popt[2]:.4f} ± {pstd[2]:.4f}")
print(f"sum of squared differences = {ss_res:.4f}")

plot_fit(x, y, popt)
plot_residuals(x, y, popt)
