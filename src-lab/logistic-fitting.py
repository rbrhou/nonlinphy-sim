import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

data = np.array([
    (0.05, 1), (0.1, 1), (0.2, 0.95), (0.3, 0.92), (0.4, 0.82), (0.5, 0.6),
    (0.6, 0.4), (0.7, 0.15), (0.8, 0.15), (0.9, 0.1), (1.0, 0.1),
], dtype=float)


def model_function(x, a_1, a_2, a_3, a_4, a_5):
    """
    R = a_1 + a_2 / (1 + a_3 * exp(a_4 * x + a_5))
    """
    return a_1 + a_2 / (1 + a_3 * np.exp(a_4 * x + a_5))

def load_data(data):
    x = data[:, 0]
    y = data[:, 1]
    return x, y

def plot_fit(x, y, popt):
    x_smooth = np.linspace(x.min(), x.max(), 200)
    plt.plot(x, y, "o", label="data")
    plt.plot(x_smooth, model_function(x_smooth, *popt), label=r"$a_1 + \frac{a_2}{1 + a_3 e^{a_4 x + a_5}}$")
    plt.xlabel(r"Temperature ($10^3$ °C)")
    plt.ylabel("Reduction factor R")
    plt.title("Steel yield strength reduction vs. temperature")
    plt.legend()
    plt.show()

def plot_residuals(x, y, popt):
    residuals = y - model_function(x, *popt)
    plt.plot(x, residuals, "o")
    plt.axhline(0, color="black")
    plt.xlabel(r"Temperature ($10^3$ °C)")
    plt.ylabel("Residual")
    plt.title("Residuals plot")
    plt.show()


x, y = load_data(data)

popt, pcov = curve_fit(model_function, x, y, maxfev=10000)

print(f"a_1 = {popt[0]:.4f}")
print(f"a_2 = {popt[1]:.4f}")
print(f"a_3 = {popt[2]:.4f}")
print(f"a_4 = {popt[3]:.4f}")
print(f"a_5 = {popt[4]:.4f}")
print(f"a_3 * exp(a_5) = {popt[2] * np.exp(popt[4]):.6f}")
print(f"sum of squared differences = {ss_res:.5f}")

plot_fit(x, y, popt)
plot_residuals(x, y, popt)
