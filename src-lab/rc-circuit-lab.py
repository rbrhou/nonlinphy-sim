import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

data = "rc_data - Sheet1.csv"

def model_function(t, V_0, RC):
    return V_0 * np.exp(-t / RC)

def load_data(path):
    data = np.loadtxt(path, delimiter=",", skiprows=1, usecols=(2, 3, 4))
    return data[:, 0], data[:, 1], data[:, 2]

def chi2_test(model, popt, x, y_measured, sigma_y):
    chi2 = np.sum(((y_measured - model(x, *popt)) / sigma_y) ** 2)
    dof = len(x) - len(popt)
    return chi2, dof

def plot_fit(t, V, sigma_V, popt):
    t_sorted = np.sort(t)
    plt.errorbar(t, V, yerr=sigma_V, fmt="o", capsize=3, label="data")
    plt.plot(t_sorted, model_function(t_sorted, *popt), label=r"$V_0 e^{-t/RC}$")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Voltage (Volt)")
    plt.title("Voltage vs. Time: RC Circuit Discharge")
    plt.legend()
    plt.show()

def plot_residuals(t, V, sigma_V, popt):
    residuals = V - model_function(t, *popt)
    plt.errorbar(t, residuals, yerr=sigma_V, fmt="o", capsize=3)
    plt.axhline(0, color="black")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Residual (Volt)")
    plt.title("Residuals plot")
    plt.show()

t, V, sigma_V = load_data(data)

popt, pcov = curve_fit(model_function, t, V, p0=[V[0], t.max() / 3], sigma=sigma_V, absolute_sigma=True)
pstd = np.sqrt(np.diag(pcov))
chi_2, dof = chi2_test(model_function, popt, t, V, sigma_V)

print(f"V_0 = {popt[0]:} ± {pstd[0]:} V")
print(f"RC  = {popt[1]:} ± {pstd[1]:} µs")
print(f"chi2 = {chi_2:}, dof = {dof}, reduced chi2 = {chi_2 / dof:}")

plot_fit(t, V, sigma_V, popt)
plot_residuals(t, V, sigma_V, popt)
