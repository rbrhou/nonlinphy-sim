import numpy as np
from scipy.optimize import curve_fit
from scipy import stats
import matplotlib.pyplot as plt

N_THEORY = 3 / 5

DATA_FILE = "Circuit Lightbulb-1.csv"


def load_data(path):
    # columns: Trial, Voltage, Current, Unc of Vol, Unc of Current, ...
    data = np.loadtxt(path, delimiter=",", skiprows=1, usecols=(1, 2, 4))
    return data[:, 0], data[:, 1], data[:, 2]


def model_fixed(V, k):
    return k * V**N_THEORY


def model_free(V, k, n):
    return k * V**n


def chi2_test(model, popt, V, I, sigma_I):
    chi2 = np.sum(((I - model(V, *popt)) / sigma_I) ** 2)
    dof = len(V) - len(popt)
    p = stats.chi2.sf(chi2, dof)
    return chi2, dof, p


def plot_fit(V, I, sigma_I, popt):
    Vg = np.linspace(V.min(), V.max(), 300)
    plt.errorbar(V, I, yerr=sigma_I, fmt="o", capsize=3, label="data")
    plt.plot(Vg, model_fixed(Vg, *popt), label=r"$I = kV^{3/5}$")
    plt.xlabel("V")
    plt.ylabel("I")
    plt.legend()
    plt.show()


V, I, sigma_I = load_data(DATA_FILE)

# Fixed exponent n = 3/5
p_fixed, cov_fixed = curve_fit(model_fixed, V, I, p0=[1.0], sigma=sigma_I, absolute_sigma=True)
err_fixed = np.sqrt(np.diag(cov_fixed))
chi2, dof, p = chi2_test(model_fixed, p_fixed, V, I, sigma_I)
print(f"k = {p_fixed[0]:.4f} ± {err_fixed[0]:.4f}")
print(f"chi^2 = {chi2:.2f}, dof = {dof}, chi^2/dof = {chi2 / dof:.2f}, p = {p:.3f}")

# Free exponent, compared with 3/5
p_free, cov_free = curve_fit(model_free, V, I, p0=[1.0, 0.5], sigma=sigma_I, absolute_sigma=True)
err_free = np.sqrt(np.diag(cov_free))
print(f"n = {p_free[1]:.4f} ± {err_free[1]:.4f}  (theory {N_THEORY})")

plot_fit(V, I, sigma_I, p_fixed)
