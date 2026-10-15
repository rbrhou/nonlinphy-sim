import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

n_theory = 3/5

data = "Circuit Lightbulb - 1.csv"

def load_data(path):
    data = np.loadtxt(path, delimiter=",", skiprows=1, usecols=(1, 2, 4))
    return data[:, 0], data[:, 1], data[:, 2]


def model_fixed(V, k):
    return k * V**n_theory


def model_free(V, k, n):
    return k * V**n


def chi2_test(model, popt, V, I, sigma_I):
    chi2 = np.sum(((I - model(V, *popt)) / sigma_I) ** 2)
    dof = 8
    return chi2, dof


def plot_fit(V, I, sigma_I, popt):
    V_sorted = np.sort(V)
    plt.errorbar(V, I, yerr=sigma_I, fmt="o", capsize=3, label="data")
    plt.plot(V_sorted, model_fixed(V_sorted, *popt), label=r"$I = kV^{3/5}$")
    plt.xlabel("V")
    plt.ylabel("I")
    plt.title("Parameters fitting plot")
    plt.legend()
    plt.show()
    
def plot_residuals(V, I, sigma_I, popt):
    residuals = (I - model_fixed(V, *popt)) / sigma_I
    plt.plot(V, residuals, "o")
    plt.errorbar(V, residuals, yerr=np.ones_like(V), fmt=".", capsize=3)
    plt.xlabel("V")
    plt.ylabel(r"$\text{Normalized residuals } (I - kV^{3/5}) / \sigma_I$")
    plt.title("residuals plot")
    plt.show()
 
 
def plot_loglog(V, I, sigma_I, popt):
    V_sorted = np.sort(V)
    plt.errorbar(np.log(V), np.log(I), yerr=sigma_I / I, fmt="o", capsize=3, label="data")
    plt.plot(np.log(V_sorted), np.log(model_fixed(V_sorted, *popt)), label="slope 3/5")
    plt.xlabel(r"$\ln V$")
    plt.ylabel(r"$\ln I$")
    plt.title("Loglog plot")
    plt.legend()
    plt.show()


V, I, sigma_I = load_data(data)

p_fixed, cov_fixed = curve_fit(model_fixed, V, I, p0=[1.0], sigma=sigma_I, absolute_sigma=True)
err_fixed = np.sqrt(np.diag(cov_fixed))
chi2, dof = chi2_test(model_fixed, p_fixed, V, I, sigma_I)
print(f"k = {p_fixed[0]} ± {err_fixed[0]}")
print(f"chi^2 = {chi2}, dof = {dof}, chi^2/dof = {chi2 / dof}")

p_free, cov_free = curve_fit(model_free, V, I, p0=[1.0, 0.5], sigma=sigma_I, absolute_sigma=True)
err_free = np.sqrt(np.diag(cov_free))
print(f"n = {p_free[1]} ± {err_free[1]}")

plot_fit(V, I, sigma_I, p_fixed)
plot_residuals(V, I, sigma_I, p_fixed)
plot_loglog(V, I, sigma_I, p_fixed)
