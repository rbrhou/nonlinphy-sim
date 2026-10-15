import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

data = np.array([
    (1, 16), (2, 12), (3, 14), (4, 8), (5, 24), (6, 92), (7, 311), (8, 243),
    (9, 558), (10, 644), (11, 768), (12, 838), (13, 911), (14, 897), (15, 853),
    (16, 860), (17, 853), (18, 875), (19, 673), (20, 378), (21, 207), (22, 142),
    (23, 108), (24, 62),
], dtype=float)


def model_function(x, a_1, a_2, a_3):
    """
    y = a_1 * exp(-(x - a_2)^2 / a_3)
    """
    return a_1 * np.exp(-(x - a_2)**2 / a_3)
    
def load_data(data):
    x = data[:, 0]
    y = data[:, 1]
    sigma_y = np.sqrt(y)
    return x, y, sigma_y

def chi2_test(model, popt, x, y_measured, sigma_y):
    chi2 = np.sum(((y_measured - model(x, *popt)) / sigma_y) ** 2)
    dof = len(x) - len(popt)
    return chi2, dof

def plot_fit(x, y, sigma_y, popt):
    plt.errorbar(x, y, yerr=sigma_y, fmt='o', label="data")
    plt.plot(x, model_function(x, *popt), label=r"$a_1 e^{-(x - a_2)^2 / a_3}$")
    plt.xlabel("Hour of the day")
    plt.ylabel("Number of trucks")
    plt.title("Trucks per hour: Gaussian model vs. data")
    plt.legend()
    plt.show()

def plot_residuals(x, y, sigma_y, popt):
    residuals = y - model_function(x, *popt)
    plt.errorbar(x, residuals, yerr=sigma_y, fmt="o")
    plt.axhline(0, color="black")
    plt.xlabel("Hour of the day")
    plt.ylabel("Residual (trucks)")
    plt.title("Residuals plot")
    plt.show()
    
    
x, y, sigma_y = load_data(data)

popt, pcov = curve_fit(model_function, x, y, sigma=sigma_y, absolute_sigma=True)
pstd = np.sqrt(np.diag(pcov))

chi_2, dof = chi2_test(model_function, popt, x, y, sigma_y)

print(f"a_1 = {popt[0]:.2f} ± {pstd[0]:.2f} trucks")
print(f"a_2 = {popt[1]:.2f} ± {pstd[1]:.2f} hours")
print(f"a_3 = {popt[2]:.2f} ± {pstd[2]:.2f} hours^2")
print(f"chi2 = {chi_2:.1f}, dof = {dof}, reduced chi2 = {chi_2 / dof:.2f}")

plot_fit(x, y, sigma_y, popt)
plot_residuals(x, y, sigma_y, popt)
