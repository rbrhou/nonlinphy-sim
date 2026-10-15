import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


data = "data"


c = 2.99792458e8             # speed of light (m/s)
e = 1.602176634e-19          # elementary charge (C)
h_accepted = 6.62607015e-34  # accepted Planck constant (J s)


def model_function(f, h_over_e, V_offset):
    """Threshold voltage vs. frequency: eV = hf - E_loss  ->  V = (h/e) f + V_offset."""
    return h_over_e * f + V_offset

def load_data(path):
    raw = np.loadtxt(path, delimiter=",", skiprows=1, usecols=(0, 1, 2, 4))
    lam = raw[:, 0] * 1e-9             # nm -> m
    V = raw[:, 1]
    sigma_V = raw[:, 2]
    sigma_lam = raw[:, 3] * 1e-9       # nm -> m
    f = c / lam                        # Hz
    sigma_f = c * sigma_lam / lam**2   # propagated: |df/dlam| * sigma_lam
    return f, sigma_f, V, sigma_V

def effective_sigma(popt, sigma_f, sigma_V):
    """Fold the frequency uncertainty into the voltage uncertainty: slope * sigma_f."""
    return np.sqrt(sigma_V**2 + (popt[0] * sigma_f) ** 2)

def chi2_test(model, popt, x, y_measured, sigma_y):
    chi2 = np.sum(((y_measured - model(x, *popt)) / sigma_y) ** 2)
    dof = len(x) - len(popt)
    return chi2, dof

def plot_fit(f, sigma_f, V, sigma_V, popt):
    f_line = np.linspace(f.min() * 0.97, f.max() * 1.03, 200)
    plt.errorbar(f / 1e14, V, xerr=sigma_f / 1e14, yerr=sigma_V, fmt="o", capsize=3, label="data")
    plt.plot(f_line / 1e14, model_function(f_line, *popt), label=r"$V = (h/e)\,f + V_{\text{intercept}}$")
    plt.xlabel(r"Frequency ($10^{14}$ Hz)")
    plt.ylabel("Threshold voltage (V)")
    plt.title("LED Threshold Voltage vs. Frequency")
    plt.legend()
    plt.show()

def plot_residuals(f, V, sigma_eff, popt):
    residuals = V - model_function(f, *popt)
    plt.errorbar(f / 1e14, residuals, yerr=sigma_eff, fmt="o", capsize=3)
    plt.axhline(0, color="black")
    plt.xlabel(r"Frequency ($10^{14}$ Hz)")
    plt.ylabel("Residual (V)")
    plt.title("Residuals plot")
    plt.show()


f, sigma_f, V, sigma_V = load_data(data)

popt, pcov = curve_fit(model_function, f, V, p0=[4e-15, 0.0], sigma=sigma_V, absolute_sigma=True)

for _ in range(50):
    sigma_eff = effective_sigma(popt, sigma_f, sigma_V)
    popt, pcov = curve_fit(model_function, f, V, p0=popt, sigma=sigma_eff, absolute_sigma=True)

pstd = np.sqrt(np.diag(pcov))
sigma_eff = effective_sigma(popt, sigma_f, sigma_V)
chi_2, dof = chi2_test(model_function, popt, f, V, sigma_eff)

h = popt[0] * e
sigma_h = pstd[0] * e
n_sigma = abs(h - h_accepted) / sigma_h

print(f"h/e      = {popt[0]:.4e} ± {pstd[0]:.4e} V s")
print(f"V_offset = {popt[1]:.4f} ± {pstd[1]:.4f} V")
print(f"h        = {h:.4e} ± {sigma_h:.4e} J s")
print(f"accepted h = {h_accepted:.5e} J s,  {n_sigma:.2f} sigma from accepted value")
print(f"chi2 = {chi_2:.3f}, dof = {dof}, reduced chi2 = {chi_2 / dof:.3f}")

plot_fit(f, sigma_f, V, sigma_V, popt)
plot_residuals(f, V, sigma_eff, popt)
