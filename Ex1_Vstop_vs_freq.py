import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def linear_model(x, m, b):
    return m * x + b

x_data_wave = np.array([395, 455, 508, 530, 590, 617, 660], dtype=float)
y_data = np.array([1.378, 1.020, 0.774, 0.791, 0.622, 0.689, 0.635], dtype=float)
x_err_wave = np.array([16, 18, 12, 35, 15, 16, 10], dtype=float)
y_err = np.array([0.048, 0.052, 0.026, 0.027, 0.048, 0.060, 0.087], dtype=float)

c = 3.00e8
x_data = c / (x_data_wave * 1e-9)
x_err = (c / ((x_data_wave * 1e-9) ** 2)) * (x_err_wave * 1e-9)

popt, pcov = curve_fit(linear_model, x_data, y_data, sigma=y_err, absolute_sigma=True)
m_fit, b_fit = popt

sigma_tot = np.sqrt(y_err**2 + (m_fit * x_err)**2)

popt, pcov = curve_fit(linear_model, x_data, y_data, sigma=sigma_tot, absolute_sigma=True)
m_fit, b_fit = popt
m_err, b_err = np.sqrt(np.diag(pcov))

y_fit = linear_model(x_data, m_fit, b_fit)
residuals = y_data - y_fit

chi2 = np.sum((residuals / sigma_tot)**2)
dof = len(x_data) - 2
chi2_red = chi2 / dof

print(f"Fit Results: m = ({m_fit:.3e} ± {m_err:.3e}) V·s")
print(f"             b = ({b_fit:.3f} ± {b_err:.3f}) V")
print(f"Chi-squared: {chi2:.2f} (Reduced Chi-squared: {chi2_red:.2f})")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True, 
                               gridspec_kw={'height_ratios': [3, 1]})
fig.subplots_adjust(hspace=0.05)

ax1.errorbar(x_data, y_data, xerr=x_err, yerr=y_err, fmt='o', color='black', 
             ecolor='gray', capsize=3, label='Data with Errors')

x_fit = np.linspace(min(x_data), max(x_data), 100)
fit_label = f'Fit: $V = ({m_fit/1e-15:.2f} \\pm {m_err/1e-15:.2f}) \\times 10^{{-15}} f + ({b_fit:.2f} \\pm {b_err:.2f})$\n$\\chi^2 = {chi2:.2f}$ (red $\\chi^2$ = {chi2_red:.2f})'
ax1.plot(x_fit, linear_model(x_fit, m_fit, b_fit), color='blue', linewidth=2, label=fit_label)

ax1.set_ylabel('Stopping Voltage (Volts)', fontsize=24)
ax1.set_title('Stopping Voltage vs Frequency', fontsize=28)
ax1.legend(loc='upper left', fontsize=20)
ax1.grid(True, linestyle='--', alpha=0.5)

ax2.errorbar(x_data, residuals, yerr=sigma_tot, fmt='o', color='black', 
             ecolor='black', capsize=3)
ax2.axhline(0, color='blue', linestyle='--', linewidth=1.5)

ax2.set_xlabel('Frequency (Hz)', fontsize=24)
ax2.set_ylabel('Residuals (V)', fontsize=24)
ax2.grid(True, linestyle='--', alpha=0.5)

plt.show()