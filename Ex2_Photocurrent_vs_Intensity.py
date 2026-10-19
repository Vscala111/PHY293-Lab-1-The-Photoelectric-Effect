import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def power_model(x, a, b):
    return a * (x ** b)

x_data = np.array([33.3, 66.6, 100, 133.3, 166.6, 200], dtype=float)
y_data_v = np.array([23.5, 113.5, 166.5, 302.5, 406.5, 439.3], dtype=float)
x_err = np.array([0, 0, 0, 0, 0, 0], dtype=float)
y_err_v = np.array([1.3, 3.9, 3.9, 7.2, 6.2, 4.0], dtype=float)

y_data = y_data_v / (100 * 1000)
y_err = y_err_v / (100 * 1000)
# Initial guess a = 1, b = 1
popt, pcov = curve_fit(power_model, x_data, y_data, p0=(1, 1), sigma=y_err, absolute_sigma=True)
a_fit, b_fit = popt
a_err, b_err = np.sqrt(np.diag(pcov))

y_fit = power_model(x_data, a_fit, b_fit)
residuals = y_data - y_fit

chi2 = np.sum((residuals / y_err)**2)
dof = len(x_data) - 2
chi2_red = chi2 / dof

print(f"Fit Results: y = ({a_fit:.2f} ± {a_err:.2f}) * x^({b_fit:.2f} ± {b_err:.2f})")
print(f"Chi-squared: {chi2:.2f} (Reduced Chi-squared: {chi2_red:.2f})")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True, 
                               gridspec_kw={'height_ratios': [3, 1]})
fig.subplots_adjust(hspace=0.05)

ax1.errorbar(x_data, y_data, xerr=x_err, yerr=y_err, fmt='o', color='black', 
             ecolor='gray', capsize=3, label='Data with Errors')

x_fit = np.linspace(min(x_data), max(x_data), 100)
fit_label = f'Fit: $I_{{photocurrent}} = ({a_fit/1e-6:.2f} \\pm {a_err/1e-6:.2f}) I^{{{b_fit:.2f} \\pm {b_err:.2f}}} \\times 10^{{-6}}$ \n$\\chi^2 = {chi2:.2f}$ (red $\\chi^2$ = {chi2_red:.2f})'
ax1.plot(x_fit, power_model(x_fit, a_fit, b_fit), color='blue', linewidth=2, label=fit_label)

ax1.set_ylabel('Photocurrent (mA)', fontsize=24)
ax1.set_title('Photocurrent vs Intensity', fontsize=28)
ax1.legend(loc='upper left', fontsize=20)
ax1.grid(True, linestyle='--', alpha=0.5)

ax2.errorbar(x_data, residuals, yerr=y_err, fmt='o', color='black', 
             ecolor='black', capsize=3)
ax2.axhline(0, color='blue', linestyle='--', linewidth=1.5)

ax2.set_xlabel('Intensity (mA)', fontsize=24)
ax2.set_ylabel('Residuals', fontsize=24)
ax2.grid(True, linestyle='--', alpha=0.5)

plt.show()