import numpy as np
import matplotlib.pyplot as plt

x_data_wave = np.array([395, 455, 508, 530, 590, 617, 660], dtype=float)
y_data_v = np.array([87.3, 57.3, 26.8, 24.3, 1.63, 3.8, 0.48], dtype=float)
x_err_wave = np.array([16, 18, 12, 35, 15, 16, 10], dtype=float)
y_err_v = np.array([30.3, 10.9, 5.7, 7.2, 0.26, 0.82, 0.15], dtype=float)

c = 3.00e8
x_data = c / (x_data_wave * 1e-9)
y_data = y_data_v / (100 * 1000)
x_err = (c / ((x_data_wave * 1e-9) ** 2)) * (x_err_wave * 1e-9)
y_err = y_err_v / (100 * 1000)

fig, ax = plt.subplots(figsize=(8, 6))

ax.errorbar(x_data, y_data, xerr=x_err, yerr=y_err, fmt='o', color='black', 
            ecolor='gray', capsize=3, label='Data with errors')

ax.set_xlabel('Frequency (Hz)', fontsize=24)
ax.set_ylabel('Photocurrent (mA)', fontsize=24)
ax.set_title('Photocurrent vs Frequency', fontsize=28)
ax.legend(loc='upper left', fontsize = 20)
ax.grid(True, linestyle='--', alpha=0.5)

plt.show()