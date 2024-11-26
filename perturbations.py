# Importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# Constants
c = 3e8  # Speed of light (in m/s)
Z = 1.69  # Effective nuclear charge for Helium-4
a = 5.29177e-11  # Bohr radius (in meters)
G = 6.67430e-11  # Gravitational constant (in m^3/kg/s^2)
rho_0 = 145  # Density of Helium-4 (in kg/m^3)
lambda_ = 0.5e-9  # de Broglie wavelength (in meters)
h = 6.626e-34  # Planck's constant (in J·s)

# Prefactor for numerical integration (adjusted for energy E' instead of E'/p)
pre_factor = ((Z**3) / (np.pi * a**3))**2 * (np.pi / 3) * G * (h / (lambda_ * c)) * rho_0 * (4 * np.pi)**2

# Define the inner integral over r_2
def inner_integral(r_1, a, Z):
    integrand = lambda r_2: (1 + (r_2 / r_1)**2) * np.exp(-2 * Z * r_2 / a) * r_2**2
    result, _ = quad(integrand, 0, r_1)
    return result

# Define the outer integral over r_1
def energy_integral(r_1, a, Z):
    return r_1**4 * np.exp(-2 * Z * r_1 / a) * inner_integral(r_1, a, Z)

# Numerical integration for a range of r_1 values
r_1_values = np.linspace(1e-12, 1e-9, 500)  # r_1 from 1 pm to 1 nm
E_values = [pre_factor * energy_integral(r_1, a, Z) for r_1 in r_1_values]

# Convert numerical energy values to eV
E_values_eV = [E * 6.242e18 for E in E_values]

# Analytical approximation for energy E'
analytical_value = pre_factor * (15 * lambda_**8) / (32 * Z**8)
analytical_value_eV = analytical_value * 6.242e18

# Plot the result with larger text sizes
plt.figure(figsize=(10, 6))
plt.plot(r_1_values * 1e9, E_values_eV, label=r"Numerical $E'$", linewidth=2, color='blue')
plt.axhline(analytical_value_eV, color='red', linestyle='--', label="Analytical Approximation", linewidth=1.5)

# Adjusting text sizes
plt.title(r"Energy Correction $E'$ vs $r_1$", fontsize=18)
plt.xlabel(r"$r_1$ (in nm)", fontsize=16)
plt.ylabel(r"Energy Correction $E'$ (eV)", fontsize=16)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=16)  # Larger legend text
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()

# Display the plot
plt.show()
