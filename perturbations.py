import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# Constants
c = 3e8  # Speed of light (in m/s)
Z = 1.69  # Atomic number for Helium-4 (or another constant you prefer)
a = 5.29177e-11  # Bohr radius (in meters)
G = 6.67430e-11  # Gravitational constant (in m^3/kg/s^2)
rho_0 = 1e-27  # Example density (in kg/m^3)
pre_factor = c * ((Z**3) / (np.pi * a**3))**2 *G* (np.pi / 3) * rho_0 * (4 * np.pi)**2

# Define the inner integral over r_2
def inner_integral(r_1, a, Z):
    integrand = lambda r_2: (1 + (r_2 / r_1)**2) * np.exp(-2 * Z * r_2 / a) * r_2**2
    result, _ = quad(integrand, 0, r_1)  # Perform numerical integration
    return result

# Define the outer integral over r_1
def energy_per_momentum_integral(r_1, a, Z):
    return r_1**4 * np.exp(-2 * Z * r_1 / a) * inner_integral(r_1, a, Z)

# Generate r_1 values and compute the energy correction per unit momentum
r_1_values = np.linspace(0, 1e-9, 500)  # r_1 from 0 to 1 nm
E_per_p_values = [pre_factor * energy_per_momentum_integral(r_1, a, Z) for r_1 in r_1_values]

# Plot the result
plt.figure(figsize=(8, 6))
plt.plot(r_1_values, E_per_p_values, label=r"$\frac{E'}{p}$")
plt.title(r"Energy Correction per Unit Momentum vs $r_1$")
plt.xlabel(r"$r_1$ (in meters)")
plt.ylabel(r"Energy per Unit Momentum $\frac{E'}{p}$ (J·s/m)")
plt.grid(True)
plt.legend()
plt.show()