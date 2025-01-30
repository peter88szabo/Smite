import random
import numpy as np
import matplotlib.pyplot as plt

def thermal_rot_symmetric_top(RT, Ixyz):
    Lrot = np.zeros(3)
    
    # Sample the Lz component
    rnd_gauss = random.gauss(0.0, 1.0)
    Lrot[2] = np.sqrt(Ixyz[2] * RT) * rnd_gauss
    
    return Lrot

RT = 1.0
Ixyz = [2.0, 2.0, 3.0]
num_samples = 1000

samples_Lz = [thermal_rot_symmetric_top(RT, Ixyz)[2] for _ in range(num_samples)]

# Generate Lz values for the theoretical distribution
Lz_vals = np.linspace(min(samples_Lz), max(samples_Lz), 500)
Iz = Ixyz[2]
P_Lz = np.exp(-Lz_vals**2 / (2 * Iz * RT)) / np.sqrt(2 * np.pi * Iz * RT)  # Normalized Gaussian

plt.figure(figsize=(8, 5))
plt.hist(samples_Lz, bins=30, density=True, alpha=0.6, color='skyblue', label='Sampled Lz Histogram')
plt.plot(Lz_vals, P_Lz, 'r-', linewidth=2, label='Theoretical Distribution (P(Lz))')

plt.xlabel('Lz')
plt.ylabel('Probability Density')
plt.title('Comparison of Sampled and Theoretical Distributions of Lz')
plt.legend()
plt.show()

