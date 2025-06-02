import numpy as np
import matplotlib.pyplot as plt

# Define the x range for plotting
x = np.linspace(0, 1, 500)

# Define the smooth function using tanh
y = 0.5 * (np.tanh((10 * (x - 0.3))) - np.tanh((10 * (x - 0.4))))

# Compute the derivative
dy_dx = 0.5 * (1 - np.tanh((10 * (x - 0.3)))**2) - 0.5 * (1 - np.tanh((10 * (x - 0.4)))**2)

# Create the plot with two subplots: function and derivative
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Plot the function
ax1.plot(x, y, 'b-', label='Smooth Square Wave Approximation (tanh)')
ax1.grid(True)
ax1.set_xlabel('x')
ax1.set_ylabel('u')
ax1.set_title('Smooth Boundary Condition with Large Gradients')
ax1.legend()
ax1.axvline(x=0.1, color='r', linestyle='--', alpha=0.3)
ax1.axvline(x=0.2, color='r', linestyle='--', alpha=0.3)
ax1.text(0.105, 0.5, 'Rising', rotation=90, color='r', alpha=0.5)
ax1.text(0.205, 0.5, 'Falling', rotation=90, color='r', alpha=0.5)

# Plot the derivative
ax2.plot(x, dy_dx, 'g-', label='Derivative of Smooth Function')
ax2.grid(True)
ax2.set_xlabel('x')
ax2.set_ylabel('du/dx')
ax2.set_title('Derivative of Smooth Boundary Condition')
ax2.legend()
ax2.axvline(x=0.1, color='r', linestyle='--', alpha=0.3)
ax2.axvline(x=0.2, color='r', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.show()