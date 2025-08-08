import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# --- Config ---
activation_type = "fixed"  # or "fixed"
mu_values = [2.0, 4.0, 6.0]
colors = {2: 'green', 4: 'blue', 6: 'red'}

# --- Plot ---
plt.figure(figsize=(7, 6))

for mu in mu_values:
    file_path = Path(f'./pinn_output/mu1_{mu}/{activation_type}/plot_data_mu1_{mu}_{activation_type}.npz')
    data = np.load(file_path)
    
    y_coords = data['y_coords']
    u_exact_norm = data['u_exact_normalized']
    u_pinn_norm = data['u_pinn_normalized']
    
    # Line (predicted)
    plt.plot(u_pinn_norm, y_coords, '-', color=colors[mu],
             linewidth=2, label=f'μ₁={mu} PINN')
    
    # Circles (exact points)
    num_points = 20
    indices = np.linspace(0, len(y_coords) - 1, num_points, dtype=int)
    plt.plot(u_exact_norm[indices], y_coords[indices], 'o',
             color=colors[mu], markerfacecolor='none',
             markersize=6, markeredgewidth=1.5,
             label=f'μ₁={mu} Analytical')

# Axis and formatting
plt.axhline(0, color='k', linestyle=':', label='Interface')
plt.ylim(-0.55, 0.55)
plt.yticks(np.linspace(-0.5, 0.5, 11))
plt.xlim(-0.1, 2.1)
plt.xticks(np.arange(0, 2.1, 0.5))
plt.title(f'Steady-State Velocity Profiles ({activation_type})')
plt.xlabel('$u / \\bar{u}_{exact}$')
plt.ylabel('y')
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()

# Save
plot_save_path = Path(f'./pinn_output/combined_profiles_{activation_type}.png')
plt.savefig(plot_save_path, dpi=300)
print(f"Plot saved to: {plot_save_path}")

plt.show()