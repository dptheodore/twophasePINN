import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import mean_squared_error
import glob
import re
import sys

# Set data type for TensorFlow and NumPy
DTYPE = 'float32'
tf.keras.backend.set_floatx(DTYPE)

# --- Configuration (CHANGE THESE BASED ON WHAT YOU WANT TO PLOT)---
mu1 = 2.0
USE_ADAPTIVE_ACTIVATION = False
# ---------------------

# --- Path Setup ---
ACTIVATION_TYPE = "adaptive" if USE_ADAPTIVE_ACTIVATION else "fixed"
# This is the base directory for a specific experiment configuration
experiment_dir = Path("./pinn_output") / f"mu1_{mu1}" / ACTIVATION_TYPE

# =============================================================================
# 1. PROBLEM PARAMETERS
# =============================================================================
t_min, t_max = 0.0, 0.5
x_min, x_max = -0.1, 0.1
y_min, y_max = -0.5, 0.5

mu2 = 1.0
rho1 = 1.0
rho2 = 1.0
p_W = 3.4
p_E = 1.0
dp_dx = (p_E - p_W) / (x_max - x_min)
Re = 1.0 / mu2


def find_best_model_across_runs(directory: Path) -> str | None:
    """
    Searches recursively through all subdirectories (timestamped runs) of a given
    experiment directory to find the single best model weight file, determined
    by the lowest MSE value in the filename.
    """
    if not directory.is_dir():
        return None

    # Use rglob to search recursively in all subdirectories for the weight files.
    # The pattern '**/...' matches in the current directory and all subdirectories.
    weight_files = list(directory.rglob("best_pinn_model_*.weights.h5"))

    if not weight_files:
        return None

    min_mse = float('inf')
    best_file_path = None

    # Regex to extract the scientific notation number (MSE) from the filename
    mse_regex = re.compile(r'best_pinn_model_(\d+\.?\d*e[+-]?\d+)\.weights\.h5')

    for f_path in weight_files:
        match = mse_regex.search(f_path.name)
        if match:
            try:
                current_mse = float(match.group(1))
                if current_mse < min_mse:
                    min_mse = current_mse
                    best_file_path = str(f_path)
            except (ValueError, IndexError):
                print(f"Warning: Could not parse MSE from filename: {f_path.name}")
                continue
    
    if best_file_path:
        print(f"Found best model across all runs with MSE: {min_mse:.4e}")

    return best_file_path

# =============================================================================
# 2. NEURAL NETWORK MODEL AND HELPER FUNCTIONS
# =============================================================================
class PINN(tf.keras.Model):
    def __init__(self, use_adaptive_activation=True, **kwargs):
        super().__init__(**kwargs)
        self.use_adaptive_activation = use_adaptive_activation
        initializer = tf.keras.initializers.GlorotNormal()
        n_hidden_layers = 10
        n_neurons = 100
        self.hidden_layers = [
            tf.keras.layers.Dense(n_neurons, activation=None, kernel_initializer=initializer, bias_initializer='zeros')
            for _ in range(n_hidden_layers)
        ]
        self.u_output_layer = tf.keras.layers.Dense(1, activation='linear', kernel_initializer=initializer, bias_initializer='zeros')
        self.p_output_layer = tf.keras.layers.Dense(1, activation='exponential', kernel_initializer=initializer, bias_initializer='zeros')
        self.alpha_output_layer = tf.keras.layers.Dense(1, activation='sigmoid', kernel_initializer=initializer, bias_initializer='zeros')
        if self.use_adaptive_activation:
            self.n = tf.constant(10.0, dtype=DTYPE)
            self.a_s = []
            initial_a_value = 0.1
            a_initializer = tf.keras.initializers.Constant(initial_a_value)
            for i in range(n_hidden_layers):
                a = self.add_weight(name=f'a_{i}', shape=(n_neurons,), initializer=a_initializer, trainable=True)
                self.a_s.append(a)

    def call(self, inputs):
        x = inputs
        for i, layer in enumerate(self.hidden_layers):
            x = layer(x)
            if self.use_adaptive_activation:
                a = self.a_s[i]
                x = tf.tanh(self.n * a * x)
            else:
                x = tf.tanh(x)
        u = self.u_output_layer(x)
        p = self.p_output_layer(x)
        alpha = self.alpha_output_layer(x)
        return u, p, alpha

def analytical_solution(y, mu_1, mu_2, pressure_drop):
    b = y_max
    A1 = (-pressure_drop * b / (2 * mu_1)) * ((mu_1 - mu_2) / (mu_1 + mu_2))
    A2 = (-pressure_drop * b / (2 * mu_2)) * ((mu_1 - mu_2) / (mu_1 + mu_2))
    B  = pressure_drop * b**2 / (mu_1 + mu_2)
    u_analytical = np.zeros_like(y, dtype=np.float32)
    mask1, mask2 = y >= 0, y < 0
    u_analytical[mask1] = (-pressure_drop/(2*mu_1))*y[mask1]**2 + A1*y[mask1] + B
    u_analytical[mask2] = (-pressure_drop/(2*mu_2))*y[mask2]**2 + A2*y[mask2] + B
    return u_analytical

# =============================================================================
# 3. MAIN EXECUTION: VISUALIZATION
# =============================================================================
print(f"Searching for the best model in all subdirectories of: {experiment_dir}")
best_model_path = find_best_model_across_runs(experiment_dir)

if best_model_path is None:
    print(f"Error: No valid '.weights.h5' files found in any subdirectories of '{experiment_dir}'.")
    print("Please ensure the training script has been run successfully for this configuration.")
    sys.exit(1)

print(f"Loading best model: {Path(best_model_path).name}\n")

# Instantiate and build the model before loading weights
pinn_model = PINN(use_adaptive_activation=USE_ADAPTIVE_ACTIVATION)
pinn_model.build(input_shape=(None, 3)) 
pinn_model.load_weights(best_model_path)

print("Generating verification plot...")
# Evaluation line: x = 0, t = t_max, y ∈ [y_min, y_max]
N_plot = 100
y_plot = np.linspace(y_min, y_max, N_plot).astype(DTYPE)
t_plot = np.full_like(y_plot, t_max, dtype=DTYPE)
x_plot = np.zeros_like(y_plot, dtype=DTYPE)

# Model prediction at (t_max, x=0, y)
inputs = tf.convert_to_tensor(np.stack([t_plot, x_plot, y_plot], axis=1))
u_pred, _, _ = pinn_model(inputs)
u_pred = u_pred.numpy().flatten()

# Analytical steady-state solution
u_exact = analytical_solution(y_plot, mu1, mu2, -dp_dx)

# Normalize both profiles by the mean of the *exact* solution for consistent comparison
mean_u_exact = np.mean(u_exact)
u_pred_norm = u_pred / mean_u_exact
u_exact_norm = u_exact / mean_u_exact

# --- Calculate and Print Errors ---
MSE = mean_squared_error(u_exact_norm, u_pred_norm)
L1_rel = np.mean(np.abs(u_pred_norm - u_exact_norm)) / np.mean(np.abs(u_exact_norm))
L2_rel = np.linalg.norm(u_pred_norm - u_exact_norm) / np.linalg.norm(u_exact_norm)

print(f"--- Steady-State Error Metrics (Normalized by mean of u_exact) ---")
print(f"MSE: {MSE:.3e}")
print(f"L1 relative error: {L1_rel:.3e}")
print(f"L2 relative error: {L2_rel:.3e}")
print(f"Mean of u_exact: {mean_u_exact:.4f}, Mean of u_pred: {np.mean(u_pred):.4f}")

# --- Create and Save the Plot ---
plt.figure(figsize=(7, 6))
# Plot analytical solution as a dashed line
plt.plot(u_exact_norm, y_plot, 'b--', label='Analytical Solution', linewidth=2.5)
# Plot PINN prediction as a solid line
plt.plot(u_pred_norm, y_plot, 'r-', label='PINN Prediction (Line)', linewidth=2)

# Plot PINN prediction as discrete points for emphasis
num_points = 20
indices = np.linspace(0, len(y_plot) - 1, num_points, dtype=int)
plt.plot(u_pred_norm[indices], y_plot[indices], 'o', color='red', markerfacecolor='none', markersize=8, markeredgewidth=1.5, label='PINN Prediction (Points)')

plt.axhline(0, color='k', linestyle=':', label='Interface')
plt.ylim(y_min - 0.05, y_max + 0.05)
plt.yticks(np.linspace(y_min, y_max, 11))
plt.xlim(-0.1, 2.1)
plt.xticks(np.arange(0, 2.1, 0.5))
plt.title(f'Steady-State Velocity Profile ($\\mu_1/\\mu_2 = {mu1/mu2}$, {ACTIVATION_TYPE})')
plt.xlabel('$u / \\bar{u}_{exact}$')
plt.ylabel('y')
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()

# Save the plot to the main experiment directory with a unique name
plot_filename = f"steady_state_comparison_mu1_{mu1}_{ACTIVATION_TYPE}.png"
plot_save_path = experiment_dir / plot_filename
plt.savefig(plot_save_path, dpi=300)
print(f"\nPlot saved to: {plot_save_path}")
plt.show()

# --- Save Plot Data to the main experiment directory ---
data_filename = f"plot_data_mu1_{mu1}_{ACTIVATION_TYPE}.npz"
plot_data_file = experiment_dir / data_filename
np.savez_compressed(
    plot_data_file,
    y_coords=y_plot,
    u_exact_normalized=u_exact_norm,
    u_pinn_normalized=u_pred_norm
)
print(f"Plot data saved to: {plot_data_file}")