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
BASE_SAVE_DIR = Path("./pinn_output")
run_save_dir = BASE_SAVE_DIR / f"mu1_{mu1}" / ACTIVATION_TYPE

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


def find_best_weights(directory: Path) -> str:
    """
    Searches a directory for weight files and returns the path to the one
    with the lowest MSE value in its filename.
    """
    if not directory.is_dir():
        return None

    # Pattern to find files like 'best_pinn_model_1.2345e-05.weights.h5'
    pattern = str(directory / "best_pinn_model_*.weights.h5")
    weight_files = glob.glob(pattern)

    if not weight_files:
        return None

    min_mse = float('inf')
    best_file_path = None

    # Regex to extract the scientific notation number (MSE)
    # This looks for a floating point number in scientific notation
    mse_regex = re.compile(r'best_pinn_model_(\d+\.?\d*e[+-]?\d+)\.weights\.h5')

    for f_path in weight_files:
        filename = Path(f_path).name
        match = mse_regex.search(filename)
        if match:
            try:
                # Extract the MSE value (the first captured group)
                current_mse = float(match.group(1))
                if current_mse < min_mse:
                    min_mse = current_mse
                    best_file_path = f_path
            except (ValueError, IndexError):
                # Ignore files with malformed numbers
                continue

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
print(f"Searching for best model in: {run_save_dir}")
best_model_path = find_best_weights(run_save_dir)

if best_model_path is None:
    print(f"Error: No valid '.weights.h5' files found in '{run_save_dir}'.")
    print("Please ensure the training script has been run for this configuration.")
    sys.exit(1)

print(f"Found best model: {Path(best_model_path).name}\n")

# Instantiate and load the model
pinn_model = PINN(use_adaptive_activation=USE_ADAPTIVE_ACTIVATION)
pinn_model(tf.zeros((1, 3)))
pinn_model.load_weights(str(best_model_path))

print("Generating verification plot...")
# Evaluation line: x = 0, t = t_max, y ∈ [y_min, y_max]
N_plot = 100
y_plot = np.linspace(y_min, y_max, N_plot).astype(DTYPE)
t_plot = np.full_like(y_plot, t_max, dtype=DTYPE)
x_plot = np.zeros_like(y_plot, dtype=DTYPE)

# Model prediction at (t, x=0, y)
inputs = tf.convert_to_tensor(np.stack([t_plot, x_plot, y_plot], axis=1))
u_pred, _, _ = pinn_model(inputs)
u_pred = u_pred.numpy().flatten()

# Analytical steady-state solution
u_exact = analytical_solution(y_plot, mu1, mu2, -dp_dx)

# Normalize both by mean of u_exact
mean_u_exact = np.mean(u_exact)
u_pred_norm = u_pred / mean_u_exact
u_exact_norm = u_exact / mean_u_exact

# --- Calculate and Print Errors ---
MSE = mean_squared_error(u_exact_norm, u_pred_norm)
L1 = np.mean(np.abs(u_pred_norm - u_exact_norm)) / np.mean(np.abs(u_exact_norm))
L2 = np.linalg.norm(u_pred_norm - u_exact_norm) / np.linalg.norm(u_exact_norm)

print(f"--- Steady-State Error Metrics (Normalized by mean of u_exact) ---")
print(f"MSE: {MSE:.3e}")
print(f"L1 relative error: {L1:.3e}")
print(f"L2 relative error: {L2:.3e}")
print(f"Mean of u_exact: {mean_u_exact:.4f}, Mean of u_pred: {np.mean(u_pred):.4f}")

# --- Create and Save the Plot ---
plt.figure(figsize=(6, 5))
plt.plot(u_exact_norm, y_plot, 'b--', label='Analytical Solution', linewidth=2)

num_points = 20
indices = np.linspace(0, len(y_plot) - 1, num_points, dtype=int)
plt.plot(u_pred_norm[indices], y_plot[indices], 'o', color='b', markerfacecolor='none', markersize=8, label='PINN Prediction')

plt.axhline(0, color='k', linestyle=':', label='Interface')
plt.ylim(y_min - 0.05, y_max + 0.05)
plt.yticks(np.linspace(y_min, y_max, 11))
plt.xlim(-0.1, 2.1)
plt.xticks(np.arange(0, 2.1, 0.5))
plt.title(f'Steady-State Velocity Profile ($\\mu_1/\\mu_2 = {mu1/mu2}$)')
plt.xlabel('$u / \\bar{u}_{exact}$')
plt.ylabel('y')
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()

# Save the plot to the specific run directory
plot_save_path = run_save_dir / "steady_state_velocity_comparison.png"
plt.savefig(plot_save_path, dpi=300)
print(f"\nPlot saved to: {plot_save_path}")
plt.show()

# --- Save Plot Data to the Run-Specific Folder ---
plot_data_file = run_save_dir / f"steady_state_plot_data_{mu1}-{mu2}.npz"
np.savez_compressed(
    plot_data_file,
    y_coords=y_plot,
    u_exact_normalized=u_exact_norm,
    y_pinn_points=y_plot[indices],
    u_pinn_normalized=u_pred_norm[indices]
)
print(f"Plot data saved to: {plot_data_file}")