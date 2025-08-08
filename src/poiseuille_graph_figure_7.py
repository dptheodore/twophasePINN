import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re

# ==== Config ====
mu1 = 4.0
base_dir = Path("./pinn_output") / f"mu1_{mu1}"
act_types = ["fixed", "adaptive"]
colors = {"fixed": "blue", "adaptive": "red"}

# Regex to extract MSE from a filename
mse_regex = re.compile(r'best_pinn_model_(\d+\.?\d*e[+-]?\d+)\.weights\.h5')

def find_best_model_path(act_dir: Path):
    """
    Finds the path to the best model's .weights.h5 file by recursively searching
    for the file with the lowest MSE value in its name.
    """
    weight_files = list(act_dir.rglob("best_pinn_model_*.weights.h5"))
    if not weight_files:
        return None, None

    min_mse = float('inf')
    best_file_path = None

    for f_path in weight_files:
        m = mse_regex.search(f_path.name)
        if m:
            try:
                v = float(m.group(1))
                if v < min_mse:
                    min_mse = v
                    best_file_path = f_path
            except ValueError:
                continue
    
    # Extract the final best MSE string from the best file path
    best_mse_str = None
    if best_file_path:
        m = mse_regex.search(best_file_path.name)
        if m:
            best_mse_str = m.group(1)

    return best_file_path, best_mse_str, min_mse

# --- Find the best model and its corresponding run directory ---
best_model_details = {}
for t in act_types:
    act_dir = base_dir / t
    path, mse_str, mse_val = find_best_model_path(act_dir)
    if path is None:
        raise FileNotFoundError(f"No 'best_pinn_model_*.weights.h5' files found under {act_dir}")
    
    best_model_details[t] = {
        'path': path,
        'mse_str': mse_str,
        'mse_val': mse_val,
        'run_dir': path.parent
    }
    print(f"{t}: best MSE = {mse_val:.4e} (from run: {path.parent.name})")

# --- Locate and load the history files from the correct run directory ---
loss_icbc = {}
loss_pde = {}
a_history = None

for t in act_types:
    details = best_model_details[t]
    run_dir = details['run_dir']
    mse_s = details['mse_str']

    # Construct the full, direct path to the history files
    icbc_path = run_dir / f"loss_boundary_initial_history_{mu1}_{t}_{mse_s}.npy"
    pde_path  = run_dir / f"loss_pde_history_{mu1}_{t}_{mse_s}.npy"

    if not icbc_path.is_file():
        raise FileNotFoundError(f"Could not find required file: {icbc_path}")
    if not pde_path.is_file():
        raise FileNotFoundError(f"Could not find required file: {pde_path}")

    loss_icbc[t] = np.load(icbc_path)
    loss_pde[t]  = np.load(pde_path)
    print(f"Loaded {icbc_path.name} ({loss_icbc[t].shape}), {pde_path.name} ({loss_pde[t].shape})")

# --- Adaptive-only a_history ---
adaptive_details = best_model_details['adaptive']
a_path = adaptive_details['run_dir'] / f"a_history_{mu1}_{adaptive_details['mse_str']}.npy"

if a_path.is_file():
    a_raw = np.load(a_path)
    # Take the mean of the absolute values if it's a 2D array
    a_history = np.mean(np.abs(a_raw), axis=1) if a_raw.ndim > 1 else a_raw
    print(f"Loaded a_history from {a_path} with shape {a_raw.shape} -> plotting shape {a_history.shape}")
else:
    print(f"Optional a_history file not found at {a_path} (skipping A subplot).")

# --- Prepare epoch axes ---
epochs_icbc = {t: np.arange(len(loss_icbc[t])) / 1e4 for t in act_types}
epochs_pde  = {t: np.arange(len(loss_pde[t])) / 1e4 for t in act_types}
epochs_a    = (np.arange(len(a_history)) / 1e4) if a_history is not None else None

# --- Plotting ---
fig, axes = plt.subplots(1, 3, figsize=(20, 8), sharex=True)
fig.suptitle(f"Loss History Comparison for $\mu_1 = {mu1}$", fontsize=16)
plt.subplots_adjust(top=0.92, hspace=0.3)


# 1) IC/BC Loss
axes[0].plot(epochs_icbc["fixed"], loss_icbc["fixed"], color=colors["fixed"], label="Fixed", zorder=1)
axes[0].plot(epochs_icbc["adaptive"], loss_icbc["adaptive"], color=colors["adaptive"], label="Adaptive", zorder=2)
axes[0].set_yscale('log')
axes[0].set_ylabel("Log MSE")
axes[0].set_title("Initial/Boundary Condition Loss")
axes[0].set_ylim(10e-5, 10e3)
axes[0].legend()

# 2) PDE Loss
axes[1].plot(epochs_pde["fixed"], loss_pde["fixed"], color=colors["fixed"], label="Fixed", zorder=1)
axes[1].plot(epochs_pde["adaptive"], loss_pde["adaptive"], color=colors["adaptive"], label="Adaptive", zorder=2)
axes[1].set_yscale('log')
axes[1].set_ylabel("Log MSE")
axes[1].set_ylim(10e-5, 10e1)
axes[1].set_title("PDE Residual Loss")
axes[1].legend()

# 3) Adaptive Coefficient 'a' History
if a_history is not None:
    axes[2].plot(epochs_a, a_history, color=colors["adaptive"], label=r"Mean of $|a|$")
    axes[2].set_ylabel("Value")
    axes[2].set_ylim(0.08, 0.11)  # Fixed y-range
    axes[2].set_title("Adaptive Activation Coefficient History")
    axes[2].legend()
else:
    axes[2].text(0.5, 0.5, 'No a_history data found', ha='center', va='center', transform=axes[2].transAxes)
    axes[2].set_title("Adaptive Activation Coefficient History")
    axes[2].set_ylim(0.08, 0.11) 

# --- Axis Formatting ---
axes[2].set_xlim(left=0)
axes[2].set_xlabel("Epochs (× 10⁴)")
for ax in axes:
    ax.grid(True, which='both', linestyle='--', linewidth=0.4)
    ax.set_xlim(-0.05, 2.05)  # Small white space left/right of 0 and 2

# --- Save and Show ---
plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust for suptitle
save_path = base_dir / f"loss_histories_mu1_{mu1}_comparison.png"
plt.savefig(save_path, dpi=300)
print(f"Saved comparison plot to {save_path}")
plt.show()