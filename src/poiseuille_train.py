import os
import glob
import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import mean_squared_error

# Set data type for TensorFlow and NumPy
DTYPE = 'float32'
tf.keras.backend.set_floatx(DTYPE)

# --- Configuration (CHANGE THESE BASED ON WHAT YOU WANT TO PLOT)---
mu1 = 6.0
USE_ADAPTIVE_ACTIVATION = False
tf.random.set_seed(1234)
w_ic, w_bc, w_f = 1.0, 1.0, 1.0 # According to paper, these are a flat 1.0, we can tweak these in the future
best_weights_file_path = "" # declared later in __main__
# =============================================================================
# 1. PROBLEM PARAMETERS AND DOMAIN DEFINITION
# =============================================================================
# As specified in Section 3.1.2 of Buhendwa et al. (2021)
t_min, t_max = 0.0, 0.5
x_min, x_max = -0.1, 0.1
y_min, y_max = -0.5, 0.5

# Fluid properties (viscosity ratio can be varied as in the paper)
# Using mu1/mu2 = 2 as an example case from the paper
mu2 = 1.0  # Fluid 2
rho1 = 1.0
rho2 = 1.0

# Pressure gradient parameters
p_W = 3.4
p_E = 1.0
dp_dx = (p_E - p_W) / (x_max - x_min) # Should be -12

# Reynolds number is based on reference quantities, all set to 1.0
# Re = rho_r * u_r * L_r / mu_r. With all ref quantities = 1, Re = 1/mu_r.
# We will use mu2 as the reference viscosity.
Re = 1.0 / mu2


def plot_initial_data(data):
    """
    Plots spatial distribution of training points at t=0.
    CORRECTED to handle tensor shapes correctly during masking.
    """
    print("Plotting initial data...")
    # --- 1. Extract and Filter Data ---
    (t_ic, x_ic, y_ic) = data['ic']
    (_, _, alpha_ic) = data['ic_targets']
    
    x_ic_plot = x_ic.numpy().flatten()
    y_ic_plot = y_ic.numpy().flatten()
    alpha_ic_plot = alpha_ic.numpy().flatten()
    colors = np.where(alpha_ic_plot > 0.5, 'green', 'purple')

    # Extract boundary points that exist at t=0
    bc_t0_coords = []
    
    # North/South Walls
    (t_ns, x_ns, y_ns) = data['ns']

    mask_ns_t0 = tf.squeeze(tf.equal(t_ns, 0.0))
    if tf.reduce_any(mask_ns_t0):
        x_ns_t0 = tf.boolean_mask(x_ns, mask_ns_t0)
        y_ns_t0 = tf.boolean_mask(y_ns, mask_ns_t0)
        bc_t0_coords.append(tf.concat([x_ns_t0, y_ns_t0], axis=1))

    # East/West Walls
    (t_ew, x_ew_west, x_ew_east, y_ew) = data['ew']

    mask_ew_t0 = tf.squeeze(tf.equal(t_ew, 0.0))
    if tf.reduce_any(mask_ew_t0):
        y_ew_t0 = tf.boolean_mask(y_ew, mask_ew_t0)
        # West boundary at t=0
        x_ew_west_t0 = tf.boolean_mask(x_ew_west, mask_ew_t0)
        bc_t0_coords.append(tf.concat([x_ew_west_t0, y_ew_t0], axis=1))
        # East boundary at t=0
        x_ew_east_t0 = tf.boolean_mask(x_ew_east, mask_ew_t0)
        bc_t0_coords.append(tf.concat([x_ew_east_t0, y_ew_t0], axis=1))
    
    # --- 2. Create Plot ---
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(x_ic_plot, y_ic_plot, c=colors, s=10, label='Interior IC')
    if bc_t0_coords:
        bc_t0 = np.vstack([b.numpy() for b in bc_t0_coords])
        x_bc, y_bc = bc_t0[:,0], bc_t0[:,1]
        ax.scatter(x_bc, y_bc, c='red', s=15, marker='x', label='Boundary (t=0)')

    # --- 3. Style Plot ---
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect((x_max - x_min) / (y_max - y_min))
    ax.set_title("Training Points at t=0")
    ax.legend()
    plt.tight_layout()
    plt.savefig(run_save_dir / "initial_data.png", dpi=150)
    plt.close(fig) # Close figure to free memory


def plot_residual_points_t0(data):
    """
    Plots the spatial distribution of the residual (collocation) points at t=0.
    """
    print("Plotting residual points at t=0...")
    t_f, x_f, y_f = data['f']
    mask_t0 = tf.equal(t_f, 0.0)
    x_f_t0 = tf.boolean_mask(x_f, mask_t0).numpy()
    y_f_t0 = tf.boolean_mask(y_f, mask_t0).numpy()

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(x_f_t0, y_f_t0, c='dodgerblue', s=5, label=f'Residual Points (N={len(x_f_t0)})')

    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect((x_max - x_min) / (y_max - y_min))
    ax.set_title("Residual (Collocation) Points at t=0")
    ax.legend()
    plt.tight_layout()
    plt.savefig(run_save_dir / "residual_points_t0.png", dpi=150)
    plt.close(fig) # Close figure to free memory

# =============================================================================
# 2. NEURAL NETWORK MODEL CONFIGURATION
# =============================================================================
class PINN(tf.keras.Model):
    """
    PINN model using adaptive activation with the softplus function.
    Now with specific activation functions for each physical output.
    """
    def __init__(self, use_adaptive_activation=True, **kwargs):
        super().__init__(**kwargs)
        self.use_adaptive_activation = use_adaptive_activation
        initializer = tf.keras.initializers.GlorotNormal()
        n_hidden_layers = 10
        n_neurons = 100
        
        # Hidden layers
        self.hidden_layers = [
            tf.keras.layers.Dense(n_neurons, activation=None, kernel_initializer=initializer, bias_initializer='zeros')
            for _ in range(n_hidden_layers)
        ]
        
        # --- MULTI-OUTPUT LAYERS WITH SPECIFIC ACTIVATIONS ---
        # Output layer for u velocity, with linear activation (unbounded)
        self.u_output_layer = tf.keras.layers.Dense(1, activation='linear', kernel_initializer=initializer, bias_initializer='zeros')
        
        # Output layer for p pressure, with exponential activation (P >= 0)
        self.p_output_layer = tf.keras.layers.Dense(1, activation='exponential', kernel_initializer=initializer, bias_initializer='zeros')
        
        # Output layer for alpha volume fraction, with sigmoid activation (0 <= alpha <= 1)
        self.alpha_output_layer = tf.keras.layers.Dense(1, activation='sigmoid', kernel_initializer=initializer, bias_initializer='zeros')

        if self.use_adaptive_activation:
            self.n = tf.constant(10.0, dtype=DTYPE)
            self.a_s = []
            initial_a_value = 0.1
            a_initializer = tf.keras.initializers.Constant(initial_a_value)

            for i in range(n_hidden_layers):
                a = self.add_weight(
                    name=f'a_{i}',
                    shape=(n_neurons,),  #One per neuron
                    initializer=a_initializer,
                    trainable=True
                )
                self.a_s.append(a)

    def call(self, inputs):
        x = inputs
        for i, layer in enumerate(self.hidden_layers):
            x = layer(x)
            if self.use_adaptive_activation:
                a = self.a_s[i]
                # Apply adaptive activation with tanh
                x = tf.tanh(self.n * a * x)
            else:
                x = tf.tanh(x)
        
        # Get outputs from the specific output layers
        u = self.u_output_layer(x)
        p = self.p_output_layer(x)
        alpha = self.alpha_output_layer(x)
        
        # Return all three outputs as separate tensors
        return u, p, alpha

# =============================================================================
# 3. COLLOCATION POINT SAMPLING STRATEGY
# =============================================================================
def generate_training_data():
    """
    Generates training data points according to the sampling strategy.
    """
    print("Generating training data points...")
    N_IC, N_IC_interface, N_IC_bulk = 700, 300, 400
    t_ic_interface = tf.zeros((N_IC_interface, 1), dtype=DTYPE)
    x_ic_interface = tf.random.uniform((N_IC_interface, 1), x_min, x_max, dtype=DTYPE)
    y_ic_interface = tf.random.uniform((N_IC_interface, 1), -0.02, 0.02, dtype=DTYPE)
    t_ic_bulk = tf.zeros((N_IC_bulk, 1), dtype=DTYPE)
    x_ic_bulk = tf.random.uniform((N_IC_bulk, 1), x_min, x_max, dtype=DTYPE)
    y_ic_bulk_pos = tf.random.uniform((N_IC_bulk // 2, 1), 0.02, y_max, dtype=DTYPE)
    y_ic_bulk_neg = tf.random.uniform((N_IC_bulk // 2, 1), y_min, -0.02, dtype=DTYPE)
    y_ic_bulk = tf.concat([y_ic_bulk_pos, y_ic_bulk_neg], axis=0)
    t_ic = tf.concat([t_ic_interface, t_ic_bulk], axis=0)
    x_ic = tf.concat([x_ic_interface, x_ic_bulk], axis=0)
    y_ic = tf.concat([y_ic_interface, y_ic_bulk], axis=0)
    u_ic = tf.zeros_like(t_ic)
    p_ic = p_W + dp_dx * (x_ic - x_min)
    alpha_ic = tf.cast(y_ic >= 0, dtype=DTYPE)

    N_t_snapshots = 20
    t_snapshots = tf.sort(tf.concat([[[t_min]], [[t_max]], tf.random.uniform((N_t_snapshots - 2, 1), t_min, t_max, dtype=DTYPE)], axis=0), axis=0)

    N_spat_ns = 20
    t_ns = tf.repeat(t_snapshots, N_spat_ns * 2, axis=0)
    x_ns = tf.random.uniform((N_t_snapshots * N_spat_ns * 2, 1), x_min, x_max, dtype=DTYPE)
    y_ns_block_north = tf.ones((N_spat_ns, 1), dtype=DTYPE) * y_max
    y_ns_block_south = tf.ones((N_spat_ns, 1), dtype=DTYPE) * y_min
    y_ns_block = tf.concat([y_ns_block_north, y_ns_block_south], axis=0)
    y_ns = tf.tile(y_ns_block, (N_t_snapshots, 1))
    u_ns = tf.zeros_like(t_ns)
    p_ns = p_W + dp_dx * (x_ns - x_min)

    N_spat_ew = 20
    t_ew = tf.repeat(t_snapshots, N_spat_ew, axis=0)
    y_ew = tf.tile(tf.linspace(y_min, y_max, N_spat_ew)[:, tf.newaxis], (N_t_snapshots, 1))
    x_ew_west = tf.ones_like(t_ew) * x_min
    x_ew_east = tf.ones_like(t_ew) * x_max
    p_ew_west_true = p_W * tf.ones_like(t_ew)
    p_ew_east_true = p_E * tf.ones_like(t_ew)

    N_res_interface_per_t, N_res_bulk_per_t = 800, 4000
    t_res_list, x_res_list, y_res_list = [], [], []
    for t_snap in tf.unstack(t_snapshots):
        t_res_list.append(tf.ones((N_res_interface_per_t, 1), dtype=DTYPE) * t_snap)
        x_res_list.append(tf.random.uniform((N_res_interface_per_t, 1), x_min, x_max, dtype=DTYPE))
        y_res_list.append(tf.random.uniform((N_res_interface_per_t, 1), -0.04, 0.04, dtype=DTYPE))
        t_res_list.append(tf.ones((N_res_bulk_per_t, 1), dtype=DTYPE) * t_snap)
        x_res_list.append(tf.random.uniform((N_res_bulk_per_t, 1), x_min, x_max, dtype=DTYPE))
        y_res_bulk_pos = tf.random.uniform((N_res_bulk_per_t // 2, 1), 0.04, y_max, dtype=DTYPE)
        y_res_bulk_neg = tf.random.uniform((N_res_bulk_per_t // 2, 1), y_min, -0.04, dtype=DTYPE)
        y_res_list.append(tf.concat([y_res_bulk_pos, y_res_bulk_neg], axis=0))

    t_res, x_res, y_res = tf.concat(t_res_list, axis=0), tf.concat(x_res_list, axis=0), tf.concat(y_res_list, axis=0)
    t_f = tf.concat([t_res, t_ic, t_ns, t_ew, t_ew], axis=0)
    x_f = tf.concat([x_res, x_ic, x_ns, x_ew_west, x_ew_east], axis=0)
    y_f = tf.concat([y_res, y_ic, y_ns, y_ew, y_ew], axis=0)
    
    print(f"Total IC points: {t_ic.shape[0]}, NS BC points: {t_ns.shape[0]}, EW BC pairs: {t_ew.shape[0]}, Residual points: {t_f.shape[0]}")
    return {'ic':(t_ic,x_ic,y_ic),'ic_targets':(u_ic,p_ic,alpha_ic),'ns':(t_ns,x_ns,y_ns),'ns_targets':(u_ns,p_ns),'ew':(t_ew,x_ew_west,x_ew_east,y_ew),'ew_targets':(p_ew_west_true,p_ew_east_true),'f':(t_f,x_f,y_f)}

# =============================================================================
# 4. LOSS FUNCTION IMPLEMENTATION
# =============================================================================
def get_residuals(model, t, x, y):
    with tf.GradientTape(persistent=True) as tape:
        tape.watch([t, x, y])
        u, p, alpha = model(tf.concat([t, x, y], axis=1))
        
        # First derivatives
        u_t = tape.gradient(u, t)
        u_x = tape.gradient(u, x)
        u_y = tape.gradient(u, y)
        p_x = tape.gradient(p, x)
        alpha_t = tape.gradient(alpha, t)
        alpha_x = tape.gradient(alpha, x)
        
        # For mu_y
        mu = mu2 + (mu1 - mu2) * alpha
        mu_y = tape.gradient(mu, y)
        
        # Second derivative
        u_yy = tape.gradient(u_y, y)
        
    del tape

    # Physical property (density) from alpha
    rho = rho2 + (rho1 - rho2) * alpha

    # Residuals
    f_m = u_x
    f_u = rho * u_t + p_x - (1.0 / Re) * (mu * u_yy + mu_y * u_y)
    f_a = alpha_t + u * alpha_x

    return f_m, f_u, f_a

def compute_loss(model, data, w_ic, w_bc, w_f):
    t_ic, x_ic, y_ic = data['ic']
    u_ic_true, p_ic_true, alpha_ic_true = data['ic_targets']
    u_ic_pred, p_ic_pred, alpha_ic_pred = model(tf.concat([t_ic, x_ic, y_ic], axis=1))
    loss_ic = tf.reduce_mean(tf.square(u_ic_pred - u_ic_true)) + tf.reduce_mean(tf.square(p_ic_pred - p_ic_true)) + tf.reduce_mean(tf.square(alpha_ic_pred - alpha_ic_true))

    t_ns, x_ns, y_ns = data['ns']
    u_ns_true, p_ns_true = data['ns_targets']
    u_ns_pred, p_ns_pred, _ = model(tf.concat([t_ns, x_ns, y_ns], axis=1))
    loss_ns = tf.reduce_mean(tf.square(u_ns_pred - u_ns_true)) + tf.reduce_mean(tf.square(p_ns_pred - p_ns_true))

    t_ew, x_ew_west, x_ew_east, y_ew = data['ew']
    p_ew_west_true, p_ew_east_true = data['ew_targets']
    _, p_ew_west, _ = model(tf.concat([t_ew, x_ew_west, y_ew], axis=1))
    u_ew_west, _, _ = model(tf.concat([t_ew, x_ew_west, y_ew], axis=1))
    u_ew_east, p_ew_east, _ = model(tf.concat([t_ew, x_ew_east, y_ew], axis=1))
    loss_ew = tf.reduce_mean(tf.square(u_ew_west - u_ew_east)) + tf.reduce_mean(tf.square(p_ew_west - p_ew_west_true)) + tf.reduce_mean(tf.square(p_ew_east - p_ew_east_true))
    loss_bc = loss_ns + loss_ew

    t_f, x_f, y_f = data['f']
    f_m, f_u, f_a = get_residuals(model, t_f, x_f, y_f)
    loss_f = tf.reduce_mean(tf.square(f_m)) + tf.reduce_mean(tf.square(f_u)) + tf.reduce_mean(tf.square(f_a))
    total_loss = w_ic * loss_ic + w_bc * loss_bc + w_f * loss_f
    return total_loss, loss_ic, loss_bc, loss_f

# =============================================================================
# 5. TRAINING PROCEDURE
# =============================================================================
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

def evaluate_model_mse(model):
    N_eval = 200
    y_eval = np.linspace(y_min, y_max, N_eval).astype(DTYPE)
    t_eval = np.ones_like(y_eval) * t_max
    x_eval = np.zeros_like(y_eval)
    
    eval_points = tf.concat([t_eval[:,None], x_eval[:,None], y_eval[:,None]], axis=1)
    u_pred, _, _ = model(eval_points, training=False)
    u_pred = u_pred.numpy().flatten()
    u_exact = analytical_solution(y_eval, mu1, mu2, -dp_dx)

    u_pred_norm = u_pred / np.mean(u_pred) if np.mean(u_pred) != 0 else u_pred
    u_exact_norm = u_exact / np.mean(u_exact) if np.mean(u_exact) != 0 else u_exact

    return mean_squared_error(u_exact_norm, u_pred_norm)

def create_batches(data_dict, num_batches=10):
    """
    Shuffles and splits all training data into batches.
    """
    (t_ic, x_ic, y_ic), (u_ic, p_ic, alpha_ic) = data_dict['ic'], data_dict['ic_targets']
    (t_ns, x_ns, y_ns), (u_ns, p_ns) = data_dict['ns'], data_dict['ns_targets']
    (t_ew, x_ew_w, x_ew_e, y_ew), (p_ew_w, p_ew_e) = data_dict['ew'], data_dict['ew_targets']
    (t_f, x_f, y_f) = data_dict['f']

    ic_indices = tf.random.shuffle(tf.range(t_ic.shape[0]))
    ns_indices = tf.random.shuffle(tf.range(t_ns.shape[0]))
    ew_indices = tf.random.shuffle(tf.range(t_ew.shape[0]))
    f_indices = tf.random.shuffle(tf.range(t_f.shape[0]))

    def split(tensor, indices): return tf.split(tf.gather(tensor, indices), num_batches)

    t_ic_b, x_ic_b, y_ic_b = split(t_ic, ic_indices), split(x_ic, ic_indices), split(y_ic, ic_indices)
    u_ic_b, p_ic_b, alpha_ic_b = split(u_ic, ic_indices), split(p_ic, ic_indices), split(alpha_ic, ic_indices)
    t_ns_b, x_ns_b, y_ns_b = split(t_ns, ns_indices), split(x_ns, ns_indices), split(y_ns, ns_indices)
    u_ns_b, p_ns_b = split(u_ns, ns_indices), split(p_ns, ns_indices)
    t_ew_b, x_ew_w_b, x_ew_e_b, y_ew_b = split(t_ew, ew_indices), split(x_ew_w, ew_indices), split(x_ew_e, ew_indices), split(y_ew, ew_indices)
    p_ew_w_b, p_ew_e_b = split(p_ew_w, ew_indices), split(p_ew_e, ew_indices)
    t_f_b, x_f_b, y_f_b = split(t_f, f_indices), split(x_f, f_indices), split(y_f, f_indices)

    batches = []
    for i in range(num_batches):
        batches.append({
            'ic': (t_ic_b[i], x_ic_b[i], y_ic_b[i]), 'ic_targets': (u_ic_b[i], p_ic_b[i], alpha_ic_b[i]),
            'ns': (t_ns_b[i], x_ns_b[i], y_ns_b[i]), 'ns_targets': (u_ns_b[i], p_ns_b[i]),
            'ew': (t_ew_b[i], x_ew_w_b[i], x_ew_e_b[i], y_ew_b[i]), 'ew_targets': (p_ew_w_b[i], p_ew_e_b[i]),
            'f': (t_f_b[i], x_f_b[i], y_f_b[i])})
    return batches

@tf.function
def train_step(model, optimizer, data, w_ic, w_bc, w_f):
    with tf.GradientTape() as tape:
        total_loss, loss_ic, loss_bc, loss_f = compute_loss(model, data, w_ic, w_bc, w_f)
    optimizer.apply_gradients(zip(tape.gradient(total_loss, model.trainable_variables), model.trainable_variables))
    return total_loss, loss_ic, loss_bc, loss_f

if __name__ == "__main__":    
    ACTIVATION_TYPE = "adaptive" if USE_ADAPTIVE_ACTIVATION else "fixed"
    BASE_SAVE_DIR = Path("./pinn_output")

    # --- NEW: Generate a timestamp for the run ---
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    
    run_save_dir = BASE_SAVE_DIR / f"mu1_{mu1}" / ACTIVATION_TYPE / timestamp
    
    run_save_dir.mkdir(parents=True, exist_ok=True)
    print(f"All outputs for this run will be saved to: {run_save_dir}")

    # --- Model and Data Setup ---
    pinn_model = PINN(use_adaptive_activation=USE_ADAPTIVE_ACTIVATION)
    training_data = generate_training_data()
    # plot_initial_data(training_data) # Optional plotting of data
    # plot_residual_points_t0(training_data) # Optional plotting of data

    # --- Training Configuration ---
    training_stages = [{'epochs': 3000, 'lr': 5e-4}, {'epochs': 3000, 'lr': 1e-4}, {'epochs': 3000, 'lr': 5e-5}, {'epochs': 3000, 'lr': 1e-5}, {'epochs': 10000, 'lr': 5e-6}]
    NUM_BATCHES = 10
    NUM_EPOCHS_BEFORE_CHECKPOINT = 10
    print("\nStarting training...")
    total_epochs = 0
    history = {'loss':[], 'loss_ic':[], 'loss_bc':[], 'loss_f':[]}
    if USE_ADAPTIVE_ACTIVATION: 
        history['a'] = []
        
    start_time = time.time()
    optimizer = tf.keras.optimizers.Adam()
    best_mse = float('inf')
    best_model_basename = "best_pinn_model" # Base name for saved weight files
        
    total_loss, loss_ic, loss_bc, loss_f = 0, 0, 0, 0

    for stage in training_stages:
        epochs, lr = stage['epochs'], stage['lr']
        optimizer.learning_rate.assign(lr)
        print(f"\n--- Training Stage: {epochs} epochs with LR = {lr} ---")
        for epoch in range(epochs):
            batches = create_batches(training_data, num_batches=NUM_BATCHES)
            for batch_data in batches:
                total_loss, loss_ic, loss_bc, loss_f = train_step(pinn_model, optimizer, batch_data, w_ic, w_bc, w_f)
            
            total_epochs += 1
            history['loss'].append(total_loss.numpy())
            history['loss_ic'].append(loss_ic.numpy())
            history['loss_bc'].append(loss_bc.numpy())
            history['loss_f'].append(loss_f.numpy())
            if USE_ADAPTIVE_ACTIVATION:
                layer_means = [tf.reduce_mean(a).numpy() for a in pinn_model.a_s]
                total_mean = sum(layer_means) / len(layer_means)
                history['a'].append(total_mean)
            
            # --- Checkpointing Logic (every 10 epochs for efficiency) ---
            if total_epochs % NUM_EPOCHS_BEFORE_CHECKPOINT == 0:
                current_mse = evaluate_model_mse(pinn_model)
                if USE_ADAPTIVE_ACTIVATION:
                    layer_means = [tf.reduce_mean(a).numpy() for a in pinn_model.a_s]
                    total_mean = sum(layer_means) / len(layer_means)
                    print(f"Epoch: {total_epochs:05d}, Loss: {total_loss:.4e}, IC: {loss_ic:.4e}, BC: {loss_bc:.4e}, PDE: {loss_f:.4e}, Mean A: {total_mean:.4f}, MSE: {current_mse:.4e}")
                else:
                    print(f"Epoch: {total_epochs:05d}, Loss: {total_loss:.4e}, IC: {loss_ic:.4e}, BC: {loss_bc:.4e}, PDE: {loss_f:.4e}, MSE: {current_mse:.4e}")
                
                # --- Scoped Model Saving (Works as intended with new directory structure) ---
                if current_mse < best_mse:
                    # 1. First, remove all old weight files in this specific run's directory
                    # The pattern now correctly looks only inside the timestamped `run_save_dir`
                    cleanup_pattern = str(run_save_dir / f"{best_model_basename}_*.weights.h5")
                    for f_path in glob.glob(cleanup_pattern):
                        os.remove(f_path)
                    
                    # 2. Now, update the best MSE and save the new best model
                    best_mse = current_mse
                    best_weights_filename = f"{best_model_basename}_{best_mse:.4e}.weights.h5"
                    best_weights_file_path = str(run_save_dir / best_weights_filename) # Update the global tracker
                    
                    print(f"     >>> New best MSE: {best_mse:.4e}. Saving model to {best_weights_file_path}")
                    pinn_model.save_weights(best_weights_file_path)

    print(f"\nTraining finished in {time.time() - start_time:.2f} seconds.")

    # --- Save Training History to the Run-Specific Folder ---
    if USE_ADAPTIVE_ACTIVATION:
        np.save(run_save_dir / f"a_history_{mu1}_{best_mse:.4e}.npy", np.array(history['a']))
        
    loss_ic_bc = np.array(history['loss_ic']) + np.array(history['loss_bc'])
    np.save(run_save_dir / f"loss_boundary_initial_history_{mu1}_{ACTIVATION_TYPE}_{best_mse:.4e}.npy", loss_ic_bc)
    np.save(run_save_dir / f"loss_pde_history_{mu1}_{ACTIVATION_TYPE}_{best_mse:.4e}.npy", np.array(history['loss_f']))

    # =========================================================================
    # 6. VERIFICATION AND VISUALIZATION
    # =========================================================================
    if not best_weights_file_path:
        print("\nNo best model was saved. Skipping verification.")
    else:
        print(f"\nLoading best model from {best_weights_file_path} for verification...")
        pinn_model.load_weights(best_weights_file_path)

        N_plot = 100
        y_plot = np.linspace(y_min, y_max, N_plot).astype(DTYPE)
        t_plot = np.ones_like(y_plot) * t_max
        x_plot = np.zeros_like(y_plot)
        u_pred, _, _ = pinn_model(tf.concat([t_plot[:,None], x_plot[:,None], y_plot[:,None]], axis=1))
        u_pred = u_pred.numpy().flatten()
        u_exact = analytical_solution(y_plot, mu1, mu2, -dp_dx)
        
        u_pred_norm = u_pred / np.mean(u_pred) if np.mean(u_pred) != 0 else u_pred
        u_exact_norm = u_exact / np.mean(u_exact) if np.mean(u_exact) != 0 else u_exact
        
        MSE = mean_squared_error(u_exact_norm, u_pred_norm)
        L1 = np.mean(np.abs(u_pred_norm - u_exact_norm)) / np.mean(np.abs(u_exact_norm))
        L2 = np.linalg.norm(u_pred_norm - u_exact_norm) / np.linalg.norm(u_exact_norm)
        print(f"\n--- Error Metrics (on best model) ---\nMSE: {MSE:.3e}\nL1 relative error: {L1:.3e}\nL2 relative error: {L2:.3e}")

        plt.figure(figsize=(16, 10))
        plt.subplot(2, 2, 1)
        loss_ic_bc = np.array(history['loss_ic']) + np.array(history['loss_bc'])
        plt.plot(loss_ic_bc, label=f"{ACTIVATION_TYPE} IC/BC Loss")
        plt.yscale('log'); plt.title('IC + BC Loss History'); plt.xlabel('Epochs'); plt.ylabel('MSE'); plt.legend(); plt.grid(True, which="both", ls="--")

        plt.subplot(2, 2, 2)
        plt.plot(history['loss_f'], label=f"{ACTIVATION_TYPE} PDE Loss")
        plt.yscale('log'); plt.title('PDE Loss History'); plt.xlabel('Epochs'); plt.ylabel('MSE'); plt.legend(); plt.grid(True, which="both", ls="--")


        if USE_ADAPTIVE_ACTIVATION:
            plt.subplot(2, 2, 3)
            plt.plot(history['a']); plt.title('Mean of "a"'); plt.xlabel('Epochs'); plt.ylabel('Mean Value of "a"'); plt.grid(True)
        
        plt.subplot(2, 2, (4, 5))
        plt.plot(u_pred_norm, y_plot, 'r-', label='PINN Prediction', linewidth=2)
        plt.plot(u_exact_norm, y_plot, 'b--', label='Analytical Solution', linewidth=2)
        plt.axhline(0, color='k', linestyle=':', label='Interface')
        plt.title(f'Steady-State Velocity Profile ($\\mu_1/\\mu_2 = {mu1/mu2}$)')
        plt.xlabel('$u / \\bar{u}$'); plt.ylabel('y'); plt.legend(); plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(run_save_dir / "final_results.png", dpi=300)
        plt.show()