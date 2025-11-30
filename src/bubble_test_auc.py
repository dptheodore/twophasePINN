import sys
sys.path.append("../utilities")
from utilities import *

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import csv  # <--- Added import

def update_contourf_fixed(frame, x_list, y_list, data_list, ax_list, pcfsets, kwargs):
    """
    Updates the contour plots for each frame.
    """
    artists = []
    
    for i, (ax, data, x, y) in enumerate(zip(ax_list, data_list, x_list, y_list)):
        # --- 1. CLEANUP PREVIOUS FRAME ---
        current_plot = pcfsets[i]
        
        if isinstance(current_plot, (list, tuple)):
            current_plot = current_plot[0] if len(current_plot) > 0 else None

        if current_plot is not None:
            try:
                current_plot.remove() # Modern Matplotlib
            except (AttributeError, ValueError):
                # Fallback for older Matplotlib
                if hasattr(current_plot, 'collections'):
                    for c in current_plot.collections:
                        c.remove()

        # --- 2. PREPARE DATA ---
        current_data = data[frame, :, :]

        # --- 3. FIX KWARGS (Handle List vs Dict) ---
        if isinstance(kwargs, (list, tuple)):
            if i < len(kwargs):
                plot_kwargs = kwargs[i]
            elif len(kwargs) > 0:
                plot_kwargs = kwargs[0]
            else:
                plot_kwargs = {}
        else:
            plot_kwargs = kwargs

        # --- 4. PLOT NEW FRAME ---
        new_contour = ax.contourf(x, y, current_data, **plot_kwargs)
        
        pcfsets[i] = new_contour
        artists.append(new_contour)

    return artists

def compute_mae_over_time(cfd_data, nn_data):
    """
    Compute MAE at each time point. 
    Assumes cfd_data, nn_data: shape [nt, nx, ny]
    """
    nt, nx, ny = cfd_data.shape
    # print(f'CFD SHAPE: {cfd_data.shape}') # Commented out to reduce noise during CSV generation
    mae = np.zeros(nt)
    l1_norm = np.zeros(nt)
    l2_norm = np.zeros(nt)
    l1_true_norm = np.zeros(nt)
    l2_true_norm = np.zeros(nt)
    for i in range(nt):
        error_vector = cfd_data[i, :, :] - nn_data[i, :, :]
        mae[i] = np.mean(np.abs(error_vector))
        l1_norm[i] = np.linalg.norm(error_vector, ord=1)
        l1_true_norm[i] = np.linalg.norm(cfd_data[i, :, :], ord=1)
        l2_norm[i] = np.linalg.norm(error_vector, ord=2)
        l2_true_norm[i] = np.linalg.norm(cfd_data[i, :, :], ord=2)
        
        if l1_true_norm[i] != 0:
            l1_norm[i] = l1_norm[i] / l1_true_norm[i]
        if l2_true_norm[i] != 0:
            l2_norm[i] = l2_norm[i] / l2_true_norm[i]
    return mae, l1_norm, l2_norm

def save_error_auc_to_csv(t, cfd_data_list, nn_data_list, var_names, filename="error_auc.csv"):
    """
    Calculates the Area Under the Curve (AUC) for MAE, L1, and L2 errors using the 
    Trapezoidal rule and saves the results to a CSV file.
    """
    print(f"\nCalculating AUC for errors and saving to '{filename}'...")
    
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write Header
        writer.writerow(["Variable", "Error_Type", "AUC_Value"])

        for cfd, nn, name in zip(cfd_data_list, nn_data_list, var_names):
            # 1. Compute the error vectors over time
            mae, l1_norm, l2_norm = compute_mae_over_time(cfd, nn)

            # 2. Integrate using Trapezoidal rule (np.trapz)
            auc_mae = np.trapz(mae, x=t)
            auc_l1 = np.trapz(l1_norm, x=t)
            auc_l2 = np.trapz(l2_norm, x=t)

            # 3. Write rows to CSV
            writer.writerow([name, "MAE", auc_mae])
            writer.writerow([name, "L1_Norm", auc_l1])
            writer.writerow([name, "L2_Norm", auc_l2])

            print(f"  Processed {name}: MAE AUC={auc_mae:.4f}")

    print("CSV save complete.")

def plot_mae_and_norms_scaled(t, cfd_data, nn_data, var_names):
    """
    Plot MAE (red, left y-axis) and Norms (blue, right y-axis) over time.
    """
    n_vars = len(var_names)
    fig, axes = plt.subplots(n_vars, 1, figsize=(8, 4*n_vars), sharex=True)

    if n_vars == 1:
        axes = [axes]

    for i, (cfd, nn, name, ax) in enumerate(zip(cfd_data, nn_data, var_names, axes)):
        mae, l1_norm, l2_norm = compute_mae_over_time(cfd, nn)

        ax.plot(t, mae, 'r-', label=f'{name} MAE')
        ax.set_ylabel(f'{name} MAE', color='red')
        ax.tick_params(axis='y', labelcolor='red')

        ax2 = ax.twinx()
        ax2.plot(t, l1_norm, 'b-', label=f'{name} L1 Norm')
        ax2.plot(t, l2_norm, 'b--', label=f'{name} L2 Norm')
        ax2.set_ylabel(f'{name} Norms', color='blue')
        ax2.tick_params(axis='y', labelcolor='blue')

        ax.grid(True)
        ax.set_xticks([0, 5, 10])

    axes[-1].set_xlabel("Time")
    plt.suptitle("MAE (red) and Norms (blue) over Time", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def main():

    # LOAD CFD SOLUTION
    pressure_cfd, velocityX_cfd, velocityY_cfd, levelset_cfd, x, y, t = load_cfd(
        start_index=0, end_index=151,
        temporal_step_size=10, spatial_step_size=2
    )

    # REFERENCE PARAMETERS FOR NON-DIMENSIONALIZATION
    L_ref = 0.25
    rho_ref = 1000

    # NON-DIMENSIONALIZATION
    x /= L_ref 
    y /= L_ref 
    t /= L_ref 
    pressure_cfd /= rho_ref

    # BUILD THE MODEL ARCHITECTURE
    inputs = tf.keras.Input(shape=(3,), name="input_tensor")
    z = tf.keras.layers.Dense(400, activation='tanh')(inputs)
    z = tf.keras.layers.Dense(400, activation='tanh')(z)
    z = tf.keras.layers.Dense(400, activation='tanh')(z)
    z = tf.keras.layers.Dense(400, activation='tanh')(z)
    z = tf.keras.layers.Dense(400, activation='tanh')(z)
    z = tf.keras.layers.Dense(400, activation='tanh')(z)
    z = tf.keras.layers.Dense(400, activation='tanh')(z)
    z = tf.keras.layers.Dense(400, activation='tanh')(z)

    output_u = tf.keras.layers.Dense(1, activation='linear', name="output_u")(z)
    output_v = tf.keras.layers.Dense(1, activation='linear', name="output_v")(z)
    output_p = tf.keras.layers.Dense(1, activation='exponential', name="output_p")(z)
    output_a = tf.keras.layers.Dense(1, activation='sigmoid', name="output_a")(z)

    model = tf.keras.Model(inputs=inputs, outputs=[output_u, output_v, output_p, output_a])

    # LOAD TRAINED WEIGHTS
    weights_path = "./hyperparam/hyperparam.weights.h5"
    if not os.path.exists(weights_path):
        print(f"Warning: Weights file {weights_path} not found.")
    else:
        model.load_weights(weights_path)

    # PREPARE PREDICTION DATA
    test_data = reshape_test_data(x, y, t)
    
    # PREDICT AND RESHAPE SOLUTION
    print("\nPredicting nn solution")
    velocityX_nn, velocityY_nn, pressure_nn, volume_fraction_nn = model.predict(
        test_data, batch_size=int(1e6), verbose=1
    )

    velocityX_nn = reshape_prediction(x, y, t, velocityX_nn)
    velocityY_nn = reshape_prediction(x, y, t, velocityY_nn)
    pressure_nn = reshape_prediction(x, y, t, pressure_nn)
    volume_fraction_nn = reshape_prediction(x, y, t, volume_fraction_nn)

    # CONTOURPLOT PARAMETERS
    data = [pressure_nn, velocityX_nn, velocityY_nn, pressure_cfd, velocityX_cfd, velocityY_cfd]
    titles = ["p_pred", "u_pred", "v_pred", "p_cfd", "u_cfd", "v_cfd"]
    nrows_ncols = (2, 3)

    # CREATE FIGURE
    fig, grid, pcfsets, kwargs = grid_contour_plots(data, nrows_ncols, titles, x, y)

    # ANIMATE AND SAVE
    print("Starting animation generation...")
    
    ani = FuncAnimation(
        fig, update_contourf_fixed, frames=len(t),
        fargs=([x] * np.prod(nrows_ncols), [y] * np.prod(nrows_ncols),
               data, [ax for ax in grid], pcfsets, kwargs),
        interval=50, blit=False, repeat=False
    )

    try:
        ani.save("bubble_comparison.gif", writer='pillow', fps=10)
        print("Success: Saved 'bubble_comparison.gif'")
    except Exception as e:
        print(f"Failed to save GIF: {e}")

    # DEFINE LISTS FOR ERROR CALCULATION
    cfd_list = [pressure_cfd, velocityY_cfd, velocityX_cfd ]
    nn_list  = [pressure_nn, velocityY_nn, velocityX_nn ]
    var_names = ['p (pressure)', 'u (velocityX)', 'v (velocityY)' ]
    
    # 1. SAVE AUC TO CSV (New Step)
    save_error_auc_to_csv(t, cfd_list, nn_list, var_names, filename="error_auc.csv")

    # 2. PLOT MAE GRAPHS
    plot_mae_and_norms_scaled(t, cfd_list, nn_list, var_names)

if __name__ == "__main__":
    main()