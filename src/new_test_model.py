import sys
sys.path.append("../utilities")
from utilities import *

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def compute_mae_over_time(cfd_data, nn_data):
    """
    Compute mean absolute error over time.
    cfd_data, nn_data: arrays of shape [ny, nx, nt]
    returns: array of shape [nt]
    """
    ny, nx, nt = cfd_data.shape
    mae = np.zeros(nt)
    for i in range(nt):
        mae[i] = np.mean(np.abs(cfd_data[:, :, i] - nn_data[:, :, i]))
    return mae

def plot_mae_and_norms_scaled(t, cfd_data, nn_data, var_names):
    """
    Plot MAE (red, left y-axis) and Norms (blue, right y-axis) over time.
    MAE scale: 0 to 0.15
    Norms scale: 0 to 0.06
    """
    n_vars = len(var_names)
    fig, axes = plt.subplots(n_vars, 1, figsize=(8, 4*n_vars), sharex=True)

    if n_vars == 1:
        axes = [axes]

    for i, (cfd, nn, name, ax) in enumerate(zip(cfd_data, nn_data, var_names, axes)):
        mae = compute_mae_over_time(cfd, nn)
        norm = np.linalg.norm(cfd.reshape(-1, cfd.shape[-1]), axis=0) / np.prod(cfd.shape[:2])

        # adjust time array if needed
        if mae.shape[0] != t.shape[0]:
            t_resized = np.linspace(t[0], t[-1], mae.shape[0])
        else:
            t_resized = t

        # Left axis (red) for MAE
        ax.plot(t_resized, mae, 'r-', label=f'{name} MAE')
        ax.set_ylabel(f'{name} MAE', color='red')
        ax.tick_params(axis='y', labelcolor='red')
        ax.set_ylim(0, 0.15)

        # Right axis (blue) for Norm
        ax2 = ax.twinx()
        ax2.plot(t_resized, norm, 'b-', label=f'{name} Norm')
        ax2.set_ylabel(f'{name} Norm', color='blue')
        ax2.tick_params(axis='y', labelcolor='blue')
        ax2.set_ylim(0, 0.06)

        ax.grid(True)

    axes[-1].set_xlabel("Time")
    plt.suptitle("MAE (red, 0–0.15) and Norms (blue, 0–0.06) over Time", fontsize=14)
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
    z = tf.keras.layers.Dense(350, activation='tanh')(inputs)
    z = tf.keras.layers.Dense(350, activation='tanh')(z)
    z = tf.keras.layers.Dense(350, activation='tanh')(z)
    z = tf.keras.layers.Dense(350, activation='tanh')(z)
    z = tf.keras.layers.Dense(350, activation='tanh')(z)
    z = tf.keras.layers.Dense(350, activation='tanh')(z)
    z = tf.keras.layers.Dense(350, activation='tanh')(z)
    z = tf.keras.layers.Dense(350, activation='tanh')(z)

    output_u = tf.keras.layers.Dense(1, activation='linear', name="output_u")(z)
    output_v = tf.keras.layers.Dense(1, activation='linear', name="output_v")(z)
    output_p = tf.keras.layers.Dense(1, activation='exponential', name="output_p")(z)
    output_a = tf.keras.layers.Dense(1, activation='sigmoid', name="output_a")(z)

    model = tf.keras.Model(inputs=inputs, outputs=[output_u, output_v, output_p, output_a])

    # LOAD TRAINED WEIGHTS
    #weights_path = "./checkpoints/Jul-14-2025_20-53-04/loss_2.3736em02.weights.h5"
    weights_path = "../trained_model/loss_9.7710e-03_weights.h5"
    model.load_weights(weights_path)

    # PREPARE PREDICTION DATA (x, y, t are numeric arrays here!)
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
    #fig, grid, pcfsets, kwargs = grid_contour_plots(data, nrows_ncols, titles, x, y)

    # ANIMATE
    # ani = FuncAnimation(
    #     fig, update_contourf, frames=len(t),
    #     fargs=([x] * np.prod(nrows_ncols), [y] * np.prod(nrows_ncols),
    #            data, [ax for ax in grid], pcfsets, kwargs),
    #     interval=50, blit=True, repeat=True
    # )

    # #plt.show()

    mae_u = compute_mae_over_time(velocityX_cfd, velocityX_nn)
    mae_v = compute_mae_over_time(velocityY_cfd, velocityY_nn)
    mae_p = compute_mae_over_time(pressure_cfd, pressure_nn)

    print(f"t.shape: {t.shape}, mae_u.shape: {mae_u.shape}")

    if mae_u.shape[0] != t.shape[0]:
        # adjust time array to match MAE length
        t_resized = np.linspace(t[0], t[-1], mae_u.shape[0])
    else:
        t_resized = t

    cfd_list = [velocityX_cfd, velocityY_cfd, pressure_cfd]
    nn_list  = [velocityX_nn, velocityY_nn, pressure_nn]
    var_names = ['u (velocityX)', 'v (velocityY)', 'p (pressure)']
    
    plot_mae_and_norms_scaled(t, cfd_list, nn_list, var_names)

if __name__ == "__main__":
    main()