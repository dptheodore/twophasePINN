import sys
sys.path.append("../utilities")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
GPU_ID = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID

import tensorflow as tf
import numpy as np
import pandas as pd
import scipy.io
from generate_points_3d import get_training_data_3d
from utilities import NNCreator, writeToJSONFile
import time
import math
import glob
from datetime import datetime
import shutil
import logging
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
from tensorflow.keras.utils import get_custom_objects

# Define and register the custom sine activation function so Keras can find it by name
def sine_activation(x):
    return K.sin(x)
get_custom_objects().update({'sine': sine_activation})


# Set random seeds for reproducibility
np.random.seed(1234)
tf.random.set_seed(1234)

class TwoPhasePinn(tf.keras.Model):
    """
    This class implements a physics-informed neural network in TensorFlow 2.
    It approximates the incompressible two-phase Navier-Stokes equations in 3D
    using a Volume-of-Fluid approach.
    """

    def __init__(self, hidden_layers, activation_functions, adaptive_activation_coeff,
                 adaptive_activation_n, adaptive_activation_init, use_ad_act,
                 loss_weights_PDE, mu, sigma, g, rho, u_ref, L_ref, rho_mean):
        super(TwoPhasePinn, self).__init__()

        # Physical Parameters
        self.mu1, self.mu2 = mu
        self.sigma = sigma
        self.g = g
        self.rho1, self.rho2 = rho
        self.U_ref = u_ref
        self.L_ref = L_ref
        self.rho_ref = self.rho2
        # Volume-averaged density, used to remove the mean body force (see PDE_caller).
        self.rho_mean = rho_mean

        # Loss weights
        self.loss_weights_PDE = tf.constant(loss_weights_PDE, dtype=tf.float32)
        # Adaptive activation coefficients
        self.use_ad_act = use_ad_act
        self.ad_act_coeff = {}
        if self.use_ad_act:
            for key, initial_value in adaptive_activation_init.items():
                self.ad_act_coeff[key] = tf.Variable(initial_value, trainable=True, name=key, dtype=tf.float32)

        # --- Correctly build the model using NNCreator ---
        # 1. Create the activation function dictionary needed by NNCreator
        activation_functions_dict = self._get_activation_function_dict(
            hidden_layers, activation_functions, adaptive_activation_coeff, adaptive_activation_n
        )

        # 2. Define the output layer structure
        outputs = ["output_u", "output_v", "output_w", "output_p", "output_a"]
        activations_output = [None, None, None, "exponential", "sigmoid"]
        output_layer = list(zip(outputs, activations_output))

        # 3. Instantiate NNCreator and build the model
        nn_creator = NNCreator(tf.float32)
        self.nn = nn_creator.get_model_dnn(4, hidden_layers, output_layer, activation_functions_dict, self.use_ad_act)

    def _get_activation_function_dict(self, hidden_layers, activation_functions, adaptive_activation_coeff, adaptive_activation_n):
        """Helper to create the activation dictionary for NNCreator."""
        activation_dict = {i: [None, None, 0] for i in range(1, len(hidden_layers) + 1)}
        for layer_no in activation_dict:
            activation_dict[layer_no][2] = adaptive_activation_n[layer_no - 1]
            for func_name, layers in activation_functions.items():
                if layer_no in layers:
                    activation_dict[layer_no][0] = func_name
            if self.use_ad_act:
                for coeff_name, layers in adaptive_activation_coeff.items():
                    if layer_no in layers:
                        activation_dict[layer_no][1] = self.ad_act_coeff[coeff_name]
        return activation_dict

    def call(self, inputs):
        # The model built by NNCreator is now stored in self.nn
        return self.nn(inputs)

    @tf.function
    def compute_gradients(self, x, y, z, t):
        # Use a nested tape to compute second-order derivatives
        with tf.GradientTape(persistent=True) as tape2:
            tape2.watch([x, y, z, t])
            with tf.GradientTape(persistent=True) as tape1:
                tape1.watch([x, y, z, t])
                u, v, w, p, a = self.call(tf.concat([x, y, z, t], axis=1))

            # First-order gradients computed with the inner tape
            u_x = tape1.gradient(u, x)
            u_y = tape1.gradient(u, y)
            u_z = tape1.gradient(u, z)
            u_t = tape1.gradient(u, t)
            v_x = tape1.gradient(v, x)
            v_y = tape1.gradient(v, y)
            v_z = tape1.gradient(v, z)
            v_t = tape1.gradient(v, t)
            w_x = tape1.gradient(w, x)
            w_y = tape1.gradient(w, y)
            w_z = tape1.gradient(w, z)
            w_t = tape1.gradient(w, t)
            p_x = tape1.gradient(p, x)
            p_y = tape1.gradient(p, y)
            p_z = tape1.gradient(p, z)
            a_x = tape1.gradient(a, x)
            a_y = tape1.gradient(a, y)
            a_z = tape1.gradient(a, z)
            a_t = tape1.gradient(a, t)

        # Second-order gradients computed with the outer tape
        u_xx = tape2.gradient(u_x, x)
        u_yy = tape2.gradient(u_y, y)
        u_zz = tape2.gradient(u_z, z)
        v_xx = tape2.gradient(v_x, x)
        v_yy = tape2.gradient(v_y, y)
        v_zz = tape2.gradient(v_z, z)
        w_xx = tape2.gradient(w_x, x)
        w_yy = tape2.gradient(w_y, y)
        w_zz = tape2.gradient(w_z, z)
        a_xx = tape2.gradient(a_x, x)
        a_yy = tape2.gradient(a_y, y)
        a_zz = tape2.gradient(a_z, z)
        a_xy = tape2.gradient(a_x, y)
        a_xz = tape2.gradient(a_x, z)
        a_yz = tape2.gradient(a_y, z)

        # Clean up tapes
        del tape1, tape2

        return (u, u_x, u_y, u_z, u_t, u_xx, u_yy, u_zz), \
               (v, v_x, v_y, v_z, v_t, v_xx, v_yy, v_zz), \
               (w, w_x, w_y, w_z, w_t, w_xx, w_yy, w_zz), \
               (p, p_x, p_y, p_z), \
               (a, a_x, a_y, a_z, a_t, a_xx, a_yy, a_zz, a_xy, a_xz, a_yz)

    @tf.function
    def PDE_caller(self, x, y, z, t):
        u_grads, v_grads, w_grads, p_grads, a_grads = self.compute_gradients(x, y, z, t)
        u, u_x, u_y, u_z, u_t, u_xx, u_yy, u_zz = u_grads
        v, v_x, v_y, v_z, v_t, v_xx, v_yy, v_zz = v_grads
        w, w_x, w_y, w_z, w_t, w_xx, w_yy, w_zz = w_grads
        p, p_x, p_y, p_z = p_grads
        a, a_x, a_y, a_z, a_t, a_xx, a_yy, a_zz, a_xy, a_xz, a_yz = a_grads

        mu = self.mu2 + (self.mu1 - self.mu2) * a
        mu_x = (self.mu1 - self.mu2) * a_x
        mu_y = (self.mu1 - self.mu2) * a_y
        mu_z = (self.mu1 - self.mu2) * a_z
        rho = self.rho2 + (self.rho1 - self.rho2) * a

        abs_interface_grad = tf.sqrt(a_x**2 + a_y**2 + a_z**2 + np.finfo(float).eps)
        # 3D mean curvature of the levelset -div(grad(a)/|grad(a)|)
        curvature = -(((a_yy + a_zz) * a_x**2 + (a_xx + a_zz) * a_y**2 + (a_xx + a_yy) * a_z**2
                        - 2 * a_x * a_y * a_xy - 2 * a_x * a_z * a_xz - 2 * a_y * a_z * a_yz)
                      / tf.pow(abs_interface_grad, 3))

        one_Re = mu / (self.rho_ref * self.U_ref * self.L_ref)
        one_Re_x = mu_x / (self.rho_ref * self.U_ref * self.L_ref)
        one_Re_y = mu_y / (self.rho_ref * self.U_ref * self.L_ref)
        one_Re_z = mu_z / (self.rho_ref * self.U_ref * self.L_ref)
        one_We = self.sigma / (self.rho_ref * self.U_ref**2 * self.L_ref)
        one_Fr = self.g * self.L_ref / self.U_ref**2

        PDE_m = u_x + v_y + w_z
        PDE_a = a_t + u * a_x + v * a_y + w * a_z
        PDE_u = (u_t + u * u_x + v * u_y + w * u_z) * rho / self.rho_ref + p_x - \
                one_We * curvature * a_x - one_Re * (u_xx + u_yy + u_zz) - \
                2.0 * one_Re_x * u_x - one_Re_y * (u_y + v_x) - one_Re_z * (u_z + w_x)
        PDE_v = (v_t + u * v_x + v * v_y + w * v_z) * rho / self.rho_ref + p_y - \
                one_We * curvature * a_y - one_Re * (v_xx + v_yy + v_zz) - \
                2.0 * one_Re_y * v_y - one_Re_x * (u_y + v_x) - one_Re_z * (v_z + w_y)
        # Gravity acts along z (see generate_points_3d.py), so the body-force term
        # lives in the w-momentum equation here instead of the v-momentum equation.
        #
        # REDUCED GRAVITY. The LBPM domain is triply periodic (Domain/BC = 0), so p is
        # periodic and grad(p) has zero volume average -- there is no mean pressure
        # gradient available to balance a mean body force. Using the full rho*g would
        # leave an unbalanced mean forcing of <rho>/rho_ref * (1/Fr) that nothing can
        # cancel except a uniform acceleration, which the data rules out (the domain-mean
        # w is constant in time). LBPM removes the mean force itself, so only the density
        # deviation from the volume average drives the flow. The 2D case never needed this:
        # its north boundary was a pressure Dirichlet, which does permit a mean vertical
        # pressure gradient. If the .db is ever switched to a pressure BC in z, drop this
        # correction and use rho directly, as the 2D script does.
        rho_reduced = rho - self.rho_mean
        PDE_w = (w_t + u * w_x + v * w_y + w * w_z) * rho / self.rho_ref + p_z - \
                one_We * curvature * a_z - one_Re * (w_xx + w_yy + w_zz) - \
                rho_reduced / self.rho_ref * one_Fr - 2.0 * one_Re_z * w_z - \
                one_Re_x * (u_z + w_x) - one_Re_y * (v_z + w_y)

        return PDE_m, PDE_u, PDE_v, PDE_w, PDE_a

    @tf.function
    def _periodic_mismatch(self, plus_coords, minus_coords, t):
        ''' Squared mismatch in u, v, w and p between the two faces of a periodic pair. '''
        x_p, y_p, z_p = plus_coords
        x_m, y_m, z_m = minus_coords
        pred_plus = self.call(tf.concat([x_p, y_p, z_p, t], axis=1))
        pred_minus = self.call(tf.concat([x_m, y_m, z_m, t], axis=1))
        loss = 0.0
        for i in range(4):  # u, v, w, p
            loss += tf.reduce_mean(tf.square(pred_plus[i] - pred_minus[i]))
        return loss

    @tf.function
    def compute_loss(self, data_A, data_PDE, data_XPER, data_YPER, data_ZPER, data_GAUGE):
        # Unpack tensor tuples
        x_A, y_A, z_A, t_A, a_A = data_A
        x_PDE, y_PDE, z_PDE, t_PDE = data_PDE
        x_Xp, y_Xp, z_Xp, x_Xm, y_Xm, z_Xm, t_XPER = data_XPER
        x_Yp, y_Yp, z_Yp, x_Ym, y_Ym, z_Ym, t_YPER = data_YPER
        x_Zp, y_Zp, z_Zp, x_Zm, y_Zm, z_Zm, t_ZPER = data_ZPER
        x_G, y_G, z_G, t_G, p_G = data_GAUGE

        f_PDE = tf.zeros_like(x_PDE)

        # Loss A (Volume Fraction)
        output_tensors = self.call(tf.concat([x_A, y_A, z_A, t_A], axis=1))
        loss_a_A = tf.reduce_mean(tf.square(a_A - output_tensors[4]))

        # Loss XPER/YPER/ZPER. The LBPM run sets Domain/BC = 0, so all six faces are
        # periodic; there are no walls and no pressure outlet.
        loss_XPER = self._periodic_mismatch((x_Xp, y_Xp, z_Xp), (x_Xm, y_Xm, z_Xm), t_XPER)
        loss_YPER = self._periodic_mismatch((x_Yp, y_Yp, z_Yp), (x_Ym, y_Ym, z_Ym), t_YPER)
        loss_ZPER = self._periodic_mismatch((x_Zp, y_Zp, z_Zp), (x_Zm, y_Zm, z_Zm), t_ZPER)

        # Loss GAUGE. With every face periodic the pressure is only determined up to an
        # additive constant, so it is pinned at a single corner point.
        _, _, _, pred_p_G, _ = self.call(tf.concat([x_G, y_G, z_G, t_G], axis=1))
        loss_p_GAUGE = tf.reduce_mean(tf.square(p_G - pred_p_G))

        loss_BC = loss_XPER + loss_YPER + loss_ZPER + loss_p_GAUGE

        # Loss PDE (Physics-Informed)
        PDE_m, PDE_u, PDE_v, PDE_w, PDE_a = self.PDE_caller(x_PDE, y_PDE, z_PDE, t_PDE)
        loss_PDE_m = tf.reduce_mean(tf.square(f_PDE - PDE_m))
        loss_PDE_u = tf.reduce_mean(tf.square(f_PDE - PDE_u))
        loss_PDE_v = tf.reduce_mean(tf.square(f_PDE - PDE_v))
        loss_PDE_w = tf.reduce_mean(tf.square(f_PDE - PDE_w))
        loss_PDE_a = tf.reduce_mean(tf.square(f_PDE - PDE_a))

        loss_PDE = tf.tensordot(tf.stack([loss_PDE_m, loss_PDE_u, loss_PDE_v, loss_PDE_w, loss_PDE_a]), self.loss_weights_PDE, 1)

        # Total Loss
        total_loss = loss_a_A + loss_BC + loss_PDE

        return total_loss, loss_a_A, loss_BC, loss_PDE_m, loss_PDE_u, loss_PDE_v, loss_PDE_w, loss_PDE_a


    @tf.function
    def train_step(self, optimizer, data_A, data_PDE, data_XPER, data_YPER, data_ZPER, data_GAUGE):
        with tf.GradientTape() as tape:
            # Pass the tensor tuples directly to compute_loss
            losses = self.compute_loss(data_A, data_PDE, data_XPER, data_YPER, data_ZPER, data_GAUGE)
            total_loss = losses[0]

        gradients = tape.gradient(total_loss, self.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return losses


def setup_output_directory():
    if not os.path.exists("checkpoints"):
        os.mkdir("checkpoints")
    dirname = os.path.abspath(os.path.join("checkpoints", datetime.now().strftime("%b-%d-%Y_%H-%M-%S")))
    os.mkdir(dirname)

    shutil.copyfile(__file__, os.path.join(dirname, os.path.basename(__file__)))
    if os.path.exists("generate_points_3d.py"):
        shutil.copyfile("generate_points_3d.py", os.path.join(dirname, "generate_points_3d.py"))

    logpath = os.path.join(dirname, "output.log")
    return dirname, logpath

def get_logger(logpath):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(sh)

    fh = logging.FileHandler(logpath)
    fh.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(fh)

    return logger

def get_proportional_batch_sizes(total_batch_size, training_data, logger):
    """Calculates proportional batch sizes for each dataset."""
    num_samples_total = sum(len(df) for df in training_data.values())
    num_batches = math.ceil(num_samples_total / total_batch_size)

    batch_sizes = {}
    for key, df in training_data.items():
        if len(df) > 0:
            proportion = len(df) / num_samples_total
            batch_sizes[key] = math.ceil(proportion * total_batch_size)
        else:
            batch_sizes[key] = 0

    logger.info(f"Total samples: {num_samples_total}, Desired batch size: {total_batch_size}")
    logger.info(f"Calculated num_batches: {num_batches}, Proportional batch sizes: {batch_sizes}")
    return batch_sizes, num_batches

def main():
    """
    This script trains a PINN for the 3D rising bubble case.
    """
    dirname, logpath = setup_output_directory()
    logger = get_logger(logpath)

    NOP_a = (500, 400)
    NOP_PDE = (400, 2000, 3000)
    # A 3D periodic face is a surface, not an edge, so it needs far more than the 20
    # spatial points the 2D script used per boundary.
    NOP_xper = (400, 20)
    NOP_yper = (400, 20)
    NOP_zper = (400, 20)

    training_data, gauge_df, scales = get_training_data_3d(
        NOP_a, NOP_PDE, NOP_xper, NOP_yper, NOP_zper)

    # --- NN Architecture and Hyperparameters --- #
    no_layers = 8
    hidden_layers = [400] * no_layers

    # --- CHOOSE YOUR CONFIGURATION ---
    activation_choice = 'tanh'  # Options: 'tanh' or 'sine'
    use_aac_1 = False
    use_aac_2 = False #If you declare both false, fixed/no activation is used

    # Build activation function dictionary based on choice
    activation_functions = {activation_choice: range(1, no_layers + 1)}

    # Determine adaptive activation mode for file naming and logic
    adaptive_mode = 'fixed'
    if use_aac_1:
        adaptive_mode = 'activation1'
    elif use_aac_2:
        adaptive_mode = 'activation2'

    logger.info(f"Configuration: Activation='{activation_choice}', Mode='{adaptive_mode}'")

    # Conditionally define the adaptive activation parameters
    if use_aac_1:
        adaptive_activation_coeff = {"aac_1": range(1, no_layers + 1)}
        adaptive_activation_init = {"aac_1": 0.1}
        adaptive_activation_n = [10] * no_layers
    elif use_aac_2:
        adaptive_activation_coeff = {
            "aac_2_a": range(1, 5),
            "aac_2_b": range(5, 9)
        }
        adaptive_activation_init = {
            "aac_2_a": 0.05,
            "aac_2_b": 0.1
        }
        adaptive_activation_n = [20] * 4 + [10] * 4
    else: # 'fixed' mode
        adaptive_activation_coeff = {}
        adaptive_activation_init = {}
        adaptive_activation_n = [10] * no_layers

    use_adaptive_activation = use_aac_1 or use_aac_2

    # Physical parameters of the LBPM run (t14_forced_sphere.db), together with the
    # reference scales measured from the dataset. These replace the 2D Hysing benchmark
    # values, which were SI quantities for a light bubble in water and had no
    # correspondence to this lattice-unit run of a dense drop in gas.
    mu = scales["mu"]            # [drop, carrier] dynamic viscosity = rho * nu
    sigma = scales["sigma"]      # measured by Laplace's law (emergent from the PR EOS)
    g = scales["g"]              # Color/F = 0,0,+2e-6, per unit mass, along +z
    rho = scales["rho"]          # [rhoA drop, rhoB carrier]
    u_ref = scales["U_ref"]
    L_ref = scales["L_ref"]
    logger.info(f"Reference scales: L_ref={L_ref:.4f} cells, U_ref={u_ref:.6g} cells/step, "
                f"T_ref={scales['T_ref']:.1f} steps")
    logger.info(f"mu={mu}, rho={rho}, sigma={sigma}, g={g}, rho_mean={scales['rho_mean']:.5f}")
    loss_weights_PDE = [1.0, 10.0, 10.0, 10.0, 1.0]
    epochs_list = [5000] * 5
    learning_rates = [1e-4, 5e-5, 1e-5, 5e-6, 1e-6]
    checkpoint_interval = 10
    num_of_batches = 20

    num_samples_total = sum(len(df) for df in training_data.values())
    total_batch_size = math.ceil(num_samples_total / num_of_batches)

    pinn = TwoPhasePinn(hidden_layers, activation_functions, adaptive_activation_coeff,
                      adaptive_activation_n, adaptive_activation_init, use_adaptive_activation,
                      loss_weights_PDE, mu, sigma, g, rho, u_ref, L_ref, scales["rho_mean"])

    pinn.nn.load_weights('initial_weights_3d.h5')

    start_total = time.time()

    history_loss_a = []
    history_loss_f_uvw = []
    history_loss_f_ma = []

    def to_tensor_tuple(df, columns):
        return tuple(tf.constant(df[c].to_numpy().reshape(-1, 1), dtype=tf.float32) for c in columns)

    # The pressure gauge is a single spatial point over a handful of times, so it is
    # fed whole on every step rather than batched (a proportional batch would round it
    # down to one row and make the gauge unreliably sampled).
    data_GAUGE = to_tensor_tuple(gauge_df, gauge_df.columns)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rates[0], epsilon=1e-8)
    current_best_total_loss = float('inf')
    for i, (epochs, lr) in enumerate(zip(epochs_list, learning_rates)):
        logger.info(f"\n--- Starting Training Phase {i+1}/{len(epochs_list)} ---")
        logger.info(f"Epochs: {epochs}, Learning Rate: {lr}")
        optimizer.learning_rate.assign(lr)
        prop_batch_sizes, num_batches = get_proportional_batch_sizes(total_batch_size, training_data, logger)
        start_checkpoint_time = time.time()
        for epoch in range(1, epochs + 1):
            epoch_losses = []
            shuffled_data = {}
            for key, df in training_data.items():
                length = len(df)
                shuffled_indices = np.random.choice(length, length, replace=False)
                shuffled_data[key] = pd.DataFrame(
                    data=df.to_numpy()[shuffled_indices, :],
                    columns=df.columns
                )

            for b in range(num_batches):
                batch_dict = {}
                for key, df in shuffled_data.items():
                    start_idx = b * prop_batch_sizes[key]
                    end_idx = (b + 1) * prop_batch_sizes[key]
                    batch_dict[key] = df.iloc[start_idx:end_idx]

                if all(batch.empty for batch in batch_dict.values()):
                    continue

                data_A = to_tensor_tuple(batch_dict['A'], batch_dict['A'].columns)
                data_PDE = to_tensor_tuple(batch_dict['PDE'], ['x_PDE', 'y_PDE', 'z_PDE', 't_PDE'])
                data_XPER = to_tensor_tuple(batch_dict['XPER'], ['x_Xp', 'y_Xp', 'z_Xp', 'x_Xm', 'y_Xm', 'z_Xm', 't_XPER'])
                data_YPER = to_tensor_tuple(batch_dict['YPER'], ['x_Yp', 'y_Yp', 'z_Yp', 'x_Ym', 'y_Ym', 'z_Ym', 't_YPER'])
                data_ZPER = to_tensor_tuple(batch_dict['ZPER'], ['x_Zp', 'y_Zp', 'z_Zp', 'x_Zm', 'y_Zm', 'z_Zm', 't_ZPER'])

                batch_loss_values = pinn.train_step(optimizer, data_A, data_PDE, data_XPER, data_YPER, data_ZPER, data_GAUGE)
                epoch_losses.append([l.numpy() for l in batch_loss_values])

            sum_losses = np.sum(epoch_losses, axis=0)
            total_loss, loss_a, loss_bc, loss_m, loss_u, loss_v, loss_w, loss_pde_a = sum_losses

            history_loss_a.append(loss_a)
            history_loss_f_uvw.append(loss_u + loss_v + loss_w)
            history_loss_f_ma.append(loss_m + loss_pde_a)

            if epoch % checkpoint_interval == 0:
                current_time = time.time()
                time_for_epoch = current_time - start_checkpoint_time
                start_checkpoint_time = current_time
                log_msg = f"Epoch: {epoch}/{epochs} - Time: {time_for_epoch:.2f}s - Loss: {total_loss:.4e}"
                log_msg += f" | a: {loss_a:.4e}, BC: {loss_bc:.4e}, m: {loss_m:.4e}"
                log_msg += f", u: {loss_u:.4e}, v: {loss_v:.4e}, w: {loss_w:.4e}, pde_a: {loss_pde_a:.4e}"
                logger.info(log_msg)

            # Saves weights every 'checkpoint_interval' epochs, overwriting the previous file.
            if epoch % checkpoint_interval == 0 and total_loss < current_best_total_loss:
                logger.info(f"Saving checkpoint at epoch {epoch} with loss {total_loss:.4e}")
                # Remove previous checkpoint file to ensure only one exists
                for f in glob.glob(os.path.join(dirname, "*_weights.h5")):
                    os.remove(f)
                # Save new checkpoint with loss in the filename
                safe_loss = f"{total_loss:.4e}".replace("+", "").replace("-", "m")
                weight_filename = f"loss_{safe_loss}.weights.h5"
                pinn.nn.save_weights(os.path.join(dirname, weight_filename))
                current_best_total_loss = total_loss

    total_training_time = time.time() - start_total
    logger.info(f"\nTotal training time: {total_training_time:.3f}s")

    logger.info("\n" + "="*50)
    logger.info("PERFORMING FINAL EVALUATION AND REPORTING")
    logger.info("="*50)

    list_of_files = glob.glob(os.path.join(dirname, '*.weights.h5'))
    if list_of_files:
        latest_file = max(list_of_files, key=os.path.getctime)
        logger.info(f"Loading best model weights from: {os.path.basename(latest_file)}\n")
        pinn.nn.load_weights(latest_file)
    else:
        logger.info("No checkpoint file found. Evaluating with final weights from training.\n")

    logger.info("Calculating final loss...")
    final_evaluation_losses = []
    # Using batching parameters from the last training phase
    for b in range(num_batches):
        batch_dict = {}
        for key, df in training_data.items(): # Using original, non-shuffled data
            start_idx = b * prop_batch_sizes[key]
            end_idx = (b + 1) * prop_batch_sizes[key]
            batch_dict[key] = df.iloc[start_idx:end_idx]

        if all(batch.empty for batch in batch_dict.values()):
            continue

        data_A = to_tensor_tuple(batch_dict['A'], batch_dict['A'].columns)
        data_PDE = to_tensor_tuple(batch_dict['PDE'], ['x_PDE', 'y_PDE', 'z_PDE', 't_PDE'])
        data_XPER = to_tensor_tuple(batch_dict['XPER'], ['x_Xp', 'y_Xp', 'z_Xp', 'x_Xm', 'y_Xm', 'z_Xm', 't_XPER'])
        data_YPER = to_tensor_tuple(batch_dict['YPER'], ['x_Yp', 'y_Yp', 'z_Yp', 'x_Ym', 'y_Ym', 'z_Ym', 't_YPER'])
        data_ZPER = to_tensor_tuple(batch_dict['ZPER'], ['x_Zp', 'y_Zp', 'z_Zp', 'x_Zm', 'y_Zm', 'z_Zm', 't_ZPER'])

        # Compute loss for the batch without training
        batch_losses = pinn.compute_loss(data_A, data_PDE, data_XPER, data_YPER, data_ZPER, data_GAUGE)
        final_evaluation_losses.append([l.numpy() for l in batch_losses])

    # Calculate the mean loss across all batches
    sum_final_losses = np.sum(final_evaluation_losses, axis=0)
    _, loss_a, loss_bc, loss_m, loss_u, loss_v, loss_w, loss_pde_a = sum_final_losses

    logger.info("--- Final Loss Breakdown ---")
    logger.info(f"MSE_alpha (volume fraction): {loss_a:.4e}")
    logger.info(f"MSE_BC                     : {loss_bc:.4e}")
    logger.info(f"MSE_f,m                    : {loss_m:.4e}")
    logger.info(f"MSE_f,u                    : {loss_u:.4e}")
    logger.info(f"MSE_f,v                    : {loss_v:.4e}")
    logger.info(f"MSE_f,w                    : {loss_w:.4e}")
    logger.info(f"MSE_f,a                    : {loss_pde_a:.4e}")
    logger.info("----------------------------\n")

    # --- Plotting and Saving History (No changes needed here) ---
    logger.info("Generating and saving loss history plots...")
    epochs_range = range(1, len(history_loss_a) + 1)

    # Plot 1: MSE_alpha
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_range, history_loss_a)
    plt.title(f'MSE of Volume Fraction (alpha) vs. Epochs ({adaptive_mode} - {activation_choice})')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.yscale('log')
    plt.grid(True, which="both", ls="--")
    plt.savefig(os.path.join(dirname, 'loss_history_alpha.png'))

    # Plot 2: MSE_f,u + MSE_f,v + MSE_f,w
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_range, history_loss_f_uvw)
    plt.title(f'MSE of Momentum (u,v,w) vs. Epochs ({adaptive_mode} - {activation_choice})')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss (f_u + f_v + f_w)')
    plt.yscale('log')
    plt.grid(True, which="both", ls="--")
    plt.savefig(os.path.join(dirname, 'loss_history_momentum_uvw.png'))

    # Plot 3: MSE_f,m + MSE_f,a
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_range, history_loss_f_ma)
    plt.title(f'MSE of Conservation (m,a) vs. Epochs ({adaptive_mode} - {activation_choice})')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss (f_m + f_a)')
    plt.yscale('log')
    plt.grid(True, which="both", ls="--")
    plt.savefig(os.path.join(dirname, 'loss_history_conservation_ma.png'))

    plt.close('all') # Close all figures to free memory
    logger.info("Plots saved successfully.")

    # Save history to a CSV file with a descriptive name
    history_filename = f"loss_history_{adaptive_mode}_{activation_choice}.csv"
    history_filepath = os.path.join(dirname, history_filename)

    history_df = pd.DataFrame({
        'epoch': epochs_range,
        'MSE_alpha': history_loss_a,
        'MSE_f_uvw': history_loss_f_uvw,
        'MSE_f_ma': history_loss_f_ma
    })

    history_df.to_csv(history_filepath, index=False)
    logger.info(f"Loss history data saved to: {history_filepath}")

    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)

if __name__ == "__main__":
    main()
