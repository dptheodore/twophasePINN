import sys
# Assuming 'utilities' and 'generate_points' are in this path
# You might need to adjust this path based on your project structure.
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
from generate_points import get_training_data
# The 'utilities' module from the original script is not fully provided.
# A placeholder class for NNCreator and a function for writeToJSONFile are added.
# You will need to ensure the actual 'utilities' module is available and compatible.
from utilities import NNCreator, writeToJSONFile 
import time
import math
import glob
from datetime import datetime
import shutil
import logging

# Set random seeds for reproducibility
np.random.seed(1234)
tf.random.set_seed(1234)

class TwoPhasePinn(tf.keras.Model):
    """
    This class implements a physics-informed neural network in TensorFlow 2.
    It approximates the incompressible two-phase Navier-Stokes equations in 2D
    using a Volume-of-Fluid approach.
    """

    def __init__(self, hidden_layers, activation_functions, adaptive_activation_coeff,
                 adaptive_activation_n, adaptive_activation_init, use_ad_act,
                 loss_weights_PDE, mu, sigma, g, rho, u_ref, L_ref):
        super(TwoPhasePinn, self).__init__()

        # Physical Parameters
        self.mu1, self.mu2 = mu
        self.sigma = sigma
        self.g = g
        self.rho1, self.rho2 = rho
        self.U_ref = u_ref
        self.L_ref = L_ref
        self.rho_ref = self.rho2

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
        outputs = ["output_u", "output_v", "output_p", "output_a"]
        activations_output = [None, None, "exponential", "sigmoid"]
        output_layer = list(zip(outputs, activations_output))

        # 3. Instantiate NNCreator and build the model
        nn_creator = NNCreator(tf.float32)
        self.nn = nn_creator.get_model_dnn(3, hidden_layers, output_layer, activation_functions_dict, self.use_ad_act)

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
    def compute_gradients(self, x, y, t):
        # Use a nested tape to compute second-order derivatives
        with tf.GradientTape(persistent=True) as tape2:
            tape2.watch([x, y, t])
            with tf.GradientTape(persistent=True) as tape1:
                tape1.watch([x, y, t])
                u, v, p, a = self.call(tf.concat([x, y, t], axis=1))

            # First-order gradients computed with the inner tape
            u_x = tape1.gradient(u, x)
            u_y = tape1.gradient(u, y)
            u_t = tape1.gradient(u, t)
            v_x = tape1.gradient(v, x)
            v_y = tape1.gradient(v, y)
            v_t = tape1.gradient(v, t)
            p_x = tape1.gradient(p, x)
            p_y = tape1.gradient(p, y)
            a_x = tape1.gradient(a, x)
            a_y = tape1.gradient(a, y)
            a_t = tape1.gradient(a, t)

        # Second-order gradients computed with the outer tape
        u_xx = tape2.gradient(u_x, x)
        u_yy = tape2.gradient(u_y, y)
        v_xx = tape2.gradient(v_x, x)
        v_yy = tape2.gradient(v_y, y)
        a_xx = tape2.gradient(a_x, x)
        a_yy = tape2.gradient(a_y, y)
        a_xy = tape2.gradient(a_x, y)

        # Clean up tapes
        del tape1, tape2

        return (u, u_x, u_y, u_t, u_xx, u_yy), \
               (v, v_x, v_y, v_t, v_xx, v_yy), \
               (p, p_x, p_y), \
               (a, a_x, a_y, a_t, a_xx, a_yy, a_xy)

    @tf.function
    def PDE_caller(self, x, y, t):
        u_grads, v_grads, p_grads, a_grads = self.compute_gradients(x, y, t)
        u, u_x, u_y, u_t, u_xx, u_yy = u_grads
        v, v_x, v_y, v_t, v_xx, v_yy = v_grads
        p, p_x, p_y = p_grads
        a, a_x, a_y, a_t, a_xx, a_yy, a_xy = a_grads

        mu = self.mu2 + (self.mu1 - self.mu2) * a
        mu_x = (self.mu1 - self.mu2) * a_x
        mu_y = (self.mu1 - self.mu2) * a_y
        rho = self.rho2 + (self.rho1 - self.rho2) * a

        abs_interface_grad = tf.sqrt(a_x**2 + a_y**2 + np.finfo(float).eps)
        curvature = -((a_xx + a_yy) / abs_interface_grad -
                      (a_x**2 * a_xx + a_y**2 * a_yy + 2 * a_x * a_y * a_xy) / tf.pow(abs_interface_grad, 3))

        one_Re = mu / (self.rho_ref * self.U_ref * self.L_ref)
        one_Re_x = mu_x / (self.rho_ref * self.U_ref * self.L_ref)
        one_Re_y = mu_y / (self.rho_ref * self.U_ref * self.L_ref)
        one_We = self.sigma / (self.rho_ref * self.U_ref**2 * self.L_ref)
        one_Fr = self.g * self.L_ref / self.U_ref**2

        PDE_m = u_x + v_y
        PDE_a = a_t + u * a_x + v * a_y
        PDE_u = (u_t + u * u_x + v * u_y) * rho / self.rho_ref + p_x - \
                one_We * curvature * a_x - one_Re * (u_xx + u_yy) - \
                2.0 * one_Re_x * u_x - one_Re_y * (u_y + v_x)
        PDE_v = (v_t + u * v_x + v * v_y) * rho / self.rho_ref + p_y - \
                one_We * curvature * a_y - one_Re * (v_xx + v_yy) - \
                rho / self.rho_ref * one_Fr - 2.0 * one_Re_y * v_y - one_Re_x * (u_y + v_x)

        return PDE_m, PDE_u, PDE_v, PDE_a

    @tf.function
    def compute_loss(self, data_A, data_PDE, data_N, data_EW, data_NSEW):
        # Unpack tensor tuples
        x_A, y_A, t_A, a_A = data_A
        x_PDE, y_PDE, t_PDE = data_PDE
        x_N, y_N, t_N, p_N = data_N
        x_E, y_E, t_EW, x_W, y_W = data_EW
        x_NSEW, y_NSEW, t_NSEW, u_NSEW, v_NSEW = data_NSEW

        f_PDE = tf.zeros_like(x_PDE)

        # Loss A (Volume Fraction)
        output_tensors = self.call(tf.concat([x_A, y_A, t_A], axis=1))
        loss_a_A = tf.reduce_mean(tf.square(a_A - output_tensors[3]))
        #tf.print(loss_a_A)

        # Loss NSEW (Boundary Conditions)
        pred_u_NSEW, pred_v_NSEW, _, _ = self.call(tf.concat([x_NSEW, y_NSEW, t_NSEW], axis=1))
        loss_u_NSEW = tf.reduce_mean(tf.square(u_NSEW - pred_u_NSEW))
        loss_v_NSEW = tf.reduce_mean(tf.square(v_NSEW - pred_v_NSEW))

        # Loss N (Pressure at North boundary)
        _, _, pred_p_N, _ = self.call(tf.concat([x_N, y_N, t_N], axis=1))
        loss_p_N = tf.reduce_mean(tf.square(p_N - pred_p_N))

        # Loss EW (Periodic Boundary)
        pred_east = self.call(tf.concat([x_E, y_E, t_EW], axis=1))
        pred_west = self.call(tf.concat([x_W, y_W, t_EW], axis=1))
        loss_u_EW = tf.reduce_mean(tf.square(pred_east[0] - pred_west[0]))
        loss_v_EW = tf.reduce_mean(tf.square(pred_east[1] - pred_west[1]))
        loss_p_EW = tf.reduce_mean(tf.square(pred_east[2] - pred_west[2]))

        loss_BC = loss_u_NSEW + loss_v_NSEW + loss_p_N + loss_u_EW + loss_v_EW + loss_p_EW

        # Loss PDE (Physics-Informed)
        PDE_m, PDE_u, PDE_v, PDE_a = self.PDE_caller(x_PDE, y_PDE, t_PDE)
        loss_PDE_m = tf.reduce_mean(tf.square(f_PDE - PDE_m))
        loss_PDE_u = tf.reduce_mean(tf.square(f_PDE - PDE_u))
        loss_PDE_v = tf.reduce_mean(tf.square(f_PDE - PDE_v))
        loss_PDE_a = tf.reduce_mean(tf.square(f_PDE - PDE_a))

        loss_PDE = tf.tensordot(tf.stack([loss_PDE_m, loss_PDE_u, loss_PDE_v, loss_PDE_a]), self.loss_weights_PDE, 1)

        # Total Loss
        total_loss = loss_a_A + loss_BC + loss_PDE

        return total_loss, loss_a_A, loss_BC, loss_PDE_m, loss_PDE_u, loss_PDE_v, loss_PDE_a


    @tf.function
    def train_step(self, optimizer, data_A, data_PDE, data_N, data_EW, data_NSEW):
        with tf.GradientTape() as tape:
            # Pass the tensor tuples directly to compute_loss
            losses = self.compute_loss(data_A, data_PDE, data_N, data_EW, data_NSEW)
            total_loss = losses[0]

        gradients = tape.gradient(total_loss, self.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return losses


def setup_output_directory():
    if not os.path.exists("checkpoints"):
        os.mkdir("checkpoints")
    dirname = os.path.abspath(os.path.join("checkpoints", datetime.now().strftime("%b-%d-%Y_%H-%M-%S")))
    os.mkdir(dirname)
    
    # Copy essential files for reproducibility
    shutil.copyfile(__file__, os.path.join(dirname, os.path.basename(__file__)))
    if os.path.exists("generate_points.py"):
        shutil.copyfile("generate_points.py", os.path.join(dirname, "generate_points.py"))
    
    logpath = os.path.join(dirname, "output.log")
    return dirname, logpath

def get_logger(logpath):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # StreamHandler for console output
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(sh)
    
    # FileHandler for logging to a file
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
    This script trains a PINN for the rising bubble case.
    """
    dirname, logpath = setup_output_directory()
    logger = get_logger(logpath)

    # --- Parameters and Data Loading (no changes here) ---
    NOP_a = (500, 400)
    NOP_PDE = (400, 2000, 3000)
    NOP_north = (20, 20)
    NOP_south = (20, 20)
    NOP_east = (20, 20)
    NOP_west = (20, 20)
    training_data = get_training_data(NOP_a, NOP_PDE, NOP_north, NOP_south, NOP_east, NOP_west)

    # --- NN Architecture and Hyperparameters (no changes here) ---
    no_layers = 8
    hidden_layers = [400] * no_layers
    activation_functions = {'tanh': range(1, no_layers + 1)}
    adaptive_activation_coeff = {"aac_1": range(1, no_layers + 1)}
    adaptive_activation_init = {"aac_1": 0.1}
    adaptive_activation_n = [10] * no_layers
    use_ad_act = False
    mu = [1.0, 10.0]
    sigma = 24.5
    g = -0.98
    rho = [100, 1000]
    u_ref = 1.0
    L_ref = 0.25
    loss_weights_PDE = [1.0, 10.0, 10.0, 1.0]
    epochs_list = [5000] * 5
    learning_rates = [1e-4, 5e-5, 1e-5, 5e-6, 1e-6]
    checkpoint_interval = 100
    num_of_batches = 20
    # --- Set Total Batch Size ---
    # The original script used 20 batches. We calculate the equivalent total batch size.
    num_samples_total = sum(len(df) for df in training_data.values())
    total_batch_size = math.ceil(num_samples_total / num_of_batches) 

    # --- Instantiate PINN (no changes here) ---
    pinn = TwoPhasePinn(hidden_layers, activation_functions, adaptive_activation_coeff,
                      adaptive_activation_n, adaptive_activation_init, use_ad_act,
                      loss_weights_PDE, mu, sigma, g, rho, u_ref, L_ref)

    pinn.nn.load_weights('initial_weights.h5')

    # first_layer_weights = pinn.nn.layers[1].get_weights()[0]
    # print("TF2 First Layer Weights (slice):\n", first_layer_weights[:2, :2])
    # Comment above out to verify same weights as paper should be:
    # TF2 First Layer Weights (slice):
    #    [[-0.09551109 -0.03110073]
    #    [-0.08348112  0.0810886 ]]


    # --- REVISED TRAINING LOOP WITH MINI-BATCHING ---
    start_total = time.time()
    epoch_loss_checkpoints = 1e10
    
    # Helper for converting a DataFrame batch to tensors
    def to_tensor_tuple(df, columns):
        return tuple(tf.constant(df[c].to_numpy().reshape(-1, 1), dtype=tf.float32) for c in columns)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rates[0])
    for i, (epochs, lr) in enumerate(zip(epochs_list, learning_rates)):
        logger.info(f"\n--- Starting Training Phase {i+1}/{len(epochs_list)} ---")
        logger.info(f"Epochs: {epochs}, Learning Rate: {lr}")
        optimizer.learning_rate.assign(lr)
        # Get proportional batch sizes for this training phase
        prop_batch_sizes, num_batches = get_proportional_batch_sizes(total_batch_size, training_data, logger)
        start_epoch = time.time()
        for epoch in range(1, epochs + 1):
            epoch_losses = []

            # Shuffle data at the beginning of each epoch
            shuffled_data = {key: df.sample(frac=1) for key, df in training_data.items()}

            for b in range(num_batches):
                # Create the mini-batch for each data type
                batch_dict = {}
                for key, df in shuffled_data.items():
                    start_idx = b * prop_batch_sizes[key]
                    end_idx = (b + 1) * prop_batch_sizes[key]
                    batch_dict[key] = df.iloc[start_idx:end_idx]

                # Skip empty batches
                if all(batch.empty for batch in batch_dict.values()):
                    continue

                # Convert the mini-batch DataFrames to tensors
                data_A = to_tensor_tuple(batch_dict['A'], batch_dict['A'].columns)
                data_PDE = to_tensor_tuple(batch_dict['PDE'], ['x_PDE', 'y_PDE', 't_PDE'])
                data_N = to_tensor_tuple(batch_dict['N'], batch_dict['N'].columns)
                data_EW = to_tensor_tuple(batch_dict['EW'], ['x_E', 'y_E', 't_EW', 'x_W', 'y_W'])
                data_NSEW = to_tensor_tuple(batch_dict['NSEW'], batch_dict['NSEW'].columns)
                
                tensor_data_tuple = (data_A, data_PDE, data_N, data_EW, data_NSEW)
                
                # Perform one training step on the mini-batch
                batch_loss_values = pinn.train_step(optimizer, *tensor_data_tuple)
                epoch_losses.append([l.numpy() for l in batch_loss_values])

            # Calculate average loss for the epoch
            avg_losses = np.sum(epoch_losses, axis=0)
            total_loss, loss_a, loss_bc, loss_m, loss_u, loss_v, loss_pde_a = avg_losses
            
            
            # Logging
            num_epochs_per_log = 1
            if epoch % num_epochs_per_log == 0:
                time_for_epoch = time.time() - start_epoch
                start_epoch = time.time()
                log_msg = f"Epoch: {epoch}/{epochs} - Time for {num_epochs_per_log} epochs: {time_for_epoch:.2f}s - Loss: {total_loss:.4e}"
                log_msg += f" | a: {loss_a:.4e}, NSEW: {loss_bc:.4e}, m: {loss_m:.4e}"
                log_msg += f", u: {loss_u:.4e}, v: {loss_v:.4e}, pde_a: {loss_pde_a:.4e}"
                logger.info(log_msg)
            
            if total_loss < epoch_loss_checkpoints and epoch % checkpoint_interval == 0:
                logger.info(f"Saving checkpoint at epoch {epoch} with loss {total_loss:.4e}")
                for f in glob.glob(os.path.join(dirname, "*_weights.h5")):
                    os.remove(f)
                # sanitize the loss for the filename
                safe_loss = f"{total_loss:.4e}".replace("+", "").replace("-", "m")
                weight_filename = f"loss_{safe_loss}.weights.h5"
                pinn.nn.save_weights(os.path.join(dirname, weight_filename))
                epoch_loss_checkpoints = total_loss

    total_training_time = time.time() - start_total
    logger.info(f"\nTotal training time: {total_training_time:.3f}s")
    
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)

if __name__ == "__main__":
    main()