import sys
sys.path.append("../utilities")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
GPU_ID = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID

import tensorflow as tf
import numpy as np
import optuna
from optuna.pruners import MedianPruner
import math
import time
from datetime import datetime
from generate_points import get_training_data
from utilities import NNCreator
import tensorflow.keras.backend as K
from tensorflow.keras.utils import get_custom_objects

# Register custom activation
def sine_activation(x):
    return K.sin(x)
get_custom_objects().update({'sine': sine_activation})

# Set seeds
np.random.seed(1234)
tf.random.set_seed(1234)

class TwoPhasePinn(tf.keras.Model):
    def __init__(self, hidden_layers, activation_functions, loss_weights_PDE, 
                 mu, sigma, g, rho, u_ref, L_ref):
        super(TwoPhasePinn, self).__init__()
        
        self.mu1, self.mu2 = mu
        self.sigma = sigma
        self.g = g
        self.rho1, self.rho2 = rho
        self.U_ref = u_ref
        self.L_ref = L_ref
        self.rho_ref = self.rho2
        self.loss_weights_PDE = tf.constant(loss_weights_PDE, dtype=tf.float32)
        
        # Build model
        activation_dict = {i: ['tanh', None, 10] for i in range(1, len(hidden_layers) + 1)}
        outputs = ["output_u", "output_v", "output_p", "output_a"]
        activations_output = [None, None, "exponential", "sigmoid"]
        output_layer = list(zip(outputs, activations_output))
        
        nn_creator = NNCreator(tf.float32)
        self.nn = nn_creator.get_model_dnn(3, hidden_layers, output_layer, activation_dict, False)

    def call(self, inputs):
        return self.nn(inputs)

    @tf.function
    def compute_gradients(self, x, y, t):
        with tf.GradientTape(persistent=True) as tape2:
            tape2.watch([x, y, t])
            with tf.GradientTape(persistent=True) as tape1:
                tape1.watch([x, y, t])
                u, v, p, a = self.call(tf.concat([x, y, t], axis=1))
            
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
        
        u_xx = tape2.gradient(u_x, x)
        u_yy = tape2.gradient(u_y, y)
        v_xx = tape2.gradient(v_x, x)
        v_yy = tape2.gradient(v_y, y)
        a_xx = tape2.gradient(a_x, x)
        a_yy = tape2.gradient(a_y, y)
        a_xy = tape2.gradient(a_x, y)
        
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
        x_A, y_A, t_A, a_A = data_A
        x_PDE, y_PDE, t_PDE = data_PDE
        x_N, y_N, t_N, p_N = data_N
        x_E, y_E, t_EW, x_W, y_W = data_EW
        x_NSEW, y_NSEW, t_NSEW, u_NSEW, v_NSEW = data_NSEW
        
        f_PDE = tf.zeros_like(x_PDE)
        
        output_tensors = self.call(tf.concat([x_A, y_A, t_A], axis=1))
        loss_a_A = tf.reduce_mean(tf.square(a_A - output_tensors[3]))
        
        pred_u_NSEW, pred_v_NSEW, _, _ = self.call(tf.concat([x_NSEW, y_NSEW, t_NSEW], axis=1))
        loss_u_NSEW = tf.reduce_mean(tf.square(u_NSEW - pred_u_NSEW))
        loss_v_NSEW = tf.reduce_mean(tf.square(v_NSEW - pred_v_NSEW))
        
        _, _, pred_p_N, _ = self.call(tf.concat([x_N, y_N, t_N], axis=1))
        loss_p_N = tf.reduce_mean(tf.square(p_N - pred_p_N))
        
        pred_east = self.call(tf.concat([x_E, y_E, t_EW], axis=1))
        pred_west = self.call(tf.concat([x_W, y_W, t_EW], axis=1))
        loss_u_EW = tf.reduce_mean(tf.square(pred_east[0] - pred_west[0]))
        loss_v_EW = tf.reduce_mean(tf.square(pred_east[1] - pred_west[1]))
        loss_p_EW = tf.reduce_mean(tf.square(pred_east[2] - pred_west[2]))
        
        loss_BC = loss_u_NSEW + loss_v_NSEW + loss_p_N + loss_u_EW + loss_v_EW + loss_p_EW
        
        PDE_m, PDE_u, PDE_v, PDE_a = self.PDE_caller(x_PDE, y_PDE, t_PDE)
        loss_PDE_m = tf.reduce_mean(tf.square(f_PDE - PDE_m))
        loss_PDE_u = tf.reduce_mean(tf.square(f_PDE - PDE_u))
        loss_PDE_v = tf.reduce_mean(tf.square(f_PDE - PDE_v))
        loss_PDE_a = tf.reduce_mean(tf.square(f_PDE - PDE_a))
        
        loss_PDE = tf.tensordot(tf.stack([loss_PDE_m, loss_PDE_u, loss_PDE_v, loss_PDE_a]), 
                                self.loss_weights_PDE, 1)
        
        total_loss = loss_a_A + loss_BC + loss_PDE
        
        return total_loss

    @tf.function
    def train_step(self, optimizer, data_A, data_PDE, data_N, data_EW, data_NSEW):
        with tf.GradientTape() as tape:
            total_loss = self.compute_loss(data_A, data_PDE, data_N, data_EW, data_NSEW)
        
        gradients = tape.gradient(total_loss, self.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        return total_loss


def objective(trial):
    """Optuna objective function with early stopping."""
    
    try:
        # Sample hyperparameters
        n_layers = trial.suggest_int('n_layers', 6, 10)
        n_neurons = trial.suggest_int('n_neurons', 256, 512, step=32)
        
        # Loss weights (logarithmic scale)
        weight_m = trial.suggest_float('weight_m', 0.1, 10.0, log=True)
        weight_u = trial.suggest_float('weight_u', 0.1, 10.0, log=True)
        weight_v = trial.suggest_float('weight_v', 0.1, 10.0, log=True)
        weight_a = trial.suggest_float('weight_a', 0.1, 10.0, log=True)
        
        loss_weights_PDE = [weight_m, weight_u, weight_v, weight_a]
        hidden_layers = [n_neurons] * n_layers
        
        print(f"\nTrial {trial.number}: layers={n_layers}, neurons={n_neurons}")
        print(f"  Loss weights: m={weight_m:.3f}, u={weight_u:.3f}, v={weight_v:.3f}, a={weight_a:.3f}")
    except:
        pass
    
    # Load training data
    NOP_a = (500, 400)
    NOP_PDE = (400, 2000, 3000)
    NOP_north = (20, 20)
    NOP_south = (20, 20)
    NOP_east = (20, 20)
    NOP_west = (20, 20)
    
    training_data = get_training_data(NOP_a, NOP_PDE, NOP_north, NOP_south, NOP_east, NOP_west)
    
    # Setup model
    mu = [1.0, 10.0]
    sigma = 24.5
    g = -0.98
    rho = [100, 1000]
    u_ref = 1.0
    L_ref = 0.25
    
    pinn = TwoPhasePinn(hidden_layers, {}, loss_weights_PDE, mu, sigma, g, rho, u_ref, L_ref)
    #pinn.nn.load_weights('initial_weights.h5')
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    
    # Training setup
    num_of_batches = 15
    num_samples_total = sum(len(df) for df in training_data.values())
    total_batch_size = math.ceil(num_samples_total / num_of_batches)
    
    # Calculate batch sizes
    batch_sizes = {}
    for key, df in training_data.items():
        if len(df) > 0:
            proportion = len(df) / num_samples_total
            batch_sizes[key] = math.ceil(proportion * total_batch_size)
        else:
            batch_sizes[key] = 0
    
    def to_tensor_tuple(df, columns):
        return tuple(tf.constant(df[c].to_numpy().reshape(-1, 1), dtype=tf.float32) for c in columns)
    
    # Early stopping parameters
    max_epochs = 2500
    patience = 100
    best_loss = float('inf')
    patience_counter = 0
    print_checkpoint = 10
    start_time = time.time()
    
    # Training loop
    for epoch in range(1, max_epochs + 1):
        epoch_losses = []
        shuffled_data = {key: df.sample(frac=1) for key, df in training_data.items()}
        
        for b in range(num_of_batches):
            batch_dict = {}
            for key, df in shuffled_data.items():
                start_idx = b * batch_sizes[key]
                end_idx = (b + 1) * batch_sizes[key]
                batch_dict[key] = df.iloc[start_idx:end_idx]
            
            if all(batch.empty for batch in batch_dict.values()):
                continue
            
            data_A = to_tensor_tuple(batch_dict['A'], batch_dict['A'].columns)
            data_PDE = to_tensor_tuple(batch_dict['PDE'], ['x_PDE', 'y_PDE', 't_PDE'])
            data_N = to_tensor_tuple(batch_dict['N'], batch_dict['N'].columns)
            data_EW = to_tensor_tuple(batch_dict['EW'], ['x_E', 'y_E', 't_EW', 'x_W', 'y_W'])
            data_NSEW = to_tensor_tuple(batch_dict['NSEW'], batch_dict['NSEW'].columns)
            
            loss = pinn.train_step(optimizer, data_A, data_PDE, data_N, data_EW, data_NSEW)
            epoch_losses.append(loss.numpy())
        
        avg_loss = np.mean(epoch_losses)
        if (epoch%print_checkpoint == 0):
            print(f"Epoch: {epoch}/{max_epochs} - Loss: {avg_loss:.4e}")
        # Report to Optuna for pruning
        trial.report(avg_loss, epoch)
        
        if trial.should_prune():
            raise optuna.TrialPruned()
        
        # Early stopping check
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            break
    
    training_time = time.time() - start_time
    
    # Calculate weighted score: accuracy (2x weight) and time (1x weight)
    # Lower loss is better, lower time is better
    score = 2.0 * best_loss + training_time / 1000.0  # Normalize time
    
    return score


def main():
    """Run Optuna optimization."""
    
    # Create study
    study = optuna.create_study(
        direction='minimize',
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=100)
    )
    
    print("Starting hyperparameter optimization...")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Run optimization
    study.optimize(objective, n_trials=50, timeout=None)
    
    # Print results
    print("\n" + "="*50)
    print("OPTIMIZATION COMPLETE")
    print("="*50)
    print(f"\nBest trial: {study.best_trial.number}")
    print(f"Best score: {study.best_value:.6e}")
    print("\nBest hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # Save results
    results_file = f"optuna_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    study.trials_dataframe().to_csv(results_file, index=False)
    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()