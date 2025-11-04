import sys
sys.path.append("../utilities")
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
GPU_ID = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import scipy.io
from generate_points import get_training_data
from pytorch_utilities import NNCreator, writeToJSONFile
import time
import math
import glob
from datetime import datetime
import shutil
import logging
import matplotlib.pyplot as plt

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set random seeds for reproducibility
np.random.seed(1234)
torch.manual_seed(1234)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class SineActivation(nn.Module):
    """Custom sine activation function"""
    def forward(self, x):
        return torch.sin(x)


class AdaptiveActivation(nn.Module):
    """Adaptive activation wrapper"""
    def __init__(self, activation_fn, coeff=0.1, n=10):
        super().__init__()
        self.activation_fn = activation_fn
        self.coeff = nn.Parameter(torch.tensor(coeff, dtype=torch.float32))
        self.n = n
    
    def forward(self, x):
        return self.activation_fn(self.n * self.coeff * x)


class DenseBlock(nn.Module):
    """Dense layer with optional adaptive activation"""
    def __init__(self, in_features, out_features, activation=None, use_adaptive=False, 
                 adaptive_coeff=0.1, adaptive_n=10):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.use_adaptive = use_adaptive
        
        if activation is not None:
            if activation == 'tanh':
                act_fn = nn.Tanh()
            elif activation == 'sine':
                act_fn = SineActivation()
            elif activation == 'relu':
                act_fn = nn.ReLU()
            else:
                act_fn = nn.Tanh()
            
            if use_adaptive:
                self.activation = AdaptiveActivation(act_fn, adaptive_coeff, adaptive_n)
            else:
                self.activation = act_fn
        else:
            self.activation = None
    
    def forward(self, x):
        x = self.linear(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class TwoPhasePinnNet(nn.Module):
    """Neural network architecture for two-phase PINN"""
    def __init__(self, hidden_layers, activation_functions, adaptive_activation_coeff,
                 adaptive_activation_n, adaptive_activation_init, use_ad_act):
        super().__init__()
        
        layers = []
        in_features = 3  # x, y, t
        
        # Build hidden layers
        for i, h in enumerate(hidden_layers):
            layer_idx = i + 1
            activation = None
            for act_name, layer_list in activation_functions.items():
                if layer_idx in layer_list:
                    activation = act_name
                    break
            
            adaptive_coeff = 0.1
            adaptive_n = adaptive_activation_n[i] if i < len(adaptive_activation_n) else 10
            
            if use_ad_act:
                for coeff_name, layer_list in adaptive_activation_coeff.items():
                    if layer_idx in layer_list:
                        adaptive_coeff = adaptive_activation_init[coeff_name]
                        break
            
            layers.append(DenseBlock(in_features, h, activation, use_ad_act, 
                                    adaptive_coeff, adaptive_n))
            in_features = h
        
        self.hidden_layers = nn.ModuleList(layers)
        
        # Output layers: u, v, p, a
        self.output_u = nn.Linear(in_features, 1)
        self.output_v = nn.Linear(in_features, 1)
        self.output_p = nn.Linear(in_features, 1)
        self.output_a = nn.Linear(in_features, 1)
        
    def forward(self, x):
        for layer in self.hidden_layers:
            x = layer(x)
        
        u = self.output_u(x)
        v = self.output_v(x)
        p = torch.exp(self.output_p(x))  # exponential activation
        a = torch.sigmoid(self.output_a(x))  # sigmoid activation
        
        return u, v, p, a


class TwoPhasePinn(nn.Module):
    """
    Physics-informed neural network for incompressible two-phase 
    Navier-Stokes equations in 2D using Volume-of-Fluid approach.
    """
    def __init__(self, hidden_layers, activation_functions, adaptive_activation_coeff,
                 adaptive_activation_n, adaptive_activation_init, use_ad_act,
                 loss_weights_PDE, mu, sigma, g, rho, u_ref, L_ref):
        super().__init__()
        
        # Physical Parameters
        self.mu1, self.mu2 = mu
        self.sigma = sigma
        self.g = g
        self.rho1, self.rho2 = rho
        self.U_ref = u_ref
        self.L_ref = L_ref
        self.rho_ref = self.rho2
        
        # Loss weights
        self.loss_weights_PDE = torch.tensor(loss_weights_PDE, dtype=torch.float32, device=device)
        
        # Build network
        self.nn = TwoPhasePinnNet(hidden_layers, activation_functions, 
                                  adaptive_activation_coeff, adaptive_activation_n,
                                  adaptive_activation_init, use_ad_act)
        self.nn.to(device)
    
    def forward(self, inputs):
        return self.nn(inputs)
    
    def tf_median(self, x):
        """Calculate median"""
        x_flat = x.reshape(-1)
        return torch.median(x_flat)
    
    def compute_gradients(self, x, y, t):
        """Compute first and second order derivatives - OPTIMIZED"""
        x = x.requires_grad_(True)
        y = y.requires_grad_(True)
        t = t.requires_grad_(True)
        
        inputs = torch.cat([x, y, t], dim=1)
        u, v, p, a = self(inputs)
        
        # Compute all first-order gradients in one pass using create_graph=True
        grad_outputs = torch.ones_like(u)
        
        # U gradients
        u_grads = torch.autograd.grad(u, [x, y, t], grad_outputs, create_graph=True)
        u_x, u_y, u_t = u_grads
        
        # V gradients  
        v_grads = torch.autograd.grad(v, [x, y, t], grad_outputs, create_graph=True)
        v_x, v_y, v_t = v_grads
        
        # P gradients
        p_grads = torch.autograd.grad(p, [x, y], grad_outputs, create_graph=True)
        p_x, p_y = p_grads
        
        # A gradients (including second order for curvature)
        a_grads = torch.autograd.grad(a, [x, y, t], grad_outputs, create_graph=True)
        a_x, a_y, a_t = a_grads
        
        # Second order gradients - only compute what's needed
        u_xx = torch.autograd.grad(u_x, x, grad_outputs, create_graph=True)[0]
        u_yy = torch.autograd.grad(u_y, y, grad_outputs, create_graph=True)[0]
        
        v_xx = torch.autograd.grad(v_x, x, grad_outputs, create_graph=True)[0]
        v_yy = torch.autograd.grad(v_y, y, grad_outputs, create_graph=True)[0]
        
        a_xx = torch.autograd.grad(a_x, x, grad_outputs, create_graph=True)[0]
        a_yy = torch.autograd.grad(a_y, y, grad_outputs, create_graph=True)[0]
        a_xy = torch.autograd.grad(a_x, y, grad_outputs, create_graph=True)[0]
        
        return (u, u_x, u_y, u_t, u_xx, u_yy), \
               (v, v_x, v_y, v_t, v_xx, v_yy), \
               (p, p_x, p_y), \
               (a, a_x, a_y, a_t, a_xx, a_yy, a_xy)
    
    def PDE_caller(self, x, y, t):
        """Compute PDE residuals"""
        u_grads, v_grads, p_grads, a_grads = self.compute_gradients(x, y, t)
        u, u_x, u_y, u_t, u_xx, u_yy = u_grads
        v, v_x, v_y, v_t, v_xx, v_yy = v_grads
        p, p_x, p_y = p_grads
        a, a_x, a_y, a_t, a_xx, a_yy, a_xy = a_grads
        
        mu = self.mu2 + (self.mu1 - self.mu2) * a
        mu_x = (self.mu1 - self.mu2) * a_x
        mu_y = (self.mu1 - self.mu2) * a_y
        rho = self.rho2 + (self.rho1 - self.rho2) * a
        
        abs_interface_grad = torch.sqrt(a_x**2 + a_y**2 + np.finfo(float).eps)
        curvature = -((a_xx + a_yy) / abs_interface_grad -
                      (a_x**2 * a_xx + a_y**2 * a_yy + 2 * a_x * a_y * a_xy) / 
                      torch.pow(abs_interface_grad, 3))
        
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
                rho / self.rho_ref * one_Fr - 2.0 * one_Re_y * v_y - \
                one_Re_x * (u_y + v_x)
        
        return PDE_m, PDE_u, PDE_v, PDE_a
    
    def compute_loss(self, data_A, data_PDE, data_N, data_EW, data_NSEW):
        """Compute total loss"""
        x_A, y_A, t_A, a_A = data_A
        x_PDE, y_PDE, t_PDE = data_PDE
        x_N, y_N, t_N, p_N = data_N
        x_E, y_E, t_EW, x_W, y_W = data_EW
        x_NSEW, y_NSEW, t_NSEW, u_NSEW, v_NSEW = data_NSEW
        
        f_PDE = torch.zeros_like(x_PDE)
        
        # Loss A (Volume Fraction)
        _, _, _, pred_a_A = self(torch.cat([x_A, y_A, t_A], dim=1))
        loss_a_A = torch.mean((a_A - pred_a_A)**2)
        
        # Loss NSEW (Boundary Conditions)
        pred_u_NSEW, pred_v_NSEW, _, _ = self(torch.cat([x_NSEW, y_NSEW, t_NSEW], dim=1))
        loss_u_NSEW = torch.mean((u_NSEW - pred_u_NSEW)**2)
        loss_v_NSEW = torch.mean((v_NSEW - pred_v_NSEW)**2)
        
        # Loss N (Pressure at North boundary)
        _, _, pred_p_N, _ = self(torch.cat([x_N, y_N, t_N], dim=1))
        loss_p_N = torch.mean((p_N - pred_p_N)**2)
        
        # Loss EW (Periodic Boundary)
        pred_east = self(torch.cat([x_E, y_E, t_EW], dim=1))
        pred_west = self(torch.cat([x_W, y_W, t_EW], dim=1))
        loss_u_EW = torch.mean((pred_east[0] - pred_west[0])**2)
        loss_v_EW = torch.mean((pred_east[1] - pred_west[1])**2)
        loss_p_EW = torch.mean((pred_east[2] - pred_west[2])**2)
        
        loss_BC = loss_u_NSEW + loss_v_NSEW + loss_p_N + loss_u_EW + loss_v_EW + loss_p_EW
        
        # Loss PDE (Physics-Informed)
        PDE_m, PDE_u, PDE_v, PDE_a = self.PDE_caller(x_PDE, y_PDE, t_PDE)
        loss_PDE_m = torch.mean((f_PDE - PDE_m)**2)
        loss_PDE_u = torch.mean((f_PDE - PDE_u)**2)
        loss_PDE_v = torch.mean((f_PDE - PDE_v)**2)
        loss_PDE_a = torch.mean((f_PDE - PDE_a)**2)
        
        loss_PDE = torch.dot(torch.stack([loss_PDE_m, loss_PDE_u, loss_PDE_v, loss_PDE_a]), 
                            self.loss_weights_PDE)
        
        total_loss = loss_a_A + loss_BC + loss_PDE
        
        return total_loss, loss_a_A, loss_BC, loss_PDE_m, loss_PDE_u, loss_PDE_v, loss_PDE_a
    
    def train_step_with_accumulation(self, optimizer, data_batches, accumulation_steps):
        """Training step with gradient accumulation across batches"""
        optimizer.zero_grad()
        
        total_losses = None
        num_accumulated = 0
        
        for batch_data in data_batches[:accumulation_steps]:
            data_A, data_PDE, data_N, data_EW, data_NSEW = batch_data
            
            losses = self.compute_loss(data_A, data_PDE, data_N, data_EW, data_NSEW)
            
            # Scale loss by accumulation steps
            total_loss = losses[0] / accumulation_steps
            total_loss.backward()
            
            # Accumulate losses for reporting
            if total_losses is None:
                total_losses = [l.item() for l in losses]
            else:
                total_losses = [tl + l.item() for tl, l in zip(total_losses, losses)]
            
            num_accumulated += 1
        
        optimizer.step()
        
        # Average the accumulated losses
        total_losses = [tl / num_accumulated for tl in total_losses]
        return [torch.tensor(l) for l in total_losses]


def setup_output_directory():
    """Setup output directory for checkpoints and logs"""
    if not os.path.exists("checkpoints"):
        os.mkdir("checkpoints")
    dirname = os.path.abspath(os.path.join("checkpoints", 
                                           datetime.now().strftime("%b-%d-%Y_%H-%M-%S")))
    os.mkdir(dirname)
    
    shutil.copyfile(__file__, os.path.join(dirname, os.path.basename(__file__)))
    if os.path.exists("generate_points.py"):
        shutil.copyfile("generate_points.py", os.path.join(dirname, "generate_points.py"))
    
    logpath = os.path.join(dirname, "output.log")
    return dirname, logpath


def get_logger(logpath):
    """Setup logger"""
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
    """Calculate proportional batch sizes for each dataset"""
    num_samples_total = sum(len(df) for df in training_data.values())
    num_batches = math.ceil(num_samples_total / total_batch_size)
    
    batch_sizes = {}
    for key, df in training_data.items():
        if len(df) > 0:
            proportion = len(df) / num_samples_total
            batch_sizes[key] = math.ceil(proportion * total_batch_size)
        else:
            batch_sizes[key] = 0
    
    batch_sizes = {
        'A': max(16, batch_sizes['A']),
        'PDE': max(32, batch_sizes['PDE']),
        'N': max(16, batch_sizes['N']),
        'EW': max(16, batch_sizes['EW']),
        'NSEW': max(16, batch_sizes['NSEW'])
    }
    
    logger.info(f"Total samples: {num_samples_total}, Desired batch size: {total_batch_size}")
    logger.info(f"Calculated num_batches: {num_batches}, Proportional batch sizes: {batch_sizes}")
    return batch_sizes, num_batches


def to_tensor_tuple(df, columns):
    """Convert DataFrame columns to tensor tuple"""
    return tuple(torch.tensor(df[c].to_numpy().reshape(-1, 1), 
                             dtype=torch.float32, device=device) for c in columns)


def preload_training_data_to_gpu(training_data):
    """Convert all training data to GPU tensors once"""
    gpu_data = {}
    lengths = {}
    
    for key, df in training_data.items():
        gpu_data[key] = {}
        for col in df.columns:
            gpu_data[key][col] = torch.tensor(
                df[col].to_numpy().reshape(-1, 1), 
                dtype=torch.float32, 
                device=device
            )
        lengths[key] = len(df)
    
    return gpu_data, lengths


def get_batch_tensors(gpu_data, indices, key, start_idx, end_idx, columns=None):
    """Extract batch from pre-loaded GPU tensors"""
    batch_indices = indices[key][start_idx:end_idx]
    if columns is None:
        columns = gpu_data[key].keys()
    return tuple(gpu_data[key][col][batch_indices] for col in columns)

def main():
    """Main training loop"""
    dirname, logpath = setup_output_directory()
    logger = get_logger(logpath)
    
    NOP_a = (500, 400)
    NOP_PDE = (400, 2000, 3000)
    NOP_north = (20, 20)
    NOP_south = (20, 20)
    NOP_east = (20, 20)
    NOP_west = (20, 20)
    
    training_data = get_training_data(NOP_a, NOP_PDE, NOP_north, NOP_south, NOP_east, NOP_west)
    
    logger.info("Pre-loading training data to GPU...")
    gpu_training_data, data_lengths = preload_training_data_to_gpu(training_data)
    logger.info("Data loaded to GPU successfully.")
    

    # NN Architecture and Hyperparameters
    no_layers = 8
    hidden_layers = [400] * no_layers
    
    activation_choice = 'tanh'
    use_aac_1 = False
    use_aac_2 = False
    
    activation_functions = {activation_choice: range(1, no_layers + 1)}
    
    adaptive_mode = 'fixed'
    if use_aac_1:
        adaptive_mode = 'activation1'
    elif use_aac_2:
        adaptive_mode = 'activation2'
    
    logger.info(f"Configuration: Activation='{activation_choice}', Mode='{adaptive_mode}'")
    
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
    else:
        adaptive_activation_coeff = {}
        adaptive_activation_init = {}
        adaptive_activation_n = [10] * no_layers
    
    use_adaptive_activation = use_aac_1 or use_aac_2
    
    mu = [1.0, 10.0]
    sigma = 24.5
    g = -0.98
    rho = [100, 1000]
    u_ref = 1.0
    L_ref = 0.25
    loss_weights_PDE = [1.0, 10.0, 10.0, 1.0]
    epochs_list = [5000] * 5
    learning_rates = [1e-4, 5e-5, 1e-5, 5e-6, 1e-6]
    checkpoint_interval = 10
    num_of_batches = 15
    
    num_samples_total = sum(len(df) for df in training_data.values())
    total_batch_size = math.ceil(num_samples_total / num_of_batches)
    
    pinn = TwoPhasePinn(hidden_layers, activation_functions, adaptive_activation_coeff,
                       adaptive_activation_n, adaptive_activation_init, use_adaptive_activation,
                       loss_weights_PDE, mu, sigma, g, rho, u_ref, L_ref)
    pinn = torch.compile(pinn, mode='reduce-overhead')  # or mode='max-autotune'

    
    # Load initial weights if available
    if os.path.exists('initial_weights.pth'):
        pinn.load_state_dict(torch.load('initial_weights.pth'))
        logger.info("Loaded initial weights from initial_weights.pth")
    
    start_total = time.time()
    
    history_loss_a = []
    history_loss_f_uv = []
    history_loss_f_ma = []
    
    optimizer = optim.Adam(pinn.parameters(), lr=learning_rates[0])
    current_best_total_loss = float('inf')
    
    for i, (epochs, lr) in enumerate(zip(epochs_list, learning_rates)):
        logger.info(f"\n--- Starting Training Phase {i+1}/{len(epochs_list)} ---")
        logger.info(f"Epochs: {epochs}, Learning Rate: {lr}")
        
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        prop_batch_sizes, num_batches = get_proportional_batch_sizes(total_batch_size, 
                                                                     training_data, logger)
        start_checkpoint_time = time.time()
        
        accumulation_steps = 3  # Add this near the top of main(), after num_of_batches

        for epoch in range(1, epochs + 1):
            epoch_losses = []
            shuffled_indices = {
                key: torch.randperm(data_lengths[key], device=device) 
                for key in gpu_training_data.keys()
            }
            
            optimizer.zero_grad()  # Zero once at start
            accumulated_count = 0
            
            for b in range(num_batches):
                data_A = get_batch_tensors(
                    gpu_training_data, shuffled_indices, 'A',
                    b * prop_batch_sizes['A'], 
                    (b + 1) * prop_batch_sizes['A']
                )
                
                data_PDE = get_batch_tensors(
                    gpu_training_data, shuffled_indices, 'PDE',
                    b * prop_batch_sizes['PDE'], 
                    (b + 1) * prop_batch_sizes['PDE'],
                    columns=['x_PDE', 'y_PDE', 't_PDE']
                )
                
                data_N = get_batch_tensors(
                    gpu_training_data, shuffled_indices, 'N',
                    b * prop_batch_sizes['N'], 
                    (b + 1) * prop_batch_sizes['N']
                )
                
                # For EW, we need specific columns in specific order
                ew_start = b * prop_batch_sizes['EW']
                ew_end = (b + 1) * prop_batch_sizes['EW']
                ew_indices = shuffled_indices['EW'][ew_start:ew_end]
                data_EW = (
                    gpu_training_data['EW']['x_E'][ew_indices],
                    gpu_training_data['EW']['y_E'][ew_indices],
                    gpu_training_data['EW']['t_EW'][ew_indices],
                    gpu_training_data['EW']['x_W'][ew_indices],
                    gpu_training_data['EW']['y_W'][ew_indices]
                )
                
                data_NSEW = get_batch_tensors(
                    gpu_training_data, shuffled_indices, 'NSEW',
                    b * prop_batch_sizes['NSEW'], 
                    (b + 1) * prop_batch_sizes['NSEW']
                )
                
                # Forward and backward (accumulate gradients)
                losses = pinn.compute_loss(data_A, data_PDE, data_N, data_EW, data_NSEW)
                total_loss = losses[0] / accumulation_steps  # Scale loss
                total_loss.backward()  # Accumulate gradients
                
                epoch_losses.append([l.item() for l in losses])
                accumulated_count += 1
                
                # Only step optimizer every accumulation_steps batches
                if accumulated_count >= accumulation_steps or (b + 1) == num_batches:
                    optimizer.step()
                    optimizer.zero_grad()
                    accumulated_count = 0
            
            avg_losses = np.mean(epoch_losses, axis=0)
            total_loss, loss_a, loss_bc, loss_m, loss_u, loss_v, loss_pde_a = avg_losses
            
            history_loss_a.append(loss_a)
            history_loss_f_uv.append(loss_u + loss_v)
            history_loss_f_ma.append(loss_m + loss_pde_a)
            
            if epoch % checkpoint_interval == 0:
                current_time = time.time()
                time_for_epoch = current_time - start_checkpoint_time
                start_checkpoint_time = current_time
                log_msg = f"Epoch: {epoch}/{epochs} - Time: {time_for_epoch:.2f}s - Loss: {total_loss:.4e}"
                log_msg += f" | a: {loss_a:.4e}, BC: {loss_bc:.4e}, m: {loss_m:.4e}"
                log_msg += f", u: {loss_u:.4e}, v: {loss_v:.4e}, pde_a: {loss_pde_a:.4e}"
                logger.info(log_msg)
            
            # Save best model
            if epoch % checkpoint_interval == 0 and total_loss < current_best_total_loss:
                logger.info(f"Saving checkpoint at epoch {epoch} with loss {total_loss:.4e}")
                # Remove previous checkpoint
                for f in glob.glob(os.path.join(dirname, "*.pth")):
                    os.remove(f)
                # Save new checkpoint
                safe_loss = f"{total_loss:.4e}".replace("+", "").replace("-", "m")
                weight_filename = f"loss_{safe_loss}.pth"
                torch.save(pinn.state_dict(), os.path.join(dirname, weight_filename))
                current_best_total_loss = total_loss
    
    total_training_time = time.time() - start_total
    logger.info(f"\nTotal training time: {total_training_time:.3f}s")
    
    logger.info("\n" + "="*50)
    logger.info("PERFORMING FINAL EVALUATION AND REPORTING")
    logger.info("="*50)
    
    list_of_files = glob.glob(os.path.join(dirname, '*.pth'))
    if list_of_files:
        latest_file = max(list_of_files, key=os.path.getctime)
        logger.info(f"Loading best model weights from: {os.path.basename(latest_file)}\n")
        pinn.load_state_dict(torch.load(latest_file))
    else:
        logger.info("No checkpoint file found. Evaluating with final weights from training.\n")
    
    logger.info("Calculating final loss...")
    final_evaluation_losses = []
    
    pinn.eval()
    with torch.no_grad():
        for b in range(num_batches):
            batch_dict = {}
            for key, df in training_data.items():
                start_idx = b * prop_batch_sizes[key]
                end_idx = (b + 1) * prop_batch_sizes[key]
                batch_dict[key] = df.iloc[start_idx:end_idx]
            
            if all(batch.empty for batch in batch_dict.values()):
                continue
            
            data_A = to_tensor_tuple(batch_dict['A'], batch_dict['A'].columns)
            data_PDE = to_tensor_tuple(batch_dict['PDE'], ['x_PDE', 'y_PDE', 't_PDE'])
            data_N = to_tensor_tuple(batch_dict['N'], batch_dict['N'].columns)
            data_EW = to_tensor_tuple(batch_dict['EW'], ['x_E', 'y_E', 't_EW', 'x_W', 'y_W'])
            data_NSEW = to_tensor_tuple(batch_dict['NSEW'], batch_dict['NSEW'].columns)
            
            batch_losses = pinn.compute_loss(data_A, data_PDE, data_N, data_EW, data_NSEW)
            final_evaluation_losses.append([l.item() for l in batch_losses])
    
    avg_final_losses = np.mean(final_evaluation_losses, axis=0)
    _, loss_a, loss_bc, loss_m, loss_u, loss_v, loss_pde_a = avg_final_losses
    
    logger.info("--- Final Loss Breakdown ---")
    logger.info(f"MSE_alpha (volume fraction): {loss_a:.4e}")
    logger.info(f"MSE_BC                     : {loss_bc:.4e}")
    logger.info(f"MSE_f,m                    : {loss_m:.4e}")
    logger.info(f"MSE_f,u                    : {loss_u:.4e}")
    logger.info(f"MSE_f,v                    : {loss_v:.4e}")
    logger.info(f"MSE_f,a                    : {loss_pde_a:.4e}")
    logger.info("----------------------------\n")
    
    # Plotting and saving history
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
    
    # Plot 2: MSE_f,u + MSE_f,v
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_range, history_loss_f_uv)
    plt.title(f'MSE of Momentum (u,v) vs. Epochs ({adaptive_mode} - {activation_choice})')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss (f_u + f_v)')
    plt.yscale('log')
    plt.grid(True, which="both", ls="--")
    plt.savefig(os.path.join(dirname, 'loss_history_momentum_uv.png'))
    
    # Plot 3: MSE_f,m + MSE_f,a
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_range, history_loss_f_ma)
    plt.title(f'MSE of Conservation (m,a) vs. Epochs ({adaptive_mode} - {activation_choice})')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss (f_m + f_a)')
    plt.yscale('log')
    plt.grid(True, which="both", ls="--")
    plt.savefig(os.path.join(dirname, 'loss_history_conservation_ma.png'))
    
    plt.close('all')
    logger.info("Plots saved successfully.")
    
    # Save history to CSV
    history_filename = f"loss_history_{adaptive_mode}_{activation_choice}.csv"
    history_filepath = os.path.join(dirname, history_filename)
    
    history_df = pd.DataFrame({
        'epoch': epochs_range,
        'MSE_alpha': history_loss_a,
        'MSE_f_uv': history_loss_f_uv,
        'MSE_f_ma': history_loss_f_ma
    })
    
    history_df.to_csv(history_filepath, index=False)
    logger.info(f"Loss history data saved to: {history_filepath}")
    
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)


if __name__ == "__main__":
    main()