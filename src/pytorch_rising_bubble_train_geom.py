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
from generate_points_geom import get_training_data
from utilities import NNCreator, writeToJSONFile
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
        """Compute first and second order derivatives"""
        x = x.requires_grad_(True)
        y = y.requires_grad_(True)
        t = t.requires_grad_(True)
        
        inputs = torch.cat([x, y, t], dim=1)
        u, v, p, a = self(inputs)
        
        # First order gradients
        u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
        u_y = torch.autograd.grad(u, y, torch.ones_like(u), create_graph=True)[0]
        u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
        
        v_x = torch.autograd.grad(v, x, torch.ones_like(v), create_graph=True)[0]
        v_y = torch.autograd.grad(v, y, torch.ones_like(v), create_graph=True)[0]
        v_t = torch.autograd.grad(v, t, torch.ones_like(v), create_graph=True)[0]
        
        p_x = torch.autograd.grad(p, x, torch.ones_like(p), create_graph=True)[0]
        p_y = torch.autograd.grad(p, y, torch.ones_like(p), create_graph=True)[0]
        
        a_x = torch.autograd.grad(a, x, torch.ones_like(a), create_graph=True)[0]
        a_y = torch.autograd.grad(a, y, torch.ones_like(a), create_graph=True)[0]
        a_t = torch.autograd.grad(a, t, torch.ones_like(a), create_graph=True)[0]
        
        # Second order gradients
        u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
        u_yy = torch.autograd.grad(u_y, y, torch.ones_like(u_y), create_graph=True)[0]
        
        v_xx = torch.autograd.grad(v_x, x, torch.ones_like(v_x), create_graph=True)[0]
        v_yy = torch.autograd.grad(v_y, y, torch.ones_like(v_y), create_graph=True)[0]
        
        a_xx = torch.autograd.grad(a_x, x, torch.ones_like(a_x), create_graph=True)[0]
        a_yy = torch.autograd.grad(a_y, y, torch.ones_like(a_y), create_graph=True)[0]
        a_xy = torch.autograd.grad(a_x, y, torch.ones_like(a_x), create_graph=True)[0]
        
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
    
    def compute_geom_loss(self, data_GEOM, geom_weight=1e1, delta=0.5 * 0.00390625,
                          eps_soft=0.1, min_weight=0.05, huber_delta=1e-3, debug=False):
        """Compute stabilized geometric loss"""
        xg, yg, tg, grad_x_gt, grad_y_gt, normal_x_gt, normal_y_gt = data_GEOM
        
        # Evaluate at offset positions
        pts_left = torch.cat([xg - delta, yg, tg], dim=1)
        pts_right = torch.cat([xg + delta, yg, tg], dim=1)
        pts_bottom = torch.cat([xg, yg - delta, tg], dim=1)
        pts_top = torch.cat([xg, yg + delta, tg], dim=1)
        
        with torch.no_grad():
            _, _, _, a_left = self(pts_left)
            _, _, _, a_right = self(pts_right)
            _, _, _, a_bottom = self(pts_bottom)
            _, _, _, a_top = self(pts_top)
        
        # Central difference derivative
        denom = 2.0 * delta
        grad_x_pred = (a_right - a_left) / denom
        grad_y_pred = (a_top - a_bottom) / denom
        
        grad_vec_pred_mag = torch.cat([torch.abs(grad_x_pred), torch.abs(grad_y_pred)], dim=1)
        
        # Predicted normal
        norm_mag = torch.sqrt(grad_x_pred**2 + grad_y_pred**2 + 1e-8)
        normal_pred = torch.cat([-grad_x_pred / norm_mag, -grad_y_pred / norm_mag], dim=1)
        
        # Soft-contour weighting
        _, _, _, a_center = self(torch.cat([xg, yg, tg], dim=1))
        a_center = torch.clamp(a_center, 0.0, 1.0)
        weight = torch.exp(-a_center**2 / (2.0 * eps_soft**2))
        weight = weight.reshape(-1)
        weight = torch.maximum(weight, torch.tensor(min_weight, device=device))
        w2 = weight.unsqueeze(1)
        
        # Prepare GT
        grad_x_gt_s = grad_x_gt.squeeze(-1)
        grad_y_gt_s = grad_y_gt.squeeze(-1)
        grad_vec_gt_mag = torch.cat([grad_x_gt_s.unsqueeze(1), grad_y_gt_s.unsqueeze(1)], dim=1)
        
        # Huber loss on gradients
        diff_grad = grad_vec_pred_mag - grad_vec_gt_mag
        abs_diff = torch.abs(diff_grad)
        huber_mask = (abs_diff <= huber_delta).float()
        loss_grad_vec_point = huber_mask * 0.5 * diff_grad**2 + \
                              (1.0 - huber_mask) * (huber_delta * (abs_diff - 0.5 * huber_delta))
        loss_grad_vec = torch.mean(w2 * loss_grad_vec_point.sum(dim=1, keepdim=True))
        
        # Normal (direction) loss
        normal_x_gt_s = normal_x_gt.squeeze(-1)
        normal_y_gt_s = normal_y_gt.squeeze(-1)
        normal_gt = torch.stack([normal_x_gt_s, normal_y_gt_s], dim=1)
        
        gt_norms = torch.sqrt((normal_gt**2).sum(dim=1, keepdim=True)) + 1e-12
        normal_gt_unit = normal_gt / gt_norms
        
        normal_pred_unit = normal_pred / (torch.sqrt((normal_pred**2).sum(dim=1, keepdim=True)) + 1e-12)
        
        cos_sim = (normal_pred_unit * normal_gt_unit).sum(dim=1, keepdim=True)
        cos_sim = torch.clamp(cos_sim, -1.0, 1.0)
        loss_normal_point = 1.0 - cos_sim
        loss_normal = torch.mean(w2 * loss_normal_point)
        
        loss_geom = geom_weight * (loss_grad_vec + loss_normal)
        
        return loss_geom
    
    def compute_loss(self, data_A, data_PDE, data_N, data_EW, data_NSEW, data_GEOM):
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
        
        loss_geom = self.compute_geom_loss(data_GEOM, geom_weight=1e1, debug=False)
        
        total_loss = loss_a_A + loss_BC + loss_PDE + loss_geom
        
        return total_loss, loss_a_A, loss_BC, loss_PDE_m, loss_PDE_u, loss_PDE_v, loss_PDE_a, loss_geom
    
    def train_step(self, optimizer, data_A, data_PDE, data_N, data_EW, data_NSEW, data_GEOM):
        """Single training step"""
        optimizer.zero_grad()
        losses = self.compute_loss(data_A, data_PDE, data_N, data_EW, data_NSEW, data_GEOM)
        total_loss = losses[0]
        total_loss.backward()
        optimizer.step()
        return losses


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
        'NSEW': max(16, batch_sizes['NSEW']),
        'GEOM': max(16, batch_sizes['GEOM'])
    }
    
    logger.info(f"Total samples: {num_samples_total}, Desired batch size: {total_batch_size}")
    logger.info(f"Calculated num_batches: {num_batches}, Proportional batch sizes: {batch_sizes}")
    return batch_sizes, num_batches


def to_tensor_tuple(df, columns):
    """Convert DataFrame columns to tensor tuple"""
    return tuple(torch.tensor(df[c].to_numpy().reshape(-1, 1), 
                             dtype=torch.float32, device=device) for c in columns)


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
    checkpoint_interval = 1
    num_of_batches = 20
    
    num_samples_total = sum(len(df) for df in training_data.values())
    total_batch_size = math.ceil(num_samples_total / num_of_batches)
    
    pinn = TwoPhasePinn(hidden_layers, activation_functions, adaptive_activation_coeff,
                       adaptive_activation_n, adaptive_activation_init, use_adaptive_activation,
                       loss_weights_PDE, mu, sigma, g, rho, u_ref, L_ref)
    
    # Load initial weights if available
    if os.path.exists('initial_weights.pth'):
        pinn.load_state_dict(torch.load('initial_weights.pth'))
        logger.info("Loaded initial weights from initial_weights.pth")
    
    start_total = time.time()
    
    history_loss_a = []
    history_loss_f_uv = []
    history_loss_f_ma = []
    history_loss_geom = []
    
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
        
        for epoch in range(1, epochs + 1):
            epoch_losses = []
            shuffled_data = {key: df.sample(frac=1).reset_index(drop=True) 
                           for key, df in training_data.items()}
            
            for b in range(num_batches):
                batch_dict = {}
                for key, df in shuffled_data.items():
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
                data_GEOM = to_tensor_tuple(batch_dict['GEOM'], batch_dict['GEOM'].columns)
                
                batch_loss_values = pinn.train_step(optimizer, data_A, data_PDE, data_N, 
                                                   data_EW, data_NSEW, data_GEOM)
                epoch_losses.append([l.item() for l in batch_loss_values])
            
            avg_losses = np.mean(epoch_losses, axis=0)
            total_loss, loss_a, loss_bc, loss_m, loss_u, loss_v, loss_pde_a, loss_geom = avg_losses
            
            history_loss_a.append(loss_a)
            history_loss_f_uv.append(loss_u + loss_v)
            history_loss_f_ma.append(loss_m + loss_pde_a)
            history_loss_geom.append(loss_geom)
            
            if epoch % checkpoint_interval == 0:
                current_time = time.time()
                time_for_epoch = current_time - start_checkpoint_time
                start_checkpoint_time = current_time
                log_msg = f"Epoch: {epoch}/{epochs} - Time: {time_for_epoch:.2f}s - Loss: {total_loss:.4e}"
                log_msg += f" | a: {loss_a:.4e}, BC: {loss_bc:.4e}, m: {loss_m:.4e}"
                log_msg += f", u: {loss_u:.4e}, v: {loss_v:.4e}, pde_a: {loss_pde_a:.4e}"
                log_msg += f" geom: {loss_geom:.4e}"
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
            data_GEOM = to_tensor_tuple(batch_dict['GEOM'], batch_dict['GEOM'].columns)
            
            batch_losses = pinn.compute_loss(data_A, data_PDE, data_N, data_EW, data_NSEW, data_GEOM)
            final_evaluation_losses.append([l.item() for l in batch_losses])
    
    avg_final_losses = np.mean(final_evaluation_losses, axis=0)
    _, loss_a, loss_bc, loss_m, loss_u, loss_v, loss_pde_a, loss_geom = avg_final_losses
    
    logger.info("--- Final Loss Breakdown ---")
    logger.info(f"MSE_alpha (volume fraction): {loss_a:.4e}")
    logger.info(f"MSE_BC                     : {loss_bc:.4e}")
    logger.info(f"MSE_f,m                    : {loss_m:.4e}")
    logger.info(f"MSE_f,u                    : {loss_u:.4e}")
    logger.info(f"MSE_f,v                    : {loss_v:.4e}")
    logger.info(f"MSE_f,a                    : {loss_pde_a:.4e}")
    logger.info(f"MSE_GEOM                   : {loss_geom:.4e}")
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
    
    # Plot 4: MSE GEOM
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_range, history_loss_geom)
    plt.title(f'MSE of Geometric Loss vs. Epochs ({adaptive_mode} - {activation_choice})')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss Geom')
    plt.yscale('log')
    plt.grid(True, which="both", ls="--")
    plt.savefig(os.path.join(dirname, 'loss_history_geom.png'))
    
    plt.close('all')
    logger.info("Plots saved successfully.")
    
    # Save history to CSV
    history_filename = f"loss_history_{adaptive_mode}_{activation_choice}.csv"
    history_filepath = os.path.join(dirname, history_filename)
    
    history_df = pd.DataFrame({
        'epoch': epochs_range,
        'MSE_alpha': history_loss_a,
        'MSE_f_uv': history_loss_f_uv,
        'MSE_f_ma': history_loss_f_ma,
        'MSE_geom': history_loss_geom
    })
    
    history_df.to_csv(history_filepath, index=False)
    logger.info(f"Loss history data saved to: {history_filepath}")
    
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)


if __name__ == "__main__":
    main()