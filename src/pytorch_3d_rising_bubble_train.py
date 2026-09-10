import sys
sys.path.append("../utilities")
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import math
import time
import glob
import shutil
import logging
from collections import namedtuple
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from generate_points_3d import get_training_data_3d
from pytorch_utilities import NNCreator, writeToJSONFile

# Set random seeds for reproducibility. Each rank ends up with its own
# independently-sampled collocation points (see get_training_data_3d), which is
# fine for data-parallel training -- only the per-rank dataset SIZES need to
# match across ranks, not their contents, and those sizes are deterministic
# given the NOP_* point counts below.
np.random.seed(1234)
torch.manual_seed(1234)

Physics = namedtuple(
    "Physics",
    ["mu1", "mu2", "sigma", "g", "rho1", "rho2", "U_ref", "L_ref", "rho_ref", "loss_weights_PDE"],
)


def build_activation_dict(hidden_layers, activation_functions, adaptive_activation_n):
    """Builds the {layer_no: [func_name, ad_act_coeff, n]} dict NNCreator expects."""
    activation_dict = {}
    for layer_no in range(1, len(hidden_layers) + 1):
        func_name = None
        for name, layers in activation_functions.items():
            if layer_no in layers:
                func_name = name
        activation_dict[layer_no] = [func_name, None, adaptive_activation_n[layer_no - 1]]
    return activation_dict


def compute_gradients(model, x, y, z, t):
    """Computes first- and second-order derivatives of (u, v, w, p, a) w.r.t. (x, y, z, t).

    `model` is called directly (not through a `self.forward` alias) so that this
    works correctly whether `model` is the raw net or a DistributedDataParallel
    wrapper -- DDP's gradient-sync hooks are tied to going through `model(...)`.
    """
    x = x.requires_grad_(True)
    y = y.requires_grad_(True)
    z = z.requires_grad_(True)
    t = t.requires_grad_(True)

    u, v, w, p, a = model(torch.cat([x, y, z, t], dim=1))

    u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    u_y = torch.autograd.grad(u, y, torch.ones_like(u), create_graph=True)[0]
    u_z = torch.autograd.grad(u, z, torch.ones_like(u), create_graph=True)[0]
    u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]

    v_x = torch.autograd.grad(v, x, torch.ones_like(v), create_graph=True)[0]
    v_y = torch.autograd.grad(v, y, torch.ones_like(v), create_graph=True)[0]
    v_z = torch.autograd.grad(v, z, torch.ones_like(v), create_graph=True)[0]
    v_t = torch.autograd.grad(v, t, torch.ones_like(v), create_graph=True)[0]

    w_x = torch.autograd.grad(w, x, torch.ones_like(w), create_graph=True)[0]
    w_y = torch.autograd.grad(w, y, torch.ones_like(w), create_graph=True)[0]
    w_z = torch.autograd.grad(w, z, torch.ones_like(w), create_graph=True)[0]
    w_t = torch.autograd.grad(w, t, torch.ones_like(w), create_graph=True)[0]

    p_x = torch.autograd.grad(p, x, torch.ones_like(p), create_graph=True)[0]
    p_y = torch.autograd.grad(p, y, torch.ones_like(p), create_graph=True)[0]
    p_z = torch.autograd.grad(p, z, torch.ones_like(p), create_graph=True)[0]

    a_x = torch.autograd.grad(a, x, torch.ones_like(a), create_graph=True)[0]
    a_y = torch.autograd.grad(a, y, torch.ones_like(a), create_graph=True)[0]
    a_z = torch.autograd.grad(a, z, torch.ones_like(a), create_graph=True)[0]
    a_t = torch.autograd.grad(a, t, torch.ones_like(a), create_graph=True)[0]

    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, torch.ones_like(u_y), create_graph=True)[0]
    u_zz = torch.autograd.grad(u_z, z, torch.ones_like(u_z), create_graph=True)[0]

    v_xx = torch.autograd.grad(v_x, x, torch.ones_like(v_x), create_graph=True)[0]
    v_yy = torch.autograd.grad(v_y, y, torch.ones_like(v_y), create_graph=True)[0]
    v_zz = torch.autograd.grad(v_z, z, torch.ones_like(v_z), create_graph=True)[0]

    w_xx = torch.autograd.grad(w_x, x, torch.ones_like(w_x), create_graph=True)[0]
    w_yy = torch.autograd.grad(w_y, y, torch.ones_like(w_y), create_graph=True)[0]
    w_zz = torch.autograd.grad(w_z, z, torch.ones_like(w_z), create_graph=True)[0]

    a_xx = torch.autograd.grad(a_x, x, torch.ones_like(a_x), create_graph=True)[0]
    a_yy = torch.autograd.grad(a_y, y, torch.ones_like(a_y), create_graph=True)[0]
    a_zz = torch.autograd.grad(a_z, z, torch.ones_like(a_z), create_graph=True)[0]
    a_xy = torch.autograd.grad(a_x, y, torch.ones_like(a_x), create_graph=True)[0]
    a_xz = torch.autograd.grad(a_x, z, torch.ones_like(a_x), create_graph=True)[0]
    a_yz = torch.autograd.grad(a_y, z, torch.ones_like(a_y), create_graph=True)[0]

    return (u, u_x, u_y, u_z, u_t, u_xx, u_yy, u_zz), \
           (v, v_x, v_y, v_z, v_t, v_xx, v_yy, v_zz), \
           (w, w_x, w_y, w_z, w_t, w_xx, w_yy, w_zz), \
           (p, p_x, p_y, p_z), \
           (a, a_x, a_y, a_z, a_t, a_xx, a_yy, a_zz, a_xy, a_xz, a_yz)


def PDE_caller(model, physics, x, y, z, t):
    u_grads, v_grads, w_grads, p_grads, a_grads = compute_gradients(model, x, y, z, t)
    u, u_x, u_y, u_z, u_t, u_xx, u_yy, u_zz = u_grads
    v, v_x, v_y, v_z, v_t, v_xx, v_yy, v_zz = v_grads
    w, w_x, w_y, w_z, w_t, w_xx, w_yy, w_zz = w_grads
    p, p_x, p_y, p_z = p_grads
    a, a_x, a_y, a_z, a_t, a_xx, a_yy, a_zz, a_xy, a_xz, a_yz = a_grads

    mu = physics.mu2 + (physics.mu1 - physics.mu2) * a
    mu_x = (physics.mu1 - physics.mu2) * a_x
    mu_y = (physics.mu1 - physics.mu2) * a_y
    mu_z = (physics.mu1 - physics.mu2) * a_z
    rho = physics.rho2 + (physics.rho1 - physics.rho2) * a

    abs_interface_grad = torch.sqrt(a_x ** 2 + a_y ** 2 + a_z ** 2 + np.finfo(float).eps)
    # 3D mean curvature of the levelset -div(grad(a)/|grad(a)|)
    curvature = -(((a_yy + a_zz) * a_x ** 2 + (a_xx + a_zz) * a_y ** 2 + (a_xx + a_yy) * a_z ** 2
                   - 2 * a_x * a_y * a_xy - 2 * a_x * a_z * a_xz - 2 * a_y * a_z * a_yz)
                  / torch.pow(abs_interface_grad, 3))

    one_Re = mu / (physics.rho_ref * physics.U_ref * physics.L_ref)
    one_Re_x = mu_x / (physics.rho_ref * physics.U_ref * physics.L_ref)
    one_Re_y = mu_y / (physics.rho_ref * physics.U_ref * physics.L_ref)
    one_Re_z = mu_z / (physics.rho_ref * physics.U_ref * physics.L_ref)
    one_We = physics.sigma / (physics.rho_ref * physics.U_ref ** 2 * physics.L_ref)
    one_Fr = physics.g * physics.L_ref / physics.U_ref ** 2

    PDE_m = u_x + v_y + w_z
    PDE_a = a_t + u * a_x + v * a_y + w * a_z
    PDE_u = (u_t + u * u_x + v * u_y + w * u_z) * rho / physics.rho_ref + p_x - \
            one_We * curvature * a_x - one_Re * (u_xx + u_yy + u_zz) - \
            2.0 * one_Re_x * u_x - one_Re_y * (u_y + v_x) - one_Re_z * (u_z + w_x)
    PDE_v = (v_t + u * v_x + v * v_y + w * v_z) * rho / physics.rho_ref + p_y - \
            one_We * curvature * a_y - one_Re * (v_xx + v_yy + v_zz) - \
            2.0 * one_Re_y * v_y - one_Re_x * (u_y + v_x) - one_Re_z * (v_z + w_y)
    # Gravity acts along z, so the body-force term lives in the w-momentum equation.
    PDE_w = (w_t + u * w_x + v * w_y + w * w_z) * rho / physics.rho_ref + p_z - \
            one_We * curvature * a_z - one_Re * (w_xx + w_yy + w_zz) - \
            rho / physics.rho_ref * one_Fr - 2.0 * one_Re_z * w_z - \
            one_Re_x * (u_z + w_x) - one_Re_y * (v_z + w_y)

    return PDE_m, PDE_u, PDE_v, PDE_w, PDE_a


def compute_loss(model, physics, data_A, data_PDE, data_TOP, data_NOSLIP, data_XPER, data_YPER):
    x_A, y_A, z_A, t_A, a_A = data_A
    x_PDE, y_PDE, z_PDE, t_PDE = data_PDE
    x_TOP, y_TOP, z_TOP, t_TOP, u_TOP, v_TOP, w_TOP, p_TOP = data_TOP
    x_NS, y_NS, z_NS, t_NS, u_NS, v_NS, w_NS = data_NOSLIP
    x_Xp, y_Xp, z_Xp, x_Xm, y_Xm, z_Xm, t_XPER = data_XPER
    x_Yp, y_Yp, z_Yp, x_Ym, y_Ym, z_Ym, t_YPER = data_YPER

    f_PDE = torch.zeros_like(x_PDE)

    # Loss A (Volume Fraction)
    _, _, _, _, pred_a_A = model(torch.cat([x_A, y_A, z_A, t_A], dim=1))
    loss_a_A = torch.mean((a_A - pred_a_A) ** 2)

    # Loss NOSLIP (No-slip walls at z_min and z_max)
    pred_u_NS, pred_v_NS, pred_w_NS, _, _ = model(torch.cat([x_NS, y_NS, z_NS, t_NS], dim=1))
    loss_u_NS = torch.mean((u_NS - pred_u_NS) ** 2)
    loss_v_NS = torch.mean((v_NS - pred_v_NS) ** 2)
    loss_w_NS = torch.mean((w_NS - pred_w_NS) ** 2)

    # Loss TOP (Pressure at open/pressure boundary z = z_max)
    _, _, _, pred_p_TOP, _ = model(torch.cat([x_TOP, y_TOP, z_TOP, t_TOP], dim=1))
    loss_p_TOP = torch.mean((p_TOP - pred_p_TOP) ** 2)

    # Loss XPER (Periodic boundary along x)
    pred_x_plus = model(torch.cat([x_Xp, y_Xp, z_Xp, t_XPER], dim=1))
    pred_x_minus = model(torch.cat([x_Xm, y_Xm, z_Xm, t_XPER], dim=1))
    loss_u_XPER = torch.mean((pred_x_plus[0] - pred_x_minus[0]) ** 2)
    loss_v_XPER = torch.mean((pred_x_plus[1] - pred_x_minus[1]) ** 2)
    loss_w_XPER = torch.mean((pred_x_plus[2] - pred_x_minus[2]) ** 2)
    loss_p_XPER = torch.mean((pred_x_plus[3] - pred_x_minus[3]) ** 2)

    # Loss YPER (Periodic boundary along y)
    pred_y_plus = model(torch.cat([x_Yp, y_Yp, z_Yp, t_YPER], dim=1))
    pred_y_minus = model(torch.cat([x_Ym, y_Ym, z_Ym, t_YPER], dim=1))
    loss_u_YPER = torch.mean((pred_y_plus[0] - pred_y_minus[0]) ** 2)
    loss_v_YPER = torch.mean((pred_y_plus[1] - pred_y_minus[1]) ** 2)
    loss_w_YPER = torch.mean((pred_y_plus[2] - pred_y_minus[2]) ** 2)
    loss_p_YPER = torch.mean((pred_y_plus[3] - pred_y_minus[3]) ** 2)

    loss_BC = (loss_u_NS + loss_v_NS + loss_w_NS + loss_p_TOP +
               loss_u_XPER + loss_v_XPER + loss_w_XPER + loss_p_XPER +
               loss_u_YPER + loss_v_YPER + loss_w_YPER + loss_p_YPER)

    # Loss PDE (Physics-Informed)
    PDE_m, PDE_u, PDE_v, PDE_w, PDE_a = PDE_caller(model, physics, x_PDE, y_PDE, z_PDE, t_PDE)
    loss_PDE_m = torch.mean((f_PDE - PDE_m) ** 2)
    loss_PDE_u = torch.mean((f_PDE - PDE_u) ** 2)
    loss_PDE_v = torch.mean((f_PDE - PDE_v) ** 2)
    loss_PDE_w = torch.mean((f_PDE - PDE_w) ** 2)
    loss_PDE_a = torch.mean((f_PDE - PDE_a) ** 2)

    loss_PDE = torch.dot(
        torch.stack([loss_PDE_m, loss_PDE_u, loss_PDE_v, loss_PDE_w, loss_PDE_a]),
        physics.loss_weights_PDE,
    )

    total_loss = loss_a_A + loss_BC + loss_PDE

    return total_loss, loss_a_A, loss_BC, loss_PDE_m, loss_PDE_u, loss_PDE_v, loss_PDE_w, loss_PDE_a


def train_step(model, optimizer, physics, data_A, data_PDE, data_TOP, data_NOSLIP, data_XPER, data_YPER):
    optimizer.zero_grad()
    losses = compute_loss(model, physics, data_A, data_PDE, data_TOP, data_NOSLIP, data_XPER, data_YPER)
    losses[0].backward()
    optimizer.step()
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


def get_logger(logpath, name=__name__, console=True):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if console:
        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(sh)

    if logpath is not None:
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


def to_tensor_tuple(df, columns, device):
    return tuple(
        torch.tensor(df[c].to_numpy().reshape(-1, 1), dtype=torch.float32, device=device)
        for c in columns
    )


def build_batch_tensors(batch_dict, device):
    return (
        to_tensor_tuple(batch_dict['A'], batch_dict['A'].columns, device),
        to_tensor_tuple(batch_dict['PDE'], ['x_PDE', 'y_PDE', 'z_PDE', 't_PDE'], device),
        to_tensor_tuple(batch_dict['TOP'], batch_dict['TOP'].columns, device),
        to_tensor_tuple(batch_dict['NOSLIP'], batch_dict['NOSLIP'].columns, device),
        to_tensor_tuple(batch_dict['XPER'], ['x_Xp', 'y_Xp', 'z_Xp', 'x_Xm', 'y_Xm', 'z_Xm', 't_XPER'], device),
        to_tensor_tuple(batch_dict['YPER'], ['x_Yp', 'y_Yp', 'z_Yp', 'x_Ym', 'y_Ym', 'z_Ym', 't_YPER'], device),
    )


def slice_batch(data, prop_batch_sizes, b):
    batch_dict = {}
    for key, df in data.items():
        start_idx = b * prop_batch_sizes[key]
        end_idx = (b + 1) * prop_batch_sizes[key]
        batch_dict[key] = df.iloc[start_idx:end_idx]
    return batch_dict


def main():
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    is_distributed = world_size > 1
    is_main = rank == 0

    if is_distributed:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)

    if is_main:
        dirname, logpath = setup_output_directory()
        logger = get_logger(logpath)
    else:
        dirname, logpath = None, None
        logger = get_logger(None, name=f"{__name__}.rank{rank}", console=True)

    logger.info(f"[rank {rank}/{world_size}] using device {device}")

    NOP_a = (500, 400)
    NOP_PDE = (400, 2000, 3000)
    NOP_top = (20, 20)
    NOP_bottom = (20, 20)
    NOP_xper = (20, 20)
    NOP_yper = (20, 20)

    training_data = get_training_data_3d(NOP_a, NOP_PDE, NOP_top, NOP_bottom, NOP_xper, NOP_yper)

    # --- NN Architecture and Hyperparameters --- #
    no_layers = 8
    hidden_layers = [400] * no_layers
    activation_choice = 'tanh'
    adaptive_mode = 'fixed'  # adaptive activation is not wired up for the 3D case yet
    adaptive_activation_n = [10] * no_layers
    activation_functions = {activation_choice: range(1, no_layers + 1)}
    activation_functions_dict = build_activation_dict(hidden_layers, activation_functions, adaptive_activation_n)

    logger.info(f"Configuration: Activation='{activation_choice}', Mode='{adaptive_mode}'")

    output_layer = [
        ("output_u", None), ("output_v", None), ("output_w", None),
        ("output_p", "exponential"), ("output_a", "sigmoid"),
    ]

    nn_creator = NNCreator(torch.float32)
    net = nn_creator.get_model_dnn(4, hidden_layers, output_layer, activation_functions_dict, False)
    net.to(device)

    if is_distributed:
        model = DDP(net, device_ids=[local_rank] if torch.cuda.is_available() else None)
    else:
        model = net

    logger.info("Starting from a freshly initialized network -- no PyTorch-format pretrained "
                "weights exist yet for the 3D case (initial_weights_3d.h5 is a Keras checkpoint "
                "and isn't portable to this architecture).")

    mu = [1.0, 10.0]
    sigma = 24.5
    g = -0.98
    rho = [100, 1000]
    u_ref = 1.0
    L_ref = 0.25
    loss_weights_PDE = [1.0, 10.0, 10.0, 10.0, 1.0]
    physics = Physics(
        mu1=mu[0], mu2=mu[1], sigma=sigma, g=g, rho1=rho[0], rho2=rho[1],
        U_ref=u_ref, L_ref=L_ref, rho_ref=rho[1],
        loss_weights_PDE=torch.tensor(loss_weights_PDE, dtype=torch.float32, device=device),
    )

    epochs_list = [5000] * 5
    learning_rates = [1e-4, 5e-5, 1e-5, 5e-6, 1e-6]
    checkpoint_interval = 10
    num_of_batches = 20

    num_samples_total = sum(len(df) for df in training_data.values())
    total_batch_size = math.ceil(num_samples_total / num_of_batches)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rates[0], eps=1e-8)

    start_total = time.time()
    history_loss_a, history_loss_f_uvw, history_loss_f_ma = [], [], []
    current_best_total_loss = float('inf')

    num_batches, prop_batch_sizes = None, None

    for i, (epochs, lr) in enumerate(zip(epochs_list, learning_rates)):
        logger.info(f"\n--- Starting Training Phase {i + 1}/{len(epochs_list)} ---")
        logger.info(f"Epochs: {epochs}, Learning Rate: {lr}")
        for group in optimizer.param_groups:
            group['lr'] = lr

        prop_batch_sizes, num_batches = get_proportional_batch_sizes(total_batch_size, training_data, logger)
        # Each rank owns a full independent copy of training_data (see module
        # docstring note on seeding); splitting num_batches across ranks keeps
        # every rank's per-epoch step count identical, which DDP's collective
        # all-reduce during backward() requires to stay in lockstep.
        num_local_batches = max(1, num_batches // world_size)
        start_checkpoint_time = time.time()

        for epoch in range(1, epochs + 1):
            epoch_losses = []
            shuffled_data = {key: df.sample(frac=1).reset_index(drop=True) for key, df in training_data.items()}

            for b in range(num_local_batches):
                batch_dict = slice_batch(shuffled_data, prop_batch_sizes, b)
                if all(batch.empty for batch in batch_dict.values()):
                    continue

                data_A, data_PDE, data_TOP, data_NOSLIP, data_XPER, data_YPER = build_batch_tensors(batch_dict, device)
                losses = train_step(model, optimizer, physics, data_A, data_PDE, data_TOP, data_NOSLIP, data_XPER, data_YPER)
                epoch_losses.append([l.item() for l in losses])

            mean_losses = torch.tensor(np.mean(epoch_losses, axis=0), dtype=torch.float64, device=device)
            if is_distributed:
                dist.all_reduce(mean_losses, op=dist.ReduceOp.AVG)
            total_loss, loss_a, loss_bc, loss_m, loss_u, loss_v, loss_w, loss_pde_a = mean_losses.tolist()

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

            if is_main and epoch % checkpoint_interval == 0 and total_loss < current_best_total_loss:
                logger.info(f"Saving checkpoint at epoch {epoch} with loss {total_loss:.4e}")
                for f in glob.glob(os.path.join(dirname, "*.pth")):
                    os.remove(f)
                safe_loss = f"{total_loss:.4e}".replace("+", "").replace("-", "m")
                weight_filename = f"loss_{safe_loss}.pth"
                state_dict = (model.module if is_distributed else model).state_dict()
                torch.save(state_dict, os.path.join(dirname, weight_filename))
                current_best_total_loss = total_loss

    if is_distributed:
        dist.barrier()

    total_training_time = time.time() - start_total
    logger.info(f"\nTotal training time: {total_training_time:.3f}s")

    if is_main:
        logger.info("\n" + "=" * 50)
        logger.info("PERFORMING FINAL EVALUATION AND REPORTING")
        logger.info("=" * 50)

        underlying_net = model.module if is_distributed else model

        list_of_files = glob.glob(os.path.join(dirname, '*.pth'))
        if list_of_files:
            latest_file = max(list_of_files, key=os.path.getctime)
            logger.info(f"Loading best model weights from: {os.path.basename(latest_file)}\n")
            underlying_net.load_state_dict(torch.load(latest_file, map_location=device))
        else:
            logger.info("No checkpoint file found. Evaluating with final weights from training.\n")

        logger.info("Calculating final loss...")
        underlying_net.eval()
        final_evaluation_losses = []
        for b in range(num_batches):
            batch_dict = slice_batch(training_data, prop_batch_sizes, b)
            if all(batch.empty for batch in batch_dict.values()):
                continue

            data_A, data_PDE, data_TOP, data_NOSLIP, data_XPER, data_YPER = build_batch_tensors(batch_dict, device)
            # Not wrapped in torch.no_grad(): the PDE residual needs first/second
            # derivatives of the network output w.r.t. its inputs, which requires
            # autograd to be active even though we never call .backward() here.
            batch_losses = compute_loss(underlying_net, physics, data_A, data_PDE, data_TOP, data_NOSLIP, data_XPER, data_YPER)
            final_evaluation_losses.append([l.item() for l in batch_losses])

        avg_final_losses = np.mean(final_evaluation_losses, axis=0)
        _, loss_a, loss_bc, loss_m, loss_u, loss_v, loss_w, loss_pde_a = avg_final_losses

        logger.info("--- Final Loss Breakdown ---")
        logger.info(f"MSE_alpha (volume fraction): {loss_a:.4e}")
        logger.info(f"MSE_BC                     : {loss_bc:.4e}")
        logger.info(f"MSE_f,m                    : {loss_m:.4e}")
        logger.info(f"MSE_f,u                    : {loss_u:.4e}")
        logger.info(f"MSE_f,v                    : {loss_v:.4e}")
        logger.info(f"MSE_f,w                    : {loss_w:.4e}")
        logger.info(f"MSE_f,a                    : {loss_pde_a:.4e}")
        logger.info("----------------------------\n")

        logger.info("Generating and saving loss history plots...")
        import matplotlib.pyplot as plt
        epochs_range = range(1, len(history_loss_a) + 1)

        plt.figure(figsize=(10, 6))
        plt.plot(epochs_range, history_loss_a)
        plt.title(f'MSE of Volume Fraction (alpha) vs. Epochs ({adaptive_mode} - {activation_choice})')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.yscale('log')
        plt.grid(True, which="both", ls="--")
        plt.savefig(os.path.join(dirname, 'loss_history_alpha.png'))

        plt.figure(figsize=(10, 6))
        plt.plot(epochs_range, history_loss_f_uvw)
        plt.title(f'MSE of Momentum (u,v,w) vs. Epochs ({adaptive_mode} - {activation_choice})')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss (f_u + f_v + f_w)')
        plt.yscale('log')
        plt.grid(True, which="both", ls="--")
        plt.savefig(os.path.join(dirname, 'loss_history_momentum_uvw.png'))

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

        history_filename = f"loss_history_{adaptive_mode}_{activation_choice}.csv"
        history_filepath = os.path.join(dirname, history_filename)
        history_df = pd.DataFrame({
            'epoch': epochs_range,
            'MSE_alpha': history_loss_a,
            'MSE_f_uvw': history_loss_f_uvw,
            'MSE_f_ma': history_loss_f_ma,
        })
        history_df.to_csv(history_filepath, index=False)
        logger.info(f"Loss history data saved to: {history_filepath}")

        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)

    if is_distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
