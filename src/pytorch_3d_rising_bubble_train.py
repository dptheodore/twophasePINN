import sys
sys.path.append("../utilities")
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import math
import contextlib
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

import generate_points_3d
from generate_points_3d import get_training_data_3d, get_validation_data_3d
from pytorch_utilities import NNCreator, writeToJSONFile

# Every rank seeds identically and so generates the SAME training points. The data is
# then partitioned at batch level (rank r trains batches r, r + N, r + 2N, ... of a
# shuffle shared by all ranks), so each epoch visits every batch exactly once whatever the
# number of GPUs.
np.random.seed(1234)
torch.manual_seed(1234)

Physics = namedtuple(
    "Physics",
    ["mu1", "mu2", "sigma", "g", "rho1", "rho2", "U_ref", "L_ref", "rho_ref",
     "loss_weights_PDE"],
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
    # The 2D curvature expression extended term by term, -div(grad(a)/|grad(a)|). Kept in
    # the same lap/|g| - (g.H.g)/|g|^3 form as 2D (not the algebraically combined one) so
    # that the eps inside |g| enters identically where grad(a) ~ 0.
    curvature = -((a_xx + a_yy + a_zz) / abs_interface_grad -
                  (a_x ** 2 * a_xx + a_y ** 2 * a_yy + a_z ** 2 * a_zz +
                   2 * a_x * a_y * a_xy + 2 * a_x * a_z * a_xz + 2 * a_y * a_z * a_yz)
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
    # Gravity acts along z (the 2D y), so the body-force term is in the w equation.
    PDE_w = (w_t + u * w_x + v * w_y + w * w_z) * rho / physics.rho_ref + p_z - \
            one_We * curvature * a_z - one_Re * (w_xx + w_yy + w_zz) - \
            rho / physics.rho_ref * one_Fr - 2.0 * one_Re_z * w_z - \
            one_Re_x * (u_z + w_x) - one_Re_y * (v_z + w_y)

    return PDE_m, PDE_u, PDE_v, PDE_w, PDE_a


def compute_loss(model, physics, data_A, data_PDE, data_N, data_EW, data_FB, data_NSEW):
    x_A, y_A, z_A, t_A, a_A = data_A
    x_PDE, y_PDE, z_PDE, t_PDE = data_PDE
    x_N, y_N, z_N, t_N, p_N = data_N
    x_E, y_E, z_E, x_W, y_W, z_W, t_EW = data_EW
    x_F, y_F, z_F, x_B, y_B, z_B, t_FB = data_FB
    x_NSEW, y_NSEW, z_NSEW, t_NSEW, u_NSEW, v_NSEW, w_NSEW = data_NSEW

    f_PDE = torch.zeros_like(x_PDE)

    # Loss A (Volume Fraction)
    _, _, _, _, pred_a_A = model(torch.cat([x_A, y_A, z_A, t_A], dim=1))
    loss_a_A = torch.mean((a_A - pred_a_A) ** 2)

    # Loss NSEW (Boundary Conditions)
    pred_u_NSEW, pred_v_NSEW, pred_w_NSEW, _, _ = model(torch.cat([x_NSEW, y_NSEW, z_NSEW, t_NSEW], dim=1))
    loss_u_NSEW = torch.mean((u_NSEW - pred_u_NSEW) ** 2)
    loss_v_NSEW = torch.mean((v_NSEW - pred_v_NSEW) ** 2)
    loss_w_NSEW = torch.mean((w_NSEW - pred_w_NSEW) ** 2)

    # Loss N (Pressure at North boundary)
    _, _, _, pred_p_N, _ = model(torch.cat([x_N, y_N, z_N, t_N], dim=1))
    loss_p_N = torch.mean((p_N - pred_p_N) ** 2)

    # Loss EW (Periodic Boundary)
    pred_east = model(torch.cat([x_E, y_E, z_E, t_EW], dim=1))
    pred_west = model(torch.cat([x_W, y_W, z_W, t_EW], dim=1))
    loss_u_EW = torch.mean((pred_east[0] - pred_west[0]) ** 2)
    loss_v_EW = torch.mean((pred_east[1] - pred_west[1]) ** 2)
    loss_w_EW = torch.mean((pred_east[2] - pred_west[2]) ** 2)
    loss_p_EW = torch.mean((pred_east[3] - pred_west[3]) ** 2)

    # Loss FB (Periodic Boundary on the y faces, the 3D counterpart of EW)
    pred_front = model(torch.cat([x_F, y_F, z_F, t_FB], dim=1))
    pred_back = model(torch.cat([x_B, y_B, z_B, t_FB], dim=1))
    loss_u_FB = torch.mean((pred_front[0] - pred_back[0]) ** 2)
    loss_v_FB = torch.mean((pred_front[1] - pred_back[1]) ** 2)
    loss_w_FB = torch.mean((pred_front[2] - pred_back[2]) ** 2)
    loss_p_FB = torch.mean((pred_front[3] - pred_back[3]) ** 2)

    loss_BC = loss_u_NSEW + loss_v_NSEW + loss_w_NSEW + loss_p_N + \
              loss_u_EW + loss_v_EW + loss_w_EW + loss_p_EW + \
              loss_u_FB + loss_v_FB + loss_w_FB + loss_p_FB

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


VALIDATION_COLUMNS = ["mse_a", "rel_l2_u", "rel_l2_v", "rel_l2_w", "rel_l2_p"]


def evaluate_validation(net, val, chunk=20000):
    """Compares the network with the CFD on the held-out validation snapshots: MSE of the
    volume fraction (the quantity loss_a fits on the training snapshots) and relative L2
    errors of u, v, w, p, which the training loss never sees as data."""
    preds = []
    with torch.no_grad():
        for s in range(0, len(val["xyzt"]), chunk):
            preds.append(torch.cat(net(val["xyzt"][s:s + chunk]), dim=1))
    u, v, w, p, a = torch.cat(preds).T
    ref = val["uvwpa"]

    def rel_l2(pred, true):
        return (torch.linalg.norm(pred - true) / torch.linalg.norm(true)).item()

    return dict(mse_a=torch.mean((a - ref[:, 4]) ** 2).item(),
                rel_l2_u=rel_l2(u, ref[:, 0]), rel_l2_v=rel_l2(v, ref[:, 1]),
                rel_l2_w=rel_l2(w, ref[:, 2]), rel_l2_p=rel_l2(p, ref[:, 3]))


def train_step(model, optimizer, physics, batch):
    optimizer.zero_grad()
    losses = compute_loss(model, physics, *batch)
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
        to_tensor_tuple(batch_dict['N'], batch_dict['N'].columns, device),
        to_tensor_tuple(batch_dict['EW'], batch_dict['EW'].columns, device),
        to_tensor_tuple(batch_dict['FB'], batch_dict['FB'].columns, device),
        to_tensor_tuple(batch_dict['NSEW'], batch_dict['NSEW'].columns, device),
    )


def slice_batch(data, num_batches, b):
    """Splits every dataset evenly over num_batches. Fixed ceil-rounded per-key sizes ran
    the small boundary sets (400 rows at 9 per batch) dry before the last batches, and the
    mean over an empty slice made loss_BC NaN."""
    batch_dict = {}
    for key, df in data.items():
        start_idx = len(df) * b // num_batches
        end_idx = len(df) * (b + 1) // num_batches
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
    if not is_main:
        logger.setLevel(logging.WARNING)  # rank 0 reports the all-reduced losses for everyone

    NOP_a = (500, 400)
    NOP_PDE = (400, 2000, 3000)
    NOP_north = (20, 20)
    NOP_south = (20, 20)
    NOP_east = (20, 20)
    NOP_west = (20, 20)
    NOP_front_back = (20, 20)

    # Generation prints only from rank 0; the rest generate silently.
    with contextlib.redirect_stdout(sys.stdout if is_main else open(os.devnull, "w")):
        training_data = get_training_data_3d(NOP_a, NOP_PDE, NOP_north, NOP_south, NOP_east,
                                             NOP_west, NOP_front_back)
    logger.info(f"Training data: {os.path.abspath(generate_points_3d.H5_PATH)}")

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
    # Fresh Xavier init, no 2D starting weights: the lifted 2D network had a zero y row in
    # its first layer, and on the y-symmetric 3D bubble the gradient on that row cancels,
    # so it could not leave the y-independent (cylinder) solution.
    net = nn_creator.get_model_dnn(4, hidden_layers, output_layer, activation_functions_dict, False)
    net.to(device)

    if is_distributed:
        model = DDP(net, device_ids=[local_rank] if torch.cuda.is_available() else None)
    else:
        model = net

    # Physical parameters of the 2D Hysing benchmark, as in rising_bubble_train.py, and of
    # the Basilisk run (index 0 = bubble, where a = 1; index 1 = liquid)
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
    # Optimizer steps per epoch, as in rising_bubble_train.py (num_of_batches = 20 on one
    # process). Each DDP step consumes one batch per rank, so the data is split into
    # steps_per_epoch * world_size batches; every rank takes the same number of steps
    # (unequal counts deadlock DDP's all-reduce). Peak memory is ~1.35 GiB per 1000 points
    # per GPU (measured).
    steps_per_epoch = 20
    num_of_batches = steps_per_epoch * world_size

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
        # Equal step counts on every rank, or DDP's all-reduce deadlocks.
        num_local_batches = num_batches // world_size
        if num_batches % world_size:
            logger.info(f"Dropping {num_batches % world_size} remainder batch(es) per epoch "
                        f"to keep {world_size} ranks in lockstep")
        start_checkpoint_time = time.time()

        for epoch in range(1, epochs + 1):
            epoch_losses = []
            # Same permutation on every rank (seeded by phase and epoch), different batches.
            rng = np.random.default_rng([1234, i, epoch])
            shuffled_data = {key: df.sample(frac=1, random_state=rng).reset_index(drop=True)
                             for key, df in training_data.items()}

            # No skipping of empty batches here: a rank that skips a step deadlocks the rest.
            # slice_batch splits every set evenly, so no batch is empty anyway.
            for k in range(num_local_batches):
                batch_dict = slice_batch(shuffled_data, num_batches, rank + k * world_size)
                batch = build_batch_tensors(batch_dict, device)
                losses = train_step(model, optimizer, physics, batch)
                epoch_losses.append([l.item() for l in losses])

            # Reported exactly as rising_bubble_train.py does: the SUM of the per-batch losses
            # over its 20 batches. Here that is the mean over every batch of the epoch on all
            # ranks times steps_per_epoch (= 20), so it does not depend on the GPU count.
            sum_losses = torch.tensor(np.sum(epoch_losses, axis=0), dtype=torch.float64, device=device)
            if is_distributed:
                dist.all_reduce(sum_losses, op=dist.ReduceOp.SUM)
            epoch_losses_2d = sum_losses / (num_local_batches * world_size) * steps_per_epoch
            total_loss, loss_a, loss_bc, loss_m, loss_u, loss_v, loss_w, loss_pde_a = epoch_losses_2d.tolist()

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
            batch_dict = slice_batch(training_data, num_batches, b)
            if all(batch.empty for batch in batch_dict.values()):
                continue

            batch = build_batch_tensors(batch_dict, device)
            # Not wrapped in torch.no_grad(): the PDE residual needs first/second
            # derivatives of the network output w.r.t. its inputs, which requires
            # autograd to be active even though we never call .backward() here.
            batch_losses = compute_loss(underlying_net, physics, *batch)
            final_evaluation_losses.append([l.item() for l in batch_losses])

        # Same convention as the per-epoch log: 2D's sum over 20 batches.
        final_losses_2d = np.mean(final_evaluation_losses, axis=0) * steps_per_epoch
        _, loss_a, loss_bc, loss_m, loss_u, loss_v, loss_w, loss_pde_a = final_losses_2d

        logger.info("--- Final Loss Breakdown ---")
        logger.info(f"MSE_alpha (volume fraction): {loss_a:.4e}")
        logger.info(f"MSE_BC                     : {loss_bc:.4e}")
        logger.info(f"MSE_f,m                    : {loss_m:.4e}")
        logger.info(f"MSE_f,u                    : {loss_u:.4e}")
        logger.info(f"MSE_f,v                    : {loss_v:.4e}")
        logger.info(f"MSE_f,w                    : {loss_w:.4e}")
        logger.info(f"MSE_f,a                    : {loss_pde_a:.4e}")
        logger.info("----------------------------")
        # The 10 held-out snapshots, never used for training and only looked at here, once
        # training is done: (near-interface, anywhere) points per snapshot.
        NOP_validation = (2000, 8000)
        val_df = get_validation_data_3d(NOP_validation)
        val = dict(xyzt=torch.tensor(val_df[["x", "y", "z", "t"]].to_numpy(), dtype=torch.float32, device=device),
                   uvwpa=torch.tensor(val_df[["u", "v", "w", "p", "a"]].to_numpy(), dtype=torch.float32, device=device))
        logger.info(f"Validation: {len(val_df)} points on the held-out snapshots "
                    f"t = {np.unique(np.round(val_df['t'].to_numpy() * generate_points_3d.L_REF, 2))}")
        metrics = evaluate_validation(underlying_net, val)
        pd.DataFrame([metrics])[VALIDATION_COLUMNS].to_csv(os.path.join(dirname, "validation.csv"), index=False)
        logger.info("--- Held-out validation snapshots (vs CFD) ---")
        logger.info(f"MSE_alpha                  : {metrics['mse_a']:.4e}")
        for k in "uvwp":
            logger.info(f"rel. L2 error {k}            : {metrics['rel_l2_' + k]:.4e}")
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
