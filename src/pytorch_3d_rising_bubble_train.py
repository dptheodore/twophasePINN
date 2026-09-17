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
    ["mu1", "mu2", "kappa", "g", "rho1", "rho2", "U_ref", "L_ref", "rho_ref",
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

    # THIRD derivatives of a, for the Korteweg force kappa*rho*grad(lap(rho)). The mixed
    # second derivatives a_xy/a_xz/a_yz the sharp-interface curvature needed are gone with
    # it -- grad(lap(a)) is all the free-energy model asks for.
    lap_a = a_xx + a_yy + a_zz
    lap_a_x = torch.autograd.grad(lap_a, x, torch.ones_like(lap_a), create_graph=True)[0]
    lap_a_y = torch.autograd.grad(lap_a, y, torch.ones_like(lap_a), create_graph=True)[0]
    lap_a_z = torch.autograd.grad(lap_a, z, torch.ones_like(lap_a), create_graph=True)[0]

    # Velocity divergence and its gradient. Both are needed now that the flow is NOT
    # incompressible: div(u) is a residual term in continuity, and its gradient appears in
    # the compressible part of the viscous stress.
    div_u = u_x + v_y + w_z
    div_x = torch.autograd.grad(div_u, x, torch.ones_like(div_u), create_graph=True)[0]
    div_y = torch.autograd.grad(div_u, y, torch.ones_like(div_u), create_graph=True)[0]
    div_z = torch.autograd.grad(div_u, z, torch.ones_like(div_u), create_graph=True)[0]

    return (u, u_x, u_y, u_z, u_t, u_xx, u_yy, u_zz), \
           (v, v_x, v_y, v_z, v_t, v_xx, v_yy, v_zz), \
           (w, w_x, w_y, w_z, w_t, w_xx, w_yy, w_zz), \
           (p, p_x, p_y, p_z), \
           (a, a_x, a_y, a_z, a_t, a_xx, a_yy, a_zz), \
           (lap_a_x, lap_a_y, lap_a_z, div_u, div_x, div_y, div_z)


def PDE_caller(model, physics, x, y, z, t):
    """Residuals of the equations LBPM actually integrates.

    The dataset comes from eos_type = "cp": a SINGLE-COMPONENT Peng-Robinson free-energy
    model (see the repo-root t14_forced_sphere.db). Three consequences, all of which the
    earlier sharp-interface incompressible form got wrong:

    1. Both phases are the same substance, so the interface exchanges mass by evaporation
       and condensation instead of being advected as a material surface. Continuity is
       d(rho)/dt + div(rho u) = 0, and there is no separate a_t + u.grad(a) = 0 equation --
       it was forcing the interface to move with the fluid, which it does not. Measured on
       this data div(u) is 1.29x the size of its own terms, so the incompressible form
       could never have driven its residual to zero honestly.
    2. Surface tension is not a sharp-interface sigma*curvature force; it is the Korteweg
       (density-gradient) force kappa*rho*grad(lap(rho)), with kappa = Color/pr_kappa taken
       straight from the .db. No sigma calibration is involved -- which is just as well,
       since measuring sigma from this data is ambiguous by a factor of ~30 depending on
       which cells are averaged.
    3. With div(u) != 0 the viscous stress carries its compressible part, -(2/3)*mu*div(u)
       on the diagonal (Stokes' hypothesis; LBPM's MRT bulk viscosity is not exactly this,
       so it is an approximation, but a far smaller one than omitting the term).

    p is the network's own field and is now interpretable as the thermodynamic pressure,
    since the Korteweg force carries the interfacial stress that the CSF term used to.
    """
    u_grads, v_grads, w_grads, p_grads, a_grads, extra = compute_gradients(model, x, y, z, t)
    u, u_x, u_y, u_z, u_t, u_xx, u_yy, u_zz = u_grads
    v, v_x, v_y, v_z, v_t, v_xx, v_yy, v_zz = v_grads
    w, w_x, w_y, w_z, w_t, w_xx, w_yy, w_zz = w_grads
    p, p_x, p_y, p_z = p_grads
    a, a_x, a_y, a_z, a_t, a_xx, a_yy, a_zz = a_grads
    lap_a_x, lap_a_y, lap_a_z, div_u, div_x, div_y, div_z = extra

    mu = physics.mu2 + (physics.mu1 - physics.mu2) * a
    mu_x = (physics.mu1 - physics.mu2) * a_x
    mu_y = (physics.mu1 - physics.mu2) * a_y
    mu_z = (physics.mu1 - physics.mu2) * a_z
    rho = physics.rho2 + (physics.rho1 - physics.rho2) * a
    # Nondimensional density and its slope w.r.t. a: rho* = 1 + drho_da * a.
    rho_star = rho / physics.rho_ref
    drho_da = (physics.rho1 - physics.rho2) / physics.rho_ref

    one_Re = mu / (physics.rho_ref * physics.U_ref * physics.L_ref)
    one_Re_x = mu_x / (physics.rho_ref * physics.U_ref * physics.L_ref)
    one_Re_y = mu_y / (physics.rho_ref * physics.U_ref * physics.L_ref)
    one_Re_z = mu_z / (physics.rho_ref * physics.U_ref * physics.L_ref)
    # Korteweg coefficient: kappa*rho_ref / (L_ref^2 * U_ref^2) makes kappa*rho*grad(lap(rho))
    # dimensionless on the same rho_ref*U_ref^2/L_ref scale as every other momentum term.
    Ko = physics.kappa * physics.rho_ref / (physics.L_ref ** 2 * physics.U_ref ** 2)
    one_Fr = physics.g * physics.L_ref / physics.U_ref ** 2

    # CONTINUITY with mass exchange across the interface. Expanding
    # d(rho*)/dt + div(rho* u) with rho* = 1 + drho_da * a gives the advective part of a
    # (which is NOT zero here) plus the compressible part.
    PDE_m = drho_da * (a_t + u * a_x + v * a_y + w * a_z) + rho_star * div_u

    # Korteweg force per unit volume, grad(lap(rho*)) = drho_da * grad(lap(a)).
    F_ko_x = Ko * rho_star * drho_da * lap_a_x
    F_ko_y = Ko * rho_star * drho_da * lap_a_y
    F_ko_z = Ko * rho_star * drho_da * lap_a_z

    PDE_u = (u_t + u * u_x + v * u_y + w * u_z) * rho_star + p_x - \
            F_ko_x - one_Re * (u_xx + u_yy + u_zz) - \
            2.0 * one_Re_x * u_x - one_Re_y * (u_y + v_x) - one_Re_z * (u_z + w_x) + \
            (2.0 / 3.0) * (one_Re * div_x + one_Re_x * div_u)
    PDE_v = (v_t + u * v_x + v * v_y + w * v_z) * rho_star + p_y - \
            F_ko_y - one_Re * (v_xx + v_yy + v_zz) - \
            2.0 * one_Re_y * v_y - one_Re_x * (u_y + v_x) - one_Re_z * (v_z + w_y) + \
            (2.0 / 3.0) * (one_Re * div_y + one_Re_y * div_u)
    # Gravity acts along z, so the body-force term lives in the w-momentum equation.
    #
    # FULL GRAVITY, exactly as the 2D script does it.
    #
    # An earlier revision used a reduced gravity (rho - rho_mean) here. The argument was
    # that the domain is triply periodic, so p is periodic, grad(p) has zero volume
    # average, and a mean body force therefore cannot be balanced -- unless the whole
    # domain accelerates uniformly, which that revision asserted the data ruled out
    # ("the domain-mean w is constant in time").
    #
    # The current dataset falsifies the premise, not the reasoning: the domain-mean w is
    # NOT constant, it ramps from 7e-6 to 2.0e-2 cells/step. The measured bulk
    # acceleration is 1.98e-6 cells/step^2 against a body force of 2.0e-6, i.e.
    # dw/dt = g to within 1.2%. Nothing removes the mean force; the fluid simply
    # accelerates, and both phases accelerate together (drop-to-carrier slip is 0.2% of
    # sqrt(g'D)), which is why the drop barely rises relative to its surroundings.
    #
    # That is precisely what the volume mean of this equation asks for with full gravity:
    # <rho/rho_ref * w_t> balances <rho/rho_ref> * (1/Fr) while <p_z> = 0. Subtracting
    # rho_mean would instead demand a mean acceleration of zero, which the data
    # contradicts directly.
    PDE_w = (w_t + u * w_x + v * w_y + w * w_z) * rho_star + p_z - \
            F_ko_z - one_Re * (w_xx + w_yy + w_zz) - \
            rho_star * one_Fr - 2.0 * one_Re_z * w_z - \
            one_Re_x * (u_z + w_x) - one_Re_y * (v_z + w_y) + \
            (2.0 / 3.0) * (one_Re * div_z + one_Re_z * div_u)

    return PDE_m, PDE_u, PDE_v, PDE_w


def compute_loss(model, physics, data_A, data_PDE, data_GAUGE):
    x_A, y_A, z_A, t_A, a_A = data_A
    x_PDE, y_PDE, z_PDE, t_PDE = data_PDE
    x_G, y_G, z_G, t_G, p_G = data_GAUGE

    f_PDE = torch.zeros_like(x_PDE)

    # Loss A (Volume Fraction)
    _, _, _, _, pred_a_A = model(torch.cat([x_A, y_A, z_A, t_A], dim=1))
    loss_a_A = torch.mean((a_A - pred_a_A) ** 2)

    # Loss GAUGE. The LBPM run sets Domain/BC = 0, so all six faces are periodic; there
    # are no walls and no pressure outlet. Periodicity is now exact by construction --
    # the network takes a sin/cos embedding of x, y and z with the domain lengths as
    # periods (see get_model_dnn(periodic_lengths=...)), which makes every output and
    # every derivative periodic identically. That replaces the XPER/YPER/ZPER face-pair
    # mismatch losses this script used to carry, matching the TensorFlow version.
    #
    # With every face periodic the pressure is only determined up to an additive
    # constant, so it is pinned at a single corner point. That is the only BC left.
    _, _, _, pred_p_G, _ = model(torch.cat([x_G, y_G, z_G, t_G], dim=1))
    loss_p_GAUGE = torch.mean((p_G - pred_p_G) ** 2)

    loss_BC = loss_p_GAUGE

    # Loss PDE (Physics-Informed)
    # Four residuals now, not five: continuity absorbs what used to be the separate
    # volume-fraction advection equation (see PDE_caller).
    PDE_m, PDE_u, PDE_v, PDE_w = PDE_caller(model, physics, x_PDE, y_PDE, z_PDE, t_PDE)
    loss_PDE_m = torch.mean((f_PDE - PDE_m) ** 2)
    loss_PDE_u = torch.mean((f_PDE - PDE_u) ** 2)
    loss_PDE_v = torch.mean((f_PDE - PDE_v) ** 2)
    loss_PDE_w = torch.mean((f_PDE - PDE_w) ** 2)

    loss_PDE = torch.dot(
        torch.stack([loss_PDE_m, loss_PDE_u, loss_PDE_v, loss_PDE_w]),
        physics.loss_weights_PDE,
    )

    total_loss = loss_a_A + loss_BC + loss_PDE

    return total_loss, loss_a_A, loss_BC, loss_PDE_m, loss_PDE_u, loss_PDE_v, loss_PDE_w


def train_step(model, optimizer, physics, data_A, data_PDE, data_GAUGE):
    optimizer.zero_grad()
    losses = compute_loss(model, physics, data_A, data_PDE, data_GAUGE)
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
    # The pressure gauge is not batched -- it is a single spatial point over a handful
    # of times and is fed whole on every step (see data_GAUGE in main()).
    return (
        to_tensor_tuple(batch_dict['A'], batch_dict['A'].columns, device),
        to_tensor_tuple(batch_dict['PDE'], ['x_PDE', 'y_PDE', 'z_PDE', 't_PDE'], device),
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

    training_data, gauge_df, scales = get_training_data_3d(NOP_a, NOP_PDE)

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
    # `periodic_lengths` makes the network periodic in x, y and z by construction,
    # replacing the face-pair mismatch losses (see compute_loss).
    net = nn_creator.get_model_dnn(4, hidden_layers, output_layer, activation_functions_dict, False,
                                   periodic_lengths=scales["period"])
    net.to(device)
    logger.info(f"Periodic lengths (nondimensional): {scales['period']}")

    if is_distributed:
        model = DDP(net, device_ids=[local_rank] if torch.cuda.is_available() else None)
    else:
        model = net

    logger.info("Starting from a freshly initialized network -- no PyTorch-format pretrained "
                "weights exist yet for the 3D case (initial_weights_3d.h5 is a Keras checkpoint "
                "and isn't portable to this architecture).")

    # Physical parameters of the LBPM run (t14_forced_sphere.db), together with the
    # reference scales measured from the dataset. These replace the 2D Hysing benchmark
    # values, which were SI quantities for a light bubble in water and had no
    # correspondence to this lattice-unit run of a dense drop in gas.
    mu = scales["mu"]            # [drop, carrier] dynamic viscosity = rho * nu
    kappa = scales["kappa"]      # Color/pr_kappa: the Korteweg coefficient, straight from the .db
    g = scales["g"]              # Color/F = 0,0,+2e-6, per unit mass, along +z
    rho = scales["rho"]          # [rhoA drop, rhoB carrier]
    u_ref = scales["U_ref"]
    L_ref = scales["L_ref"]
    logger.info(f"Reference scales: L_ref={L_ref:.4f} cells, U_ref={u_ref:.6g} cells/step, "
                f"T_ref={scales['T_ref']:.1f} steps")
    logger.info(f"mu={mu}, rho={rho}, kappa={kappa}, g={g}, rho_mean={scales['rho_mean']:.5f}")
    logger.info(f"Ko = {kappa * rho[1] / (L_ref ** 2 * u_ref ** 2):.4f} "
                f"(measured sigma {scales['sigma']:.4g} is diagnostic only now)")
    # [continuity, u, v, w] -- the volume-fraction advection residual is gone.
    loss_weights_PDE = [1.0, 10.0, 10.0, 10.0]
    physics = Physics(
        mu1=mu[0], mu2=mu[1], kappa=kappa, g=g, rho1=rho[0], rho2=rho[1],
        U_ref=u_ref, L_ref=L_ref, rho_ref=rho[1],
        loss_weights_PDE=torch.tensor(loss_weights_PDE, dtype=torch.float32, device=device),
    )

    # Fed whole on every step rather than batched -- a proportional batch would round the
    # 40-row gauge down to one or two rows and make it unreliably sampled.
    data_GAUGE = to_tensor_tuple(gauge_df, gauge_df.columns, device)

    epochs_list = [5000] * 5
    learning_rates = [1e-4, 5e-5, 1e-5, 5e-6, 1e-6]
    checkpoint_interval = 10
    num_of_batches = 20

    num_samples_total = sum(len(df) for df in training_data.values())
    total_batch_size = math.ceil(num_samples_total / num_of_batches)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rates[0], eps=1e-8)

    start_total = time.time()
    history_loss_a, history_loss_f_uvw, history_loss_f_m = [], [], []
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

                data_A, data_PDE = build_batch_tensors(batch_dict, device)
                losses = train_step(model, optimizer, physics, data_A, data_PDE, data_GAUGE)
                epoch_losses.append([l.item() for l in losses])

            # Summed over batches (not averaged) to match rising_bubble_train.py. Each rank
            # runs num_batches // world_size of the batches, so summing across ranks gives
            # the same epoch total a single-process run would print.
            sum_losses = torch.tensor(np.sum(epoch_losses, axis=0), dtype=torch.float64, device=device)
            if is_distributed:
                dist.all_reduce(sum_losses, op=dist.ReduceOp.SUM)
            total_loss, loss_a, loss_bc, loss_m, loss_u, loss_v, loss_w = sum_losses.tolist()

            history_loss_a.append(loss_a)
            history_loss_f_uvw.append(loss_u + loss_v + loss_w)
            history_loss_f_m.append(loss_m)

            if epoch % checkpoint_interval == 0:
                current_time = time.time()
                time_for_epoch = current_time - start_checkpoint_time
                start_checkpoint_time = current_time
                log_msg = f"Epoch: {epoch}/{epochs} - Time: {time_for_epoch:.2f}s - Loss: {total_loss:.4e}"
                log_msg += f" | a: {loss_a:.4e}, BC: {loss_bc:.4e}, m: {loss_m:.4e}"
                log_msg += f", u: {loss_u:.4e}, v: {loss_v:.4e}, w: {loss_w:.4e}"
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

            data_A, data_PDE = build_batch_tensors(batch_dict, device)
            # Not wrapped in torch.no_grad(): the PDE residual needs first/second
            # derivatives of the network output w.r.t. its inputs, which requires
            # autograd to be active even though we never call .backward() here.
            batch_losses = compute_loss(underlying_net, physics, data_A, data_PDE, data_GAUGE)
            final_evaluation_losses.append([l.item() for l in batch_losses])

        # Summed over batches to match rising_bubble_train.py.
        sum_final_losses = np.sum(final_evaluation_losses, axis=0)
        _, loss_a, loss_bc, loss_m, loss_u, loss_v, loss_w = sum_final_losses

        logger.info("--- Final Loss Breakdown ---")
        logger.info(f"MSE_alpha (volume fraction): {loss_a:.4e}")
        logger.info(f"MSE_BC                     : {loss_bc:.4e}")
        logger.info(f"MSE_f,m (continuity)       : {loss_m:.4e}")
        logger.info(f"MSE_f,u                    : {loss_u:.4e}")
        logger.info(f"MSE_f,v                    : {loss_v:.4e}")
        logger.info(f"MSE_f,w                    : {loss_w:.4e}")
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
        plt.plot(epochs_range, history_loss_f_m)
        plt.title(f'MSE of Continuity vs. Epochs ({adaptive_mode} - {activation_choice})')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss (f_m)')
        plt.yscale('log')
        plt.grid(True, which="both", ls="--")
        plt.savefig(os.path.join(dirname, 'loss_history_continuity.png'))

        plt.close('all')
        logger.info("Plots saved successfully.")

        history_filename = f"loss_history_{adaptive_mode}_{activation_choice}.csv"
        history_filepath = os.path.join(dirname, history_filename)
        history_df = pd.DataFrame({
            'epoch': epochs_range,
            'MSE_alpha': history_loss_a,
            'MSE_f_uvw': history_loss_f_uvw,
            'MSE_f_m': history_loss_f_m,
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
