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

import generate_points_3d_pr as pr
from generate_points_3d_pr import get_training_data_3d_pr
from pytorch_utilities import NNCreator

''' Peng-Robinson counterpart of pytorch_3d_rising_bubble_train.py.

Inputs (x, y, z, t); outputs (u, v, w, rho, p, a). Density is now a network output
supervised directly from the h5, instead of being reconstructed from a volume fraction
as rho2 + (rho1 - rho2) * a. That matters for a single-component Peng-Robinson fluid: rho
is THE state variable, the interface is a smooth density profile set by kappa, and the
bulk densities are not pinned at the .db's rhoA/rhoB (the data runs 0.45 .. 5.95).

Residuals, all nondimensional (see generate_points_3d_pr for the derivation from LBPM):

  continuity  rho_t + u . grad(rho) + rho div(u) = 0
  momentum    rho Du/Dt = -Eu grad(p) + (Ko rho + Kb) grad(lap(rho)) + div(tau) + rho g(t)/Fr
  EOS         p = P_PR(rho) / p_scale

Eu = k^2 p_scale / (rho_ref U_ref^2) carries LBPM's k-transformation factor. Kb =
LATTICE_BETA / (L_ref U_ref)^2 is the rho-independent Korteweg-like term the CFD obeys on
top of kappa (see generate_points_3d_pr); without it the CFD's own rho profile is not a
solution and the network trades rho accuracy for momentum loss. The EOS
residual holds p to the thermodynamic pressure, which also fixes its gauge -- no pressure
gauge point is needed any more. a takes no part in the physics: it is a supervised phase
indicator kept for interface tracking and post-processing.

Viscous stress tau = mu (grad u + grad u^T) + BULK_LAMBDA mu div(u) I with mu = rho nu.
BULK_LAMBDA = 0 is the lattice-Boltzmann stress (bulk viscosity 2 mu / 3); -2/3 would be
Stokes' hypothesis, which the earlier 3D script used. '''

# Every rank seeds identically and so generates the SAME training points. The data is
# then partitioned at batch level (rank r trains batches r, r + N, r + 2N, ... of a
# shuffle shared by all ranks), so each epoch visits every batch exactly once whatever the
# number of GPUs.
np.random.seed(1234)
torch.manual_seed(1234)

BULK_LAMBDA = 0.0

# Body force along z.
#   "ramp":    LBPM's nominal g(t) = 2e-6 * floor(t / 20000), imposed.
#   "learned": g(t) is a trainable piecewise-linear curve, uniform in space, with one knot
#              per h5 snapshot, initialised at the ramp. The stored velocities accelerate at
#              only ~0.74 of the nominal force (even in pure gas), so this lets the momentum
#              equation find the force the data actually reflects. Compare the result with
#              the domain momentum budget logged alongside it.
GRAVITY_MODE = "learned"

# Which fields besides a are supervised from the h5. ("rho",) infers u, v, w and p from
# the physics alone; add "u", "v", "w" (and/or "p") for a data-assisted run. A learned
# g(t) needs the velocities: with density alone the network can invent velocities to suit
# whatever g it lands on, and g is barely identifiable.
SUPERVISED = ("rho", "u", "v", "w") if GRAVITY_MODE == "learned" else ("rho",)

Physics = namedtuple(
    "Physics",
    ["one_Re", "Eu", "Ko", "Kb", "U_ref", "L_ref", "rho_ref", "p_scale",
     "drho_star", "loss_weights_PDE", "loss_weight_EOS", "loss_weights_DATA"],
)


class PINN(torch.nn.Module):
    """The field network plus the body force g(t), so that DDP sees (and syncs) the
    learned force's parameters like any other weight. forward() returns
    (u, v, w, rho, p, a, g_star), g_star = g(t) L_ref / U_ref^2."""

    def __init__(self, net, mode, t_knots, T_ref, t_origin, L_ref, U_ref):
        super().__init__()
        self.net = net
        self.mode = mode
        self.T_ref, self.t_origin = T_ref, t_origin
        self.g_to_star = L_ref / U_ref ** 2
        self.register_buffer("t_knots", torch.tensor(t_knots, dtype=torch.float32))
        if mode == "learned":
            # theta = g / FORCE_INCREMENT, O(1), starting from the nominal ramp.
            t_lattice = np.asarray(t_knots) * T_ref + t_origin
            theta0 = np.floor(t_lattice / pr.FORCE_RAMP_INTERVAL)
            self.theta = torch.nn.Parameter(torch.tensor(theta0, dtype=torch.float32))
        elif mode != "ramp":
            raise ValueError(f"unknown GRAVITY_MODE {mode!r}")

    def g_lattice(self, t):
        """g(t) in lattice units (cells/step^2), t nondimensional, shape (N, 1)."""
        t = t.detach()
        if self.mode == "ramp":
            return pr.FORCE_INCREMENT * torch.floor((t * self.T_ref + self.t_origin) / pr.FORCE_RAMP_INTERVAL)
        k = self.t_knots
        idx = torch.bucketize(t.contiguous(), k).clamp(1, len(k) - 1)
        t0, t1 = k[idx - 1], k[idx]
        frac = ((t - t0) / (t1 - t0)).clamp(0.0, 1.0)
        theta = self.theta[idx - 1] * (1.0 - frac) + self.theta[idx] * frac
        return pr.FORCE_INCREMENT * theta

    def forward(self, X):
        return (*self.net(X), self.g_lattice(X[:, 3:4]) * self.g_to_star)


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


def grad(f, x):
    return torch.autograd.grad(f, x, torch.ones_like(f), create_graph=True)[0]


def PDE_caller(model, physics, x, y, z, t):
    """Residuals of continuity, the three momentum components, and the EOS.

    `model` is called directly (not through a `self.forward` alias) so that this works
    whether `model` is the raw net or its DDP wrapper.
    """
    x = x.requires_grad_(True)
    y = y.requires_grad_(True)
    z = z.requires_grad_(True)
    t = t.requires_grad_(True)
    X = (x, y, z)

    u, v, w, rho, p, _, gz = model(torch.cat([x, y, z, t], dim=1))
    vel = (u, v, w)

    # First derivatives: dvel[i][j] = d vel_i / d x_j.
    dvel = [[grad(c, xj) for xj in X] for c in vel]
    vel_t = [grad(c, t) for c in vel]
    drho = [grad(rho, xj) for xj in X]
    rho_t = grad(rho, t)
    dp = [grad(p, xj) for xj in X]

    div_u = dvel[0][0] + dvel[1][1] + dvel[2][2]
    ddiv = [grad(div_u, xj) for xj in X]
    lap_vel = [sum(grad(dvel[i][j], X[j]) for j in range(3)) for i in range(3)]

    # Korteweg force needs THIRD derivatives of rho: grad(lap(rho)).
    lap_rho = sum(grad(drho[j], X[j]) for j in range(3))
    dlap_rho = [grad(lap_rho, xj) for xj in X]

    PDE_m = rho_t + u * drho[0] + v * drho[1] + w * drho[2] + rho * div_u

    lam = BULK_LAMBDA
    PDE_mom = []
    for i in range(3):
        accel = vel_t[i] + u * dvel[i][0] + v * dvel[i][1] + w * dvel[i][2]
        # div(tau)_i / (rho_ref U^2 / L) with mu* = rho* nu:
        # rho (lap u_i + (1 + lam) d_i div) + d_j rho (d_j u_i + d_i u_j) + lam d_i rho div
        visc = rho * (lap_vel[i] + (1.0 + lam) * ddiv[i]) + \
            sum(drho[j] * (dvel[i][j] + dvel[j][i]) for j in range(3)) + \
            lam * drho[i] * div_u
        res = rho * accel + physics.Eu * dp[i] - (physics.Ko * rho + physics.Kb) * dlap_rho[i] - \
            physics.one_Re * visc
        if i == 2:
            res = res - rho * gz
        PDE_mom.append(res)

    PDE_eos = p - pr.pressure_pr(rho * physics.rho_ref) / physics.p_scale

    return PDE_m, PDE_mom[0], PDE_mom[1], PDE_mom[2], PDE_eos


def compute_loss(model, physics, data_DATA, data_PDE):
    x_D, y_D, z_D, t_D = data_DATA[:4]
    targets = data_DATA[4:-1]
    a_D = data_DATA[-1]

    # DATA LOSSES. rho is compared in units of the liquid-gas density jump so its loss is
    # on the same O(1) footing as a's; velocities and p are already O(1) in u*, p*.
    u, v, w, rho, p, a, _ = model(torch.cat([x_D, y_D, z_D, t_D], dim=1))
    pred = dict(rho=rho, u=u, v=v, w=w, p=p)
    loss_a = torch.mean((a_D - a) ** 2)
    loss_fields = []
    for name, target in zip(SUPERVISED, targets):
        err = pred[name] - target
        if name == "rho":
            err = err / physics.drho_star
        loss_fields.append(torch.mean(err ** 2))
    loss_data = loss_a + torch.dot(torch.stack(loss_fields), physics.loss_weights_DATA)

    x_P, y_P, z_P, t_P = data_PDE
    PDE_m, PDE_u, PDE_v, PDE_w, PDE_eos = PDE_caller(model, physics, x_P, y_P, z_P, t_P)
    loss_m = torch.mean(PDE_m ** 2)
    loss_u = torch.mean(PDE_u ** 2)
    loss_v = torch.mean(PDE_v ** 2)
    loss_w = torch.mean(PDE_w ** 2)
    loss_eos = torch.mean(PDE_eos ** 2)

    loss_PDE = torch.dot(torch.stack([loss_m, loss_u, loss_v, loss_w]), physics.loss_weights_PDE) + \
        physics.loss_weight_EOS * loss_eos

    total_loss = loss_data + loss_PDE
    return (total_loss, loss_a, *loss_fields, loss_m, loss_u, loss_v, loss_w, loss_eos)


def loss_names():
    return ["total", "a"] + list(SUPERVISED) + ["m", "u_mom", "v_mom", "w_mom", "eos"]


def train_step(model, optimizer, physics, data_DATA, data_PDE):
    optimizer.zero_grad()
    losses = compute_loss(model, physics, data_DATA, data_PDE)
    losses[0].backward()
    optimizer.step()
    return losses


def setup_output_directory():
    if not os.path.exists("checkpoints"):
        os.mkdir("checkpoints")
    dirname = os.path.abspath(os.path.join("checkpoints", "pr_" + datetime.now().strftime("%b-%d-%Y_%H-%M-%S")))
    os.mkdir(dirname)

    shutil.copyfile(__file__, os.path.join(dirname, os.path.basename(__file__)))
    for f in ("generate_points_3d_pr.py", "generate_points_3d.py"):
        if os.path.exists(f):
            shutil.copyfile(f, os.path.join(dirname, f))

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


def to_tensor_tuple(df, device):
    return tuple(
        torch.tensor(df[c].to_numpy().reshape(-1, 1), dtype=torch.float32, device=device)
        for c in df.columns
    )


def build_batch_tensors(batch_dict, device):
    return to_tensor_tuple(batch_dict['DATA'], device), to_tensor_tuple(batch_dict['PDE'], device)


def slice_batch(data, prop_batch_sizes, b):
    batch_dict = {}
    for key, df in data.items():
        start_idx = b * prop_batch_sizes[key]
        end_idx = (b + 1) * prop_batch_sizes[key]
        batch_dict[key] = df.iloc[start_idx:end_idx]
    return batch_dict


def format_losses(values):
    return ", ".join(f"{n}: {v:.4e}" for n, v in zip(loss_names(), values))


def gravity_table(pinn, scales, device):
    """g(t) at every knot: the model's (learned or imposed), LBPM's nominal ramp, and the
    momentum-budget estimate from the data (interpolated from interval midpoints)."""
    t_knots = np.asarray(scales["t_knots"])
    t_lattice = t_knots * scales["T_ref"] + scales["t_origin"]
    with torch.no_grad():
        t = torch.tensor(t_knots, dtype=torch.float32, device=device).reshape(-1, 1)
        g_model = pinn.g_lattice(t).cpu().numpy().ravel()
    return pd.DataFrame({
        "t": t_lattice.astype(int),
        "g_model": g_model,
        "g_budget": np.interp(t_lattice, scales["t_budget"], scales["g_budget"]),
        "g_nominal": pr.body_force(t_lattice),
    })


def format_gravity(df):
    rows = df.iloc[::4]
    return "  g(t) model/budget: " + "  ".join(
        f"{t}: {gm:.2e}/{gb:.2e}" for t, gm, gb in zip(rows.t, rows.g_model, rows.g_budget))


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

    NOP_DATA = (500, 400)          # (interface band, bulk) grid nodes per snapshot
    NOP_PDE = (400, 2000, 3000)    # (interface, nearfield, domain) per snapshot

    # Generation prints only from rank 0; the rest generate silently.
    with contextlib.redirect_stdout(sys.stdout if is_main else open(os.devnull, "w")):
        training_data, scales = get_training_data_3d_pr(NOP_DATA, NOP_PDE, supervised=SUPERVISED)

    # --- NN Architecture and Hyperparameters --- #
    no_layers = 8
    hidden_layers = [400] * no_layers
    activation_choice = 'tanh'
    adaptive_activation_n = [10] * no_layers
    activation_functions = {activation_choice: range(1, no_layers + 1)}
    activation_functions_dict = build_activation_dict(hidden_layers, activation_functions, adaptive_activation_n)

    # rho* = exp(.) keeps density positive, so P_PR stays on its physical branch; p* is
    # linear because P_PR goes negative inside
    # the interface (down to -0.84 p_scale in the data).
    output_layer = [
        ("output_u", None), ("output_v", None), ("output_w", None),
        ("output_rho", "exponential"), ("output_p", None), ("output_a", "sigmoid"),
    ]

    nn_creator = NNCreator(torch.float32)
    net = nn_creator.get_model_dnn(4, hidden_layers, output_layer, activation_functions_dict, False,
                                   periodic_lengths=scales["period"])
    pinn = PINN(net, GRAVITY_MODE, scales["t_knots"], scales["T_ref"], scales["t_origin"],
                scales["L_ref"], scales["U_ref"]).to(device)
    logger.info(f"Periodic lengths (nondimensional): {scales['period']}")
    logger.info(f"Supervised fields: a + {SUPERVISED}, BULK_LAMBDA = {BULK_LAMBDA}, "
                f"GRAVITY_MODE = {GRAVITY_MODE}")

    model = DDP(pinn, device_ids=[local_rank] if torch.cuda.is_available() else None) \
        if is_distributed else pinn

    L_ref, U_ref, rho_ref, p_scale = scales["L_ref"], scales["U_ref"], scales["rho_ref"], scales["p_scale"]
    one_Re = pr.NU / (U_ref * L_ref)
    Eu = pr.K_FACTOR ** 2 * p_scale / (rho_ref * U_ref ** 2)
    Ko = pr.PR_KAPPA * rho_ref / (L_ref ** 2 * U_ref ** 2)
    Kb = pr.LATTICE_BETA / (L_ref ** 2 * U_ref ** 2)
    logger.info(f"L_ref={L_ref:.4f} cells, U_ref={U_ref:.6g} cells/step, T_ref={scales['T_ref']:.1f} steps")
    logger.info(f"1/Re={one_Re:.4f}, Eu={Eu:.4f}, Ko={Ko:.4f}, Kb={Kb:.4f}, "
                f"1/Fr(g=2e-6)={pr.FORCE_INCREMENT * L_ref / U_ref ** 2:.4f}")

    # [continuity, u, v, w] -- same weighting as the 3D script. Interface residuals are
    # dominated by the Eu grad(p) / Ko grad(lap(rho)) balance, so if the momentum losses
    # swamp everything else, these are the first knobs to turn.
    loss_weights_PDE = [1.0, 10.0, 10.0, 10.0]
    loss_weight_EOS = 1.0
    loss_weights_DATA = [1.0] * len(SUPERVISED)
    physics = Physics(
        one_Re=one_Re, Eu=Eu, Ko=Ko, Kb=Kb, U_ref=U_ref, L_ref=L_ref, rho_ref=rho_ref, p_scale=p_scale,
        drho_star=(pr.RHO_LIQ - pr.RHO_GAS) / rho_ref,
        loss_weights_PDE=torch.tensor(loss_weights_PDE, dtype=torch.float32, device=device),
        loss_weight_EOS=loss_weight_EOS,
        loss_weights_DATA=torch.tensor(loss_weights_DATA, dtype=torch.float32, device=device),
    )

    epochs_list = [5000] * 5
    learning_rates = [1e-4, 5e-5, 1e-5, 5e-6, 1e-6]
    checkpoint_interval = 10
    # Batches per epoch across ALL ranks, rounded up to a multiple of world_size so every
    # rank takes the same number of steps (unequal counts deadlock DDP's all-reduce). The
    # per-GPU batch therefore stays ~1/40 of the data; the effective batch per optimizer
    # step is world_size times that. 40 (was 20) keeps the batch the same size since the
    # generator went from every second snapshot to all 27.
    num_of_batches = math.ceil(40 / world_size) * world_size

    num_samples_total = sum(len(df) for df in training_data.values())
    total_batch_size = math.ceil(num_samples_total / num_of_batches)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rates[0], eps=1e-8)

    start_total = time.time()
    history = []
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
            for k in range(num_local_batches):
                batch_dict = slice_batch(shuffled_data, prop_batch_sizes, rank + k * world_size)
                data_DATA, data_PDE = build_batch_tensors(batch_dict, device)
                losses = train_step(model, optimizer, physics, data_DATA, data_PDE)
                epoch_losses.append([l.item() for l in losses])

            # Summed over batches and ranks, matching pytorch_3d_rising_bubble_train.py.
            sum_losses = torch.tensor(np.sum(epoch_losses, axis=0), dtype=torch.float64, device=device)
            if is_distributed:
                dist.all_reduce(sum_losses, op=dist.ReduceOp.SUM)
            values = sum_losses.tolist()
            history.append(values)
            total_loss = values[0]

            if epoch % checkpoint_interval == 0:
                current_time = time.time()
                time_for_epoch = current_time - start_checkpoint_time
                start_checkpoint_time = current_time
                logger.info(f"Epoch: {epoch}/{epochs} - Time: {time_for_epoch:.2f}s | {format_losses(values)}")
                if GRAVITY_MODE == "learned" and is_main and epoch % (10 * checkpoint_interval) == 0:
                    logger.info(format_gravity(gravity_table(pinn, scales, device)))

            if is_main and epoch % checkpoint_interval == 0 and total_loss < current_best_total_loss:
                logger.info(f"Saving checkpoint at epoch {epoch} with loss {total_loss:.4e}")
                for f in glob.glob(os.path.join(dirname, "*.pth")):
                    os.remove(f)
                safe_loss = f"{total_loss:.4e}".replace("+", "").replace("-", "m")
                state_dict = (model.module if is_distributed else model).state_dict()
                torch.save(state_dict, os.path.join(dirname, f"loss_{safe_loss}.pth"))
                current_best_total_loss = total_loss

    if is_distributed:
        dist.barrier()

    logger.info(f"\nTotal training time: {time.time() - start_total:.3f}s")

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

        underlying_net.eval()
        final_losses = []
        for b in range(num_batches):
            batch_dict = slice_batch(training_data, prop_batch_sizes, b)
            if all(batch.empty for batch in batch_dict.values()):
                continue
            data_DATA, data_PDE = build_batch_tensors(batch_dict, device)
            # No torch.no_grad(): the residuals need autograd input derivatives.
            batch_losses = compute_loss(underlying_net, physics, data_DATA, data_PDE)
            final_losses.append([l.item() for l in batch_losses])

        logger.info("--- Final Loss Breakdown ---")
        for n, v in zip(loss_names(), np.sum(final_losses, axis=0)):
            logger.info(f"MSE_{n:<8}: {v:.4e}")
        logger.info("----------------------------\n")

        g_df = gravity_table(underlying_net, scales, device)
        g_df.to_csv(os.path.join(dirname, "gravity.csv"), index=False)
        logger.info("Body force g(t), lattice units (also in gravity.csv):\n" + g_df.to_string(index=False))

        history_df = pd.DataFrame(history, columns=loss_names())
        history_df.insert(0, "epoch", range(1, len(history_df) + 1))
        history_df.to_csv(os.path.join(dirname, "loss_history.csv"), index=False)

        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        for n in loss_names()[1:]:
            plt.plot(history_df["epoch"], history_df[n], label=n)
        plt.yscale('log')
        plt.xlabel('Epoch')
        plt.ylabel('MSE (summed over batches)')
        plt.legend()
        plt.grid(True, which="both", ls="--")
        plt.savefig(os.path.join(dirname, 'loss_history.png'))
        plt.close('all')
        logger.info("Loss history saved.")

        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)

    if is_distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
