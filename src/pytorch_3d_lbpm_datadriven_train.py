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
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import generate_points_3d_lbpm_data
from generate_points_3d_lbpm_data import (get_training_data_3d, get_validation_data_3d,
                                          get_scales, get_data_scales, PHI_SCALE,
                                          BulkSampler, periodic_lengths)
from pytorch_utilities import NNCreator

''' Data-driven fit of the LBPM Peng-Robinson run (lbpm_bubble_dataset_newrun_nowrap.h5);
pytorch_3d_lbpm_datadriven_geom_train.py without the geometric constraint.

    inputs    x, y, z, t   (cells / L_ref, (t - 19600) / T_ref)
    outputs   rho_hat, phi_hat, p_hat, u_hat, v_hat, w_hat, all linear heads (the targets
              and their normalisation are in generate_points_3d_lbpm_data.py; the velocity
              is the h5 velocity + F/(2 rho), min-max normalised per component)
    loss      MSE on each output against the CFD + periodic pairing of all six outputs
              on the x, y and z faces
    network   x, y, z enter through a sin/cos embedding with the box as period, so every
              output is exactly periodic: the pairing loss is ~0 by construction (kept as a
              check)
    points    near-interface DATA points are fixed; the bulk ones are redrawn every epoch
              (BulkSampler), so no thin feature can hide between a fixed set of points

No PDE, initial condition or pressure gauge: p is fitted as the absolute P_PR(rho). '''

# Every rank seeds identically and so generates the SAME training points. The data is
# then partitioned at batch level (rank r trains batches r, r + N, r + 2N, ... of a
# shuffle shared by all ranks), so each epoch visits every batch exactly once whatever the
# number of GPUs.
np.random.seed(1234)
torch.manual_seed(1234)


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


def phase_fraction(rho_hat):
    """a of the network: rho_hat clipped to [0, 1]."""
    return torch.clamp(rho_hat, 0.0, 1.0)


def periodic_pair_loss(model, x1, y1, z1, x2, y2, z2, t):
    """Sum over all outputs of the mismatch between two opposite faces at equal time."""
    pred_1 = model(torch.cat([x1, y1, z1, t], dim=1))
    pred_2 = model(torch.cat([x2, y2, z2, t], dim=1))
    return sum(torch.mean((a - b) ** 2) for a, b in zip(pred_1, pred_2))


def compute_loss(model, data_D, data_EW, data_FB, data_NS):
    x, y, z, t, rho_gt, phi_gt, p_gt, u_gt, v_gt, w_gt = data_D

    # Loss DATA (the CFD fields)
    pred_rho, pred_phi, pred_p, pred_u, pred_v, pred_w = model(torch.cat([x, y, z, t], dim=1))
    loss_rho = torch.mean((rho_gt - pred_rho) ** 2)
    loss_phi = torch.mean((phi_gt - pred_phi) ** 2)
    loss_p = torch.mean((p_gt - pred_p) ** 2)
    loss_u = torch.mean((u_gt - pred_u) ** 2)
    loss_v = torch.mean((v_gt - pred_v) ** 2)
    loss_w = torch.mean((w_gt - pred_w) ** 2)

    # Periodic pairing on all three axes
    loss_BC = periodic_pair_loss(model, *data_EW) + periodic_pair_loss(model, *data_FB) + \
              periodic_pair_loss(model, *data_NS)

    total_loss = loss_rho + loss_phi + loss_p + loss_u + loss_v + loss_w + loss_BC

    return total_loss, loss_rho, loss_phi, loss_p, loss_u, loss_v, loss_w, loss_BC


VALIDATION_COLUMNS = ["mse_a", "rel_l2_rho", "rel_l2_phi_band", "rel_l2_p",
                      "rel_l2_u", "rel_l2_v", "rel_l2_w", "rel_l2_w_relative"]


def evaluate_validation(net, val, scales, data_scales, chunk=20000):
    """Compares the network with the CFD on the held-out validation snapshots, in physical
    (lattice) units: MSE of a, relative L2 errors of rho, p, u, v and w, and of the
    levelset inside its band (|levelset| < 8 cells; outside it the CFD levelset is the clip
    value). w is dominated by the bulk drift, so rel_l2_w_relative compares w minus its
    per-snapshot mean (over the validation points), i.e. the flow relative to the drift."""
    preds = []
    with torch.no_grad():
        for s in range(0, len(val["xyzt"]), chunk):
            preds.append(torch.cat(net(val["xyzt"][s:s + chunk]), dim=1))
    rho_hat, phi_hat, p_hat, u_hat, v_hat, w_hat = torch.cat(preds).T
    ref = val["targets"]
    band = val["band"]
    drho = scales["rho1"] - scales["rho2"]
    dp = data_scales["p_max"] - data_scales["p_min"]

    def rel_l2(pred, true):
        return (torch.linalg.norm(pred - true) / torch.linalg.norm(true)).item()

    def velocity(c, hat):
        return data_scales[f"{c}_min"] + (data_scales[f"{c}_max"] - data_scales[f"{c}_min"]) * hat

    u_pred, v_pred, w_pred = (velocity(c, h) for c, h in zip("uvw", (u_hat, v_hat, w_hat)))
    u_true, v_true, w_true = (velocity(c, ref[:, 5 + k]) for k, c in enumerate("uvw"))

    t = val["xyzt"][:, 3]
    w_pred_rel, w_true_rel = w_pred.clone(), w_true.clone()
    for tv in torch.unique(t):
        m = t == tv
        w_pred_rel[m] -= w_pred[m].mean()
        w_true_rel[m] -= w_true[m].mean()

    return dict(mse_a=torch.mean((phase_fraction(rho_hat) - ref[:, 3]) ** 2).item(),
                rel_l2_rho=rel_l2(scales["rho2"] + drho * rho_hat, scales["rho2"] + drho * ref[:, 0]),
                rel_l2_phi_band=rel_l2(phi_hat[band], ref[band, 1]),
                rel_l2_p=rel_l2(data_scales["p_min"] + dp * p_hat, data_scales["p_min"] + dp * ref[:, 2]),
                rel_l2_u=rel_l2(u_pred, u_true), rel_l2_v=rel_l2(v_pred, v_true), rel_l2_w=rel_l2(w_pred, w_true),
                rel_l2_w_relative=rel_l2(w_pred_rel, w_true_rel))


def train_step(model, optimizer, batch):
    optimizer.zero_grad()
    losses = compute_loss(model, *batch)
    losses[0].backward()
    optimizer.step()
    return losses


def setup_output_directory():
    if not os.path.exists("checkpoints"):
        os.mkdir("checkpoints")
    dirname = os.path.abspath(os.path.join("checkpoints", datetime.now().strftime("%b-%d-%Y_%H-%M-%S")))
    os.mkdir(dirname)

    shutil.copyfile(__file__, os.path.join(dirname, os.path.basename(__file__)))
    for f in ("generate_points_3d_lbpm_data.py", "generate_points_3d_lbpm.py", "generate_points_3d.py",
              "generate_points_3d_pr.py"):
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


def to_tensor_tuple(df, columns, device):
    return tuple(
        torch.tensor(df[c].to_numpy().reshape(-1, 1), dtype=torch.float32, device=device)
        for c in columns
    )


def build_batch_tensors(batch_dict, device):
    return tuple(to_tensor_tuple(batch_dict[key], batch_dict[key].columns, device)
                 for key in ("DATA", "EW", "FB", "NS"))


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

    # (near-interface, domain) data points per training snapshot. Domain was 2700 (the
    # A + PDE budget of pytorch_3d_lbpm_train.py); at ~1 point per 500 carrier cells the
    # net grew an unsampled 1-2 cell sheet with rho < 0 in run Oct-26-2026_02-00-01. At
    # 10800 fixed points it was still there (Oct-26-2026_06-37-38): a plane < 0.15 cells
    # thick that none of the fixed points touched. The domain points are now redrawn
    # every epoch by BulkSampler, NOP_bulk per snapshot, anywhere 4+ cells off the surface.
    NOP_data = (3600, 0)
    NOP_bulk = 10800
    NOP_NS = (20, 20)
    NOP_east = (20, 20)
    NOP_west = (20, 20)
    NOP_front_back = (20, 20)

    # Generation prints only from rank 0; the rest generate silently.
    with contextlib.redirect_stdout(sys.stdout if is_main else open(os.devnull, "w")):
        scales = get_scales()
        data_scales = get_data_scales(velocity=True)
        training_data = get_training_data_3d(NOP_data, NOP_NS, NOP_east, NOP_west, NOP_front_back,
                                             velocity=True, geom=False)
        bulk_sampler = BulkSampler(NOP_bulk, velocity=True)
        periods = periodic_lengths()
    near_data = training_data["DATA"]
    # A first bulk draw, for the batch sizes and the final evaluation; each epoch redraws it.
    training_data["DATA"] = pd.concat([near_data, bulk_sampler.sample(np.random.default_rng([1234]))],
                                      ignore_index=True)
    logger.info(f"Training data: {os.path.abspath(generate_points_3d_lbpm_data.H5_PATH)}")
    logger.info(f"Normalisation: rho1 = {scales['rho1']:.5f}, rho2 = {scales['rho2']:.5f}, "
                f"phi / {PHI_SCALE:g} cells, p in [{data_scales['p_min']:.6g}, {data_scales['p_max']:.6g}], "
                f"L_ref = {scales['L_ref']:.4f} cells, T_ref = {scales['T_ref']:.1f} steps")
    logger.info("Velocity (h5 + F/(2 rho)) normalisation, lattice units: " +
                ", ".join(f"{c} in [{data_scales[c + '_min']:.4g}, {data_scales[c + '_max']:.4g}]" for c in "uvw"))

    # --- NN Architecture and Hyperparameters --- #
    no_layers = 8
    hidden_layers = [400] * no_layers
    activation_choice = 'tanh'
    adaptive_mode = 'fixed'  # adaptive activation is not wired up for the 3D case yet
    adaptive_activation_n = [10] * no_layers
    activation_functions = {activation_choice: range(1, no_layers + 1)}
    activation_functions_dict = build_activation_dict(hidden_layers, activation_functions, adaptive_activation_n)

    logger.info(f"Configuration: Activation='{activation_choice}', Mode='{adaptive_mode}'")
    logger.info(f"Periodic embedding of x, y, z with periods {np.round(periods, 4)} (nondim); "
                f"bulk DATA points: {NOP_bulk} per snapshot, redrawn every epoch")

    # Linear heads: rho_hat overshoots [0, 1] slightly, phi_hat spans [-1, 1], the rest [0, 1].
    output_layer = [("output_rho", None), ("output_phi", None), ("output_p", None),
                    ("output_u", None), ("output_v", None), ("output_w", None)]

    nn_creator = NNCreator(torch.float32)
    net = nn_creator.get_model_dnn(4, hidden_layers, output_layer, activation_functions_dict, False,
                                   periodic_lengths=periods)
    net.to(device)

    if is_distributed:
        model = DDP(net, device_ids=[local_rank] if torch.cuda.is_available() else None)
    else:
        model = net

    epochs_list = [5000] * 5
    learning_rates = [1e-4, 5e-5, 1e-5, 5e-6, 1e-6]
    checkpoint_interval = 10
    # Optimizer steps per epoch, as in rising_bubble_train.py (num_of_batches = 20 on one
    # process). Each DDP step consumes one batch per rank, so the data is split into
    # steps_per_epoch * world_size batches; every rank takes the same number of steps
    # (unequal counts deadlock DDP's all-reduce).
    steps_per_epoch = 20
    num_of_batches = steps_per_epoch * world_size

    num_samples_total = sum(len(df) for df in training_data.values())
    total_batch_size = math.ceil(num_samples_total / num_of_batches)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rates[0], eps=1e-8)

    start_total = time.time()
    loss_names = ["rho", "phi", "p", "u", "v", "w", "BC"]
    history = {k: [] for k in loss_names}
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
            # Same bulk points and permutation on every rank (seeded by phase and epoch),
            # different batches.
            rng = np.random.default_rng([1234, i, epoch])
            epoch_data = dict(training_data, DATA=pd.concat([near_data, bulk_sampler.sample(rng)],
                                                            ignore_index=True))
            shuffled_data = {key: df.sample(frac=1, random_state=rng).reset_index(drop=True)
                             for key, df in epoch_data.items()}

            # No skipping of empty batches here: a rank that skips a step deadlocks the rest.
            # slice_batch splits every set evenly, so no batch is empty anyway.
            for k in range(num_local_batches):
                batch_dict = slice_batch(shuffled_data, num_batches, rank + k * world_size)
                batch = build_batch_tensors(batch_dict, device)
                losses = train_step(model, optimizer, batch)
                epoch_losses.append([l.item() for l in losses])

            # Reported as rising_bubble_train.py does: the SUM of the per-batch losses over
            # its 20 batches, i.e. the mean over every batch of the epoch on all ranks times
            # steps_per_epoch (= 20), so it does not depend on the GPU count.
            sum_losses = torch.tensor(np.sum(epoch_losses, axis=0), dtype=torch.float64, device=device)
            if is_distributed:
                dist.all_reduce(sum_losses, op=dist.ReduceOp.SUM)
            epoch_losses_2d = sum_losses / (num_local_batches * world_size) * steps_per_epoch
            total_loss, *parts = epoch_losses_2d.tolist()
            for name, value in zip(loss_names, parts):
                history[name].append(value)

            if epoch % checkpoint_interval == 0:
                current_time = time.time()
                time_for_epoch = current_time - start_checkpoint_time
                start_checkpoint_time = current_time
                log_msg = f"Epoch: {epoch}/{epochs} - Time: {time_for_epoch:.2f}s - Loss: {total_loss:.4e} | "
                log_msg += ", ".join(f"{name}: {value:.4e}" for name, value in zip(loss_names, parts))
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
        with torch.no_grad():
            for b in range(num_batches):
                batch_dict = slice_batch(training_data, num_batches, b)
                if all(batch.empty for batch in batch_dict.values()):
                    continue
                batch = build_batch_tensors(batch_dict, device)
                batch_losses = compute_loss(underlying_net, *batch)
                final_evaluation_losses.append([l.item() for l in batch_losses])

        # Same convention as the per-epoch log: 2D's sum over 20 batches.
        final_losses_2d = np.mean(final_evaluation_losses, axis=0) * steps_per_epoch

        logger.info("--- Final Loss Breakdown ---")
        for name, value in zip(loss_names, final_losses_2d[1:]):
            logger.info(f"{'MSE_' + name:<27}: {value:.4e}")
        logger.info("----------------------------")
        # The 8 held-out snapshots, never used for training and only looked at here, once
        # training is done: (near-interface, anywhere) points per snapshot.
        NOP_validation = (2000, 8000)
        val_df = get_validation_data_3d(NOP_validation, velocity=True)
        val = dict(xyzt=torch.tensor(val_df[["x", "y", "z", "t"]].to_numpy(), dtype=torch.float32, device=device),
                   targets=torch.tensor(val_df[["rho_hat", "phi_hat", "p_hat", "a",
                                                "band", "u_hat", "v_hat", "w_hat"]].to_numpy(),
                                        dtype=torch.float32, device=device),
                   band=torch.tensor(val_df["band"].to_numpy() > 0.5, device=device))
        val_steps = np.unique(np.round(val_df['t'].to_numpy() * scales["T_ref"] + scales["t_origin"]))
        logger.info(f"Validation: {len(val_df)} points on the held-out snapshots t = {val_steps.astype(int)} "
                    f"(phi on the {int(val['band'].sum())} points inside the levelset band)")
        metrics = evaluate_validation(underlying_net, val, scales, data_scales)
        pd.DataFrame([metrics])[VALIDATION_COLUMNS].to_csv(os.path.join(dirname, "validation.csv"), index=False)
        logger.info("--- Held-out validation snapshots (vs CFD) ---")
        logger.info(f"MSE_a                      : {metrics['mse_a']:.4e}")
        logger.info(f"rel. L2 error rho          : {metrics['rel_l2_rho']:.4e}")
        logger.info(f"rel. L2 error phi (band)   : {metrics['rel_l2_phi_band']:.4e}")
        logger.info(f"rel. L2 error p            : {metrics['rel_l2_p']:.4e}")
        logger.info(f"rel. L2 error u            : {metrics['rel_l2_u']:.4e}")
        logger.info(f"rel. L2 error v            : {metrics['rel_l2_v']:.4e}")
        logger.info(f"rel. L2 error w            : {metrics['rel_l2_w']:.4e}")
        logger.info(f"rel. L2 error w - <w>      : {metrics['rel_l2_w_relative']:.4e}")
        logger.info("----------------------------\n")

        logger.info("Generating and saving loss history plots...")
        import matplotlib.pyplot as plt
        epochs_range = range(1, len(history["rho"]) + 1)

        for name in loss_names:
            plt.figure(figsize=(10, 6))
            plt.plot(epochs_range, history[name])
            plt.title(f'Loss {name} vs. Epochs ({adaptive_mode} - {activation_choice})')
            plt.xlabel('Epoch')
            plt.ylabel(f'Loss {name}')
            plt.yscale('log')
            plt.grid(True, which="both", ls="--")
            plt.savefig(os.path.join(dirname, f'loss_history_{name}.png'))

        plt.close('all')
        logger.info("Plots saved successfully.")

        history_filename = f"loss_history_{adaptive_mode}_{activation_choice}.csv"
        history_filepath = os.path.join(dirname, history_filename)
        history_df = pd.DataFrame({'epoch': epochs_range, **{f'loss_{k}': v for k, v in history.items()}})
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
