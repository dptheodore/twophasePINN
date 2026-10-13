"""Compare a trained 3D PINN (pytorch_3d_rising_bubble_train.py) against the LBPM h5.

The training loss only checks the network against the volume fraction and the PDE
residuals; u, v, w and p are never compared with data. This script does that, on every
snapshot in the h5. Training uses every second snapshot, so the odd-indexed ones are
held out and are flagged as such in the output.

    python validate_pinn_3d.py checkpoints/<run>/            # picks the .pth in the dir
    python validate_pinn_3d.py checkpoints/<run>/loss_X.pth --stride 1 --plot 13

Nondimensionalisation is the one generate_points_3d applies: coordinates are in lattice
cells centred on the domain, divided by L_ref; time is relative to the first snapshot,
divided by T_ref; velocities are divided by U_ref. Pressure is only defined up to a
constant (the PINN pins it at one corner), so both are scaled by rho_ref * U_ref^2 and
gauged on the undisturbed far carrier field (see CORE_RADIUS / FAR_RADIUS).

The two pressures are NOT the same quantity. The h5 stores the Peng-Robinson
THERMODYNAMIC pressure, which swings by +/- 40 P_ref across the interface as the isotherm
crosses the two-phase loop. The network's p is the DYNAMIC pressure of the sharp-interface
form restored on 2026-09-22: the interfacial stress is carried by the singular
sigma*curvature force, not by p. So the interface band is excluded from the p comparison,
and the number to check is the Laplace jump 2/We across the drop.

w is dominated by the bulk drift of the whole domain, which would hide errors in the
flow around the drop. It is therefore also reported with each snapshot's domain-mean w
removed (column "w_rel").
"""
import sys
sys.path.append("../utilities")
import os
import glob
import argparse

import numpy as np
import pandas as pd
import h5py
import torch

from generate_points_3d import _measure_scales, RHO_GAS, T_MAX
from pytorch_utilities import NNCreator
from pytorch_3d_rising_bubble_train import build_activation_dict

INTERFACE_BAND = 4.0  # |levelset| below this counts as interface (same cut as the generator)
# The CFD is a Peng-Robinson single-component model, so its "pressure" is the THERMODYNAMIC
# pressure. The network's is not: with the sharp-interface form restored the interfacial
# stress lives in the sigma*curvature force, so p carries no PR isotherm excursion. The
# +/- 40 P_ref swing in the interface band is therefore a quantity the model does not
# represent by construction, and comparing p there measures nothing.
DEEP_PHASE = 7.5
# The GAUGE still wants quiet regions, since p is only defined up to a constant on both
# sides. |levelset| >= 7.5 is not quiet enough -- it reaches r ~ 0.73 R, where p still
# varies by +/- 14 P_ref. A core well inside the drop (p = 49.20 +/- 0.55 P_ref) and the
# far carrier (37.15 +/- 0.04) are, so the gauge uses those, in units of the drop radius.
CORE_RADIUS = 0.3     # r < CORE_RADIUS * L_ref: drop core
FAR_RADIUS = 2.0      # r > FAR_RADIUS * L_ref and deep carrier: undisturbed far field


def build_net(period, device):
    """Same architecture as pytorch_3d_rising_bubble_train.main()."""
    no_layers = 8
    hidden_layers = [400] * no_layers
    activation_functions = {"tanh": range(1, no_layers + 1)}
    activation_dict = build_activation_dict(hidden_layers, activation_functions, [10] * no_layers)
    output_layer = [
        ("output_u", None), ("output_v", None), ("output_w", None),
        ("output_p", "exponential"), ("output_a", "sigmoid"),
    ]
    net = NNCreator(torch.float32).get_model_dnn(4, hidden_layers, output_layer, activation_dict,
                                                 False, periodic_lengths=period)
    return net.to(device)


def resolve_checkpoint(path):
    if os.path.isdir(path):
        files = glob.glob(os.path.join(path, "*.pth"))
        if not files:
            raise FileNotFoundError(f"no .pth in {path}")
        return max(files, key=os.path.getctime)
    return path


@torch.no_grad()
def predict(net, pts, device, chunk=65536):
    out = [[] for _ in range(5)]
    for i in range(0, len(pts), chunk):
        x = torch.from_numpy(pts[i:i + chunk]).to(device)
        for k, o in enumerate(net(x)):
            out[k].append(o.cpu().numpy().ravel())
    return [np.concatenate(o) for o in out]


def errors(pred, ref):
    diff = pred - ref
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    ref_rms = float(np.sqrt(np.mean(ref ** 2)))
    return rmse, ref_rms, rmse / ref_rms if ref_rms > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", help=".pth file, or a run directory containing one")
    ap.add_argument("--h5", default="../cfd_data/lbpm_bubble_dataset.h5")
    ap.add_argument("--stride", type=int, default=2, help="spatial subsampling of the grid")
    ap.add_argument("--plot", type=int, default=None, help="snapshot index for an x-z slice plot")
    ap.add_argument("--out", default=None, help="output directory (default: next to the checkpoint)")
    args = ap.parse_args()

    ckpt = resolve_checkpoint(args.checkpoint)
    out_dir = args.out or os.path.dirname(os.path.abspath(ckpt))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    f = h5py.File(args.h5, "r")
    times = np.array(f["time"], dtype=float)
    n_keep = len(times) if T_MAX is None else int(np.searchsorted(times, T_MAX, side="right"))
    times = times[:n_keep]
    levelset = -np.array(f["levelset"][:n_keep])        # > 0 => drop, as in the generator
    pressure = np.array(f["pressure"][:n_keep])

    # Same scales the training run saw.
    L_ref, U_ref, _, SIGMA, _ = _measure_scales(levelset, pressure, times)
    T_ref = L_ref / U_ref
    P_ref = RHO_GAS * U_ref ** 2

    X, Y, Z = (np.array(f[c], dtype=float) for c in "XYZ")
    X, Y, Z = X / np.diff(X)[0], Y / np.diff(Y)[0], Z / np.diff(Z)[0]
    period = [len(X) / L_ref, len(Y) / L_ref, len(Z) / L_ref]
    X, Y, Z = X - X.mean(), Y - Y.mean(), Z - Z.mean()
    t_nd = (times - times[0]) / T_ref
    trained = np.zeros(len(times), dtype=bool)
    trained[::2] = True                                  # get_training_data_3d default stride

    net = build_net(period, device)
    net.load_state_dict(torch.load(ckpt, map_location=device))
    net.eval()
    print(f"\nCheckpoint: {ckpt}")
    print(f"L_ref = {L_ref:.4f} cells, U_ref = {U_ref:.6g} cells/step, T_ref = {T_ref:.1f} steps, "
          f"P_ref = {P_ref:.4g}")

    s = args.stride
    xs, ys, zs = X[::s] / L_ref, Y[::s] / L_ref, Z[::s] / L_ref
    ZZ, YY, XX = np.meshgrid(zs, ys, xs, indexing="ij")  # h5 layout is (Z, Y, X)
    grid = np.column_stack([XX.ravel(), YY.ravel(), ZZ.ravel()]).astype(np.float32)

    rows, jumps = [], []
    slices = None
    for k in range(len(times)):
        pts = np.hstack([grid, np.full((len(grid), 1), t_nd[k], np.float32)])
        u_p, v_p, w_p, p_p, a_p = predict(net, pts, device)

        sub = (k, slice(None, None, s), slice(None, None, s), slice(None, None, s))
        ls3 = levelset[sub]
        ls = ls3.ravel()
        u_c = f["velocityX"][sub].ravel() / U_ref
        v_c = f["velocityY"][sub].ravel() / U_ref
        w_c = f["velocityZ"][sub].ravel() / U_ref
        p_c = pressure[sub].ravel() / P_ref
        a_c = (ls > 0).astype(float)

        # Distance from the drop centroid, in cells, on the strided grid.
        centroid = np.argwhere(ls3 > 0).mean(axis=0)
        gz, gy, gx = np.ogrid[:ls3.shape[0], :ls3.shape[1], :ls3.shape[2]]
        radius = (np.sqrt((gz - centroid[0]) ** 2 + (gy - centroid[1]) ** 2
                          + (gx - centroid[2]) ** 2) * s).ravel()
        core = radius < CORE_RADIUS * L_ref
        far = (ls <= -DEEP_PHASE) & (radius > FAR_RADIUS * L_ref)

        # Pressure is defined up to a constant in the PINN and carries an arbitrary EOS
        # offset in the CFD, so both are gauged on the undisturbed far field (see
        # FAR_RADIUS). Gauging on any average that includes near-interface cells makes the
        # result depend on which cells are sampled -- on the drop's centroid plane it
        # swings between -0.14 and +0.51 with the subsampling stride alone.
        p_p_rel, p_c_rel = p_p - p_p[far].mean(), p_c - p_c[far].mean()
        w_p_rel, w_c_rel = w_p - w_p.mean(), w_c - w_c.mean()
        jumps.append(dict(snapshot=k, t_nd=t_nd[k],
                          pred=float(p_p_rel[core].mean()), cfd=float(p_c_rel[core].mean()),
                          cfd_core_std=float(p_c[core].std()), cfd_far_std=float(p_c[far].std())))

        regions = {
            "all": np.ones_like(ls, dtype=bool),
            "drop": ls >= INTERFACE_BAND,
            "carrier": ls <= -INTERFACE_BAND,
            "interface": np.abs(ls) < INTERFACE_BAND,
            # The only regions where p is comparable at all (see CORE_RADIUS/FAR_RADIUS).
            "core": core,
            "far_carrier": far,
        }
        fields = {
            "u": (u_p, u_c), "v": (v_p, v_c), "w": (w_p, w_c),
            "w_rel": (w_p_rel, w_c_rel), "p": (p_p_rel, p_c_rel), "a": (a_p, a_c),
        }
        for region, m in regions.items():
            for name, (pr, rf) in fields.items():
                rmse, ref_rms, rel = errors(pr[m], rf[m])
                rows.append(dict(snapshot=k, t_step=times[k], t_nd=t_nd[k], trained=trained[k],
                                 region=region, field=name, rmse=rmse, ref_rms=ref_rms, rel_l2=rel,
                                 pred_mean=float(pr[m].mean()), ref_mean=float(rf[m].mean())))

        if args.plot == k:
            shape = ZZ.shape
            slices = {name: (pr.reshape(shape), rf.reshape(shape)) for name, (pr, rf) in fields.items()}
            slices["_ls"] = ls.reshape(shape)

    df = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, "validation_vs_cfd.csv")
    df.to_csv(csv_path, index=False)

    # Summary: relative L2 error averaged over snapshots, trained vs held-out.
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.float_format", "{:.3e}".format)
    for region in ["all", "core", "far_carrier", "carrier", "drop", "interface"]:
        d = df[df.region == region]
        table = d.pivot_table(index="field", columns="trained", values=["rel_l2", "rmse", "ref_rms"],
                              aggfunc="mean")[["rel_l2", "rmse", "ref_rms"]]
        table.columns = [f"{v}_{'train' if t else 'heldout'}" for v, t in table.columns]
        print(f"\n--- region: {region} (relative L2 = rmse / rms(cfd), mean over snapshots) ---")
        print(table)
    # Bulk pressure difference between the phases. With the sharp-interface form the
    # network's jump should be the Laplace value 2*sigma/R, i.e. 2/We nondimensional -- that
    # is what sigma*curvature puts there. The CFD's own jump is reported alongside, but it
    # includes the Peng-Robinson EOS offset between the coexistence branches and so is NOT
    # the target; 2/We is.
    one_We = SIGMA / (RHO_GAS * U_ref ** 2 * L_ref)
    jd = pd.DataFrame(jumps)
    print("\n--- Bulk pressure difference: mean p(drop core) - p(far carrier), nondimensional ---")
    print(f"  PINN {jd.pred.mean():+.4f} (spread {jd.pred.std():.4f})   "
          f"CFD {jd.cfd.mean():+.4f} (spread {jd.cfd.std():.4f})")
    print(f"  Laplace target 2/We = {2 * one_We:+.4f}  (sigma = {SIGMA:.6g})")
    print(f"  ratio PINN/Laplace = {jd.pred.mean() / (2 * one_We):.3f}"
          if one_We != 0 else "")
    print(f"  CFD scatter within the gauge regions: core {jd.cfd_core_std.mean():.3f}, "
          f"far carrier {jd.cfd_far_std.mean():.3f}")

    d = df[(df.region == "all") & (df.field == "w")]
    print("\nBulk drift check, domain-mean w (nondim) pred vs cfd, first/mid/last snapshot:")
    for k in [0, len(times) // 2, len(times) - 1]:
        r = d[d.snapshot == k].iloc[0]
        print(f"  t_nd = {r.t_nd:6.3f}: pred {r.pred_mean:+.4e}  cfd {r.ref_mean:+.4e}")
    print(f"\nPer-snapshot table written to {csv_path}")

    if slices is not None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # x-z plane through the drop centroid
        drop = np.argwhere(slices["_ls"] > 0)
        iy = int(round(drop[:, 1].mean())) if len(drop) else slices["_ls"].shape[1] // 2
        names = ["u", "v", "w_rel", "p", "a"]
        fig, axes = plt.subplots(len(names), 3, figsize=(10, 4 * len(names)))
        extent = [xs[0], xs[-1], zs[0], zs[-1]]
        for row, name in zip(axes, names):
            pr, rf = slices[name]
            pr, rf = pr[:, iy, :], rf[:, iy, :]
            lim = max(np.abs(pr).max(), np.abs(rf).max())
            for ax, data, title in zip(row, [rf, pr, pr - rf], ["CFD", "PINN", "PINN - CFD"]):
                vmax = lim if title != "PINN - CFD" else max(np.abs(data).max(), 1e-12)
                im = ax.imshow(data, origin="lower", extent=extent, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
                ax.contour(np.linspace(xs[0], xs[-1], len(xs)), np.linspace(zs[0], zs[-1], len(zs)),
                           slices["_ls"][:, iy, :], levels=[0], colors="k", linewidths=0.8)
                ax.set_title(f"{name}: {title}")
                ax.set_xlabel("x/L_ref")
                ax.set_ylabel("z/L_ref")
                fig.colorbar(im, ax=ax, fraction=0.046)
        fig.suptitle(f"snapshot {args.plot} (t_nd = {t_nd[args.plot]:.3f}, "
                     f"{'trained' if trained[args.plot] else 'held out'}), y index {iy}")
        fig.tight_layout()
        png = os.path.join(out_dir, f"validation_slice_{args.plot}.png")
        fig.savefig(png, dpi=110)
        print(f"Slice plot written to {png}")


if __name__ == "__main__":
    main()
