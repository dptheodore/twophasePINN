"""Compare a trained data-driven LBPM network (pytorch_3d_lbpm_datadriven_geom_train.py)
against the h5 it was fitted to (lbpm_bubble_dataset_newrun_nowrap.h5).

Training only sees ~6300 points per snapshot; this evaluates the network on the FULL grid
of every snapshot (training and held-out alike) and compares every output with the CFD:

    rho   density, lattice units          (network: rho2 + (rho1 - rho2) rho_hat)
    phi   levelset, cells, negative in the drop (network: 8 phi_hat). Outside the band the
          CFD is the clip value +-8, so the "all"/bulk rows compare clip(phi_net, -8, 8)
    p     pressure, lattice units         (network: p_min + (p_max - p_min) p_hat)
    a     phase fraction clip(rho_hat, 0, 1), the field of the geometric loss

plus three checks the network was never told to satisfy:

    p_vs_PR(rho_net)   network p against P_PR of the network rho (the CFD obeys it to 2e-8)
    sign agreement     phi_net < 0 vs a_net > 0.5: the levelset zero is the a = 0.5 surface
    drop integrals     volume (a > 0.5), centroid z, total mass

and the geometric targets themselves: the network's face-mean differences on the
computation_regions cubes against grad_vec, exactly as compute_geom_loss forms them.

    python validate_pinn_3d_lbpm_datadriven.py checkpoints/<run>/            # newest .pth there
    python validate_pinn_3d_lbpm_datadriven.py checkpoints/<run>/loss_X.pth --plot 2 13 26

Regions: "band" = |levelset| < 8, "drop" / "carrier" = the clipped bulk. rel_l2 =
rmse / rms(cfd); for phi in the band, where the CFD crosses zero, it is the trainer's
validation metric. "split" is train / val (generate_points_3d_lbpm.VAL_INDICES).
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

import generate_points_3d_pr as pr
import generate_points_3d_lbpm as lb
import generate_points_3d_lbpm_data as gd
import pytorch_3d_lbpm_datadriven_geom_train as T
from pytorch_utilities import NNCreator

FIELDS = ["rho", "phi", "p", "a", "p_vs_PR(rho_net)"]


def resolve_checkpoint(path):
    if os.path.isdir(path):
        files = glob.glob(os.path.join(path, "*.pth"))
        if not files:
            raise FileNotFoundError(f"no .pth in {path}")
        return max(files, key=os.path.getctime)
    return path


def build_net(device, periods=None):
    """Same architecture as pytorch_3d_lbpm_datadriven_geom_train.main()."""
    no_layers = 8
    hidden_layers = [400] * no_layers
    act = T.build_activation_dict(hidden_layers, {"tanh": range(1, no_layers + 1)}, [10] * no_layers)
    output_layer = [("output_rho", None), ("output_phi", None), ("output_p", None)]
    return NNCreator(torch.float32).get_model_dnn(4, hidden_layers, output_layer, act, False,
                                                  periodic_lengths=periods).to(device)


def load_net(ckpt, device):
    """The network of a checkpoint, in eval mode. Runs from before the periodic embedding
    have no "omega" buffer; for the rest the periods are read back from it."""
    state = torch.load(ckpt, map_location=device)
    periods = (2 * np.pi / state["omega"].cpu().numpy()).tolist() if "omega" in state else None
    net = build_net(device, periods)
    net.load_state_dict(state)
    return net.eval()


@torch.no_grad()
def predict(net, pts, device, chunk=131072):
    """rho_hat, phi_hat, p_hat at the (N, 4) nondimensional points."""
    out = [[], [], []]
    for i in range(0, len(pts), chunk):
        res = net(torch.from_numpy(pts[i:i + chunk]).to(device))
        for k in range(3):
            out[k].append(res[k].cpu().numpy().ravel())
    return [np.concatenate(o).astype(float) for o in out]


@torch.no_grad()
def geom_check(net, scales, device):
    """Network face-mean differences on the computation regions vs the grad_vec targets,
    in units of the face area (dimensionless, as the trainer)."""
    G = gd.get_geom_data_3d()
    offsets, weights = (torch.tensor(q, dtype=torch.float32, device=device)
                        for q in T.face_quadrature(gd.geom_half_size(), T.GEOM_QUAD))
    centre = torch.tensor(G[["x_geom", "y_geom", "z_geom"]].to_numpy(), dtype=torch.float32, device=device)
    tg = torch.tensor(G[["t_geom"]].to_numpy(), dtype=torch.float32, device=device)
    pred, a_c = [], []
    for s in range(0, len(centre), 4096):
        c, t = centre[s:s + 4096], tg[s:s + 4096]
        n = len(c)
        pts = torch.cat([c[:, None, None, :] + offsets[None],
                         t[:, None, None, :].expand(n, 6, offsets.shape[1], 1)], dim=-1).reshape(-1, 4)
        means = (T.phase_fraction(net(pts)[0]).reshape(n, 6, -1) * weights).sum(dim=-1)
        pred.append((means[:, 0::2] - means[:, 1::2]).cpu().numpy())
        a_c.append(T.phase_fraction(net(torch.cat([c, t], dim=1))[0]).cpu().numpy().ravel())
    pred, a_c = np.vstack(pred), np.concatenate(a_c)
    gt = G[["grad_x", "grad_y", "grad_z"]].to_numpy()
    with h5py.File(gd.H5_PATH, "r") as f:
        frac = np.array(f["computation_regions"]["face_fraction"])
    cut = np.any((frac > 0) & (frac < 1), axis=1)
    rows = []
    for name, m in (("all", np.ones(len(gt), bool)), ("cut", cut), ("uncut", ~cut)):
        err = np.linalg.norm(pred[m] - gt[m], axis=1)
        rows.append(dict(regions=name, n=int(m.sum()), mean_abs_err=err.mean(),
                         p95_abs_err=np.percentile(err, 95), mean_abs_target=np.linalg.norm(gt[m], axis=1).mean(),
                         rel_l2=np.linalg.norm(pred[m] - gt[m]) / np.linalg.norm(gt[m]),
                         cos_angle=np.mean(np.sum(pred[m] * gt[m], 1) /
                                           np.maximum(np.linalg.norm(pred[m], axis=1) * np.linalg.norm(gt[m], axis=1), 1e-12)),
                         a_centre_mean=a_c[m].mean()))
    return pd.DataFrame(rows)


def errors(p, r):
    rmse = float(np.sqrt(np.mean((p - r) ** 2)))
    ref_rms = float(np.sqrt(np.mean(r ** 2)))
    return rmse, ref_rms, rmse / ref_rms if ref_rms > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", help=".pth file, or a run directory containing one")
    ap.add_argument("--stride", type=int, default=1, help="spatial subsampling of the grid")
    ap.add_argument("--plot", type=int, nargs="*", default=[0, 2, 13, 26], help="snapshots for slice plots")
    ap.add_argument("--out", default=None, help="output directory (default: next to the checkpoint)")
    args = ap.parse_args()

    ckpt = resolve_checkpoint(args.checkpoint)
    out_dir = args.out or os.path.dirname(os.path.abspath(ckpt))
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    scales, data_scales = lb.get_scales(), gd.get_data_scales()
    rho1, rho2, L_ref, T_ref = scales["rho1"], scales["rho2"], scales["L_ref"], scales["T_ref"]
    p_min, dp = data_scales["p_min"], data_scales["p_max"] - data_scales["p_min"]
    phi_clip = gd.PHI_SCALE

    net = load_net(ckpt, device)
    print(f"\nCheckpoint: {ckpt}")
    print(f"rho1 = {rho1:.5f}, rho2 = {rho2:.5f}, p in [{p_min:.6g}, {data_scales['p_max']:.6g}], "
          f"L_ref = {L_ref:.4f} cells, T_ref = {T_ref:.1f} steps")

    f = h5py.File(gd.H5_PATH, "r")
    (X, Y, Z), _ = lb._cell_axes(f)
    times = np.array(f["time"], dtype=float)
    t_nd = (times - scales["t_origin"]) / T_ref
    s = args.stride
    xs, ys, zs = X[::s], Y[::s], Z[::s]
    ZZ, YY, XX = np.meshgrid(zs, ys, xs, indexing="ij")       # h5 layout is (Z, Y, X)
    grid = (np.column_stack([XX.ravel(), YY.ravel(), ZZ.ravel()]) / L_ref).astype(np.float32)
    shape, cell_vol, nz = ZZ.shape, float(s ** 3), len(Z)
    sub = (slice(None, None, s),) * 3

    rows, integrals, slices = [], [], {}
    for k in range(len(times)):
        split = "val" if k in lb.VAL_INDICES else "train"
        pts = np.hstack([grid, np.full((len(grid), 1), t_nd[k], np.float32)])
        rho_hat, phi_hat, p_hat = predict(net, pts, device)
        rho_p, phi_p, p_p = rho2 + (rho1 - rho2) * rho_hat, phi_clip * phi_hat, p_min + dp * p_hat
        a_p = np.clip(rho_hat, 0.0, 1.0)

        rho_c = np.array(f["density"][k], dtype=float)[sub].ravel()
        phi_c = np.array(f["levelset"][k], dtype=float)[sub].ravel()
        p_c = np.array(f["pressure"][k], dtype=float)[sub].ravel()
        a_c = lb.phase_fraction(rho_c, scales)

        regions = {"all": np.ones_like(phi_c, bool), "band": np.abs(phi_c) < phi_clip,
                   "drop": phi_c <= -phi_clip, "carrier": phi_c >= phi_clip}
        phi_cmp = np.clip(phi_p, -phi_clip, phi_clip)
        fields = {"rho": (rho_p, rho_c), "phi": (phi_cmp, phi_c), "p": (p_p, p_c), "a": (a_p, a_c),
                  "p_vs_PR(rho_net)": (p_p, pr.pressure_pr(rho_p))}
        for region, m in regions.items():
            for name, (pv, rv) in fields.items():
                rmse, ref_rms, rel = errors(pv[m], rv[m])
                rows.append(dict(snapshot=k, t_step=int(times[k]), split=split, region=region,
                                 field=name, rmse=rmse, ref_rms=ref_rms, rel_l2=rel,
                                 pred_mean=float(pv[m].mean()), ref_mean=float(rv[m].mean()),
                                 max_abs_err=float(np.abs(pv[m] - rv[m]).max())))

        def drop_stats(a, rho):
            inside = a > 0.5
            ang = 2 * np.pi * ZZ.ravel()[inside] / nz             # z is periodic: circular mean
            zc = np.angle(np.mean(np.exp(1j * ang))) / (2 * np.pi) * nz if inside.any() else np.nan
            return inside.sum() * cell_vol, zc, rho.sum() * cell_vol
        vp, zp, mp = drop_stats(a_p, rho_p)
        vc, zc, mc = drop_stats(a_c, rho_c)
        sign_pred = phi_p < 0
        integrals.append(dict(snapshot=k, t_step=int(times[k]), split=split,
                              vol_pred=vp, vol_cfd=vc, zc_pred=zp, zc_cfd=zc,
                              mass_pred=mp, mass_cfd=mc,
                              rho_min_pred=rho_p.min(), rho_max_pred=rho_p.max(),
                              rho_min_cfd=rho_c.min(), rho_max_cfd=rho_c.max(),
                              phi_vs_cfd_sign_mismatch=int((sign_pred != (phi_c < 0)).sum()),
                              phi_vs_a_net_sign_mismatch=int((sign_pred != (a_p > 0.5)).sum())))

        if k in args.plot:
            slices[k] = {n: (pv.reshape(shape), rv.reshape(shape)) for n, (pv, rv) in fields.items()}
            slices[k]["phi"] = (phi_p.reshape(shape), phi_c.reshape(shape))   # unclipped net in plots
        print(f"  snapshot {k:2d} (t = {int(times[k])}, {split}) done")
    f.close()

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "validation_vs_cfd.csv"), index=False)
    ig = pd.DataFrame(integrals)
    ig.to_csv(os.path.join(out_dir, "validation_integrals.csv"), index=False)
    geom = geom_check(net, scales, device)
    geom.to_csv(os.path.join(out_dir, "validation_geom.csv"), index=False)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.float_format", "{:.3e}".format)
    for split in ("train", "val"):
        n = ig.split.eq(split).sum()
        print(f"\n=== Error vs CFD, mean over the {n} {split} snapshots (rel_l2 = rmse / rms(cfd)) ===")
        summ = df[df.split == split].groupby(["field", "region"])[["rmse", "rel_l2"]].mean()
        print(summ.unstack("region").swaplevel(axis=1).sort_index(axis=1)
              [[(r, c) for r in ("all", "band", "drop", "carrier") for c in ("rmse", "rel_l2")]]
              .reindex(FIELDS).to_string())

    r = df[df.field == "rho"].pivot_table(index="region", values="rmse", aggfunc="mean")
    print("\nrho RMSE in units of the phase jump rho1 - rho2 (the loss units): "
          + ", ".join(f"{reg} {v / (rho1 - rho2):.4f}" for reg, v in r.rmse.items()))

    print("\n=== Per-snapshot rel L2 (rho, p, a over all cells; phi in the band) ===")
    per = df[df.region == "all"].pivot(index="snapshot", columns="field", values="rel_l2")
    per["phi_band"] = df[(df.region == "band") & (df.field == "phi")].set_index("snapshot").rel_l2
    per.insert(0, "split", ig.set_index("snapshot").split)
    print(per[["split", "rho", "phi_band", "p", "a", "p_vs_PR(rho_net)"]].to_string())

    print("\n=== Drop integrals (volume in cells^3, centroid z in cells, sign mismatches in cells) ===")
    print(ig[["snapshot", "split", "vol_pred", "vol_cfd", "zc_pred", "zc_cfd", "mass_pred", "mass_cfd",
              "phi_vs_cfd_sign_mismatch", "phi_vs_a_net_sign_mismatch"]].to_string(index=False))

    print("\n=== Geometric regions: network face-mean differences vs grad_vec (training snapshots) ===")
    print(geom.to_string(index=False))

    if slices:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        names = ["rho", "phi", "p", "a"]
        extent = [xs[0], xs[-1], zs[0], zs[-1]]
        for k, sl in slices.items():
            ls3 = sl["phi"][1]
            drop = np.argwhere(ls3 < 0)
            iy = int(round(drop[:, 1].mean())) if len(drop) else ls3.shape[1] // 2
            ix = int(round(drop[:, 2].mean())) if len(drop) else ls3.shape[2] // 2
            fig, axes = plt.subplots(len(names), 3, figsize=(9, 4.2 * len(names)))
            for row, name in zip(axes, names):
                pv, rv = sl[name][0][:, iy, :], sl[name][1][:, iy, :]
                lo, hi = rv.min(), rv.max()
                for ax, data, title in zip(row, [rv, pv, pv - rv], ["CFD", "network", "network - CFD"]):
                    if title == "network - CFD":
                        e = max(np.abs(data).max(), 1e-12)
                        im = ax.imshow(data, origin="lower", extent=extent, cmap="RdBu_r", vmin=-e, vmax=e)
                    else:
                        im = ax.imshow(data, origin="lower", extent=extent, cmap="viridis", vmin=lo, vmax=hi)
                    ax.contour(xs, zs, ls3[:, iy, :], levels=[0], colors="k", linewidths=0.8)
                    ax.set_title(f"{name}: {title}")
                    fig.colorbar(im, ax=ax, fraction=0.046)
            split = "val" if k in lb.VAL_INDICES else "train"
            fig.suptitle(f"snapshot {k} (t = {int(times[k])}, {split}), x-z plane through the drop, "
                         f"y index {iy}; black = CFD levelset 0")
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, f"validation_slice_{k}.png"), dpi=100)
            plt.close(fig)

            fig, axs = plt.subplots(1, len(names), figsize=(4.6 * len(names), 3.8))
            for ax, name in zip(axs, names):
                ax.plot(zs, sl[name][1][:, iy, ix], "k-", label="CFD")
                ax.plot(zs, sl[name][0][:, iy, ix], "r--", label="network")
                ax.set_xlabel("z (cells)"); ax.set_title(f"{name} on the drop axis"); ax.legend()
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, f"validation_axis_{k}.png"), dpi=100)
            plt.close(fig)
        print(f"\nPlots: validation_slice_*.png, validation_axis_*.png in {out_dir}")
    print(f"Tables: validation_vs_cfd.csv, validation_integrals.csv, validation_geom.csv in {out_dir}")


if __name__ == "__main__":
    main()
