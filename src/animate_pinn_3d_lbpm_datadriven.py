"""Animate a trained data-driven LBPM network (pytorch_3d_lbpm_datadriven_geom_train.py)
against the h5 it was fitted to, one video per field (rho, phi, p, a).

Each video has three panels per frame (CFD | network | network - CFD) on the x-z plane
through the drop, one frame per h5 snapshot. White is the CFD levelset 0 on every panel,
black dashed the network's phi = 0 on its panel.

    python animate_pinn_3d_lbpm_datadriven.py checkpoints/Oct-26-2026_06-37-38/
    python animate_pinn_3d_lbpm_datadriven.py checkpoints/Oct-26-2026_06-37-38/ --fps 4 --y 0

Conventions are those of validate_pinn_3d_lbpm_datadriven.py (whose load_net / predict
this reuses): coordinates in cells centred on the box, divided by L_ref; t = (step -
t_origin) / T_ref; phi = 8 phi_hat; a = clip(rho_hat, 0, 1). The phi difference panel
compares clip(phi_net, -8, 8), since the CFD levelset is clipped at +-8.

The plane is at the y index of the CFD drop centroid, per frame, unless --y fixes it
(cells, centred). Colour limits are global over the run and span BOTH the CFD and the
network, so network excursions outside the CFD range (e.g. rho below rho2) are not hidden
by the colour scale; the difference limit is the 99.5th percentile of |difference|, so
the largest errors saturate. Snapshots in generate_points_3d_lbpm.VAL_INDICES were never trained
on and are marked "held out".
"""
import sys
sys.path.append("../utilities")
import os
import argparse

import numpy as np
import h5py
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

import generate_points_3d_lbpm as lb
import generate_points_3d_lbpm_data as gd
from validate_pinn_3d_lbpm_datadriven import load_net, predict, resolve_checkpoint

FIELDS = ["rho", "phi", "p", "a"]
LABELS = {"rho": "rho (lattice)", "phi": "levelset phi (cells)", "p": "p (lattice)", "a": "phase fraction a"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", help=".pth file, or a run directory containing one")
    ap.add_argument("--fields", nargs="*", default=FIELDS, choices=FIELDS)
    ap.add_argument("--y", type=float, default=None, help="fixed y of the plane, cells (default: drop centroid)")
    ap.add_argument("--fps", type=float, default=3.0)
    ap.add_argument("--dpi", type=int, default=120)
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
    print(f"checkpoint: {ckpt}")

    data = {fld: {"cfd": [], "net": []} for fld in FIELDS}
    ls_cfd, ys = [], []
    with h5py.File(gd.H5_PATH, "r") as f:
        (X, Y, Z), _ = lb._cell_axes(f)
        times = np.array(f["time"], dtype=float)
        t_nd = (times - scales["t_origin"]) / T_ref
        XX, ZZ = np.meshgrid(X, Z)   # (Z, X)
        for k in range(len(times)):
            if args.y is None:
                drop = np.argwhere(np.array(f["levelset"][k]) < 0)   # (Z, Y, X) indices
                iy = int(round(drop[:, 1].mean())) if len(drop) else len(Y) // 2
            else:
                iy = int(np.argmin(np.abs(Y - args.y)))
            ys.append(Y[iy])
            sl = (k, slice(None), iy, slice(None))
            rho_c = np.array(f["density"][sl], dtype=float)
            phi_c = np.array(f["levelset"][sl], dtype=float)
            p_c = np.array(f["pressure"][sl], dtype=float)

            pts = np.column_stack([XX.ravel() / L_ref, np.full(XX.size, Y[iy] / L_ref), ZZ.ravel() / L_ref,
                                   np.full(XX.size, t_nd[k])]).astype(np.float32)
            rho_hat, phi_hat, p_hat = (q.reshape(XX.shape) for q in predict(net, pts, device))
            cfd = {"rho": rho_c, "phi": phi_c, "p": p_c, "a": lb.phase_fraction(rho_c, scales)}
            pred = {"rho": rho2 + (rho1 - rho2) * rho_hat, "phi": phi_clip * phi_hat,
                    "p": p_min + dp * p_hat, "a": np.clip(rho_hat, 0.0, 1.0)}
            for fld in FIELDS:
                data[fld]["cfd"].append(cfd[fld])
                data[fld]["net"].append(pred[fld])
            ls_cfd.append(phi_c)
            tag = "held out" if k in lb.VAL_INDICES else "train"
            print(f"  snapshot {k:2d} (t = {int(times[k])}, {tag}), plane y = {Y[iy]:+.1f}", flush=True)
    phi_net = np.array(data["phi"]["net"])
    ext = [X[0], X[-1], Z[0], Z[-1]]

    for fld in args.fields:
        cfd, pred = np.array(data[fld]["cfd"]), np.array(data[fld]["net"])
        diff = (np.clip(pred, -phi_clip, phi_clip) if fld == "phi" else pred) - cfd
        if fld == "phi":
            lo, hi = -phi_clip, phi_clip
        else:
            both = np.concatenate([cfd.ravel(), pred.ravel()])
            lo, hi = both.min(), both.max()
        # 99.5th percentile, not the max: the worst cells (e.g. the early-snapshot rho < rho2
        # sheet) saturate instead of washing out every other frame
        dlim = np.percentile(np.abs(diff), 99.5) or 1.0
        cmap = "RdBu_r" if fld == "phi" else "viridis"

        fig, axes = plt.subplots(1, 3, figsize=(10, 6.2), constrained_layout=True)
        ims = [axes[0].imshow(cfd[0], origin="lower", extent=ext, cmap=cmap, vmin=lo, vmax=hi),
               axes[1].imshow(pred[0], origin="lower", extent=ext, cmap=cmap, vmin=lo, vmax=hi),
               axes[2].imshow(diff[0], origin="lower", extent=ext, cmap="PuOr_r", vmin=-dlim, vmax=dlim)]
        for ax, name in zip(axes, ["CFD (LBPM)", "network", "network - CFD"]):
            ax.set_title(name)
            ax.set_xlabel("x (cells)")
        axes[0].set_ylabel("z (cells)")
        fig.colorbar(ims[1], ax=axes[:2], shrink=0.6, label=LABELS[fld])
        fig.colorbar(ims[2], ax=axes[2], shrink=0.6, label="difference")
        contours = []

        def draw(k):
            for c in contours:
                c.remove()
            contours.clear()
            for im, arr in zip(ims, (cfd, pred, diff)):
                im.set_data(arr[k])
            for ax in axes:
                contours.append(ax.contour(X, Z, ls_cfd[k], [0], colors="w", linewidths=1))
            if phi_net[k].min() < 0 < phi_net[k].max():
                contours.append(axes[1].contour(X, Z, phi_net[k], [0], colors="k",
                                                linewidths=1, linestyles="--"))
            tag = "held out" if k in lb.VAL_INDICES else "train"
            fig.suptitle(f"{LABELS[fld]}   x-z plane, y = {ys[k]:+.1f}   t = {int(times[k])}  [{tag}]\n"
                         f"white: CFD levelset 0, black dashed: network phi = 0")
            return ims

        anim = FuncAnimation(fig, draw, frames=len(times), blit=False)
        path = os.path.join(out_dir, f"anim_{fld}.mp4")
        anim.save(path, writer=FFMpegWriter(fps=args.fps, bitrate=3000), dpi=args.dpi)
        plt.close(fig)
        print(f"wrote {path}   (colour range [{lo:.4g}, {hi:.4g}], difference +-{dlim:.3g})")


if __name__ == "__main__":
    main()
