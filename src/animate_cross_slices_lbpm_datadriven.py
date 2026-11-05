"""Cross-slice animation of a trained data-driven LBPM network
(pytorch_3d_lbpm_datadriven_geom_train.py) against the h5 it was fitted to, over every
snapshot, training and held-out alike. One video per field (rho, phi, p, a).

Each frame is a 3 x 3 grid: rows are the three orthogonal planes through the CFD drop
centroid of that snapshot (x-z, y-z, x-y), columns are CFD | network | network - CFD.
White is the CFD levelset 0 on every panel, black dashed the network's phi = 0 on the
network panel, and the title says whether the snapshot was trained on or held out
(generate_points_3d_lbpm.VAL_INDICES).

    python animate_cross_slices_lbpm_datadriven.py checkpoints/Oct-26-2026_18-51-26/
    python animate_cross_slices_lbpm_datadriven.py checkpoints/Oct-26-2026_18-51-26/ --split val --fps 2

Conventions are those of validate_pinn_3d_lbpm_datadriven.py (load_net / predict are
reused): coordinates in cells centred on the box, divided by L_ref; t = (step - t_origin)
/ T_ref; phi = 8 phi_hat; a = clip(rho_hat, 0, 1). The phi difference compares
clip(phi_net, -8, 8), since the CFD levelset is clipped at +-8.

Colour limits are global over the selected frames and all three planes and span BOTH the
CFD and the network, so network excursions outside the CFD range are not hidden; the
difference limit is the 99.5th percentile of |difference|, so the worst cells saturate.
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
# plane name, fixed axis (h5 layout is (Z, Y, X)), horizontal / vertical axis names
PLANES = [("x-z", 1, "x", "z"), ("y-z", 2, "y", "z"), ("x-y", 0, "x", "y")]


def plane_points(X, Y, Z, plane, c, L_ref, t):
    """(N, 4) network inputs on one plane through the cell-centred point c = (x, y, z),
    in the (vertical, horizontal) layout the h5 slice has."""
    if plane == "x-z":
        V, H = np.meshgrid(Z, X, indexing="ij"); xyz = (H, np.full_like(H, c[1]), V)
    elif plane == "y-z":
        V, H = np.meshgrid(Z, Y, indexing="ij"); xyz = (np.full_like(H, c[0]), H, V)
    else:
        V, H = np.meshgrid(Y, X, indexing="ij"); xyz = (H, V, np.full_like(H, c[2]))
    pts = np.column_stack([q.ravel() / L_ref for q in xyz] + [np.full(H.size, t)])
    return pts.astype(np.float32), H.shape


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", help=".pth file, or a run directory containing one")
    ap.add_argument("--fields", nargs="*", default=FIELDS, choices=FIELDS)
    ap.add_argument("--split", default="all", choices=["all", "train", "val"],
                    help="which snapshots to animate (default: all)")
    ap.add_argument("--fps", type=float, default=3.0)
    ap.add_argument("--dpi", type=int, default=110)
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

    # data[field][plane]["cfd" / "net"]: list over frames of 2D arrays
    data = {fld: {pl[0]: {"cfd": [], "net": []} for pl in PLANES} for fld in FIELDS}
    ls_cfd = {pl[0]: [] for pl in PLANES}
    phi_net = {pl[0]: [] for pl in PLANES}
    centres, frames = [], []
    with h5py.File(gd.H5_PATH, "r") as f:
        (X, Y, Z), _ = lb._cell_axes(f)
        axes_of = {"x": X, "y": Y, "z": Z}
        times = np.array(f["time"], dtype=float)
        t_nd = (times - scales["t_origin"]) / T_ref
        for k in range(len(times)):
            held_out = k in lb.VAL_INDICES
            if (args.split == "train" and held_out) or (args.split == "val" and not held_out):
                continue
            frames.append(k)
            rho3 = np.array(f["density"][k], dtype=float)
            phi3 = np.array(f["levelset"][k], dtype=float)
            p3 = np.array(f["pressure"][k], dtype=float)
            drop = np.argwhere(phi3 < 0)                         # (Z, Y, X) indices
            iz, iy, ix = (np.round(drop.mean(axis=0)).astype(int) if len(drop)
                          else np.array(phi3.shape) // 2)
            c = (X[ix], Y[iy], Z[iz])
            centres.append(c)
            for name, axis, _, _ in PLANES:
                sl = [slice(None)] * 3
                sl[axis] = (iz, iy, ix)[axis]
                sl = tuple(sl)
                rho_c, phi_c, p_c = rho3[sl], phi3[sl], p3[sl]
                pts, shape = plane_points(X, Y, Z, name, c, L_ref, t_nd[k])
                rho_hat, phi_hat, p_hat = (q.reshape(shape) for q in predict(net, pts, device))
                cfd = {"rho": rho_c, "phi": phi_c, "p": p_c, "a": lb.phase_fraction(rho_c, scales)}
                pred = {"rho": rho2 + (rho1 - rho2) * rho_hat, "phi": phi_clip * phi_hat,
                        "p": p_min + dp * p_hat, "a": np.clip(rho_hat, 0.0, 1.0)}
                for fld in FIELDS:
                    data[fld][name]["cfd"].append(cfd[fld])
                    data[fld][name]["net"].append(pred[fld])
                ls_cfd[name].append(phi_c)
                phi_net[name].append(pred["phi"])
            tag = "held out" if held_out else "train"
            print(f"  snapshot {k:2d} (t = {int(times[k])}, {tag}), centre "
                  f"({c[0]:+.1f}, {c[1]:+.1f}, {c[2]:+.1f})", flush=True)
    if not frames:
        raise SystemExit(f"no snapshots in split '{args.split}'")

    suffix = "" if args.split == "all" else f"_{args.split}"
    for fld in args.fields:
        cfd = {pl: np.array(data[fld][pl]["cfd"]) for pl, *_ in PLANES}
        pred = {pl: np.array(data[fld][pl]["net"]) for pl, *_ in PLANES}
        diff = {pl: (np.clip(pred[pl], -phi_clip, phi_clip) if fld == "phi" else pred[pl]) - cfd[pl]
                for pl, *_ in PLANES}
        if fld == "phi":
            lo, hi = -phi_clip, phi_clip
        else:
            both = np.concatenate([a.ravel() for pl, *_ in PLANES for a in (cfd[pl], pred[pl])])
            lo, hi = both.min(), both.max()
        dlim = np.percentile(np.abs(np.concatenate([d.ravel() for d in diff.values()])), 99.5) or 1.0
        cmap = "RdBu_r" if fld == "phi" else "viridis"

        fig, axes = plt.subplots(3, 3, figsize=(10.5, 12), constrained_layout=True,
                                 gridspec_kw=dict(height_ratios=[2, 2, 1.1]))
        ims = {}
        for row, (name, _, h, v) in zip(axes, PLANES):
            H, V = axes_of[h], axes_of[v]
            ext = [H[0], H[-1], V[0], V[-1]]
            ims[name] = [row[0].imshow(cfd[name][0], origin="lower", extent=ext, cmap=cmap, vmin=lo, vmax=hi),
                         row[1].imshow(pred[name][0], origin="lower", extent=ext, cmap=cmap, vmin=lo, vmax=hi),
                         row[2].imshow(diff[name][0], origin="lower", extent=ext, cmap="PuOr_r",
                                       vmin=-dlim, vmax=dlim)]
            for ax in row:
                ax.set_xlabel(f"{h} (cells)")
            row[0].set_ylabel(f"{name}\n{v} (cells)")
        for ax, title in zip(axes[0], ["CFD (LBPM)", "network", "network - CFD"]):
            ax.set_title(title)
        fig.colorbar(ims["x-z"][1], ax=axes[:, :2], shrink=0.5, label=LABELS[fld])
        fig.colorbar(ims["x-z"][2], ax=axes[:, 2], shrink=0.5, label="difference")
        contours = []

        def draw(n):
            for c in contours:
                c.remove()
            contours.clear()
            k = frames[n]
            for row, (name, _, h, v) in zip(axes, PLANES):
                H, V = axes_of[h], axes_of[v]
                for im, arr in zip(ims[name], (cfd[name], pred[name], diff[name])):
                    im.set_data(arr[n])
                for ax in row:
                    contours.append(ax.contour(H, V, ls_cfd[name][n], [0], colors="w", linewidths=1))
                pn = phi_net[name][n]
                if pn.min() < 0 < pn.max():
                    contours.append(row[1].contour(H, V, pn, [0], colors="k", linewidths=1, linestyles="--"))
            cx, cy, cz = centres[n]
            tag = "HELD OUT" if k in lb.VAL_INDICES else "train"
            fig.suptitle(f"{LABELS[fld]}   snapshot {k}, t = {int(times[k])}  [{tag}]\n"
                         f"planes through the CFD drop centroid ({cx:+.1f}, {cy:+.1f}, {cz:+.1f}) cells; "
                         f"white: CFD levelset 0, black dashed: network phi = 0")
            return []

        anim = FuncAnimation(fig, draw, frames=len(frames), blit=False)
        path = os.path.join(out_dir, f"anim_cross_{fld}{suffix}.mp4")
        anim.save(path, writer=FFMpegWriter(fps=args.fps, bitrate=4000), dpi=args.dpi)
        plt.close(fig)
        print(f"wrote {path}   (colour range [{lo:.4g}, {hi:.4g}], difference +-{dlim:.3g})")


if __name__ == "__main__":
    main()
