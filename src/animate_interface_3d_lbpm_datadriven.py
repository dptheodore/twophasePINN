"""3D interface video of a trained data-driven LBPM network
(pytorch_3d_lbpm_datadriven_geom_train.py) against the h5 it was fitted to, over every
snapshot of the data, training and held-out alike.

Each frame has three 3D panels, the CFD interface (levelset = 0), the network interface
(phi_net = 0, or a_net = 0.5 with --iso a) and the two overlaid, and below them the drop
volume and centroid z of both against time with the current snapshot marked. Held-out
snapshots (generate_points_3d_lbpm.VAL_INDICES) are flagged in the title and shaded on
the time plots.

    python animate_interface_3d_lbpm_datadriven.py checkpoints/Oct-26-2026_18-51-26/
    python animate_interface_3d_lbpm_datadriven.py checkpoints/Oct-26-2026_18-51-26/ --iso a --margin 0

The network is evaluated on the full h5 grid of each snapshot and meshed with marching
cubes, so any spurious zero surface away from the drop shows up as well. --margin (cells,
default 12) crops both meshes to the box enclosing the CFD drop over the whole run plus
that margin, which keeps the drop readable; --margin -1 meshes the full box. Volumes and
centroids are over the full grid either way: cells with phi < 0 (CFD), phi_net < 0 or
a_net > 0.5 (network, as --iso), the same cell counts validate_pinn_3d_lbpm_datadriven.py
reports.
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
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage.measure import marching_cubes

import generate_points_3d_lbpm as lb
import generate_points_3d_lbpm_data as gd
from validate_pinn_3d_lbpm_datadriven import load_net, predict, resolve_checkpoint

LIGHT = np.array([-0.4, -0.7, 0.6]) / np.linalg.norm([-0.4, -0.7, 0.6])
CFD_COLOUR, NET_COLOUR = np.array([0.55, 0.55, 0.6]), np.array([0.85, 0.3, 0.15])


def mesh(field, level, origin, step):
    """Triangles (n, 3, 3) in (x, y, z) cells of the level surface of a (Z, Y, X) block
    whose first cell sits at origin = (z0, y0, x0); empty if the block never crosses it."""
    if not field.min() < level < field.max():
        return np.zeros((0, 3, 3))
    verts, faces, _, _ = marching_cubes(field, level=level, step_size=step)
    return (verts + np.asarray(origin))[faces][:, :, ::-1]


def shaded(tris, colour, alpha):
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    shade = (0.3 + 0.7 * np.abs(n @ LIGHT))[:, None] * colour
    return Poly3DCollection(tris, facecolors=np.column_stack([shade, np.full(len(tris), alpha)]),
                            edgecolors="none", linewidth=0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", help=".pth file, or a run directory containing one")
    ap.add_argument("--iso", default="phi", choices=["phi", "a"],
                    help="network interface: phi_net = 0 (default) or a_net = 0.5")
    ap.add_argument("--margin", type=int, default=12,
                    help="crop to the CFD drop's bounding box over the run plus this many cells; -1 = full box")
    ap.add_argument("--step", type=int, default=1, help="marching-cubes step size (cells)")
    ap.add_argument("--elev", type=float, default=15.0)
    ap.add_argument("--azim", type=float, default=-60.0)
    ap.add_argument("--fps", type=float, default=4.0)
    ap.add_argument("--dpi", type=int, default=110)
    ap.add_argument("--out", default=None, help="output directory (default: next to the checkpoint)")
    args = ap.parse_args()

    ckpt = resolve_checkpoint(args.checkpoint)
    out_dir = args.out or os.path.dirname(os.path.abspath(ckpt))
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    scales = lb.get_scales()
    L_ref, T_ref = scales["L_ref"], scales["T_ref"]
    phi_clip = gd.PHI_SCALE

    net = load_net(ckpt, device)
    print(f"checkpoint: {ckpt}")

    with h5py.File(gd.H5_PATH, "r") as f:
        (X, Y, Z), _ = lb._cell_axes(f)
        times = np.array(f["time"], dtype=float)
        t_nd = (times - scales["t_origin"]) / T_ref
        ZZ, YY, XX = np.meshgrid(Z, Y, X, indexing="ij")          # h5 layout is (Z, Y, X)
        grid = (np.column_stack([XX.ravel(), YY.ravel(), ZZ.ravel()]) / L_ref).astype(np.float32)
        shape = ZZ.shape

        # crop box: the CFD drop over the whole run, plus the margin
        if args.margin < 0:
            box = tuple(slice(None) for _ in range(3))
        else:
            any_drop = np.zeros(shape, bool)
            for k in range(len(times)):
                any_drop |= np.array(f["levelset"][k]) < 0
            idx = np.argwhere(any_drop)
            lo = np.maximum(idx.min(axis=0) - args.margin, 0)
            hi = np.minimum(idx.max(axis=0) + args.margin + 1, shape)
            box = tuple(slice(a, b) for a, b in zip(lo, hi))
        origin = (Z[box[0]][0], Y[box[1]][0], X[box[2]][0])
        lims = [(A[s][0], A[s][-1]) for A, s in zip((X, Y, Z), box[::-1])]
        in_box = np.zeros(shape, bool)
        in_box[box] = True

        tris_cfd, tris_net, stats = [], [], []
        for k in range(len(times)):
            phi_c = np.array(f["levelset"][k], dtype=float)
            pts = np.hstack([grid, np.full((len(grid), 1), t_nd[k], np.float32)])
            rho_hat, phi_hat, _ = predict(net, pts, device)
            # meshed fields are positive in the drop, so both surfaces sit at level 0 with
            # the same orientation: -levelset for the CFD, -phi_net or a_net - 0.5 for the net
            drop_n = (-phi_clip * phi_hat if args.iso == "phi" else np.clip(rho_hat, 0.0, 1.0) - 0.5).reshape(shape)
            tris_cfd.append(mesh(-phi_c[box], 0.0, origin, args.step))
            tris_net.append(mesh(drop_n[box], 0.0, origin, args.step))
            inside_c, inside_n = phi_c < 0, drop_n > 0
            st = dict(vol_cfd=inside_c.sum(), vol_net=inside_n.sum(),
                      zc_cfd=ZZ[inside_c].mean() if inside_c.any() else np.nan,
                      zc_net=ZZ[inside_n].mean() if inside_n.any() else np.nan,
                      outside=int((inside_n & ~in_box).sum()))
            stats.append(st)
            tag = "held out" if k in lb.VAL_INDICES else "train"
            print(f"  snapshot {k:2d} (t = {int(times[k])}, {tag}): {len(tris_cfd[-1])} CFD / "
                  f"{len(tris_net[-1])} network triangles, volume {st['vol_cfd']} / {st['vol_net']} cells"
                  + (f", {st['outside']} network drop cells outside the crop" if st["outside"] else ""),
                  flush=True)

    vol_c, vol_n = np.array([s["vol_cfd"] for s in stats]), np.array([s["vol_net"] for s in stats])
    zc_c, zc_n = np.array([s["zc_cfd"] for s in stats]), np.array([s["zc_net"] for s in stats])
    net_label = "network phi = 0" if args.iso == "phi" else "network a = 0.5"

    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 3, height_ratios=[3, 1], hspace=0.25, wspace=0.05,
                          left=0.07, right=0.98, top=0.9, bottom=0.07)
    ax3 = [fig.add_subplot(gs[0, i], projection="3d") for i in range(3)]
    ax_v = fig.add_subplot(gs[1, :2])
    ax_z = fig.add_subplot(gs[1, 2])

    for ax, y_c, y_n, lab in ((ax_v, vol_c, vol_n, "drop volume (cells)"),
                              (ax_z, zc_c, zc_n, "drop centroid z (cells)")):
        for k in lb.VAL_INDICES:
            ax.axvspan(times[k] - 200, times[k] + 200, color="0.9", lw=0)
        ax.plot(times, y_c, "k-o", ms=3, label="CFD levelset < 0")
        ax.plot(times, y_n, "-o", color=NET_COLOUR, ms=3,
                label="network phi < 0" if args.iso == "phi" else "network a > 0.5")
        ax.set_xlabel("time step"); ax.set_ylabel(lab)
    ax_v.legend(loc="best", fontsize=9, title="grey bands: held out", title_fontsize=8)
    cursors = [ax.axvline(times[0], color="C0", lw=1.5) for ax in (ax_v, ax_z)]

    def draw(k):
        for ax, title in zip(ax3, ["CFD (LBPM) levelset = 0", net_label, "overlay (grey CFD, red network)"]):
            ax.cla()
            ax.set_xlim(*lims[0]); ax.set_ylim(*lims[1]); ax.set_zlim(*lims[2])
            ax.set_box_aspect([b - a for a, b in lims], zoom=1.0)
            ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
            ax.view_init(elev=args.elev, azim=args.azim)
            ax.set_title(title, fontsize=11)
        if len(tris_cfd[k]):
            ax3[0].add_collection3d(shaded(tris_cfd[k], CFD_COLOUR, 1.0))
            ax3[2].add_collection3d(shaded(tris_cfd[k], CFD_COLOUR, 0.35))
        if len(tris_net[k]):
            ax3[1].add_collection3d(shaded(tris_net[k], NET_COLOUR, 1.0))
            ax3[2].add_collection3d(shaded(tris_net[k], NET_COLOUR, 0.6))
        for c in cursors:
            c.set_xdata([times[k], times[k]])
        tag = "HELD OUT" if k in lb.VAL_INDICES else "train"
        fig.suptitle(f"interface, snapshot {k}, t = {int(times[k])}  [{tag}]   volume CFD {vol_c[k]} / "
                     f"network {vol_n[k]} cells, centroid z CFD {zc_c[k]:+.1f} / network {zc_n[k]:+.1f}",
                     fontsize=12)

    anim = FuncAnimation(fig, draw, frames=len(times), blit=False)
    path = os.path.join(out_dir, f"anim_interface_3d_{args.iso}.mp4")
    anim.save(path, writer=FFMpegWriter(fps=args.fps, bitrate=5000), dpi=args.dpi)
    plt.close(fig)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
