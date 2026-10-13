"""Render a video of the trained 3D PINN against the LBPM h5, snapshot by snapshot.

This is the animated form of validate_pinn_3d.py's --plot figure: an x-z plane through
the drop, with CFD on the top row, the PINN on the middle row and the difference on the
bottom, one column per output (u, v, w_rel, p, a) and one frame per h5 snapshot. A fourth
row repeats the PINN on its own colour range, since on the shared range a field the network
got badly wrong is indistinguishable from one it left at zero.

    python animate_pinn_3d.py ../trained_model_3d/
    python animate_pinn_3d.py ../trained_model_3d/loss_9.2355em04.pth --stride 1 --fps 3

Every convention is taken from validate_pinn_3d so the frames mean the same thing as the
numbers in validation_vs_cfd.csv:
  - coordinates in lattice cells centred on the domain, divided by L_ref; time relative
    to the first snapshot, divided by T_ref; velocities divided by U_ref.
  - w carries the bulk drift of the whole domain, so it is shown with each snapshot's
    domain-mean w removed ("w_rel"), which is what makes the flow round the drop visible.
  - p is only defined up to a constant on both sides, so both fields are gauged on the
    undisturbed far carrier (see FAR_RADIUS in validate_pinn_3d).

Colour limits are global over the whole run, computed in a first pass, so brightness in
one frame means the same as in the next. Snapshots the training run never saw (odd
indices, since get_training_data_3d strides by 2) are labelled "held out" in the title.
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
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter

from generate_points_3d import _measure_scales, RHO_GAS, T_MAX
from validate_pinn_3d import build_net, resolve_checkpoint, predict, DEEP_PHASE, FAR_RADIUS

FIELDS = ["u", "v", "w_rel", "p", "a"]
ROWS = ["CFD", "PINN", "PINN - CFD", "PINN (self-scaled)"]
# Row 3 repeats the PINN row on its own colour range instead of the shared one. On the
# shared range the network's u, v, w and p are flat to the eye, which is the honest
# comparison but says nothing about whether anything is under there; blown up to its own
# amplitude the row shows whether the field has structure in the right places or is just
# noise. The gain printed in each title is how far it had to be stretched -- a gain of
# 500x means that panel is showing something 500 times too weak to matter.


def gather_frames(net, f, times, levelset, pressure, L_ref, U_ref, P_ref, T_ref, stride, device):
    """One pass over the snapshots, collecting the x-z slice of every field."""
    X, Y, Z = (np.array(f[c], dtype=float) for c in "XYZ")
    X, Y, Z = X / np.diff(X)[0], Y / np.diff(Y)[0], Z / np.diff(Z)[0]
    X, Y, Z = X - X.mean(), Y - Y.mean(), Z - Z.mean()
    t_nd = (times - times[0]) / T_ref

    s = stride
    xs, ys, zs = X[::s] / L_ref, Y[::s] / L_ref, Z[::s] / L_ref
    ZZ, YY, XX = np.meshgrid(zs, ys, xs, indexing="ij")   # h5 layout is (Z, Y, X)
    shape = ZZ.shape
    grid = np.column_stack([XX.ravel(), YY.ravel(), ZZ.ravel()]).astype(np.float32)

    # Fix the slice plane for the whole video: the drop moves along z, not y, so letting
    # the plane follow the per-snapshot centroid would only add jitter.
    sub_all = (slice(None), slice(None, None, s), slice(None, None, s), slice(None, None, s))
    iy_per_frame = [np.argwhere(levelset[sub_all][k] > 0)[:, 1].mean() for k in range(len(times))]
    iy = int(round(float(np.median(iy_per_frame))))

    frames = []
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

        centroid = np.argwhere(ls3 > 0).mean(axis=0)
        gz, gy, gx = np.ogrid[:ls3.shape[0], :ls3.shape[1], :ls3.shape[2]]
        radius = (np.sqrt((gz - centroid[0]) ** 2 + (gy - centroid[1]) ** 2
                          + (gx - centroid[2]) ** 2) * s).ravel()
        far = (ls <= -DEEP_PHASE) & (radius > FAR_RADIUS * L_ref)

        p_p, p_c = p_p - p_p[far].mean(), p_c - p_c[far].mean()
        w_p, w_c = w_p - w_p.mean(), w_c - w_c.mean()

        pairs = {"u": (u_p, u_c), "v": (v_p, v_c), "w_rel": (w_p, w_c),
                 "p": (p_p, p_c), "a": (a_p, a_c)}
        frame = {name: (pr.reshape(shape)[:, iy, :], rf.reshape(shape)[:, iy, :])
                 for name, (pr, rf) in pairs.items()}
        frame["_ls"] = ls3[:, iy, :]
        frames.append(frame)
        print(f"  snapshot {k + 1}/{len(times)}  t_nd = {t_nd[k]:.3f}", flush=True)

    return frames, t_nd, xs, zs, iy


def colour_limits(frames):
    """Global, symmetric limits per field, so the scale never moves between frames.

    Returns (shared, diff, pinn_only) per field: the shared CFD/PINN range, the range of
    the difference, and the PINN's own range for the self-scaled row."""
    lims = {}
    for name in FIELDS:
        both = max(max(np.abs(fr[name][0]).max(), np.abs(fr[name][1]).max()) for fr in frames)
        diff = max(np.abs(fr[name][0] - fr[name][1]).max() for fr in frames)
        pinn = max(np.abs(fr[name][0]).max() for fr in frames)
        lims[name] = (float(both), float(max(diff, 1e-12)), float(max(pinn, 1e-12)))
    return lims


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", help=".pth file, or a directory containing one")
    ap.add_argument("--h5", default="../cfd_data/lbpm_bubble_dataset.h5")
    ap.add_argument("--stride", type=int, default=2, help="spatial subsampling of the grid")
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--dpi", type=int, default=110)
    ap.add_argument("--gif", action="store_true", help="also write a .gif alongside the .mp4")
    ap.add_argument("--out", default=None, help="output directory (default: next to the checkpoint)")
    args = ap.parse_args()

    ckpt = resolve_checkpoint(args.checkpoint)
    out_dir = args.out or os.path.dirname(os.path.abspath(ckpt))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    f = h5py.File(args.h5, "r")
    times = np.array(f["time"], dtype=float)
    n_keep = len(times) if T_MAX is None else int(np.searchsorted(times, T_MAX, side="right"))
    times = times[:n_keep]
    levelset = -np.array(f["levelset"][:n_keep])          # > 0 => drop, as in the generator
    pressure = np.array(f["pressure"][:n_keep])

    L_ref, U_ref, _, _, _ = _measure_scales(levelset, pressure, times)
    T_ref = L_ref / U_ref
    P_ref = RHO_GAS * U_ref ** 2

    X = np.array(f["X"], dtype=float)
    period = [len(f[c]) / L_ref for c in "XYZ"]
    net = build_net(period, device)
    net.load_state_dict(torch.load(ckpt, map_location=device))
    net.eval()
    trained = np.zeros(len(times), dtype=bool)
    trained[::2] = True                                   # get_training_data_3d default stride

    print(f"\nCheckpoint: {ckpt}")
    print(f"L_ref = {L_ref:.4f} cells, U_ref = {U_ref:.6g} cells/step, "
          f"T_ref = {T_ref:.1f} steps, P_ref = {P_ref:.4g}")
    print(f"Evaluating {len(times)} snapshots on a stride-{args.stride} grid:")

    frames, t_nd, xs, zs, iy = gather_frames(net, f, times, levelset, pressure,
                                             L_ref, U_ref, P_ref, T_ref, args.stride, device)
    lims = colour_limits(frames)

    fig, axes = plt.subplots(len(ROWS), len(FIELDS), figsize=(3.1 * len(FIELDS), 4.6 * len(ROWS)))
    extent = [xs[0], xs[-1], zs[0], zs[-1]]
    xg = np.linspace(xs[0], xs[-1], len(xs))
    zg = np.linspace(zs[0], zs[-1], len(zs))

    images, contours = {}, {}
    fr0 = frames[0]
    for col, name in enumerate(FIELDS):
        both, diff, pinn = lims[name]
        for row, label in enumerate(ROWS):
            ax = axes[row, col]
            data = {0: fr0[name][1], 1: fr0[name][0],
                    2: fr0[name][0] - fr0[name][1], 3: fr0[name][0]}[row]
            # alpha is a 0..1 indicator, so it reads better on a sequential map -- except
            # on the difference row, where the sign is the whole point.
            if name == "a" and row != 2:
                im = ax.imshow(data, origin="lower", extent=extent, cmap="viridis", vmin=0, vmax=1)
            else:
                vmax = {0: both, 1: both, 2: diff, 3: pinn}[row]
                im = ax.imshow(data, origin="lower", extent=extent, cmap="RdBu_r",
                               vmin=-vmax, vmax=vmax)
            images[(row, col)] = im
            contours[(row, col)] = ax.contour(xg, zg, fr0["_ls"], levels=[0],
                                              colors="k", linewidths=0.8)
            if row == 3:
                label = f"{label}, x{both / pinn:.4g}"
            ax.set_title(f"{name}: {label}", fontsize=10)
            ax.set_xlabel("x / L_ref", fontsize=8)
            ax.set_ylabel("z / L_ref", fontsize=8)
            ax.tick_params(labelsize=7)
            fig.colorbar(im, ax=ax, fraction=0.046)
    suptitle = fig.suptitle("")
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    def draw(k):
        fr = frames[k]
        for col, name in enumerate(FIELDS):
            pr, rf = fr[name]
            for row, data in enumerate([rf, pr, pr - rf, pr]):
                images[(row, col)].set_data(data)
                contours[(row, col)].remove()
                contours[(row, col)] = axes[row, col].contour(xg, zg, fr["_ls"], levels=[0],
                                                              colors="k", linewidths=0.8)
        suptitle.set_text(f"snapshot {k}/{len(frames) - 1}   t_nd = {t_nd[k]:.3f}   "
                          f"step {times[k]:.0f}   "
                          f"{'trained' if trained[k] else 'HELD OUT'}   (y index {iy})")
        return list(images.values())

    anim = FuncAnimation(fig, draw, frames=len(frames), blit=False)
    mp4 = os.path.join(out_dir, "pinn_vs_cfd.mp4")
    print(f"\nWriting {mp4} ...", flush=True)
    anim.save(mp4, writer=FFMpegWriter(fps=args.fps, bitrate=4000), dpi=args.dpi)
    print(f"Wrote {mp4}")
    if args.gif:
        gif = os.path.join(out_dir, "pinn_vs_cfd.gif")
        anim.save(gif, writer=PillowWriter(fps=args.fps), dpi=max(args.dpi // 2, 60))
        print(f"Wrote {gif}")


if __name__ == "__main__":
    main()
