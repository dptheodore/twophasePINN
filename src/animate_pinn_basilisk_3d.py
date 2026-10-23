"""Animate the trained 3D PINN against the Basilisk h5, one video per field (u, v, w, p).

Each video has three panels per frame (CFD | PINN | PINN - CFD) on a centre plane of the
box, one frame per h5 snapshot, with the CFD's alpha = 0.5 contour drawn on every panel
(white) and the PINN's in black on its panel.

    python animate_pinn_basilisk_3d.py checkpoints/Oct-16-2026_00-25-22/
    python animate_pinn_basilisk_3d.py checkpoints/Oct-16-2026_00-25-22/loss_1.1474em02.pth --fps 4

Conventions match generate_points_3d / pytorch_3d_rising_bubble_train:
  - inputs are physical x, y, z, t divided by L_REF = 0.25 (U_ref = 1, so velocities are
    the same in both).
  - p is divided by rho_ref U_ref^2 = 1000. The liquid hydrostatic profile
    1 + 0.98 (1 - z), which is what pins the top wall at p = 1, is subtracted from both
    the CFD and the PINN; otherwise the 2 units of hydrostatic head hide the drop. The
    difference panel is unchanged by this.
  - the h5's t = 0 pressure is all zeros (written before Basilisk's first pressure solve),
    so that frame of the p video is meaningless and is left out of the colour limits.
  - u, w, p are shown on the x-z plane through the axis (y = 0). v is identically zero
    there by symmetry, so it is shown on the y-z plane (x = 0), where it plays the role u
    plays on x-z.

Colour limits are global over the run and shared by CFD and PINN, so a frame's brightness
means the same in every frame. Snapshots tagged "validation" in the h5 were never trained
on, and are marked "held out" in the title.
"""
import sys
sys.path.append("../utilities")
import os
import glob
import argparse

import numpy as np
import h5py
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

from pytorch_utilities import NNCreator

H5_PATH = "../cfd_data/rising_bubble_3d_basilisk_L8.h5"
L_REF = 0.25
P_REF = 1000.0
G = 0.98
NO_LAYERS, WIDTH = 8, 400


def build_net(device):
    """The trainer's network: 8 x 400 tanh, outputs u, v, w, exp(p), sigmoid(a)."""
    act = {i: ["tanh", None, 10] for i in range(1, NO_LAYERS + 1)}
    out = [("output_u", None), ("output_v", None), ("output_w", None),
           ("output_p", "exponential"), ("output_a", "sigmoid")]
    net = NNCreator(torch.float32).get_model_dnn(4, [WIDTH] * NO_LAYERS, out, act, False)
    return net.to(device).eval()


def resolve_checkpoint(path):
    if os.path.isdir(path):
        found = sorted(glob.glob(os.path.join(path, "*.pth")), key=os.path.getmtime)
        if not found:
            sys.exit(f"no .pth in {path}")
        return found[-1]
    return path


@torch.no_grad()
def predict(net, pts, device, chunk=65536):
    outs = [torch.cat(net(torch.from_numpy(pts[s:s + chunk]).to(device)), dim=1).cpu()
            for s in range(0, len(pts), chunk)]
    return torch.cat(outs).numpy().T   # u, v, w, p, a


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", help=".pth file, or a run directory containing one")
    ap.add_argument("--h5", default=H5_PATH)
    ap.add_argument("--fps", type=float, default=3.0)
    ap.add_argument("--dpi", type=int, default=120)
    ap.add_argument("--out", default=None, help="output directory (default: next to the checkpoint)")
    args = ap.parse_args()

    ckpt = resolve_checkpoint(args.checkpoint)
    out_dir = args.out or os.path.dirname(os.path.abspath(ckpt))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = build_net(device)
    net.load_state_dict(torch.load(ckpt, map_location=device))
    print(f"checkpoint: {ckpt}")

    with h5py.File(args.h5, "r") as f:
        X, Y, Z = (np.array(f[c], dtype=float) for c in "XYZ")
        times = np.array(f["time"])
        split = np.array(f["split"])
        iy0, ix0 = len(Y) // 2, len(X) // 2   # cell centres nearest the axis
        p_hyd = 1.0 + G * (1.0 - Z)[:, None]  # (Z, 1), broadcasts over the horizontal axis

        # plane name -> (horizontal coordinate, h5 slice of a (Z, Y, X) snapshot, coords of the fixed axis)
        planes = {"xz": (X, (slice(None), iy0, slice(None)), ("y", Y[iy0])),
                  "yz": (Y, (slice(None), slice(None), ix0), ("x", X[ix0]))}
        field_plane = {"u": "xz", "v": "yz", "w": "xz", "p": "xz"}
        h5_name = {"u": "velocityX", "v": "velocityY", "w": "velocityZ", "p": "pressure"}
        out_idx = {"u": 0, "v": 1, "w": 2, "p": 3}

        data = {k: {"cfd": [], "pinn": []} for k in field_plane}
        alpha = {pl: {"cfd": [], "pinn": []} for pl in planes}
        for k, t in enumerate(times):
            print(f"  snapshot {k + 1}/{len(times)}  t = {t:.2f}", flush=True)
            for pl, (H, sl, (fixed_name, fixed_val)) in planes.items():
                HH, ZZ = np.meshgrid(H, Z)   # (Z, H)
                cols = {"x": HH, "y": np.full_like(HH, fixed_val)} if fixed_name == "y" else \
                       {"x": np.full_like(HH, fixed_val), "y": HH}
                pts = np.column_stack([cols["x"].ravel(), cols["y"].ravel(), ZZ.ravel(),
                                       np.full(HH.size, t)]).astype(np.float32) / L_REF
                pred = predict(net, pts, device).reshape(5, *HH.shape)
                alpha[pl]["pinn"].append(pred[4])
                alpha[pl]["cfd"].append(1.0 - f["volume_fraction"][(k,) + sl])
                for fld, fpl in field_plane.items():
                    if fpl != pl:
                        continue
                    cfd = np.array(f[h5_name[fld]][(k,) + sl], dtype=float)
                    pinn = pred[out_idx[fld]].astype(float)
                    if fld == "p":
                        cfd, pinn = cfd / P_REF - p_hyd, pinn - p_hyd
                    data[fld]["cfd"].append(cfd)
                    data[fld]["pinn"].append(pinn)

    labels = {"u": "u", "v": "v", "w": "w", "p": "p - p_hydrostatic (liquid)"}
    for fld, pl in field_plane.items():
        H = planes[pl][0]
        fixed_name, fixed_val = planes[pl][2]
        cfd, pinn = np.array(data[fld]["cfd"]), np.array(data[fld]["pinn"])
        diff = pinn - cfd
        # The t = 0 snapshot is skipped for the limits: Basilisk writes it before the first
        # pressure solve, so its CFD p is all zeros (-p_hyd after the subtraction).
        lim = np.percentile(np.abs(np.concatenate([cfd[1:].ravel(), pinn[1:].ravel()])), 99.5) or 1.0
        # The difference limit also skips the start-up (t < 0.1): at t = 0.02 and 0.04 the PINN
        # p sits ~0.1 below the CFD almost uniformly, about twice the error anywhere later,
        # so those two frames are allowed to saturate.
        dlim = np.percentile(np.abs(diff[times > 0.1]), 99.5) or 1.0
        ext = [H[0], H[-1], Z[0], Z[-1]]

        fig, axes = plt.subplots(1, 3, figsize=(11, 7.2), constrained_layout=True)
        ims = [axes[0].imshow(cfd[0], origin="lower", extent=ext, cmap="RdBu_r", vmin=-lim, vmax=lim),
               axes[1].imshow(pinn[0], origin="lower", extent=ext, cmap="RdBu_r", vmin=-lim, vmax=lim),
               axes[2].imshow(diff[0], origin="lower", extent=ext, cmap="PuOr_r", vmin=-dlim, vmax=dlim)]
        for ax, name in zip(axes, ["CFD (Basilisk)", "PINN", "PINN - CFD"]):
            ax.set_title(name)
            ax.set_xlabel(pl[0])
        axes[0].set_ylabel("z")
        fig.colorbar(ims[1], ax=axes[:2], shrink=0.6, label=labels[fld])
        fig.colorbar(ims[2], ax=axes[2], shrink=0.6, label="difference")
        contours = []

        def draw(k):
            for c in contours:
                c.remove()
            contours.clear()
            for im, arr in zip(ims, (cfd, pinn, diff)):
                im.set_data(arr[k])
            for ax in axes:
                contours.append(ax.contour(H, Z, alpha[pl]["cfd"][k], [0.5], colors="w", linewidths=1))
            contours.append(axes[1].contour(H, Z, alpha[pl]["pinn"][k], [0.5], colors="k",
                                            linewidths=1, linestyles="--"))
            tag = "held out" if split[k] == b"validation" else "train"
            if fld == "p" and k == 0:
                tag += ", CFD p not yet solved"
            fig.suptitle(f"{labels[fld]}   on {pl} plane ({fixed_name} = {fixed_val:+.4f})   "
                         f"t = {times[k]:.2f}  [{tag}]\n"
                         f"white: CFD alpha = 0.5, black dashed: PINN alpha = 0.5")
            return ims

        anim = FuncAnimation(fig, draw, frames=len(times), blit=False)
        path = os.path.join(out_dir, f"anim_{fld}.mp4")
        anim.save(path, writer=FFMpegWriter(fps=args.fps, bitrate=3000), dpi=args.dpi)
        plt.close(fig)
        print(f"wrote {path}   (shared limit +-{lim:.3g}, difference +-{dlim:.3g})")


if __name__ == "__main__":
    main()
