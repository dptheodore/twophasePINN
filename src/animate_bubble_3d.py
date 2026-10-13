"""Animates rising_bubble_3d.h5 (from revolve_2d_to_3d.py): the 2D bubble revolved
into a 3D bubble with its volume held at the initial sphere's.

Panels per frame:
  1. the bubble surface (levelset = 0) in 3D, inside the 1 x 1 x 2 box
  2. the vertical x-z slice through the bubble axis
  3. the horizontal x-y slice through the bubble centroid
The title reports the bubble volume relative to the initial sphere.

    ../.venv/bin/python animate_bubble_3d.py [--every 2] [--fps 10] [--gif]
"""
import os
import argparse

import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage.measure import marching_cubes


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--h5", default="../cfd_data/rising_bubble_3d.h5")
    ap.add_argument("--every", type=int, default=2, help="use every n-th snapshot")
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--dpi", type=int, default=100)
    ap.add_argument("--gif", action="store_true")
    ap.add_argument("--out", default="../cfd_data")
    args = ap.parse_args()

    f = h5py.File(args.h5, "r")
    X, Y, Z, times = f["X"][:], f["Y"][:], f["Z"][:], f["time"][:]
    dx = float(X[1] - X[0])
    V0 = 4.0 / 3.0 * np.pi * 0.25 ** 3
    frames = np.arange(0, len(times), args.every)
    j_mid = int(np.argmin(np.abs(Y)))
    light = np.array([-0.4, -0.7, 0.6]) / np.linalg.norm([-0.4, -0.7, 0.6])

    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.6, 0.9, 1.1], left=0.0, right=0.97, wspace=0.15)
    ax3 = fig.add_subplot(gs[0], projection="3d")
    ax_xz = fig.add_subplot(gs[1])
    ax_xy = fig.add_subplot(gs[2])

    def draw(k):
        n = frames[k]
        ls = f["levelset"][n]                        # (Z, Y, X), negative inside the bubble
        inside = ls < 0
        kz = np.nonzero(inside.any(axis=(1, 2)))[0]
        k_c = int(round(np.nonzero(inside)[0].mean()))

        # 1. 3D surface, meshed only on the z band holding the bubble
        ax3.cla()
        lo, hi = max(kz[0] - 3, 0), min(kz[-1] + 4, len(Z))
        verts, faces, _, _ = marching_cubes(ls[lo:hi], level=0.0, step_size=2)
        tris = (verts * dx + np.array([Z[lo], Y[0], X[0]]))[faces][:, :, ::-1]
        normals = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
        normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12
        shade = 0.3 + 0.7 * np.abs(normals @ light)
        ax3.add_collection3d(Poly3DCollection(tris, facecolors=shade[:, None] * np.array([0.12, 0.47, 0.71]),
                                              edgecolors="none", linewidth=0))
        ax3.set_xlim(X[0], X[-1]); ax3.set_ylim(Y[0], Y[-1]); ax3.set_zlim(Z[0], Z[-1])
        ax3.set_box_aspect((1, 1, 2), zoom=1.1)
        ax3.set_xlabel("x"); ax3.set_ylabel("y"); ax3.set_zlabel("z")
        ax3.view_init(elev=15, azim=-60)
        ax3.set_title("bubble surface")

        # 2. vertical slice through the axis
        ax_xz.cla()
        ax_xz.pcolormesh(X, Z, inside[:, j_mid, :].astype(float), cmap="Blues", vmin=0, vmax=1.4, shading="auto")
        ax_xz.contour(X, Z, ls[:, j_mid, :], levels=[0], colors="k", linewidths=1)
        ax_xz.set_aspect("equal"); ax_xz.set_xlabel("x"); ax_xz.set_ylabel("z")
        ax_xz.set_title("x-z slice through the axis (y = 0)")

        # 3. horizontal slice through the centroid
        ax_xy.cla()
        ax_xy.pcolormesh(X, Y, inside[k_c].astype(float), cmap="Blues", vmin=0, vmax=1.4, shading="auto")
        ax_xy.contour(X, Y, ls[k_c], levels=[0], colors="k", linewidths=1)
        ax_xy.set_aspect("equal"); ax_xy.set_xlabel("x"); ax_xy.set_ylabel("y")
        ax_xy.set_title(f"x-y slice at bubble-centre z = {Z[k_c]:.3f}")

        fig.suptitle(f"rising_bubble_3d.h5   t = {times[n]:.2f} s   "
                     f"bubble volume / initial sphere = {inside.sum() * dx ** 3 / V0:.3f}", fontsize=12)

    anim = FuncAnimation(fig, draw, frames=len(frames))
    os.makedirs(args.out, exist_ok=True)
    mp4 = os.path.join(args.out, "bubble_3d.mp4")
    anim.save(mp4, writer=FFMpegWriter(fps=args.fps, bitrate=4000), dpi=args.dpi)
    print("wrote", mp4)
    if args.gif:
        gif = os.path.join(args.out, "bubble_3d.gif")
        anim.save(gif, writer=PillowWriter(fps=args.fps), dpi=max(args.dpi // 2, 60))
        print("wrote", gif)
    draw(len(frames) - 1)
    png = os.path.join(args.out, "bubble_3d_last.png")
    fig.savefig(png, dpi=args.dpi)
    print("wrote", png)


if __name__ == "__main__":
    main()
