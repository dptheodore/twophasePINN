"""Animates rising_bubble_extruded_3d.h5 (from extrude_2d_to_3d.py) to show what the
extrusion is: the 2D bubble copied unchanged along a new y axis.

Panels per frame:
  1. the bubble surface (levelset = 0) in 3D -- a slab with the 2D bubble's cross
     section, the same at every y
  2. the x-z slice at mid y: the original 2D field (volume fraction + (u, w) vectors)
  3. the x-y slice through the bubble centroid height: every column is constant in y
The title reports max over the volume of |field(y) - field(y0)|, which is 0 for an
exact extrusion.

    ../.venv/bin/python animate_extruded_3d.py [--every 2] [--fps 10] [--gif]
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
    ap.add_argument("--h5", default="../cfd_data/rising_bubble_extruded_3d.h5")
    ap.add_argument("--every", type=int, default=2, help="use every n-th snapshot")
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--dpi", type=int, default=100)
    ap.add_argument("--gif", action="store_true")
    ap.add_argument("--out", default="../cfd_data")
    args = ap.parse_args()

    f = h5py.File(args.h5, "r")
    X, Y, Z, times = f["X"][:], f["Y"][:], f["Z"][:], f["time"][:]
    dx = float(X[1] - X[0])
    frames = np.arange(0, len(times), args.every)
    j_mid = len(Y) // 2
    q = 12  # quiver spacing in cells

    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.6, 1, 1.1], left=0.0, right=0.97, wspace=0.15)
    ax3 = fig.add_subplot(gs[0], projection="3d")
    ax_xz = fig.add_subplot(gs[1])
    ax_xy = fig.add_subplot(gs[2])

    def draw(k):
        n = frames[k]
        ls = f["levelset"][n]                        # (Z, Y, X), negative inside the bubble
        u, w = f["velocityX"][n], f["velocityZ"][n]
        y_var = max(np.abs(f[name][n] - f[name][n][:, :1, :]).max()
                    for name in ("levelset", "pressure", "velocityX", "velocityZ"))
        v_max = np.abs(f["velocityY"][n]).max()

        # 1. 3D bubble surface. Pad the y ends with liquid so the slab is drawn closed --
        # the field itself is periodic in y, i.e. the slab really continues indefinitely.
        ax3.cla()
        padded = np.pad(ls, ((0, 0), (1, 1), (0, 0)), constant_values=ls.max())
        verts, faces, _, _ = marching_cubes(padded, level=0.0, step_size=2)
        zyx = verts * dx + np.array([Z[0], Y[0] - dx, X[0]])
        tris = zyx[faces][:, :, ::-1]
        normals = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
        normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12
        light = np.array([-0.4, -0.7, 0.6]) / np.linalg.norm([-0.4, -0.7, 0.6])
        shade = 0.35 + 0.65 * np.abs(normals @ light)
        mesh = Poly3DCollection(tris, facecolors=shade[:, None] * np.array([0.12, 0.47, 0.71]),
                                edgecolors="none", linewidth=0)
        ax3.add_collection3d(mesh)
        ax3.set_xlim(X[0], X[-1]); ax3.set_ylim(Y[0] - dx, Y[-1] + dx); ax3.set_zlim(Z[0], Z[-1])
        ax3.set_box_aspect((X[-1] - X[0], 4 * (Y[-1] - Y[0] + 2 * dx), Z[-1] - Z[0]), zoom=1.15)
        ax3.set_yticks([Y[0], Y[-1]]); ax3.set_yticklabels([f"{Y[0]:.2f}", f"{Y[-1]:.2f}"])
        ax3.set_xlabel("x"); ax3.set_ylabel("y (4x)"); ax3.set_zlabel("z")
        ax3.view_init(elev=18, azim=-55)
        ax3.set_title("bubble surface: 2D shape extruded along y")

        # 2. x-z slice: the original 2D field
        ax_xz.cla()
        a = (ls[:, j_mid, :] < 0).astype(float)
        ax_xz.pcolormesh(X, Z, a, cmap="Blues", vmin=0, vmax=1.4, shading="auto")
        ax_xz.contour(X, Z, ls[:, j_mid, :], levels=[0], colors="k", linewidths=1)
        ax_xz.quiver(X[::q], Z[::q], u[::q, j_mid, ::q], w[::q, j_mid, ::q],
                     scale=4, width=0.003, color="tab:red")
        ax_xz.set_aspect("equal"); ax_xz.set_xlabel("x"); ax_xz.set_ylabel("z")
        ax_xz.set_title(f"x-z slice at y = {Y[j_mid]:.3f}\n= the 2D data, vectors (u, w)")

        # 3. x-y slice through the bubble centroid height
        ax_xy.cla()
        k_c = int(round(np.argwhere(ls[:, j_mid, :] < 0)[:, 0].mean()))
        ax_xy.pcolormesh(X, Y, (ls[k_c] < 0).astype(float), cmap="Blues", vmin=0, vmax=1.4,
                         shading="auto")
        ax_xy.contour(X, Y, ls[k_c], levels=[0], colors="k", linewidths=1)
        ax_xy.set_aspect(4.0); ax_xy.set_xlabel("x"); ax_xy.set_ylabel("y (drawn 4x)")
        ax_xy.set_title(f"x-y slice at bubble-centre z = {Z[k_c]:.3f}\nevery column is constant in y")

        fig.suptitle(f"rising_bubble_extruded_3d.h5   t = {times[n]:.2f} s   "
                     f"max |field(y) - field(y0)| = {y_var:.1e}   max |v| = {v_max:.1e}",
                     fontsize=12)

    anim = FuncAnimation(fig, draw, frames=len(frames))
    os.makedirs(args.out, exist_ok=True)
    mp4 = os.path.join(args.out, "extruded_bubble.mp4")
    anim.save(mp4, writer=FFMpegWriter(fps=args.fps, bitrate=4000), dpi=args.dpi)
    print("wrote", mp4)
    if args.gif:
        gif = os.path.join(args.out, "extruded_bubble.gif")
        anim.save(gif, writer=PillowWriter(fps=args.fps), dpi=max(args.dpi // 2, 60))
        print("wrote", gif)
    draw(len(frames) - 1)
    png = os.path.join(args.out, "extruded_bubble_last.png")
    fig.savefig(png, dpi=args.dpi)
    print("wrote", png)


if __name__ == "__main__":
    main()
