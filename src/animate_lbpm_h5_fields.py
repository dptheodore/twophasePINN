"""Visualise the raw fields of an LBPM h5 dataset -- no network involved.

One 2D plane through the drop per snapshot, eight panels:
    density | levelset | pressure | |u| with in-plane vectors
    u (x)   | v (y)    | w (z)    | w - <w>  (bulk drift removed)
The levelset = 0 contour (the interface) is drawn on every panel.

    python animate_lbpm_h5_fields.py                      # x-z plane, mp4 of all snapshots
    python animate_lbpm_h5_fields.py --plane yz --gif
    python animate_lbpm_h5_fields.py --snap 0 --snap 26   # static PNGs instead of a video

Array layout in the h5 is (time, z, y, x); the drop rises along z. Only the chosen plane
is read from disk, so this is cheap even on the 600 MB file. Values are raw lattice units
(cells, cells/step, LBPM pressure, LBM steps).

Notes:
  - on the x-z plane through the drop centre, v is ~0 by symmetry (and u on y-z); use
    --plane yz / xz to see the other one.
  - levelset < 0 is the drop and is clipped at +-8 cells by LBPM.
  - colour limits are global over the run so brightness means the same in every frame;
    pressure uses a 99.5th-percentile limit because the interface spike dominates.
"""
import os
import argparse

import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter

H5_DEFAULT = "../cfd_data/lbpm_bubble_dataset_newrun_nowrap.h5"

# name, title, colormap, kind ("seq" = min..max, "div" = symmetric about 0)
PANELS = [
    ("density",  r"density $\rho$",              "viridis",  "seq"),
    ("levelset", r"levelset $\phi$ (<0 = drop)", "RdBu_r",   "div"),
    ("pressure", r"pressure $p$",                "magma",    "seq"),
    ("speed",    r"$|\mathbf{u}|$ + in-plane vectors", "inferno", "seq"),
    ("u",        r"$u$ (x-velocity)",            "coolwarm", "div"),
    ("v",        r"$v$ (y-velocity)",            "coolwarm", "div"),
    ("w",        r"$w$ (z-velocity)",            "coolwarm", "div"),
    ("w_rel",    r"$w - \langle w \rangle$",     "coolwarm", "div"),
]


def read_plane(f, name, plane, idx):
    """All snapshots of one field on one plane, as (time, vertical, horizontal)."""
    d = f[name]
    if plane == "xz":
        return np.asarray(d[:, :, idx, :])        # (t, z, x)
    if plane == "yz":
        return np.asarray(d[:, :, :, idx])        # (t, z, y)
    return np.asarray(d[:, idx, :, :])            # xy: (t, y, x)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--h5", default=H5_DEFAULT)
    ap.add_argument("--plane", choices=["xz", "yz", "xy"], default="xz")
    ap.add_argument("--index", type=int, default=None,
                    help="cell index of the plane (default: domain centre for xz/yz, "
                         "drop centroid in z at the first snapshot for xy)")
    ap.add_argument("--snap", type=int, action="append",
                    help="write a static PNG of this snapshot index (repeatable) instead of a video")
    ap.add_argument("--fps", type=float, default=3.0)
    ap.add_argument("--dpi", type=int, default=110)
    ap.add_argument("--quiver", type=int, default=6, help="vector spacing in cells, 0 to disable")
    ap.add_argument("--gif", action="store_true", help="also/instead write a gif")
    ap.add_argument("--out", default="../cfd_data/h5_viz", help="output directory")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    f = h5py.File(args.h5, "r")
    times = np.asarray(f["time"], dtype=float)
    nt, nz, ny, nx = f["density"].shape

    if args.index is None:
        if args.plane == "xz":
            args.index = ny // 2
        elif args.plane == "yz":
            args.index = nx // 2
        else:
            args.index = int(round(np.argwhere(np.asarray(f["levelset"][0]) < 0)[:, 0].mean()))
    print(f"{args.h5}: {nt} snapshots, grid (z,y,x) = ({nz},{ny},{nx}); "
          f"plane {args.plane} at index {args.index}")

    # in-plane (horizontal, vertical) velocity components for the quiver
    h_comp, v_comp = {"xz": ("u", "w"), "yz": ("v", "w"), "xy": ("u", "v")}[args.plane]
    h_lab, v_lab = {"xz": ("x", "z"), "yz": ("y", "z"), "xy": ("x", "y")}[args.plane]

    data = {
        "density":  read_plane(f, "density", args.plane, args.index),
        "levelset": read_plane(f, "levelset", args.plane, args.index),
        "pressure": read_plane(f, "pressure", args.plane, args.index),
        "u":        read_plane(f, "velocityX", args.plane, args.index),
        "v":        read_plane(f, "velocityY", args.plane, args.index),
        "w":        read_plane(f, "velocityZ", args.plane, args.index),
    }
    # bulk drift: the 3D domain mean of w per snapshot (one pass over the dataset, chunked)
    mean_w = np.array([float(np.mean(f["velocityZ"][k])) for k in range(nt)])
    f.close()
    data["w_rel"] = data["w"] - mean_w[:, None, None]
    data["speed"] = np.sqrt(data["u"] ** 2 + data["v"] ** 2 + data["w"] ** 2)

    lims = {}
    for name, _, _, kind in PANELS:
        q = data[name]
        if name == "pressure":
            lo, hi = np.percentile(q, [0.25, 99.75])
        elif kind == "div":
            hi = float(np.abs(q).max()) or 1e-12
            lo = -hi
        else:
            lo, hi = float(q.min()), float(q.max())
        lims[name] = (lo, hi)

    nv, nh = data["density"].shape[1:]
    extent = [0, nh, 0, nv]
    hg, vg = np.arange(nh) + 0.5, np.arange(nv) + 0.5
    aspect = nv / nh
    fig, axes = plt.subplots(2, 4, figsize=(13.2, 2 * 2.0 * min(aspect, 2.2) + 1.2),
                             constrained_layout=True)

    images, contours, quiv = {}, {}, [None]

    def draw_contour(ax, name, k):
        if name in contours and contours[name] is not None:
            contours[name].remove()
        ls = data["levelset"][k]
        contours[name] = (ax.contour(hg, vg, ls, levels=[0.0], colors="k", linewidths=0.8)
                          if ls.min() < 0 < ls.max() else None)

    def draw_quiver(ax, k):
        if quiv[0] is not None:
            quiv[0].remove()
            quiv[0] = None
        s = args.quiver
        if s <= 0:
            return
        qh, qv = data[h_comp][k][s // 2::s, s // 2::s], data[v_comp][k][s // 2::s, s // 2::s]
        H, V = np.meshgrid(hg[s // 2::s], vg[s // 2::s])
        quiv[0] = ax.quiver(H, V, qh, qv, color="w", scale=lims["speed"][1] * 12,
                            width=0.004, alpha=0.85)

    for ax, (name, title, cmap, _) in zip(axes.ravel(), PANELS):
        lo, hi = lims[name]
        images[name] = ax.imshow(data[name][0], origin="lower", extent=extent, cmap=cmap,
                                 vmin=lo, vmax=hi, interpolation="nearest")
        fig.colorbar(images[name], ax=ax, shrink=0.85, format="%.2g")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(f"{h_lab} [cells]")
        ax.set_ylabel(f"{v_lab} [cells]")
        draw_contour(ax, name, 0)
    draw_quiver(axes.ravel()[3], 0)
    suptitle = fig.suptitle("")

    def update(k):
        for ax, (name, *_rest) in zip(axes.ravel(), PANELS):
            images[name].set_data(data[name][k])
            draw_contour(ax, name, k)
        draw_quiver(axes.ravel()[3], k)
        suptitle.set_text(
            f"{os.path.basename(args.h5)}   plane {args.plane}={args.index}   "
            f"snapshot {k}/{nt - 1}   step {times[k]:.0f}   <w> = {mean_w[k]:.3e}")
        return list(images.values())

    tag = f"{args.plane}{args.index}"
    if args.snap:
        for k in args.snap:
            update(k)
            path = os.path.join(args.out, f"h5_fields_{tag}_snap{k:02d}.png")
            fig.savefig(path, dpi=args.dpi)
            print("wrote", path)
        return

    anim = FuncAnimation(fig, update, frames=nt, blit=False)
    if not args.gif:
        try:
            path = os.path.join(args.out, f"h5_fields_{tag}.mp4")
            anim.save(path, writer=FFMpegWriter(fps=args.fps), dpi=args.dpi)
            print("wrote", path)
            return
        except (FileNotFoundError, RuntimeError) as e:
            print(f"ffmpeg unavailable ({e}); falling back to gif")
    path = os.path.join(args.out, f"h5_fields_{tag}.gif")
    anim.save(path, writer=PillowWriter(fps=args.fps), dpi=args.dpi)
    print("wrote", path)


if __name__ == "__main__":
    main()
