"""Render a video of the LBPM h5 itself -- no network involved.

An x-z plane through the drop, one frame per snapshot: the volume fraction (with the
in-plane velocity over it), u, w with the bulk drift removed, and the pressure.

    python animate_cfd_3d.py                          # lab frame, full resolution
    python animate_cfd_3d.py --follow --fps 3         # keep the drop centred instead

Everything is left in raw lattice units: lengths in cells, velocities in cells/step,
pressure in LBPM units, time in steps. (animate_pinn_3d.py shows nondimensional values,
so the colour scales are not directly comparable with it.)

Two things this is meant to make visible:
  - w carries a uniform bulk drift that ramps through the run (the domain accelerates
    under the body force), and that drift swamps everything else. Each snapshot's domain
    mean is therefore subtracted from the w panel, which is what exposes the flow
    structure at the interface. The drop still translates in the lab frame.
  - that structure is a shell of parasitic currents riding on the interface, not a
    buoyant wake -- it is ~800x the real drop-carrier slip and its amplitude does not
    change as the bulk flow grows.

v is not shown: the slice passes through the drop centroid, which is a symmetry plane for
v, so it is antisymmetric about it and reads ~1e-9 there.

Colour limits are global over the whole run (computed in a first pass) so that brightness
means the same thing in every frame; w deliberately keeps a global scale so the drift can
be seen ramping. Pressure uses a robust 99.5th-percentile limit, since the Peng-Robinson
isotherm spike at the interface is ~3x anything else in the field.
"""
import sys
sys.path.append("../utilities")
import os
import argparse

import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter

from generate_points_3d import _measure_scales, T_MAX

PANELS = ["a", "u", "w_rel", "p"]
DEEP_PHASE = 7.5
FAR_RADIUS = 2.0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--h5", default="../cfd_data/lbpm_bubble_dataset.h5")
    ap.add_argument("--stride", type=int, default=1, help="spatial subsampling of the slice")
    ap.add_argument("--follow", action="store_true",
                    help="roll z so the drop stays centred (default: lab frame)")
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--dpi", type=int, default=110)
    ap.add_argument("--quiver", type=int, default=6, help="in-plane vector spacing, 0 to disable")
    ap.add_argument("--gif", action="store_true")
    ap.add_argument("--out", default="../trained_model_3d", help="output directory")
    args = ap.parse_args()

    f = h5py.File(args.h5, "r")
    times = np.array(f["time"], dtype=float)
    n_keep = len(times) if T_MAX is None else int(np.searchsorted(times, T_MAX, side="right"))
    times = times[:n_keep]
    levelset = -np.array(f["levelset"][:n_keep])        # > 0 => drop
    pressure = np.array(f["pressure"][:n_keep])

    L_ref = _measure_scales(levelset, pressure, times)[0]   # only for the far-field mask
    print(f"\nL_ref = {L_ref:.4f} cells (raw lattice units, no normalisation)")

    X, Y, Z = (np.array(f[c], dtype=float) for c in "XYZ")
    X, Y, Z = X / np.diff(X)[0], Y / np.diff(Y)[0], Z / np.diff(Z)[0]
    X, Y, Z = X - X.mean(), Y - Y.mean(), Z - Z.mean()
    s = args.stride
    xs, zs = X[::s] , Z[::s] 

    # Fix the y plane for the whole run; the drop moves along z, not y.
    iy = int(round(float(np.median(
        [np.argwhere(levelset[k, ::s, ::s, ::s] > 0)[:, 1].mean() for k in range(len(times))]))))

    frames = []
    for k in range(len(times)):
        sub = (k, slice(None, None, s), slice(None, None, s), slice(None, None, s))
        ls3 = levelset[sub]
        u3 = f["velocityX"][sub]
        w3 = f["velocityZ"][sub]
        p3 = pressure[sub]

        # Gauge p on the undisturbed far carrier, as validate_pinn_3d does.
        centroid = np.argwhere(ls3 > 0).mean(axis=0)
        gz, gy, gx = np.ogrid[:ls3.shape[0], :ls3.shape[1], :ls3.shape[2]]
        radius = np.sqrt((gz - centroid[0]) ** 2 + (gy - centroid[1]) ** 2
                         + (gx - centroid[2]) ** 2) * s
        far = (ls3 <= -DEEP_PHASE) & (radius > FAR_RADIUS * L_ref)
        p3 = p3 - p3[far].mean()
        w_rel3 = w3 - w3.mean()

        sl = lambda q: np.asarray(q)[:, iy, :]
        mean_w = float(np.asarray(w3).mean())   # the bulk drift, kept for the title
        fr = {"a": (ls3 > 0).astype(float), "u": u3, "w_rel": w_rel3, "p": p3}
        fr = {n: sl(q) for n, q in fr.items()}
        fr["_ls"] = sl(ls3)
        fr["_meanw"] = mean_w
        if args.follow:                                  # roll the drop to mid-domain
            shift = int(round(ls3.shape[0] / 2 - centroid[0]))
            fr = {n: (q if n == "_meanw" else np.roll(q, shift, axis=0))
                  for n, q in fr.items()}
        frames.append(fr)
        print(f"  snapshot {k + 1}/{len(times)}  step {times[k]:.0f}", flush=True)

    lims = {}
    for n in PANELS:
        if n == "p":            # robust: the PR interface spike is ~3x everything else
            lims[n] = float(np.percentile(np.abs(np.stack([fr[n] for fr in frames])), 99.5))
        else:
            lims[n] = float(max(np.abs(fr[n]).max() for fr in frames))
        lims[n] = max(lims[n], 1e-12)

    fig, axes = plt.subplots(2, 2, figsize=(9, 11))
    extent = [xs[0], xs[-1], zs[0], zs[-1]]
    xg, zg = np.linspace(xs[0], xs[-1], len(xs)), np.linspace(zs[0], zs[-1], len(zs))
    q = args.quiver
    images, contours, quiv = {}, {}, [None]
    fr0 = frames[0]
    for ax, n in zip(axes.ravel(), PANELS):
        if n == "a":
            im = ax.imshow(fr0[n], origin="lower", extent=extent, cmap="viridis",
                           vmin=0, vmax=1)
        else:
            im = ax.imshow(fr0[n], origin="lower", extent=extent, cmap="RdBu_r",
                           vmin=-lims[n], vmax=lims[n])
        images[n] = im
        contours[n] = ax.contour(xg, zg, fr0["_ls"], levels=[0], colors="k", linewidths=0.8)
        ax.set_title({"a": "volume fraction w/ in-plane velocity",
                      "w_rel": "w",
                      "p": "p"}.get(n, n),
                     fontsize=10)
        ax.set_xlabel("x [cells]", fontsize=8)
        ax.set_ylabel("z [cells]", fontsize=8)
        ax.tick_params(labelsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046)
    ax_a = axes.ravel()[0]
    if q:
        quiv[0] = ax_a.quiver(xg[::q], zg[::q], fr0["u"][::q, ::q], fr0["w_rel"][::q, ::q],
                              color="w", scale=None, width=0.004)
    suptitle = fig.suptitle("")
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    def draw(k):
        fr = frames[k]
        for n in PANELS:
            images[n].set_data(fr[n])
            contours[n].remove()
            ax = axes.ravel()[PANELS.index(n)]
            contours[n] = ax.contour(xg, zg, fr["_ls"], levels=[0], colors="k", linewidths=0.8)
        if q:
            quiv[0].set_UVC(fr["u"][::q, ::q], fr["w_rel"][::q, ::q])
        suptitle.set_text(
            f"LBPM h5   snapshot {k}/{len(frames) - 1}   step {times[k]:.0f}   "
            f"bulk drift w = {frames[k]['_meanw']:+.3g}   "
            f"({'drop-following' if args.follow else 'lab frame'}, y index {iy})")
        return list(images.values())

    anim = FuncAnimation(fig, draw, frames=len(frames), blit=False)
    tag = "_follow" if args.follow else ""
    os.makedirs(args.out, exist_ok=True)
    mp4 = os.path.join(args.out, f"cfd_h5{tag}.mp4")
    print(f"\nWriting {mp4} ...", flush=True)
    anim.save(mp4, writer=FFMpegWriter(fps=args.fps, bitrate=4000), dpi=args.dpi)
    print(f"Wrote {mp4}")
    if args.gif:
        gif = os.path.join(args.out, f"cfd_h5{tag}.gif")
        anim.save(gif, writer=PillowWriter(fps=args.fps), dpi=max(args.dpi // 2, 60))
        print(f"Wrote {gif}")


if __name__ == "__main__":
    main()
