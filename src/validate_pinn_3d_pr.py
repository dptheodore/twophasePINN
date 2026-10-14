"""Compare a trained Peng-Robinson PINN (pytorch_3d_pr_train.py) against the LBPM h5.

Training only sees a few hundred grid nodes per snapshot; this evaluates the network on the
FULL grid of every snapshot and compares rho, u, v, w, p and a with the h5.

    python validate_pinn_3d_pr.py checkpoints/<run>/                 # picks the .pth in the dir
    python validate_pinn_3d_pr.py checkpoints/<run>/loss_X.pth --plot 13 --stride 1

Conventions follow get_training_data_3d_pr exactly:
  - coordinates: lattice cells centred on the domain, / L_ref; t relative to the first
    snapshot, / T_ref
  - rho / rho_ref, errors on rho also quoted in units of the liquid-gas jump (as in the loss)
  - u, v, w are the PHYSICAL (half-force corrected) velocities u_h5 + F / (2 rho), / U_ref
  - p: the h5 pressure is P_PR(rho), compared directly as p / p_scale. Current checkpoints
    have no p output and p = P_PR(rho_net); checkpoints from before 2026-10-14 carry a
    learned p, which is used and also checked against P_PR(rho_net).
Regions: "band" = diffuse interface (|levelset| < 8), "drop"/"carrier" = the bulk phases.
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
import pytorch_3d_pr_train as T
from pytorch_utilities import NNCreator

H5 = "../cfd_data/lbpm_bubble_dataset_newrun_nowrap.h5"


def resolve_checkpoint(path):
    if os.path.isdir(path):
        files = glob.glob(os.path.join(path, "*.pth"))
        if not files:
            raise FileNotFoundError(f"no .pth in {path}")
        return max(files, key=os.path.getctime)
    return path


def load_scales(f):
    """Mirrors get_training_data_3d_pr (all snapshots)."""
    times = np.array(f["time"], dtype=float)
    rho_mid = 0.5 * (pr.RHO_LIQ + pr.RHO_GAS)
    volume = np.median([(np.array(f["density"][i]) > rho_mid).sum() for i in range(len(times))])
    L_ref = float((3.0 * volume / (4.0 * np.pi)) ** (1.0 / 3.0))
    U_ref = float(np.sqrt(pr.FORCE_INCREMENT * abs(pr.RHO_LIQ - pr.RHO_GAS) / pr.RHO_GAS * 2.0 * L_ref))
    return dict(times=times, L_ref=L_ref, U_ref=U_ref, T_ref=L_ref / U_ref, rho_ref=pr.RHO_GAS,
                p_scale=float(pr.pressure_pr(pr.RHO_GAS)))


def build_pinn(sc, period, device, has_p=False):
    """Same architecture as pytorch_3d_pr_train.main(); has_p for pre-2026-10-14 checkpoints."""
    no_layers = 8
    hidden_layers = [400] * no_layers
    act = T.build_activation_dict(hidden_layers, {"tanh": range(1, no_layers + 1)}, [10] * no_layers)
    output_layer = [
        ("output_u", None), ("output_v", None), ("output_w", None),
        ("output_rho", "exponential"), ("output_a", "sigmoid"),
    ]
    if has_p:
        output_layer.insert(4, ("output_p", None))
    net = NNCreator(torch.float32).get_model_dnn(4, hidden_layers, output_layer, act, False,
                                                 periodic_lengths=period)
    t_knots = (sc["times"] - sc["times"][0]) / sc["T_ref"]
    return T.PINN(net, T.GRAVITY_MODE, t_knots, sc["T_ref"], sc["times"][0],
                  sc["L_ref"], sc["U_ref"]).to(device)


@torch.no_grad()
def predict(pinn, pts, device, chunk=131072):
    """Returns u, v, w, rho, p, a; p is None when the network has no p output."""
    names = [n for n, _ in pinn.net.output_specs]
    out = {n: [] for n in names}
    for i in range(0, len(pts), chunk):
        res = pinn.net(torch.from_numpy(pts[i:i + chunk]).to(device))
        for n, o in zip(names, res):
            out[n].append(o.cpu().numpy().ravel())
    out = {n: np.concatenate(o) for n, o in out.items()}
    return [out.get("output_" + n) for n in ("u", "v", "w", "rho", "p", "a")]


def errors(p, r):
    rmse = float(np.sqrt(np.mean((p - r) ** 2)))
    ref_rms = float(np.sqrt(np.mean(r ** 2)))
    return rmse, ref_rms, rmse / ref_rms if ref_rms > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", help=".pth file, or a run directory containing one")
    ap.add_argument("--h5", default=H5)
    ap.add_argument("--stride", type=int, default=1, help="spatial subsampling of the grid")
    ap.add_argument("--plot", type=int, nargs="*", default=[0, 13, 26], help="snapshots for slice plots")
    ap.add_argument("--out", default=None, help="output directory (default: next to the checkpoint)")
    args = ap.parse_args()

    ckpt = resolve_checkpoint(args.checkpoint)
    out_dir = args.out or os.path.dirname(os.path.abspath(ckpt))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    f = h5py.File(args.h5, "r")
    sc = load_scales(f)
    times, L_ref, U_ref, T_ref = sc["times"], sc["L_ref"], sc["U_ref"], sc["T_ref"]
    rho_ref, p_scale = sc["rho_ref"], sc["p_scale"]
    drho = (pr.RHO_LIQ - pr.RHO_GAS) / rho_ref
    rho_mid = 0.5 * (pr.RHO_LIQ + pr.RHO_GAS) / rho_ref

    X, Y, Z = (np.array(f[c], dtype=float) for c in "XYZ")
    X, Y, Z = X / np.diff(X)[0], Y / np.diff(Y)[0], Z / np.diff(Z)[0]
    period = [len(X) / L_ref, len(Y) / L_ref, len(Z) / L_ref]
    X, Y, Z = X - X.mean(), Y - Y.mean(), Z - Z.mean()
    t_nd = (times - times[0]) / T_ref

    state = torch.load(ckpt, map_location=device)
    has_p = "net.output_layers.output_p.weight" in state
    pinn = build_pinn(sc, period, device, has_p)
    pinn.load_state_dict(state)
    pinn.eval()
    print(f"\nCheckpoint: {ckpt}")
    print(f"L_ref = {L_ref:.4f} cells, U_ref = {U_ref:.6g} cells/step, T_ref = {T_ref:.1f} steps, "
          f"rho_ref = {rho_ref:.5f}, p_scale = {p_scale:.5g}")

    s = args.stride
    xs, ys, zs = X[::s] / L_ref, Y[::s] / L_ref, Z[::s] / L_ref
    ZZ, YY, XX = np.meshgrid(zs, ys, xs, indexing="ij")    # h5 layout is (Z, Y, X)
    grid = np.column_stack([XX.ravel(), YY.ravel(), ZZ.ravel()]).astype(np.float32)
    shape = ZZ.shape
    cell_vol = float(s ** 3)

    rows, integrals, slices = [], [], {}
    for k in range(len(times)):
        pts = np.hstack([grid, np.full((len(grid), 1), t_nd[k], np.float32)])
        u_p, v_p, w_p, rho_p, p_p, a_p = predict(pinn, pts, device)

        # Truth on the full grid (force correction needs the unsubsampled stencil).
        rho_full = np.array(f["density"][k], dtype=float)
        F = pr.lbpm_force(rho_full, times[k])
        sub = (slice(None, None, s),) * 3
        vel_c = [(np.array(f[n][k], dtype=float) + 0.5 * Fi / rho_full)[sub].ravel() / U_ref
                 for n, Fi in zip(("velocityX", "velocityY", "velocityZ"), F)]
        rho_c = rho_full[sub].ravel() / rho_ref
        p_c = np.array(f["pressure"][k], dtype=float)[sub].ravel() / p_scale
        ls = np.array(f["levelset"][k], dtype=float)[sub].ravel()   # raw: -8 drop, +8 carrier
        a_c = pr._phase_indicator(ls)
        p_eos = pr.pressure_pr(rho_p * rho_ref) / p_scale
        if p_p is None:
            p_p = p_eos

        regions = {"all": np.ones_like(ls, bool), "band": np.abs(ls) < 8.0,
                   "drop": ls <= -8.0, "carrier": ls >= 8.0}
        fields = {
            "rho": (rho_p, rho_c), "u": (u_p, vel_c[0]), "v": (v_p, vel_c[1]), "w": (w_p, vel_c[2]),
            "w_rel": (w_p - w_p.mean(), vel_c[2] - vel_c[2].mean()),
            "p": (p_p, p_c), "a": (a_p, a_c),
        }
        if has_p:
            fields["p_vs_PR(rho_net)"] = (p_p, p_eos)
        for region, m in regions.items():
            for name, (pv, rv) in fields.items():
                rmse, ref_rms, rel = errors(pv[m], rv[m])
                rows.append(dict(snapshot=k, t_step=int(times[k]), t_nd=t_nd[k], region=region,
                                 field=name, rmse=rmse, ref_rms=ref_rms, rel_l2=rel,
                                 pred_mean=float(pv[m].mean()), ref_mean=float(rv[m].mean()),
                                 max_abs_err=float(np.abs(pv[m] - rv[m]).max())))

        # Drop integrals: volume (rho above the midpoint), centroid z, total mass.
        def drop_stats(rho):
            inside = rho > rho_mid
            # centroid on the periodic z axis via the circular mean
            ang = 2 * np.pi * (ZZ.ravel()[inside] / period[2])
            zc = np.angle(np.mean(np.exp(1j * ang))) / (2 * np.pi) * period[2]
            return inside.sum() * cell_vol, zc, rho.sum() * cell_vol
        vp, zp, mp = drop_stats(rho_p)
        vc, zc, mc = drop_stats(rho_c)
        integrals.append(dict(snapshot=k, t_step=int(times[k]), vol_pred=vp, vol_cfd=vc,
                              zc_pred=zp, zc_cfd=zc, mass_pred=mp, mass_cfd=mc,
                              rho_min_pred=rho_p.min(), rho_max_pred=rho_p.max(),
                              rho_min_cfd=rho_c.min(), rho_max_cfd=rho_c.max()))

        if k in args.plot:
            slices[k] = {n: (pv.reshape(shape), rv.reshape(shape)) for n, (pv, rv) in fields.items()}
            slices[k]["_ls"] = ls.reshape(shape)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "validation_vs_cfd.csv"), index=False)
    ig = pd.DataFrame(integrals)
    ig.to_csv(os.path.join(out_dir, "validation_integrals.csv"), index=False)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.float_format", "{:.3e}".format)
    print("\n=== Error vs CFD, mean over all 27 snapshots (rel_l2 = rmse / rms(cfd)) ===")
    summ = df.groupby(["field", "region"])[["rmse", "ref_rms", "rel_l2", "max_abs_err"]].mean()
    print(summ.unstack("region").swaplevel(axis=1).sort_index(axis=1)
          [[(r, c) for r in ("all", "band", "drop", "carrier") for c in ("rmse", "rel_l2")]])

    r = df[(df.field == "rho")].pivot_table(index="region", values="rmse", aggfunc="mean")
    print(f"\nrho RMSE in units of the liquid-gas jump (loss units): "
          + ", ".join(f"{reg} {v / drho:.3f}" for reg, v in r.rmse.items()))

    print("\n=== Drop integrals (volume in cells^3, centroid z in L_ref) ===")
    print(ig.iloc[::3].to_string(index=False))

    print("\n=== Per-snapshot rel L2 (region all; rho also in band) ===")
    per = df[df.region == "all"].pivot(index="snapshot", columns="field", values="rel_l2")
    per["rho_band"] = df[(df.region == "band") & (df.field == "rho")].set_index("snapshot").rel_l2
    print(per[["rho", "rho_band", "u", "v", "w", "w_rel", "p", "a"]].to_string())

    gdf = T.gravity_table(pinn, dict(t_knots=t_nd, T_ref=T_ref, t_origin=times[0],
                                     **dict(zip(("t_budget", "g_budget"),
                                                pr.measure_bulk_acceleration(args.h5)))), device)
    print("\n=== Body force g(t), lattice units ===")
    print(gdf.to_string(index=False))

    if slices:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        names = ["rho", "u", "w_rel", "p", "a"]
        extent = [xs[0], xs[-1], zs[0], zs[-1]]
        for k, sl in slices.items():
            ls3 = sl["_ls"]
            drop = np.argwhere(ls3 < 0)
            iy = int(round(drop[:, 1].mean())) if len(drop) else ls3.shape[1] // 2
            fig, axes = plt.subplots(len(names), 3, figsize=(9, 4.2 * len(names)))
            for row, name in zip(axes, names):
                pv, rv = sl[name][0][:, iy, :], sl[name][1][:, iy, :]
                div = name in ("u", "w_rel")
                lo, hi = min(pv.min(), rv.min()), max(pv.max(), rv.max())
                if div:
                    hi = max(abs(lo), abs(hi)); lo = -hi
                for ax, data, title in zip(row, [rv, pv, pv - rv], ["CFD", "PINN", "PINN - CFD"]):
                    if title == "PINN - CFD":
                        e = max(np.abs(data).max(), 1e-12)
                        im = ax.imshow(data, origin="lower", extent=extent, cmap="RdBu_r", vmin=-e, vmax=e)
                    else:
                        im = ax.imshow(data, origin="lower", extent=extent,
                                       cmap="RdBu_r" if div else "viridis", vmin=lo, vmax=hi)
                    ax.contour(xs, zs, ls3[:, iy, :], levels=[0], colors="k", linewidths=0.8)
                    ax.set_title(f"{name}: {title}")
                    fig.colorbar(im, ax=ax, fraction=0.046)
            fig.suptitle(f"snapshot {k} (t = {int(times[k])}), x-z plane through drop, y index {iy}")
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, f"validation_slice_{k}.png"), dpi=100)
            plt.close(fig)

            # rho and w along the vertical line through the drop centre
            ix = int(round(drop[:, 2].mean())) if len(drop) else ls3.shape[2] // 2
            fig, axs = plt.subplots(1, 3, figsize=(14, 3.8))
            for ax, name in zip(axs, ("rho", "w", "p")):
                ax.plot(zs, sl[name][1][:, iy, ix], "k-", label="CFD")
                ax.plot(zs, sl[name][0][:, iy, ix], "r--", label="PINN")
                ax.set_xlabel("z / L_ref"); ax.set_title(f"{name} on the drop axis"); ax.legend()
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, f"validation_axis_{k}.png"), dpi=100)
            plt.close(fig)
        print(f"\nPlots: validation_slice_*.png, validation_axis_*.png in {out_dir}")
    print(f"Tables: validation_vs_cfd.csv, validation_integrals.csv in {out_dir}")


if __name__ == "__main__":
    main()
