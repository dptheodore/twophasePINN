import argparse

import h5py
import numpy as np
from scipy import ndimage

import generate_points_3d_pr as pr

''' Why does the rho data loss of pytorch_3d_pr_train.py creep UP while every PDE loss
falls? Hypothesis: the CFD data is not a solution of the trainer's residuals, so the
optimiser buys PDE loss with rho accuracy (smearing the interface shrinks the
Korteweg / grad(p) terms like 1/w^3 and 1/w).

No training here -- everything is evaluated on the h5 itself with periodic finite
differences, nondimensionalised exactly as the trainer does. Sections:

  1. scales + interface width   (is the profile even resolved?)
  2. residual floor             (trainer's losses evaluated on the truth, per region,
                                 per term, 2nd vs 4th order stencils, 400 vs 800 step dt)
  3. continuity / velocity      (div(u) over time, parasitic shell vs radius)
  4. smearing test              (blur rho by a Gaussian: what it costs in rho loss vs
                                 what it saves in PDE loss -- the trade the optimiser sees)

Defaults match the fixed trainer (corrected u, beta = 1/12); --raw-velocity --beta 0
reproduce the old one. Run from src/:  python diagnose_rho_residuals.py [--snapshots 4,12,20] '''

H5 = "../cfd_data/lbpm_bubble_dataset_newrun_nowrap.h5"
FORCE_CORRECT = True                                    # --raw-velocity turns it off
BETA = pr.LATTICE_BETA                                  # set by --beta (lattice units)
W_PDE = dict(m=1.0, u=10.0, v=10.0, w=10.0, eos=1.0)   # trainer weights
AX = dict(x=2, y=1, z=0)                                # arrays are (Z, Y, X)


# ---------------------------------------------------------------- finite differences
def d1(f, ax, order):
    r = lambda s: np.roll(f, -s, axis=ax)
    if order == 2:
        return 0.5 * (r(1) - r(-1))
    return (-r(2) + 8.0 * r(1) - 8.0 * r(-1) + r(-2)) / 12.0


def d2(f, ax, order):
    r = lambda s: np.roll(f, -s, axis=ax)
    if order == 2:
        return r(1) - 2.0 * f + r(-1)
    return (-r(2) + 16.0 * r(1) - 30.0 * f + 16.0 * r(-1) - r(-2)) / 12.0


def grad(f, order, L):
    return [d1(f, AX[c], order) * L for c in "xyz"]


def lap(f, order, L):
    return sum(d2(f, AX[c], order) for c in "xyz") * L ** 2


# ---------------------------------------------------------------- data + scales
def load_scales(h5):
    ''' Mirrors get_training_data_3d_pr (all snapshots set the scales). '''
    times = np.array(h5["time"], dtype=float)
    sub = np.arange(len(times))
    rho_mid = 0.5 * (pr.RHO_LIQ + pr.RHO_GAS)
    volume = np.median([(np.array(h5["density"][i]) > rho_mid).sum() for i in sub])
    L = float((3.0 * volume / (4.0 * np.pi)) ** (1.0 / 3.0))
    U = float(np.sqrt(pr.FORCE_INCREMENT * abs(pr.RHO_LIQ - pr.RHO_GAS) / pr.RHO_GAS * 2.0 * L))
    s = dict(L=L, U=U, T=L / U, rho_ref=pr.RHO_GAS, p_scale=float(pr.pressure_pr(pr.RHO_GAS)),
             t0=times[sub[0]], times=times)
    s["one_Re"] = pr.NU / (U * L)
    s["Eu"] = pr.K_FACTOR ** 2 * s["p_scale"] / (s["rho_ref"] * U ** 2)
    s["Ko"] = pr.PR_KAPPA * s["rho_ref"] / (L ** 2 * U ** 2)
    s["drho_star"] = (pr.RHO_LIQ - pr.RHO_GAS) / s["rho_ref"]
    return s


def snapshot(h5, i, s):
    ''' Nondimensional rho*, (u*, v*, w*), p_h5*, raw levelset at snapshot i. With
    FORCE_CORRECT the h5 velocity gets the Guo half-force term, u = u_h5 + F / (2 rho):
    the stored field is sum(f c)/rho without it. '''
    rho_lat = np.array(h5["density"][i], dtype=float)
    rho = rho_lat / s["rho_ref"]
    vel = [np.array(h5[k][i], dtype=float) for k in ("velocityX", "velocityY", "velocityZ")]
    if FORCE_CORRECT:
        F = pr.lbpm_force(rho_lat, s["times"][i])
        vel = [v + 0.5 * f / rho_lat for v, f in zip(vel, F)]
    vel = [v / s["U"] for v in vel]
    p = np.array(h5["pressure"][i], dtype=float) / s["p_scale"]
    ls = np.array(h5["levelset"][i], dtype=float)
    return rho, vel, p, ls


def regions(ls):
    ''' Masks matching the trainer's samplers (levelset raw = signed distance, clipped +-8):
    interface ~ PDE "interface" points (0-0.25 cells off the surface), near ~ "nearfield"
    (0.25-12 cells), domain ~ "domain" (|ls| >= 4), band/bulk ~ DATA band/bulk. '''
    a = np.abs(ls)
    band = a < 8.0
    near_out = ndimage.binary_dilation(band, iterations=4)
    return dict(interface=a < 1.0, near=near_out & (a >= 0.25), domain=a >= 4.0,
                band=band, bulk=~band)


def trainer_mix(mse):
    ''' Approximate per-point PDE loss with the trainer's (400, 2000, 3000) point mix. '''
    return (400 * mse["interface"] + 2000 * mse["near"] + 3000 * mse["domain"]) / 5400.0


# ---------------------------------------------------------------- residuals
def residuals(rho_m, rho, rho_p, vel_m, vel, vel_p, dt, g_star, s, order, lam=0.0):
    ''' The trainer's PDE_caller on the grid. dt is nondimensional (snapshot i-k to i+k
    spans 2 dt). p is taken AT the EOS (p* = P_PR(rho)/p_scale), as the trained network
    is pushed to; the h5 pressure is checked against the EOS separately. Returns the
    residual fields plus the individual momentum terms. '''
    L = s["L"]
    p = pr.pressure_pr(rho * s["rho_ref"]) / s["p_scale"]
    rho_t = (rho_p - rho_m) / (2.0 * dt)
    vel_t = [(a - b) / (2.0 * dt) for a, b in zip(vel_p, vel_m)]

    dvel = [grad(c, order, L) for c in vel]              # dvel[i][j] = d u_i / d x_j
    drho = grad(rho, order, L)
    dp = grad(p, order, L)
    div = dvel[0][0] + dvel[1][1] + dvel[2][2]
    ddiv = grad(div, order, L)
    lap_vel = [lap(c, order, L) for c in vel]
    dlap_rho = grad(lap(rho, order, L), order, L)

    adv_rho = vel[0] * drho[0] + vel[1] * drho[1] + vel[2] * drho[2]
    res = dict(m=rho_t + adv_rho + rho * div)
    terms = dict(rho_t=rho_t, u_grad_rho=adv_rho, rho_div_u=rho * div)
    for i, c in enumerate("uvw"):
        inertia = rho * (vel_t[i] + sum(vel[j] * dvel[i][j] for j in range(3)))
        visc = rho * (lap_vel[i] + (1.0 + lam) * ddiv[i]) + \
            sum(drho[j] * (dvel[i][j] + dvel[j][i]) for j in range(3)) + lam * drho[i] * div
        pres = s["Eu"] * dp[i]
        # BETA adds a rho-independent beta grad(lap(rho)) (lattice units) to the Korteweg
        # force, i.e. kappa -> kappa + beta / rho; nondim coefficient beta / (L U)^2.
        kort = -(s["Ko"] * rho + BETA / (L * s["U"]) ** 2) * dlap_rho[i]
        grav = -rho * g_star if i == 2 else np.zeros_like(rho)
        res[c] = inertia + pres + kort - s["one_Re"] * visc + grav
        if c in "uw":
            terms.update({f"{c}:inertia": inertia, f"{c}:Eu_grad_p": pres, f"{c}:Korteweg": kort,
                          f"{c}:viscous": s["one_Re"] * visc, f"{c}:gravity": grav,
                          f"{c}:grad_p+Kort": pres + kort})
    terms["div_u"] = div
    return res, terms


def region_mse(field, masks):
    return {k: float(np.mean(field[m] ** 2)) for k, m in masks.items()}


def rms(field, mask):
    return float(np.sqrt(np.mean(field[mask] ** 2)))


# ---------------------------------------------------------------- sections
def section_width(h5, s, idx):
    print("\n=== 1. scales and interface profile ===")
    print(f"L_ref = {s['L']:.4f} cells   U_ref = {s['U']:.6g}   T_ref = {s['T']:.1f} steps")
    print(f"1/Re = {s['one_Re']:.4f}   Eu = {s['Eu']:.4f}   Ko = {s['Ko']:.5f}   "
          f"drho* = {s['drho_star']:.3f}   1/Fr = {pr.FORCE_INCREMENT * s['L'] / s['U'] ** 2:.4f}")
    print("snapshot dt: 400 steps = %.3f T_ref (trainer uses every 2nd -> %.3f)"
          % (400 / s["T"], 800 / s["T"]))
    for i in idx:
        rho, _, _, _ = snapshot(h5, i, s)
        rho = rho * s["rho_ref"]
        Zg, Yg, Xg = np.indices(rho.shape, dtype=float)
        drop = rho > 0.5 * (pr.RHO_LIQ + pr.RHO_GAS)
        c = [g[drop].mean() for g in (Zg, Yg, Xg)]
        r = np.sqrt((Zg - c[0]) ** 2 + (Yg - c[1]) ** 2 + (Xg - c[2]) ** 2)
        bins = np.arange(0.0, 2.0 * s["L"], 0.25)
        which = np.digitize(r.ravel(), bins)
        prof = np.array([rho.ravel()[which == k].mean() if np.any(which == k) else np.nan
                         for k in range(1, len(bins))])
        rc = 0.5 * (bins[1:] + bins[:-1])
        rl, rg = np.nanmedian(prof[rc < 0.5 * s["L"]]), np.nanmedian(prof[rc > 1.5 * s["L"]])
        phi = (prof - rg) / (rl - rg)
        r90 = rc[np.argmax(phi < 0.9)]
        r10 = rc[np.argmax(phi < 0.1)]
        print(f"  t={int(s['times'][i])}: centroid z={c[0]:.1f}  rho_liq={rl:.3f} rho_gas={rg:.3f}  "
              f"10-90% width = {r10 - r90:.2f} cells = {(r10 - r90) / s['L']:.3f} L_ref  "
              f"(r90={r90:.1f}, r10={r10:.1f})  min/max rho = {rho.min():.3f}/{rho.max():.3f}")


def section_floor(h5, s, idx, ks=(1, 2), orders=(2, 4)):
    print("\n=== 2. residual floor: trainer losses evaluated on the CFD truth ===")
    print("per-point MSE (trainer logs SUM over 40 batches, i.e. ~40x these). "
          "'mix' ~ the trainer's (400, 2000, 3000) interface/near/domain sampling.")
    print("g(t) = nominal ramp. EOS uses the h5 pressure; m/u/v/w use p = P_PR(rho).")
    for i in idx:
        t = s["times"][i]
        g_star = pr.body_force(t) * s["L"] / s["U"] ** 2
        rho, vel, p_h5, ls = snapshot(h5, i, s)
        masks = regions(ls)
        eos = p_h5 - pr.pressure_pr(rho * s["rho_ref"]) / s["p_scale"]
        print(f"\n t = {int(t)}  (g* = {g_star:.4f})   EOS mse mix = {trainer_mix(region_mse(eos, masks)):.3e}")
        for k in ks:
            if i - k < 0 or i + k >= len(s["times"]):
                continue
            rho_m, vel_m, _, _ = snapshot(h5, i - k, s)
            rho_p, vel_p, _, _ = snapshot(h5, i + k, s)
            dt = 400.0 * k / s["T"]
            for order in orders:
                res, terms = residuals(rho_m, rho, rho_p, vel_m, vel, vel_p, dt, g_star, s, order)
                mse = {c: region_mse(f, masks) for c, f in res.items()}
                total = sum(W_PDE[c] * trainer_mix(mse[c]) for c in "muvw")
                print(f"  dt={400 * k:4d} steps, FD order {order}:  weighted PDE (mix) = {total:.3e}")
                for c in "muvw":
                    m = mse[c]
                    print(f"     {c}: interface {m['interface']:.3e}  near {m['near']:.3e}  "
                          f"domain {m['domain']:.3e}  mix {trainer_mix(m):.3e}")
                if k == ks[0]:
                    print("     term rms in band (|ls|<8) / domain (|ls|>=4):")
                    for name, f in terms.items():
                        print(f"       {name:<14} {rms(f, masks['band']):10.4e}  {rms(f, masks['domain']):10.4e}")


def section_velocity(h5, s, idx):
    print("\n=== 3. continuity / velocity ===")
    print("  t       rms div(u*)   rms div(u*) band   mean w*   rms |u - <u>|*")
    for i in range(len(s["times"])):
        _, vel, _, ls = snapshot(h5, i, s)
        div = sum(d1(vel[j], AX[c], 2) for j, c in enumerate("xyz")) * s["L"]
        band = np.abs(ls) < 8
        rel = np.sqrt(sum((v - v.mean()) ** 2 for v in vel))
        print(f"  {int(s['times'][i]):6d}  {rms(div, np.ones_like(band)):11.4e}  {rms(div, band):12.4e}  "
              f"{vel[2].mean():9.4e}  {np.sqrt(np.mean(rel ** 2)):9.4e}")
    print("\n  |u - <u>|* (rms) vs distance from drop centroid, r/L_ref:")
    edges = np.arange(0.0, 2.25, 0.25)
    print("  t      " + " ".join(f"{e:6.2f}" for e in edges[1:]))
    for i in idx:
        rho, vel, _, _ = snapshot(h5, i, s)
        Zg, Yg, Xg = np.indices(rho.shape, dtype=float)
        drop = rho * s["rho_ref"] > 0.5 * (pr.RHO_LIQ + pr.RHO_GAS)
        c = [g[drop].mean() for g in (Zg, Yg, Xg)]
        r = np.sqrt((Zg - c[0]) ** 2 + (Yg - c[1]) ** 2 + (Xg - c[2]) ** 2) / s["L"]
        rel = np.sqrt(sum((v - v.mean()) ** 2 for v in vel))
        row = [rms(rel, (r >= a) & (r < b)) for a, b in zip(edges[:-1], edges[1:])]
        print(f"  {int(s['times'][i]):6d} " + " ".join(f"{v:6.3f}" for v in row))


def section_smear(h5, s, i, sigmas=(0.0, 0.5, 1.0, 2.0, 3.0), order=2):
    print("\n=== 4. smearing test: blur rho (Gaussian, cells), keep the h5 velocities ===")
    print("  rho data loss uses the trainer's normalisation and its 500 band / 400 bulk mix.")
    print("  sigma   rho loss    m (mix)     u (mix)     w (mix)    weighted PDE (mix)")
    t = s["times"][i]
    g_star = pr.body_force(t) * s["L"] / s["U"] ** 2
    rho, vel, _, ls = snapshot(h5, i, s)
    rho_m, vel_m, _, _ = snapshot(h5, i - 1, s)
    rho_p, vel_p, _, _ = snapshot(h5, i + 1, s)
    masks = regions(ls)
    dt = 400.0 / s["T"]
    for sig in sigmas:
        blur = (lambda f: ndimage.gaussian_filter(f, sig, mode="wrap")) if sig > 0 else (lambda f: f)
        rb = blur(rho)
        err = ((rb - rho) / s["drho_star"]) ** 2
        rho_loss = (500 * err[masks["band"]].mean() + 400 * err[masks["bulk"]].mean()) / 900.0
        res, _ = residuals(blur(rho_m), rb, blur(rho_p), vel_m, vel, vel_p, dt, g_star, s, order)
        mix = {c: trainer_mix(region_mse(f, masks)) for c, f in res.items()}
        total = sum(W_PDE[c] * mix[c] for c in "muvw")
        print(f"  {sig:4.1f}   {rho_loss:.3e}   {mix['m']:.3e}   {mix['u']:.3e}   {mix['w']:.3e}   {total:.3e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", default=H5)
    ap.add_argument("--snapshots", default="4,13,22", help="h5 indices (need +-2 neighbours)")
    ap.add_argument("--raw-velocity", action="store_true",
                    help="use u_h5 as stored, without the F/(2 rho) correction (old trainer)")
    ap.add_argument("--beta", type=float, default=pr.LATTICE_BETA,
                    help="extra beta grad(lap(rho)) force, lattice units; 0 = old trainer")
    ap.add_argument("--skip", default="", help="comma list of sections to skip, e.g. 3,4")
    args = ap.parse_args()
    idx = [int(v) for v in args.snapshots.split(",")]
    skip = set(args.skip.split(","))
    global FORCE_CORRECT
    FORCE_CORRECT = not args.raw_velocity
    global BETA
    BETA = args.beta
    print("extra Korteweg-like term beta =", BETA)
    print("velocity:", "u_h5 + F/(2 rho)" if FORCE_CORRECT else "u_h5 as stored")

    with h5py.File(args.h5, "r") as h5:
        s = load_scales(h5)
        if "1" not in skip:
            section_width(h5, s, idx)
        if "2" not in skip:
            section_floor(h5, s, idx)
        if "3" not in skip:
            section_velocity(h5, s, idx)
        if "4" not in skip:
            section_smear(h5, s, idx[len(idx) // 2])


if __name__ == "__main__":
    main()
