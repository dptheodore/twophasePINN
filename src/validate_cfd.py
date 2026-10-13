"""Check whether a CFD dataset satisfies the equations 3d_rising_bubble_train.py solves.

Every check here corresponds to a way the previous dataset silently disagreed with the
PINN. Run it before training; a red line means no amount of loss weighting will help.

    python validate_cfd.py --h5 ../cfd_data/lbpm_bubble_dataset.h5
"""
import argparse
import numpy as np
import h5py

OK, BAD = "  OK  ", " FAIL "


def report(tag, passed, detail):
    print(f"[{OK if passed else BAD}] {tag:<34} {detail}")
    return passed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", default="../cfd_data/lbpm_bubble_dataset.h5")
    ap.add_argument("--rhoA", type=float, default=0.1, help="bubble density")
    ap.add_argument("--rhoB", type=float, default=1.0, help="surrounding density")
    ap.add_argument("--nu", type=float, default=0.042862)
    ap.add_argument("--Fz", type=float, default=-2.46914e-05)
    a = ap.parse_args()

    f = h5py.File(a.h5, "r")
    ls = -np.array(f["levelset"])
    pr = np.array(f["pressure"], dtype=float)
    t = np.array(f["time"], dtype=float)
    vx, vy, vz = (np.array(f["velocity" + c], dtype=float) for c in "XYZ")
    X, Y, Z = (np.array(f[c], dtype=float) for c in "XYZ")
    mid = len(t) // 2
    settled = slice(len(t) // 3, None)
    results = []

    print(f"\n{a.h5}\n  {len(t)} snapshots, t {t[0]:g}->{t[-1]:g}, grid {ls.shape[1:]} (Z,Y,X)\n")

    # bubble = minority phase, regardless of sign convention
    sgn = 1.0 if (ls[mid] > 0).mean() < 0.5 else -1.0
    bub = lambda k: (sgn * ls[k]) >= 0
    far = lambda k: (sgn * ls[k]) <= -6

    # ---- 1. grid aspect -------------------------------------------------------
    dx, dz = np.diff(X)[0], np.diff(Z)[0]
    results.append(report("grid axes cubic", abs(dz / dx - 1) < 1e-3,
                          f"dz/dx = {dz/dx:.3f}   (Domain/L must match the lattice aspect)"))

    # ---- 2. mass conservation -> no phase change ------------------------------
    vol = np.array([float(bub(k).sum()) for k in range(len(t))])
    s = vol[settled]
    drift = (s[-1] - s[0]) / s[0] * 100
    R = (3 * np.median(s) / (4 * np.pi)) ** (1 / 3)
    D = 2 * R
    results.append(report("bubble mass conserved", abs(drift) < 2.0,
                          f"volume drift {drift:+.2f}%   R = {R:.2f} cells"
                          f"   {'' if abs(drift)<2 else '<- phase change'}"))

    # ---- 3. incompressibility -------------------------------------------------
    u, v, w = vx[mid], vy[mid], vz[mid]
    ux = (u[1:-1, 1:-1, 2:] - u[1:-1, 1:-1, :-2]) / 2
    vy_ = (v[1:-1, 2:, 1:-1] - v[1:-1, :-2, 1:-1]) / 2
    wz = (w[2:, 1:-1, 1:-1] - w[:-2, 1:-1, 1:-1]) / 2
    core = ls[mid][1:-1, 1:-1, 1:-1]
    gas = (sgn * core) <= -6
    div = ux + vy_ + wz
    ratio = np.sqrt((div[gas] ** 2).mean()) / np.sqrt(
        (ux[gas] ** 2 + vy_[gas] ** 2 + wz[gas] ** 2).mean())
    results.append(report("incompressible (div u = 0)", ratio < 0.3,
                          f"rms(div u)/rms(terms) = {ratio:.3f} in bulk   (want << 1)"))

    # ---- 4. interface advected by the flow ------------------------------------
    dt = t[mid + 1] - t[mid - 1]
    lst = (-np.array(f["levelset"][mid + 1]) - (-np.array(f["levelset"][mid - 1]))) / dt
    c = ls[mid]
    lx = (c[1:-1, 1:-1, 2:] - c[1:-1, 1:-1, :-2]) / 2
    ly = (c[1:-1, 2:, 1:-1] - c[1:-1, :-2, 1:-1]) / 2
    lz = (c[2:, 1:-1, 1:-1] - c[:-2, 1:-1, 1:-1]) / 2
    ui, vi, wi = u[1:-1, 1:-1, 1:-1], v[1:-1, 1:-1, 1:-1], w[1:-1, 1:-1, 1:-1]
    band = np.abs(core) <= 3
    res = lst[1:-1, 1:-1, 1:-1] + ui * lx + vi * ly + wi * lz
    sig_ = np.sqrt((lst[1:-1, 1:-1, 1:-1][band] ** 2).mean())
    rr = np.sqrt((res[band] ** 2).mean()) / max(sig_, 1e-30)
    results.append(report("interface advected by u", rr < 0.4,
                          f"residual/signal = {rr:.3f}   (a_t + u.grad a = 0)"))

    # ---- 5. buoyancy: does the bubble move relative to the fluid? -------------
    slip = np.array([vz[k][bub(k)].mean() - vz[k][far(k)].mean() for k in range(len(t))])
    sl = float(np.mean(slip[settled]))
    gp = abs(a.Fz) * abs(a.rhoB - a.rhoA) / a.rhoB
    U = np.sqrt(gp * D)
    results.append(report("real buoyant slip", abs(sl) > 0.3 * U,
                          f"slip {sl:+.5f} vs sqrt(g'D) {U:.5f}   ({abs(sl)/U*100:.1f}% of expected)"))

    # ---- 6. deformation --------------------------------------------------------
    asp = []
    for k in range(len(t) // 3, len(t)):
        idx = np.argwhere(bub(k))
        e = idx.max(0) - idx.min(0) + 1
        asp.append(e[0] / e[2])
    dev = max(abs(np.array(asp) - 1))
    results.append(report("bubble deforms", dev > 0.05,
                          f"max |aspect-1| = {dev:.3f}   (a rigid sphere means no dynamics)"))

    # ---- 7. which body force does the momentum balance contain? ---------------
    p_scale = a.rhoB * U ** 2
    grads = []
    for k in range(len(t) // 3, len(t)):
        zc = np.argwhere(bub(k))[:, 0].mean()
        prof, zs = [], []
        for z in range(ls.shape[1]):
            if abs(z - zc) < 2.2 * R:
                continue
            m = (sgn * ls[k, z]) <= -6
            if m.sum() < 500:
                continue
            prof.append(pr[k, z][m].mean()); zs.append(z)
        if len(zs) > 10:
            grads.append(np.polyfit(zs, prof, 1)[0])
    if grads:
        meas = float(np.median(grads)) * R / p_scale
        one_Fr = a.Fz * R / U ** 2
        cands = {"full rho*g (2D form)": (a.rhoB / a.rhoB) * one_Fr,
                 "reduced (rho-<rho>)*g": None,
                 "no body force": 0.0}
        full = cands["full rho*g (2D form)"]
        best = "full rho*g" if abs(full - meas) < abs(meas) else "no body force"
        results.append(report("body force in momentum", abs(full - meas) < 0.5 * abs(full) + 1e-9,
                              f"dp/dz {meas:+.4f} nondim; full rho*g needs {full:+.4f}"
                              f"  -> data says '{best}'"))

    # ---- 8. surface tension ----------------------------------------------------
    offset = (a.rhoA - a.rhoB) / 3.0
    jumps = []
    for k in range(len(t) // 3, len(t)):
        zc = int(round(np.argwhere(bub(k))[:, 0].mean()))
        pl, l0 = pr[k, zc], ls[k][zc]
        pin, pout = pl[(sgn * l0) >= 7.5], pl[(sgn * l0) <= -7.5]
        if len(pin) > 10 and len(pout) > 10:
            jumps.append(pin.mean() - pout.mean())
    if jumps:
        dp = float(np.median(jumps))
        sigma = (dp - offset) * R / 2
        print(f"\n  surface tension: dp {dp:+.6g}, minus EOS offset {offset:+.6g}"
              f"  ->  sigma ~ {sigma:.6g}")
        if sigma > 0:
            muB = a.rhoB * a.nu
            drho = abs(a.rhoB - a.rhoA)
            print(f"  Eo {drho*abs(a.Fz)*D**2/sigma:8.2f}  (2D 9.0)")
            print(f"  Mo {abs(a.Fz)*muB**4*drho/(a.rhoB**2*sigma**3):8.3e}  (2D 6.0e-4)")
            print(f"  Re {a.rhoB*U*D/muB:8.1f}  (2D 33.2)     Ma {U/0.5774:.4f}")
        else:
            print("  (non-positive -- use calibrate_sigma.py with 2+ radii)")

    print(f"\n{'='*72}")
    n = sum(results)
    print(f"{n}/{len(results)} checks passed.",
          "Data is consistent with the PINN's equations." if n == len(results)
          else "Fix the FAILing rows before training -- loss weighting cannot compensate.")


if __name__ == "__main__":
    main()
