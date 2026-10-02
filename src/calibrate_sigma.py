"""Calibrate the colour-model `alpha` parameter against a target surface tension.

`Color.alpha` in LBPM is a colour-gradient coefficient, not sigma -- the mapping is
lattice dependent, so it has to be measured. Run a STATIC drop (Color.F = 0,0,0), point
this script at the resulting h5, and it reports the sigma you actually got plus the
alpha needed to reach the target.

    python calibrate_sigma.py --h5 ../cfd_data/static_drop.h5 --alpha 1e-2

Two wrinkles this handles that a naive Laplace measurement gets wrong:

1. With eos_type = "standard" the pressure is p = rho*cs^2 = rho/3, so the two bulk
   phases sit at DIFFERENT pressures purely from their densities. The measured jump is
       dp = (rhoA - rhoB)/3  +  2*sigma/R
   and the EOS offset must come out before the Laplace part is visible. With a 10:1
   density ratio that offset is ~0.3 while the Laplace term is ~4e-4, so ignoring it
   gives an answer wrong by three orders of magnitude.
2. The robust way to separate them is to fit dp against 1/R over several radii: the
   slope is 2*sigma and the intercept absorbs the offset. Pass several runs to do that.
   With a single run the script falls back to subtracting the analytic offset, which is
   far more sensitive to how well the bulk densities have equilibrated.

Multi-radius (preferred):
    python calibrate_sigma.py --alpha 1e-2 --h5 r15.h5:15 r22.h5:22.5 r30.h5:30
"""
import argparse
import sys

import h5py
import numpy as np

CS2 = 1.0 / 3.0          # lattice speed of sound squared


def load(path):
    with h5py.File(path, "r") as f:
        d = dict(levelset=-np.array(f["levelset"]),
                 pressure=np.array(f["pressure"], dtype=float),
                 time=np.array(f["time"], dtype=float))
        for c in "XYZ":
            d[c] = np.array(f[c], dtype=float)
        for c in "XYZ":
            k = "velocity" + c
            if k in f:
                d[k] = np.array(f[k], dtype=float)
    return d


def analyse(path, args):
    d = load(path)
    ls, pr, t = d["levelset"], d["pressure"], d["time"]

    # Identify the drop as the MINORITY phase rather than trusting the sign convention,
    # which is not guaranteed to survive a change of LBPM model.
    frac_pos = float((ls[len(ls) // 2] > 0).mean())
    drop_is_positive = frac_pos < 0.5
    sgn = 1.0 if drop_is_positive else -1.0
    inside = lambda a: (sgn * a) >= 7.5
    outside = lambda a: (sgn * a) <= -7.5

    print(f"\n--- {path} ---")
    print(f"  snapshots {len(t)}   t {t[0]:g} -> {t[-1]:g}   grid {ls.shape[1:]} (Z,Y,X)")
    print(f"  drop is the {'positive' if drop_is_positive else 'negative'} levelset phase"
          f"  ({100*min(frac_pos,1-frac_pos):.1f}% of the volume)")

    # dz/dx: catches the Domain/L aspect-ratio bug
    dx, dz = np.diff(d["X"])[0], np.diff(d["Z"])[0]
    print(f"  dz/dx = {dz/dx:.3f}" + ("" if abs(dz / dx - 1) < 1e-3 else "   <-- AXES NOT CUBIC"))

    # is it actually static?
    if "velocityZ" in d:
        vmax = max(np.abs(d["velocity" + c]).max() for c in "XYZ")
        verdict = "static" if vmax < 1e-3 else "NOT STATIC -- rerun with Color.F = 0,0,0"
        print(f"  max |velocity| = {vmax:.3e}   {verdict}")

    # volume conservation -> no phase change (validates the immiscible model switch)
    vol = np.array([float((sgn * s >= 0).sum()) for s in ls])
    settled = vol[len(vol) // 3:]
    drift = (settled[-1] - settled[0]) / settled[0] * 100
    print(f"  drop volume {settled.mean():.0f} cells, drift over run {drift:+.2f}%"
          + ("" if abs(drift) < 2 else "   <-- MASS NOT CONSERVED (phase change?)"))
    R = (3 * np.median(settled) / (4 * np.pi)) ** (1 / 3)

    # sphericity
    idx = np.argwhere(sgn * ls[-1] >= 0)
    ext = idx.max(0) - idx.min(0) + 1
    print(f"  R = {R:.3f} cells   bbox (z,y,x) = {tuple(int(e) for e in ext)}"
          f"   aspect z/x = {ext[0]/ext[2]:.3f}")

    # pressure jump on the plane through the drop centroid (kills any hydrostatic ramp)
    jumps = []
    for k in range(len(t) // 3, len(t)):
        pts = np.argwhere(sgn * ls[k] >= 0)
        if not len(pts):
            continue
        zc = int(round(pts[:, 0].mean()))
        pl, lsl = pr[k, zc], ls[k][zc]
        pin, pout = pl[inside(lsl)], pl[outside(lsl)]
        if len(pin) > 10 and len(pout) > 10:
            jumps.append(pin.mean() - pout.mean())
    if not jumps:
        print("  could not find deep-phase cells on the centroid plane")
        return None
    dp = float(np.median(jumps))
    print(f"  measured dp across the interface = {dp:+.6g}  (spread {np.std(jumps):.2g})")
    return dict(R=R, dp=dp)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--h5", nargs="+", required=True,
                    help="h5 file(s). Use path:R to give the nominal radius for a multi-radius fit.")
    ap.add_argument("--alpha", type=float, required=True, help="Color.alpha used for these runs")
    ap.add_argument("--target-sigma", type=float, default=5.0e-3)
    ap.add_argument("--rhoA", type=float, default=0.1, help="bubble density (Color.rhoA)")
    ap.add_argument("--rhoB", type=float, default=1.0, help="surrounding density (Color.rhoB)")
    ap.add_argument("--nu", type=float, default=0.042862, help="(tau-0.5)/3")
    ap.add_argument("--Fz", type=float, default=-2.46914e-05, help="production body force")
    args = ap.parse_args()

    results = []
    for spec in args.h5:
        path, _, rnom = spec.partition(":")
        r = analyse(path, args)
        if r:
            r["R_nominal"] = float(rnom) if rnom else r["R"]
            results.append(r)
    if not results:
        sys.exit("no usable runs")

    print("\n" + "=" * 68)
    offset = (args.rhoA - args.rhoB) * CS2
    if len(results) >= 2:
        # dp = offset + 2 sigma / R  ->  slope of dp vs 1/R is 2 sigma
        x = np.array([1.0 / r["R"] for r in results])
        y = np.array([r["dp"] for r in results])
        slope, intercept = np.polyfit(x, y, 1)
        sigma = slope / 2.0
        print(f"multi-radius fit over {len(results)} runs:")
        print(f"  dp = {intercept:+.6g} + {slope:.6g}/R   ->  sigma = {sigma:.6g}")
        print(f"  fitted intercept {intercept:+.6g}  vs analytic EOS offset {offset:+.6g}"
              f"   (agreement is a good sanity check)")
    else:
        r = results[0]
        sigma = (r["dp"] - offset) * r["R"] / 2.0
        print(f"single-radius estimate (subtracting the analytic EOS offset):")
        print(f"  sigma = (dp - (rhoA-rhoB)/3) * R/2 = ({r['dp']:+.6g} - {offset:+.6g})"
              f" * {r['R']:.3f}/2 = {sigma:.6g}")
        print(f"  WARNING: the offset {offset:+.6g} is ~{abs(offset/max(sigma,1e-12)):.0f}x the")
        print(f"           Laplace term, so this is very sensitive to bulk equilibration.")
        print(f"           Prefer 2+ radii:  --h5 r15.h5:15 r22.h5:22.5 r30.h5:30")

    if sigma <= 0:
        print("\n  sigma came out non-positive -- the jump is dominated by the EOS offset.")
        print("  Use the multi-radius fit; a single run cannot resolve it.")
        return

    scale = args.target_sigma / sigma
    print(f"\n  measured sigma {sigma:.6g}   target {args.target_sigma:.6g}   ratio {scale:.4f}")
    print(f"  ==> set  Color.alpha = {args.alpha * scale:.6g}   (was {args.alpha:g})")
    print("      alpha->sigma is close to linear in colour models; re-run this once to confirm.")

    # what the production run will then give
    R = args.target_sigma and results[0]["R"]
    D = 2 * R
    drho = abs(args.rhoB - args.rhoA)
    g = abs(args.Fz)
    muB = args.rhoB * args.nu
    Eo = drho * g * D ** 2 / args.target_sigma
    Mo = g * muB ** 4 * drho / (args.rhoB ** 2 * args.target_sigma ** 3)
    U = np.sqrt(g * drho / args.rhoB * D)
    print(f"\n  at the target sigma, with R={R:.2f} and Fz={args.Fz:g}:")
    print(f"    {'Eo':>4} {Eo:9.2f}   (2D benchmark 9.0)")
    print(f"    {'Mo':>4} {Mo:9.3e}   (2D benchmark 6.0e-4)")
    print(f"    {'Re':>4} {args.rhoB*U*D/muB:9.1f}   (2D benchmark 33.2)")
    print(f"    {'Ma':>4} {U/0.5774:9.4f}   (keep below ~0.1)")


if __name__ == "__main__":
    main()
