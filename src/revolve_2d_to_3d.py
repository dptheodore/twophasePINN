''' Turns the 2D Hysing rising-bubble dataset (rising_bubble.h5) into a 3D bubble by
REVOLVING it about the vertical axis through the bubble centre (x = 0), so the circular
bubble becomes a sphere of radius 0.25 at (0, 0, -0.5) in a 1 x 1 x 2 box -- the 3D
version of Hysing case 1 (Adelsberger et al. 2014 use exactly this setup).

Only the LEVELSET is written. It is the only field the training consumes (the volume
fraction points), and it revolves exactly: the 2D levelset is a signed distance in 2D
grid cells, and the nearest point on a surface of revolution lies in the same meridional
plane, so ls3(x, y, z) = ls2(r, z) with r = sqrt(x^2 + y^2) is the 3D signed distance in
the same units. The grid keeps the 2D spacing (256 x 256 x 512), so every cell-unit
threshold and refinement width in generate_points_3d.py means what it does in 2D.

Velocity and pressure are NOT written. Revolving a planar flow does not give a 3D flow:
planar continuity u_x + v_y = 0 is not axisymmetric continuity (1/r) d(r u_r)/dr + w_z = 0,
and on snapshot 75 the revolved field has rms div(u) = 0.39 against 0.066 for the 2D data
itself (rms |u_x| = 0.55). Storing it would pass off a non-solution as 3D ground truth.

VOLUME. Revolving alone does not conserve the bubble's volume: the 2D bubble keeps its
AREA but flattens and widens as it rises, and revolving a wider shape sweeps more volume
(it grows 44% by t = 3). An incompressible bubble cannot do that, and training would pit
the volume-fraction data against continuity + advection. Each snapshot is therefore
scaled uniformly about the bubble's (volume) centroid by s = (V0 / V)^(1/3), restoring
the initial sphere's volume while keeping the 2D shape and the centroid's path. A signed
distance scales by the same factor, so ls3 = s * ls2(r / s, z_c + (z - z_c) / s); cells
the 2D field saturates at +-8 stay at +-8.

The interface MOTION is still the 2D bubble's: a real 3D bubble with these parameters
rises somewhat differently than the 2D (cylindrical) one, so this is a 3D-shaped bubble
following 2D dynamics, not a 3D simulation.

The corners of the box (r > 0.5) lie outside the 2D domain and take the r = 0.5 value
(liquid). Storage order (time, Z, Y, X); raw sign kept (negative inside the bubble). '''

import argparse
import h5py
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--src", default="../cfd_data/rising_bubble.h5")
    parser.add_argument("--dst", default="../cfd_data/rising_bubble_3d.h5")
    args = parser.parse_args()

    with h5py.File(args.src, "r") as src, h5py.File(args.dst, "w") as dst:
        X = src["X"][:].astype(float)
        Z = src["Y"][:]
        dx = X[1] - X[0]
        assert np.allclose(X, -X[::-1], atol=1e-6), "2D x grid must be symmetric about the bubble axis"
        assert np.allclose(np.diff(Z), dx, rtol=1e-4), "2D grid is not square"

        r = np.sqrt(np.add.outer(X ** 2, X ** 2))[None]          # (1, Y, X)
        z3 = Z.astype(float)[:, None, None]                       # (Z, 1, 1)
        ax_r = np.abs(X)                                          # 2D |x| = radius

        def volume_and_centroid(ls2):
            # Pappus: each 2D bubble cell at |x| sweeps 2 pi |x| dx^2; both halves of the
            # 2D field are counted, hence pi rather than 2 pi.
            weight = np.pi * ax_r[None, :] * dx ** 2 * (ls2 < 0)
            return weight.sum(), (weight.sum(axis=1) * Z).sum() / weight.sum()

        def bilinear(F, zi, xi):
            z0 = np.clip(np.floor(zi).astype(np.int32), 0, F.shape[0] - 2)
            x0 = np.clip(np.floor(xi).astype(np.int32), 0, F.shape[1] - 2)
            wz = np.clip(zi - z0, 0, 1).astype(np.float32)
            wx = np.clip(xi - x0, 0, 1).astype(np.float32)
            return ((F[z0, x0] * (1 - wx) + F[z0, x0 + 1] * wx) * (1 - wz) +
                    (F[z0 + 1, x0] * (1 - wx) + F[z0 + 1, x0 + 1] * wx) * wz)

        nt = len(src["time"])
        dst["X"] = X.astype(np.float32)
        dst["Y"] = X.astype(np.float32)
        dst["Z"] = Z
        dst["time"] = src["time"][:]
        out = dst.create_dataset("levelset", (nt, len(Z), len(X), len(X)), dtype="f4",
                                 chunks=(1, 64, 64, len(X)), compression="gzip",
                                 compression_opts=4, shuffle=True)
        scale = dst.create_dataset("volume_scale", (nt,), dtype="f8")
        V0, _ = volume_and_centroid(src["levelset"][0])
        for n in range(nt):
            ls2 = src["levelset"][n]
            V, z_c = volume_and_centroid(ls2)
            s = (V0 / V) ** (1.0 / 3.0)
            # source point in the 2D field, as fractional (z, x) indices
            xi = (r / s - X[0]) / dx
            zi = (z_c + (z3 - z_c) / s - Z[0]) / dx
            ls3 = s * bilinear(ls2, zi, xi)
            ls3[np.abs(ls3) >= 8 * s - 1e-4] = np.sign(ls3[np.abs(ls3) >= 8 * s - 1e-4]) * 8
            out[n] = ls3
            scale[n] = s
            if n % 25 == 0:
                print("snapshot %3d  V/V0 before scaling %.4f  s %.4f" % (n, V / V0, s))

        dst.attrs["source"] = ("rising_bubble.h5 revolved about x = 0 (2D y -> 3D z), each snapshot "
                               "scaled about its centroid by volume_scale to conserve volume; levelset only")
        dst.attrs["physics"] = ("Hysing case 1 in 3D: rho = [bubble 100, liquid 1000], mu = [1, 10], "
                                "sigma = 24.5, g = -0.98 along Z, SI units")
        dst.attrs["boundaries"] = "walls on all faces, as in 2D: no-slip top/bottom, sides as the 2D east/west"
    print("wrote %s, levelset shape %s" % (args.dst, (nt, len(Z), len(X), len(X))))


if __name__ == "__main__":
    main()
