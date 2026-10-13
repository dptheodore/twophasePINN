''' Turns the 2D Hysing rising-bubble dataset (rising_bubble.h5) into a 3D volume in
the layout generate_points_3d.py reads, as a correctness check for the 3D PINN.

The 2D field is EXTRUDED along a new horizontal axis, not revolved. A planar flow
copied along y with w_y = 0 is an exact solution of the 3D incompressible
Navier-Stokes equations used in pytorch_3d_rising_bubble_train.py: every d/dy term
vanishes, the extruded velocity component is identically zero, and the 3D curvature
reduces to the 2D one. Any nonzero 3D residual on this data is therefore a bug in
the 3D equations, not physics. (Revolving about the bubble axis would give a sphere
but not a solution: planar div(u) = 0 is not axisymmetric div(u) = 0.)

Axis mapping, chosen so gravity lands in the w equation as in the 3D script:
    2D x (horizontal)  -> 3D X
    2D y (vertical)    -> 3D Z
    new extruded axis  -> 3D Y       (velocityY = 0)
Storage order is (time, Z, Y, X), matching lbpm_bubble_dataset.h5. The levelset sign
is left raw (negative inside the bubble), which both generators negate so that a = 1
is the bubble.

Every Y slice is identical, so the gzip'd file is not much bigger than the 2D one,
but the 3D generator loads levelset and pressure in full: the in-memory size scales
with --ny (32 -> ~2.5 GB levelset + ~5 GB float64 pressure). '''

import argparse
import h5py
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--src", default="../cfd_data/rising_bubble.h5")
    parser.add_argument("--dst", default="../cfd_data/rising_bubble_extruded_3d.h5")
    parser.add_argument("--ny", type=int, default=32, help="cells along the extruded axis")
    args = parser.parse_args()

    with h5py.File(args.src, "r") as src, h5py.File(args.dst, "w") as dst:
        X = src["X"][:]
        Z = src["Y"][:]
        dx = float(np.diff(X)[0])
        assert np.allclose(np.diff(X), dx) and np.allclose(np.diff(Z), dx, rtol=1e-4), \
            "2D grid is not uniform/square, extrusion would give non-cubic cells"
        Y = ((np.arange(args.ny) + 0.5 - args.ny / 2) * dx).astype(X.dtype)
        nt = len(src["time"])

        dst["X"], dst["Y"], dst["Z"] = X, Y, Z
        dst["time"] = src["time"][:]

        shape = (nt, len(Z), args.ny, len(X))
        chunks = (1, 64, args.ny, len(X))
        mapping = {"levelset": "levelset", "pressure": "pressure",
                   "velocityX": "velocityX", "velocityZ": "velocityY"}
        out = {name: dst.create_dataset(name, shape, dtype="f4", chunks=chunks,
                                        compression="gzip", compression_opts=4, shuffle=True)
               for name in list(mapping) + ["velocityY"]}

        for i in range(nt):
            for name3, name2 in mapping.items():
                out[name3][i] = np.broadcast_to(src[name2][i][:, None, :], shape[1:])
            out["velocityY"][i] = np.zeros(shape[1:], dtype="f4")

        dst.attrs["source"] = "rising_bubble.h5 extruded along Y (2D y -> 3D Z, 2D v -> velocityZ)"
        dst.attrs["physics"] = ("Hysing case 1: rho = [bubble 100, liquid 1000], mu = [1, 10], "
                                "sigma = 24.5, g = -0.98 along Z, SI units")
        dst.attrs["boundaries"] = ("X: free-slip walls, Z: no-slip walls (from the 2D run); "
                                   "Y: periodic (extrusion)")
    print("wrote %s with shape %s" % (args.dst, shape))


if __name__ == "__main__":
    main()
