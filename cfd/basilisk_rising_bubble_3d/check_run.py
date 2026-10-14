''' Benchmark quantities of a finished run, compared with Adelsberger et al. (2014),
test case 1 (the 3D extension of Hysing case 1).

    python check_run.py run_L8/log.txt ../../cfd_data/rising_bubble_3d_basilisk_L8.h5

The benchmark has no-slip walls on all six faces; this case has periodic sides, as in
the paper's 2D setup. So the numbers are a close check, not an exact one: the periodic
neighbours sit 1.0 apart (centre to centre), where the benchmark's walls are 0.5 from the
centre.

    python check_run.py --strict run_walls_L6/log.txt   # benchmark-mode gate, exit 1 on failure

--strict is for a WALLS=1 (benchmark-mode) run: it fails unless the peak rise velocity is
within 0.345-0.365 at t = 0.8-1.0 and the volume drifts by less than 0.5%. LEVEL 6 gives
0.3585 at t = 0.89 and -0.03% (benchmark: 0.352-0.358 at t ~ 0.9).

From log.txt (every step): bubble volume, centroid z, rise velocity.
From the h5 (stored snapshots): diameters (bubble extent in x, y, z) and sphericity
Psi = pi^(1/3) (6 V)^(2/3) / A, with A the area of the f = 0.5 surface. '''

import argparse

import h5py
import numpy as np
from skimage.measure import marching_cubes, mesh_surface_area

# Adelsberger et al. 2014, test case 1 (range over the DROPS / NaSt3DGPF / OpenFOAM results)
BENCH = dict(v_max=(0.352, 0.358), t_v_max=0.9, v_final=(0.33, 0.35),
             d_horizontal=0.58, d_vertical=(0.355, 0.37), sphericity_final=(0.955, 0.96))


STRICT = dict(v_max=(0.345, 0.365), t_v_max=(0.8, 1.0), drift=0.005)


def main(log_path, h5_path=None, strict=False):
    log = np.loadtxt(log_path)
    t, vol, zc, w = log[:, 1], log[:, 3], log[:, 4], log[:, 5]
    k = np.argmax(w)
    v0 = 4.0 / 3.0 * np.pi * 0.25 ** 3
    print(f"volume: start {vol[0]:.6f} (sphere {v0:.6f}), end {vol[-1]:.6f}, "
          f"drift {100 * (vol[-1] / vol[0] - 1):+.3f}%")
    print(f"rise velocity: max {w[k]:.4f} at t = {t[k]:.2f} "
          f"(benchmark {BENCH['v_max'][0]}-{BENCH['v_max'][1]} at t ~ {BENCH['t_v_max']}); "
          f"t = 3: {w[-1]:.4f} (benchmark {BENCH['v_final'][0]}-{BENCH['v_final'][1]})")
    print(f"centroid z: start {zc[0]:.4f}, t = 3: {zc[-1]:.4f} (risen {zc[-1] - zc[0]:.4f})")
    if strict:
        ok = (t[-1] > 2.999 and STRICT["v_max"][0] <= w[k] <= STRICT["v_max"][1] and
              STRICT["t_v_max"][0] <= t[k] <= STRICT["t_v_max"][1] and
              abs(vol[-1] / vol[0] - 1) < STRICT["drift"])
        print("BENCHMARK CHECK " + ("PASSED" if ok else "FAILED") +
              f" (needs peak {STRICT['v_max']} at t in {STRICT['t_v_max']}, drift < "
              f"{100 * STRICT['drift']}%, run reaching t = 3)")
        if not ok:
            raise SystemExit(1)

    if h5_path:
        with h5py.File(h5_path, "r") as h5:
            X = h5["X"][:]
            dx = float(X[1] - X[0])
            for i in (0, len(h5["time"]) - 1):
                f = h5["volume_fraction"][i]
                b = np.argwhere(f < 0.5)
                ext = (b.max(axis=0) - b.min(axis=0) + 1) * dx      # (z, y, x)
                verts, faces, _, _ = marching_cubes(f, 0.5, spacing=(dx, dx, dx))
                area = mesh_surface_area(verts, faces)
                V = (1.0 - f).sum() * dx ** 3
                psi = np.pi ** (1 / 3) * (6 * V) ** (2 / 3) / area
                print(f"t = {h5['time'][i]:.2f}: diameters x {ext[2]:.3f}, y {ext[1]:.3f}, "
                      f"z {ext[0]:.3f}; sphericity {psi:.4f}")
        print(f"benchmark at t = 3: horizontal ~{BENCH['d_horizontal']}, vertical "
              f"{BENCH['d_vertical'][0]}-{BENCH['d_vertical'][1]}, sphericity "
              f"{BENCH['sphericity_final'][0]}-{BENCH['sphericity_final'][1]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("log")
    ap.add_argument("h5", nargs="?")
    ap.add_argument("--strict", action="store_true", help="benchmark gate for a WALLS=1 run")
    a = ap.parse_args()
    main(a.log, a.h5, a.strict)
