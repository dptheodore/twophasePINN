''' Converts the raw snapshots of cfd/basilisk_rising_bubble_3d/rising3d.c into the h5
the 3D trainer reads (generate_points_3d.H5_PATH layout, like rising_bubble_3d.h5):

    X, Y, Z      cell centres, x, y in [-0.5, 0.5], z in [-1, 1] (vertical)
    time         snapshot times
    levelset     signed distance in CELLS, clipped to +-8, NEGATIVE in the bubble
                 (the 2D rising_bubble.h5 convention, so the generator's 0.75 / 4 / 8
                 cell thresholds mean what they do in 2D)
    velocityX/Y/Z, pressure
    volume_fraction  liquid fraction f from the VOF (1 = liquid, 0 = bubble)
    split        b"train" / b"validation" per snapshot
all (time, Z, Y, X) float32, gzip-compressed.

Pressure gauge: Basilisk's pressure is defined up to a constant (periodic x, y and
Neumann in z). Each snapshot is shifted so the mean over the top cell layer is 1000,
i.e. p* = p / (rho_ref U_ref^2) = 1 at the north boundary, the ambient pressure the
paper (and the trainer's p_N) prescribes. The full hydrostatic pressure is kept.

Levelset from the volume fraction: distance to the f = 0.5 isosurface (marching cubes),
periodic in x and y. On the initial sphere at 32 cells across it is within 0.13 cells
of the exact distance (rms 0.07), with no sign errors.

    python basilisk_to_h5.py RUN_DIR [--dst ../cfd_data/rising_bubble_3d_basilisk.h5] '''

import argparse
import os

import h5py
import numpy as np
from scipy import ndimage

from generate_points_3d import TRAIN_TIMES
CLIP = 8.0
P_TOP = 1000.0


def _triangle_samples(verts, faces, n=4):
    ''' Points on a barycentric grid of n subdivisions per triangle edge, so the surface
    is sampled at ~1/n of the marching-cubes edge length. '''
    bary = np.array([(i, j, n - i - j) for i in range(n + 1) for j in range(n + 1 - i)], float) / n
    tri = verts[faces]                                    # (T, 3 corners, 3 coords)
    return np.einsum("bk,tkc->tbc", bary, tri).reshape(-1, 3)


def levelset_from_fraction(f):
    ''' f: liquid fraction (Z, Y, X). Returns the signed distance in cells, < 0 in the bubble.

    Distance to the f = 0.5 isosurface (marching cubes on the trilinear f), sampled densely
    and queried with a KD-tree; the sign is the phase of the cell. A cell-centre estimate
    such as (f - 0.5) / |grad f| is up to ~0.85 cells off here, because the VOF interface
    is barely a cell thick. '''
    from scipy.spatial import cKDTree
    from skimage.measure import marching_cubes

    bubble = f < 0.5
    pad = int(CLIP) + 2
    # periodic in x and y, walls (edge padding) in z
    wrap = lambda a: np.pad(np.pad(a, ((0, 0), (pad, pad), (pad, pad)), mode="wrap"),
                            ((pad, pad), (0, 0), (0, 0)), mode="edge")
    crop = (slice(pad, -pad),) * 3

    # Coarse distance to pick the cells worth an exact query.
    b = wrap(bubble)
    coarse = np.where(bubble, ndimage.distance_transform_edt(b)[crop],
                      ndimage.distance_transform_edt(~b)[crop])
    ls = np.where(bubble, -CLIP, CLIP).astype(np.float64)
    query = coarse < CLIP + 2
    if not bubble.any() or bubble.all():
        return ls.astype(np.float32)

    verts, faces, _, _ = marching_cubes(wrap(f.astype(np.float64)), 0.5)
    tree = cKDTree(_triangle_samples(verts, faces))
    cells = np.argwhere(query) + pad                      # padded (z, y, x) index = position
    dist, _ = tree.query(cells, distance_upper_bound=CLIP + 1)
    dist = np.minimum(dist, CLIP)
    ls[query] = np.where(bubble[query], -dist, dist)
    return np.clip(ls, -CLIP, CLIP).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run_dir")
    ap.add_argument("--dst", default="../cfd_data/rising_bubble_3d_basilisk.h5")
    ap.add_argument("--delete-raw", action="store_true", help="remove each .bin once written")
    args = ap.parse_args()

    snaps = np.loadtxt(os.path.join(args.run_dir, "snapshots.txt"), ndmin=2)
    idx, times = snaps[:, 0].astype(int), np.round(snaps[:, 1], 6)
    nx, ny, nz = (int(v) for v in snaps[0, 2:5])
    dx = 1.0 / nx
    X = ((np.arange(nx) + 0.5) * dx - 0.5).astype(np.float32)
    Y = ((np.arange(ny) + 0.5) * dx - 0.5).astype(np.float32)
    Z = ((np.arange(nz) + 0.5) * dx - 1.0).astype(np.float32)
    split = [b"train" if np.isclose(t, TRAIN_TIMES, atol=1e-6).any() else b"validation" for t in times]

    shape = (len(times), nz, ny, nx)
    chunks = (1, min(nz, 64), ny, nx)
    names = dict(f="volume_fraction", u="velocityX", v="velocityY", w="velocityZ", p="pressure")
    with h5py.File(args.dst, "w") as h5:
        h5["X"], h5["Y"], h5["Z"], h5["time"] = X, Y, Z, times
        h5["split"] = np.array(split)
        ds = {k: h5.create_dataset(n, shape, np.float32, chunks=chunks, compression="gzip",
                                   compression_opts=4, shuffle=True) for k, n in names.items()}
        ds_ls = h5.create_dataset("levelset", shape, np.float32, chunks=chunks,
                                  compression="gzip", compression_opts=4, shuffle=True)
        h5.attrs["source"] = "Basilisk, cfd/basilisk_rising_bubble_3d/rising3d.c, uniform grid"
        h5.attrs["physics"] = ("3D version of the paper's rising bubble (Hysing case 1): liquid "
                               "rho 1000 mu 10, bubble rho 100 mu 1, sigma 24.5, g 0.98 along -Z, SI")
        h5.attrs["boundaries"] = "no-slip at Z = -1 and Z = 1, periodic in X and Y"
        h5.attrs["pressure_gauge"] = "mean over the top cell layer = 1000 (p* = 1 at the north wall)"

        for k, (n, t) in enumerate(zip(idx, times)):
            raw = {c: np.fromfile(os.path.join(args.run_dir, f"snap_{n:03d}_{c}.bin"), np.float32)
                   .reshape(nz, ny, nx) for c in "fuvwp"}
            raw["p"] = raw["p"] - raw["p"][-1].mean() + P_TOP
            for c in "fuvwp":
                ds[c][k] = raw[c]
            ds_ls[k] = levelset_from_fraction(raw["f"])
            vol = (1.0 - raw["f"]).sum() * dx ** 3
            print(f"t = {t:.2f} ({split[k].decode()}): bubble volume {vol:.6f}, "
                  f"max |u| {max(np.abs(raw[c]).max() for c in 'uvw'):.3f}")
            if args.delete_raw:
                for c in "fuvwp":
                    os.remove(os.path.join(args.run_dir, f"snap_{n:03d}_{c}.bin"))
    print(f"wrote {args.dst}: {len(times)} snapshots ({split.count(b'train')} train, "
          f"{split.count(b'validation')} validation), grid {nx} x {ny} x {nz}")


if __name__ == "__main__":
    main()
