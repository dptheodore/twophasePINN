''' Precomputes the geometric-constraint regions of the 3D geom trainer
(pytorch_3d_rising_bubble_train_geom.py) and writes them into the Basilisk h5 as the
group "computation_regions" -- the 3D counterpart of what compute_patches.py +
combine_data.py added to the 2D rising_bubble.h5.

Each region is an axis-aligned cube of side 5 cells (the 2D squares were 5 x 5 cells,
region_half_size = 10 * grid_scale in the nondimensional units compute_patches.py takes),
centred on an A-like point: per training snapshot, NOP[0] points 1-2 cells off the
interface on either side and NOP[1] domain points, drawn by generate_points_3d's own A
sampler (seeded here, so the trainer's random stream is untouched). For each cube:

    face_fraction      (N, 6)  fraction of each face inside the bubble (a = 1 - f > 0.5,
                               trilinear a, 24 x 24 samples per face), faces ordered
                               x-, x+, y-, y+, z-, z+
    grad_vec           (N, 3)  h^2 (f_x- - f_x+, f_y- - f_y+, f_z- - f_z+): the 3D version
                               of the 2D wetted-length differences (left - right,
                               bottom - top), now wetted AREAS. Physical units (length^2).
    integrated_normal  (N, 3)  vector area of the f = 0.5 marching-cubes surface clipped
                               to the cube, normals pointing OUT of the bubble (length^2)

Divergence theorem on (bubble inside cube): integrated_normal = grad_vec exactly, up to
the face sampling and the marching-cubes vs trilinear surface difference; the script
prints the mismatch, as compute_grad_norms.py did in 2D. The field is the VOF itself,
iso 0.5 -- the same surface the h5 levelset is the distance to (basilisk_to_h5.py).

Also stored: x_center, y_center, z_center, time_value (PHYSICAL units, unlike the 2D
file, whose time_value is t / L_ref), time_index (into the h5 time axis), and attrs.

    python compute_regions_3d.py [--h5 ../cfd_data/rising_bubble_3d_basilisk_L8.h5]
                                 [--limit-snapshots N] [--dry-run] '''

import argparse
import math
import time
from concurrent.futures import ProcessPoolExecutor

import h5py
import numpy as np
from scipy.ndimage import map_coordinates
from skimage.measure import marching_cubes

import generate_points_3d as gp

GROUP = "computation_regions"
REGION_CELLS = 5            # cube side in cells, as the 2D 5 x 5 squares
NOP_REGIONS = (500, 400)    # (interface, domain) centres per snapshot, as NOP_a
FACE_SAMPLES = 24           # per face side
ISO = 0.5
SEED = 4321


def training_indices(data):
    ''' The snapshots get_training_data_3d trains on, chosen exactly as it chooses them. '''
    times = np.array(data["time"])
    candidates = np.arange(len(times))
    if "split" in data:
        candidates = candidates[np.array(data["split"]) == b"train"]
    indices = candidates[[np.argmin(np.abs(times[candidates] - t)) for t in gp.TRAIN_TIMES]]
    assert np.abs(times[indices] - gp.TRAIN_TIMES).max() < 0.005
    return indices


def _clip_polygon(poly, axis, value, keep_greater):
    ''' Sutherland-Hodgman against one axis-aligned plane; keeps vertex order. '''
    out = []
    n = len(poly)
    for i in range(n):
        cur, nxt = poly[i], poly[(i + 1) % n]
        cur_in = cur[axis] >= value if keep_greater else cur[axis] <= value
        nxt_in = nxt[axis] >= value if keep_greater else nxt[axis] <= value
        if cur_in:
            out.append(cur)
        if cur_in != nxt_in:
            s = (value - cur[axis]) / (nxt[axis] - cur[axis])
            out.append(cur + s * (nxt - cur))
    return out


def _vector_area(poly):
    p = np.asarray(poly)
    return 0.5 * np.cross(p, np.roll(p, -1, axis=0)).sum(axis=0)


def region_geometry(a, origin, dx, center, h):
    ''' a: bubble fraction (Z, Y, X), origin = (X[0], Y[0], Z[0]). Returns
    (face_fraction (6,), integrated_normal (3,)) for the cube of side h at center (x, y, z). '''
    lo, hi = center - h / 2, center + h / 2
    shape_xyz = np.array(a.shape[::-1])
    i0 = np.clip(np.floor((lo - origin) / dx).astype(int) - 1, 0, shape_xyz - 1)
    i1 = np.clip(np.ceil((hi - origin) / dx).astype(int) + 1, 0, shape_xyz - 1)
    block = a[i0[2]:i1[2] + 1, i0[1]:i1[1] + 1, i0[0]:i1[0] + 1].astype(np.float64)
    block_origin = origin + i0 * dx

    # Face fractions from the trilinear field
    s = (np.arange(FACE_SAMPLES) + 0.5) / FACE_SAMPLES * h
    u, v = np.meshgrid(s, s, indexing="ij")
    u, v = u.ravel(), v.ravel()
    fractions = np.empty(6)
    for axis in range(3):
        others = [k for k in range(3) if k != axis]
        for side, value in enumerate((lo[axis], hi[axis])):
            pts = np.empty((len(u), 3))
            pts[:, axis] = value
            pts[:, others[0]] = lo[others[0]] + u
            pts[:, others[1]] = lo[others[1]] + v
            idx = ((pts - block_origin) / dx)[:, ::-1].T          # (z, y, x) index coords
            fractions[2 * axis + side] = (map_coordinates(block, idx, order=1) > ISO).mean()

    normal = np.zeros(3)
    if block.min() < ISO < block.max():
        verts, faces, vnormals, _ = marching_cubes(block, ISO, spacing=(dx, dx, dx))
        verts = verts[:, ::-1] + block_origin                    # physical (x, y, z)
        vnormals = vnormals[:, ::-1]                             # 'descent': out of the bubble
        tri = verts[faces]
        raw = 0.5 * np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        sign = np.sign(np.einsum("tc,tc->t", raw, vnormals[faces].sum(axis=1)))
        inside = np.all((tri >= lo) & (tri <= hi), axis=(1, 2))
        normal += (raw[inside] * sign[inside, None]).sum(axis=0)
        for t in np.flatnonzero(~inside):
            poly = list(tri[t])
            for axis in range(3):
                poly = _clip_polygon(poly, axis, lo[axis], True)
                if len(poly) < 3:
                    break
                poly = _clip_polygon(poly, axis, hi[axis], False)
                if len(poly) < 3:
                    break
            if len(poly) >= 3:
                normal += sign[t] * _vector_area(poly)
    return fractions, normal


def process_snapshot(args):
    h5_path, k, t_index = args
    np.random.seed(SEED + k)
    with h5py.File(h5_path, "r") as data:
        X, Y, Z = (np.array(data[c], dtype=float) for c in ("X", "Y", "Z"))
        t_value = float(data["time"][t_index])
        levelset = -np.array(data["levelset"][t_index], dtype=float)      # bubble positive
        a = 1.0 - np.array(data["volume_fraction"][t_index], dtype=np.float32)
    dx = np.diff(X)[0]
    interface, normal, t1, t2 = gp.compute_normals_3d(X, Y, Z, levelset[None], dx)
    pts = gp.get_points_a_3d(NOP_REGIONS, [t_value], X, Y, Z, dx, interface, normal, t1, t2,
                             levelset[None])
    h = REGION_CELLS * dx
    centers = pts[:, :3]
    lo_dom = np.array([X[0], Y[0], Z[0]])
    hi_dom = np.array([X[-1], Y[-1], Z[-1]])
    viable = np.all((centers - h / 2 >= lo_dom) & (centers + h / 2 <= hi_dom), axis=1)
    centers = centers[viable]

    origin = np.array([X[0], Y[0], Z[0]])
    fractions, normals = zip(*(region_geometry(a, origin, dx, c, h) for c in centers))
    return dict(k=k, centers=centers, t_value=t_value, t_index=t_index,
                fractions=np.array(fractions), normals=np.array(normals), h=h,
                skipped=int((~viable).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", default=gp.H5_PATH)
    ap.add_argument("--limit-snapshots", type=int, default=None)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--dry-run", action="store_true", help="compute and report, do not write")
    args = ap.parse_args()

    with h5py.File(args.h5, "r") as data:
        indices = training_indices(data)
    if args.limit_snapshots:
        indices = indices[:args.limit_snapshots]
    print(f"{len(indices)} training snapshots, {sum(NOP_REGIONS)} regions each, cube side "
          f"{REGION_CELLS} cells")

    start = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        results = sorted(ex.map(process_snapshot, [(args.h5, k, i) for k, i in enumerate(indices)]),
                         key=lambda r: r["k"])
    print(f"Computed in {time.time() - start:.0f}s")

    h = results[0]["h"]
    centers = np.vstack([r["centers"] for r in results])
    fractions = np.vstack([r["fractions"] for r in results])
    normals = np.vstack([r["normals"] for r in results])
    t_value = np.concatenate([np.full(len(r["centers"]), r["t_value"]) for r in results])
    t_index = np.concatenate([np.full(len(r["centers"]), r["t_index"]) for r in results])
    grad_vec = h ** 2 * (fractions[:, 0::2] - fractions[:, 1::2])

    # Divergence-theorem check, in units of the face area h^2
    cut = np.any((fractions > 0) & (fractions < 1), axis=1) | np.any(normals != 0, axis=1)
    err = np.linalg.norm(grad_vec - normals, axis=1) / h ** 2
    print(f"Regions: {len(centers)} ({sum(r['skipped'] for r in results)} skipped at the walls), "
          f"{int(cut.sum())} cut by the interface")
    print(f"|grad_vec - integrated_normal| / h^2 on cut regions: mean {err[cut].mean():.3e}, "
          f"95% {np.percentile(err[cut], 95):.3e}, max {err[cut].max():.3e}; "
          f"|grad_vec| / h^2 mean {np.linalg.norm(grad_vec[cut], axis=1).mean() / h ** 2:.3f}")
    print(f"Uncut regions with nonzero grad_vec: {int((np.abs(grad_vec[~cut]) > 0).any(axis=1).sum())}")

    if args.dry_run:
        return
    with h5py.File(args.h5, "a") as f:
        if GROUP in f:
            del f[GROUP]
        g = f.create_group(GROUP)
        for name, arr in (("x_center", centers[:, 0]), ("y_center", centers[:, 1]),
                          ("z_center", centers[:, 2]), ("time_value", t_value),
                          ("time_index", t_index), ("face_fraction", fractions),
                          ("grad_vec", grad_vec), ("integrated_normal", normals)):
            g.create_dataset(name, data=arr)
        g.attrs["region_half_size"] = h / 2
        g.attrs["region_cells"] = REGION_CELLS
        g.attrs["iso"] = ISO
        g.attrs["field"] = "bubble fraction a = 1 - volume_fraction, trilinear"
        g.attrs["face_order"] = "x-, x+, y-, y+, z-, z+"
        g.attrs["units"] = "physical (Basilisk); time_value is t, not t / L_ref"
        g.attrs["normal_orientation"] = "out of the bubble; grad_vec = h^2 (f_minus - f_plus)"
        g.attrs["centres"] = f"generate_points_3d.get_points_a_3d{NOP_REGIONS}, seed {SEED} + k"
    print(f"Wrote {len(centers)} regions to {args.h5}:/{GROUP}")


if __name__ == "__main__":
    main()
