''' compute_regions_3d.py for the LBPM file: precomputes the geometric-constraint regions
of pytorch_3d_lbpm_datadriven_geom_train.py and writes them into
lbpm_bubble_dataset_newrun_nowrap.h5 as the group "computation_regions".

Each region is an axis-aligned cube of side 5 cells, as in 2D and Basilisk. Per training
snapshot (the 19 not in generate_points_3d_lbpm.VAL_INDICES) NOP_REGIONS = (750, 600)
centres: 750 points 0-3 cells off the a = 0.5 surface on either side (so the cube is cut)
and 600 domain cells 4+ cells away, about 25.6k regions in all (Basilisk: 25,434).

The field is the phase fraction a = (rho - rho2) / (rho1 - rho2), clipped to [0, 1], with
the MEASURED phase densities of generate_points_3d_lbpm.get_scales (drop a = 1). Its 0.5
surface is the levelset zero of the h5 (a = 0.51 +- 0.03 there). The box is periodic, so
the field is padded by wrap-around and no cube is skipped (Basilisk skipped its walls).

The LBPM interface is ~14 cells wide (10-90%), wider than the cube, so the thresholded
face fraction (area with a > 0.5) and the face MEAN of a, which is what the trainer's
quadrature of the network a computes, are no longer the same thing (they were for the
1-2 cell Basilisk interface). Both are stored; the trainer uses the mean:

    face_mean          (N, 6)  mean of the trilinear a over each face, 24 x 24 samples,
                               faces ordered x-, x+, y-, y+, z-, z+
    grad_vec           (N, 3)  h^2 (mean_minus - mean_plus) per axis: the TRAINING target.
                               Divergence theorem: = -(integral of grad(a) over the cube),
                               exactly; checked against a sampled grad(a) below
    face_fraction      (N, 6)  fraction of each face with a > 0.5 (compute_regions_3d's)
    grad_vec_sharp     (N, 3)  h^2 (fraction_minus - fraction_plus)
    integrated_normal  (N, 3)  vector area of the a = 0.5 marching-cubes surface clipped to
                               the cube, pointing OUT of the drop; ~ grad_vec_sharp

Also stored: x_center, y_center, z_center in CELLS centred on the box (the coordinates of
generate_points_3d_lbpm._cell_axes), time_value in LBM steps, time_index into the h5.

    python compute_regions_3d_lbpm.py [--limit-snapshots N] [--dry-run] '''

import argparse
import time
from concurrent.futures import ProcessPoolExecutor

import h5py
import numpy as np
from scipy.ndimage import map_coordinates

import generate_points_3d as gp
import generate_points_3d_lbpm as lb
from compute_regions_3d import region_geometry, FACE_SAMPLES, ISO, GROUP

REGION_CELLS = 5
NOP_REGIONS = (750, 600)    # (interface, domain) centres per snapshot
INTERFACE_BAND = 3.0        # interface centres 0-3 cells off the a = 0.5 surface
PAD = 6                     # wrap-around padding, > half the cube + the 1-cell stencil margin
GRAD_SAMPLES = 12           # per cube side, for the divergence-theorem check
SEED = 4321


def face_means(a_pad, origin, center, h):
    ''' Mean of the trilinear a over each face of the cube, sampled exactly as
    region_geometry samples its face fractions. '''
    lo, hi = center - h / 2, center + h / 2
    s = (np.arange(FACE_SAMPLES) + 0.5) / FACE_SAMPLES * h
    u, v = np.meshgrid(s, s, indexing="ij")
    u, v = u.ravel(), v.ravel()
    means = np.empty(6)
    for axis in range(3):
        others = [k for k in range(3) if k != axis]
        for side, value in enumerate((lo[axis], hi[axis])):
            pts = np.empty((len(u), 3))
            pts[:, axis] = value
            pts[:, others[0]] = lo[others[0]] + u
            pts[:, others[1]] = lo[others[1]] + v
            means[2 * axis + side] = map_coordinates(a_pad, (pts - origin)[:, ::-1].T, order=1).mean()
    return means


def integrated_gradient(grads_pad, origin, center, h):
    ''' Midpoint-rule integral of the central-difference grad(a) over the cube. '''
    s = (np.arange(GRAD_SAMPLES) + 0.5) / GRAD_SAMPLES * h - h / 2
    pts = np.stack(np.meshgrid(s, s, s, indexing="ij"), axis=-1).reshape(-1, 3) + center
    idx = (pts - origin)[:, ::-1].T
    return np.array([map_coordinates(g, idx, order=1).mean() for g in grads_pad]) * h ** 3


def process_snapshot(args):
    k, t_index, scales = args
    np.random.seed(SEED + k)
    with h5py.File(lb.H5_PATH, "r") as data:
        (X, Y, Z), (nx, ny, nz) = lb._cell_axes(data)
        t_value = float(data["time"][t_index])
        levelset = -np.array(data["levelset"][t_index], dtype=float)      # drop positive
        a = lb.phase_fraction(np.array(data["density"][t_index], dtype=float), scales)

    interface, normal, t1, t2 = gp.compute_normals_3d(X, Y, Z, levelset[None], 1.0)
    near = gp.get_points_for_interface_3d(NOP_REGIONS[0], t_value, interface[0], normal[0],
                                          t1[0], t2[0], 1.0, 0.0, INTERFACE_BAND)
    far = gp.get_points_for_domain_3d(NOP_REGIONS[1], t_value, X, Y, Z, levelset, 4)
    lo_box = np.array([-nx / 2, -ny / 2, -nz / 2])
    size = np.array([nx, ny, nz], dtype=float)
    centers = np.mod(np.vstack([near, far])[:, :3] - lo_box, size) + lo_box

    a_pad = np.pad(a, PAD, mode="wrap")
    grads_pad = [np.pad(lb.pr.periodic_d1(a, axis), PAD, mode="wrap") for axis in (2, 1, 0)]
    origin = np.array([X[0], Y[0], Z[0]]) - PAD
    h = float(REGION_CELLS)

    fractions, normals = zip(*(region_geometry(a_pad, origin, 1.0, c, h) for c in centers))
    means = [face_means(a_pad, origin, c, h) for c in centers]
    grad_int = [integrated_gradient(grads_pad, origin, c, h) for c in centers]
    return dict(k=k, centers=centers, t_value=t_value, t_index=t_index,
                fractions=np.array(fractions), normals=np.array(normals),
                means=np.array(means), grad_int=np.array(grad_int))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-snapshots", type=int, default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true", help="compute and report, do not write")
    args = ap.parse_args()

    scales = lb.get_scales()
    with h5py.File(lb.H5_PATH, "r") as data:
        indices = np.setdiff1d(np.arange(len(data["time"])), lb.VAL_INDICES)
    if args.limit_snapshots:
        indices = indices[:args.limit_snapshots]
    print(f"{len(indices)} training snapshots, {sum(NOP_REGIONS)} regions each, cube side "
          f"{REGION_CELLS} cells")

    start = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        results = sorted(ex.map(process_snapshot, [(k, i, scales) for k, i in enumerate(indices)]),
                         key=lambda r: r["k"])
    print(f"Computed in {time.time() - start:.0f}s")

    h = float(REGION_CELLS)
    stack = lambda key: np.vstack([r[key] for r in results])
    centers, fractions, normals, means, grad_int = (stack(k) for k in
                                                    ("centers", "fractions", "normals", "means", "grad_int"))
    t_value = np.concatenate([np.full(len(r["centers"]), r["t_value"]) for r in results])
    t_index = np.concatenate([np.full(len(r["centers"]), r["t_index"]) for r in results])
    grad_vec = h ** 2 * (means[:, 0::2] - means[:, 1::2])
    grad_vec_sharp = h ** 2 * (fractions[:, 0::2] - fractions[:, 1::2])

    # Checks, in units of the face area h^2
    cut = np.any((fractions > 0) & (fractions < 1), axis=1) | np.any(normals != 0, axis=1)
    err_soft = np.linalg.norm(grad_vec + grad_int, axis=1) / h ** 2
    err_sharp = np.linalg.norm(grad_vec_sharp - normals, axis=1) / h ** 2
    print(f"Regions: {len(centers)}, {int(cut.sum())} cut by the a = 0.5 surface")
    print(f"|grad_vec + integral grad(a)| / h^2 (all): mean {err_soft.mean():.3e}, "
          f"95% {np.percentile(err_soft, 95):.3e}, max {err_soft.max():.3e}; "
          f"|grad_vec| / h^2 mean {np.linalg.norm(grad_vec, axis=1).mean() / h ** 2:.3f}")
    print(f"|grad_vec_sharp - integrated_normal| / h^2 on cut regions: mean {err_sharp[cut].mean():.3e}, "
          f"95% {np.percentile(err_sharp[cut], 95):.3e}, max {err_sharp[cut].max():.3e}; "
          f"|grad_vec_sharp| / h^2 mean {np.linalg.norm(grad_vec_sharp[cut], axis=1).mean() / h ** 2:.3f}")
    print(f"|grad_vec| / h^2 on cut regions: mean {np.linalg.norm(grad_vec[cut], axis=1).mean() / h ** 2:.3f}; "
          f"on uncut regions: mean {np.linalg.norm(grad_vec[~cut], axis=1).mean() / h ** 2:.3f}")

    if args.dry_run:
        return
    with h5py.File(lb.H5_PATH, "a") as f:
        if GROUP in f:
            del f[GROUP]
        g = f.create_group(GROUP)
        for name, arr in (("x_center", centers[:, 0]), ("y_center", centers[:, 1]),
                          ("z_center", centers[:, 2]), ("time_value", t_value),
                          ("time_index", t_index), ("face_mean", means), ("grad_vec", grad_vec),
                          ("face_fraction", fractions), ("grad_vec_sharp", grad_vec_sharp),
                          ("integrated_normal", normals)):
            g.create_dataset(name, data=arr)
        g.attrs["region_half_size"] = h / 2
        g.attrs["region_cells"] = REGION_CELLS
        g.attrs["iso"] = ISO
        g.attrs["field"] = (f"phase fraction a = (rho - rho2) / (rho1 - rho2), clipped, trilinear; "
                            f"rho1 = {scales['rho1']:.6f}, rho2 = {scales['rho2']:.6f} (measured)")
        g.attrs["face_order"] = "x-, x+, y-, y+, z-, z+"
        g.attrs["units"] = "cells centred on the box; time_value in LBM steps"
        g.attrs["targets"] = ("grad_vec = h^2 (face_mean_minus - face_mean_plus), the training target; "
                              "grad_vec_sharp from the a > 0.5 face fractions")
        g.attrs["normal_orientation"] = "integrated_normal points out of the drop (a = 1)"
        g.attrs["centres"] = (f"{NOP_REGIONS[0]} interface (0-{INTERFACE_BAND:g} cells) + "
                              f"{NOP_REGIONS[1]} domain per training snapshot, seed {SEED} + k")
    print(f"Wrote {len(centers)} regions to {lb.H5_PATH}:/{GROUP}")


if __name__ == "__main__":
    main()
