import numpy as np
import math
import pandas as pd
import h5py

''' 3D counterpart of generate_points.py. It ingests the LBPM-generated
"lbpm_bubble_dataset.h5" volume (X, Y, Z, time, levelset) instead of the 2D
"rising_bubble.h5" volume (X, Y, time, levelset) and produces the same kind
of training-point dataframes, extended with a z coordinate and a w
(z-velocity) component.

Grid convention (see build-lbpm/compile_silo_to_h5.py):
  - levelset/pressure/velocity* are stored as (time, Z, Y, X)
  - Body force / gravity is applied along Z (F = 0, 0, Fz in the LBPM .db),
    so Z plays the role that Y played in the 2D rising-bubble case: the
    "top" face (z = z_max) is treated as an open/pressure boundary and the
    "bottom" face (z = 0) as a no-slip wall. X and Y are both treated as
    periodic side walls.
  - Raw levelset convention is +8 = carrier phase, -8 = bubble phase; it is
    negated below (as the 2D pipeline does) so that a > 0 => bubble (a=1),
    matching the a=1 branch of mu/rho in rising_bubble_train.py.

Caveat: the "time" values in the h5 file are raw LBM timestep indices, not
physical seconds. They are nondimensionalized by L_ref below exactly like
the 2D pipeline (which assumes U_ref=1), but if a physical dt is known for
this LBPM run it should be applied before calling get_training_data_3d. '''


def get_points_for_interface_3d(NOP, t_value, interface, normal, tangent1, tangent2,
                                 cell_size, refine_start, refine_end):
    ''' 3D analogue of generate_points.get_points_for_interface. Samples points along
    the interface normal (inward = bubble side, outward = carrier side), staggered
    across a small stencil of tangential offsets (built from tangent1/tangent2) so that
    points are spread across the interface surface between grid nodes, not just at the
    interface nodes themselves. '''

    M = len(interface)
    no_points = max(1, math.ceil(NOP / (2 * M)))
    stencil = [(0.0, 0.0), (1 / 3, 0.0), (2 / 3, 0.0), (0.0, 1 / 3), (0.0, 2 / 3)]

    points_inward = []
    points_outward = []
    for x_p, n_p, t1_p, t2_p in zip(interface, normal, tangent1, tangent2):
        for f1, f2 in stencil:
            tangential_offset = cell_size * f1 * t1_p + cell_size * f2 * t2_p
            inward_n = np.random.uniform(refine_start, refine_end, no_points)
            outward_n = np.random.uniform(refine_start, refine_end, no_points)
            for in_n, out_n in zip(inward_n, outward_n):
                points_inward.append(x_p + tangential_offset + in_n * n_p)
                points_outward.append(x_p + tangential_offset - out_n * n_p)

    points_inward = np.array(points_inward)
    points_outward = np.array(points_outward)
    interface_inward = np.hstack([points_inward, t_value * np.ones((len(points_inward), 1)),
                                   np.ones((len(points_inward), 1))])
    interface_outward = np.hstack([points_outward, t_value * np.ones((len(points_outward), 1)),
                                    np.zeros((len(points_outward), 1))])
    interface_data = np.vstack([interface_inward, interface_outward])
    indices_to_keep = np.random.choice(len(interface_data), NOP, replace=False)
    return interface_data[indices_to_keep, :]


def get_points_for_domain_3d(NOP, t_value, X, Y, Z, levelset, max_levelset):
    ''' 3D analogue of generate_points.get_points_for_domain. Loops over every
    (x, y) column of the grid and samples z-values far from the interface
    (|levelset| >= max_levelset), then subsamples down to NOP points. '''

    n_xy = len(X) * len(Y)
    no_z = math.ceil(NOP / n_xy)
    domain_data = np.empty((0, 5), float)
    z_indices_all = np.arange(len(Z))
    for ix, x_value in enumerate(X):
        for iy, y_value in enumerate(Y):
            remaining = no_z
            z_kept, a_kept = [], []
            while remaining != 0:
                indices_z = np.random.choice(z_indices_all, remaining, replace=False)
                col = levelset[indices_z, iy, ix]
                keep = np.where((np.abs(col) >= max_levelset) | (np.abs(col) == 8))[0]
                z_kept.append(Z[indices_z[keep]])
                a_kept.append((col[keep] > 0).astype(float))
                remaining -= len(keep)
            z_col = np.hstack(z_kept).reshape(no_z, 1)
            a_col = np.hstack(a_kept).reshape(no_z, 1)
            x_col = x_value * np.ones((no_z, 1))
            y_col = y_value * np.ones((no_z, 1))
            t_col = t_value * np.ones((no_z, 1))
            domain_data = np.vstack([domain_data, np.hstack([x_col, y_col, z_col, t_col, a_col])])
    indices_to_keep = np.random.choice(len(domain_data), NOP, replace=False)
    return domain_data[indices_to_keep, :]


def get_points_a_3d(NOP, times, X, Y, Z, cell_size, interface_all, normal_all,
                     tangent1_all, tangent2_all, levelset_all):
    ''' Generates the points for the volume fraction loss (3D). '''

    refine_start = 0.004
    refine_end = 0.008
    domain_start_levelset = 4
    data = np.empty((0, 5), float)
    for t_value, interface, normal, t1, t2, levelset in zip(
            times, interface_all, normal_all, tangent1_all, tangent2_all, levelset_all):
        interface_data = get_points_for_interface_3d(NOP[0], t_value, interface, normal, t1, t2,
                                                       cell_size, refine_start, refine_end)
        domain_data = get_points_for_domain_3d(NOP[1], t_value, X, Y, Z, levelset, domain_start_levelset)
        data = np.vstack([data, interface_data, domain_data])
    return data


def get_points_pde_3d(NOP, times, X, Y, Z, cell_size, interface_all, normal_all,
                       tangent1_all, tangent2_all, levelset_all):
    ''' Generates the residual (collocation) points for the PDEs (3D). '''

    refine_interface_start = 0.0
    refine_interface_end = 0.001
    refine_nearfield_end = 0.1
    domain_start_levelset = 4
    data = np.empty((0, 5), float)
    for t_value, interface, normal, t1, t2, levelset in zip(
            times, interface_all, normal_all, tangent1_all, tangent2_all, levelset_all):
        interface_temp = get_points_for_interface_3d(NOP[0], t_value, interface, normal, t1, t2,
                                                       cell_size, refine_interface_start, refine_interface_end)
        nearfield_temp = get_points_for_interface_3d(NOP[1], t_value, interface, normal, t1, t2,
                                                       cell_size, refine_interface_end, refine_nearfield_end)
        domain_temp = get_points_for_domain_3d(NOP[2], t_value, X, Y, Z, levelset, domain_start_levelset)
        data = np.vstack([data, interface_temp, nearfield_temp, domain_temp])
    data[:, 4] = 0.0
    return data


def get_top(NOP, x, y, z_top, t):
    ''' Open/pressure boundary at z = z_top (analogue of get_north). '''
    low_t = t[0] + np.finfo(float).eps
    boundary_time = np.hstack([0.0, np.random.uniform(low_t, t[1], NOP[1] - 2), t[1]])
    boundary = np.empty((0, 8), float)
    for time_value in boundary_time:
        bx = np.random.uniform(x[0], x[1], (NOP[0], 1))
        by = np.random.uniform(y[0], y[1], (NOP[0], 1))
        bz = z_top * np.ones((NOP[0], 1))
        bu = np.zeros((NOP[0], 1))
        bv = np.zeros((NOP[0], 1))
        bw = np.zeros((NOP[0], 1))
        bp = np.ones((NOP[0], 1))
        bt = time_value * np.ones((NOP[0], 1))
        boundary = np.vstack([boundary, np.hstack([bx, by, bz, bt, bu, bv, bw, bp])])
    return boundary


def get_bottom(NOP, x, y, z_bottom, t):
    ''' No-slip wall at z = z_bottom (analogue of get_south). '''
    low_t = t[0] + np.finfo(float).eps
    boundary_time = np.hstack([0.0, np.random.uniform(low_t, t[1], NOP[1] - 2), t[1]])
    boundary = np.empty((0, 7), float)
    for time_value in boundary_time:
        bx = np.random.uniform(x[0], x[1], (NOP[0], 1))
        by = np.random.uniform(y[0], y[1], (NOP[0], 1))
        bz = z_bottom * np.ones((NOP[0], 1))
        bu = np.zeros((NOP[0], 1))
        bv = np.zeros((NOP[0], 1))
        bw = np.zeros((NOP[0], 1))
        bt = time_value * np.ones((NOP[0], 1))
        boundary = np.vstack([boundary, np.hstack([bx, by, bz, bt, bu, bv, bw])])
    return boundary


def _periodic_face_grid(NOP, coord1_bounds, coord2_bounds, t_bounds):
    ''' Deterministic (non-random) 2D spatial grid + time samples shared by both faces
    of a periodic pair, so that the plus/minus faces line up row-for-row. '''
    n1 = max(1, round(math.sqrt(NOP[0])))
    n2 = max(1, math.ceil(NOP[0] / n1))
    c1 = np.linspace(coord1_bounds[0], coord1_bounds[1], n1)
    c2 = np.linspace(coord2_bounds[0], coord2_bounds[1], n2)
    C1, C2 = np.meshgrid(c1, c2, indexing="ij")
    C1 = C1.reshape(-1, 1)
    C2 = C2.reshape(-1, 1)
    t_vals = np.linspace(t_bounds[0], t_bounds[1], NOP[1])
    return C1, C2, t_vals


def get_x_periodic(NOP, x, y, z, t):
    ''' Periodic pair at x = x_min / x_max (analogue of east/west), sampled over a
    shared (y, z) grid so pred(x_max) - pred(x_min) is a meaningful pairwise loss. '''
    Y_grid, Z_grid, t_vals = _periodic_face_grid(NOP, y, z, t)
    n_space = len(Y_grid)
    rows = []
    for time_value in t_vals:
        rows.append(np.hstack([
            x[1] * np.ones((n_space, 1)), Y_grid, Z_grid,
            x[0] * np.ones((n_space, 1)), Y_grid, Z_grid,
            time_value * np.ones((n_space, 1)),
        ]))
    return np.vstack(rows)


def get_y_periodic(NOP, x, y, z, t):
    ''' Periodic pair at y = y_min / y_max, sampled over a shared (x, z) grid. '''
    X_grid, Z_grid, t_vals = _periodic_face_grid(NOP, x, z, t)
    n_space = len(X_grid)
    rows = []
    for time_value in t_vals:
        rows.append(np.hstack([
            X_grid, y[1] * np.ones((n_space, 1)), Z_grid,
            X_grid, y[0] * np.ones((n_space, 1)), Z_grid,
            time_value * np.ones((n_space, 1)),
        ]))
    return np.vstack(rows)


def compute_normals_3d(X, Y, Z, levelset, dx, dy, dz):
    ''' 3D analogue of generate_points.compute_normals. For each interface cell
    (|levelset| <= 0.75) computes the surface normal from the levelset gradient and
    two tangent vectors spanning the tangent plane (via Gram-Schmidt against a helper
    axis), which are used to distribute interface/nearfield refinement points. '''

    interface, normal, tangent1, tangent2 = [], [], [], []

    for snap in levelset:
        nz, ny, nx = snap.shape
        idx_z, idx_y, idx_x = np.where(np.abs(snap) <= 0.75)

        valid = ((idx_x > 0) & (idx_x < nx - 1) &
                 (idx_y > 0) & (idx_y < ny - 1) &
                 (idx_z > 0) & (idx_z < nz - 1))
        idx_x, idx_y, idx_z = idx_x[valid], idx_y[valid], idx_z[valid]

        gx = (snap[idx_z, idx_y, idx_x + 1] - snap[idx_z, idx_y, idx_x - 1]) / (2 * dx)
        gy = (snap[idx_z, idx_y + 1, idx_x] - snap[idx_z, idx_y - 1, idx_x]) / (2 * dy)
        gz = (snap[idx_z + 1, idx_y, idx_x] - snap[idx_z - 1, idx_y, idx_x]) / (2 * dz)

        grad_abs = np.sqrt(gx ** 2 + gy ** 2 + gz ** 2)
        grad_abs = np.where(grad_abs == 0, np.finfo(float).eps, grad_abs)
        n = np.stack([gx, gy, gz], axis=1) / grad_abs[:, None]

        helper = np.tile(np.array([0.0, 0.0, 1.0]), (len(n), 1))
        near_parallel = np.abs(n[:, 2]) > 0.9
        helper[near_parallel] = np.array([1.0, 0.0, 0.0])

        t1 = np.cross(n, helper)
        t1 = t1 / np.linalg.norm(t1, axis=1, keepdims=True)
        t2 = np.cross(n, t1)

        pts = np.stack([X[idx_x], Y[idx_y], Z[idx_z]], axis=1)

        interface.append(pts)
        normal.append(n)
        tangent1.append(t1)
        tangent2.append(t2)

    return interface, normal, tangent1, tangent2


def get_training_data_3d(NOP_A, NOP_PDE, NOP_top, NOP_bottom, NOP_xper, NOP_yper,
                          h5_path="../cfd_data/lbpm_bubble_dataset.h5",
                          L_ref=0.25, time_indices=None):
    ''' Generates the training points for all losses from the LBPM 3D bubble dataset.

    Args:
        NOP_A: (interface, domain) point counts for the volume fraction loss, per snapshot
        NOP_PDE: (interface, nearfield, domain) point counts for the PDE residual, per snapshot
        NOP_top: (spatial, time) point counts for the open/pressure boundary at z_max
        NOP_bottom: (spatial, time) point counts for the no-slip wall at z_min
        NOP_xper: (spatial, time) point counts for the periodic pair at x_min/x_max
        NOP_yper: (spatial, time) point counts for the periodic pair at y_min/y_max
        h5_path: path to lbpm_bubble_dataset.h5
        L_ref: reference length used to nondimensionalize space and time (U_ref is assumed 1.0)
        time_indices: optional explicit array of time-snapshot indices to use; if None a
            manually-chosen coarse-then-fine subset of the 105 timesteps is used '''

    with h5py.File(h5_path, "r") as data:
        X = np.array(data["X"], dtype=float)
        Y = np.array(data["Y"], dtype=float)
        Z = np.array(data["Z"], dtype=float)
        times = np.array(data["time"])
        levelset = -np.array(data["levelset"])

    if time_indices is None:
        time_indices = np.sort(np.unique(np.concatenate([
            np.arange(0, 8, 2),
            np.arange(8, len(times), 5),
        ])))
    times = times[time_indices]
    levelset = levelset[time_indices]

    t_bounds = [times[0], times[-1]]
    x_bounds = [X[0], X[-1]]
    y_bounds = [Y[0], Y[-1]]
    z_bounds = [Z[0], Z[-1]]
    dx = np.diff(X)[0]
    dy = np.diff(Y)[0]
    dz = np.diff(Z)[0]
    cell_size = (dx + dy + dz) / 3.0

    print("\nDistributing points for time snapshots:\n", times)
    print("Number of time snapshots: ", len(times))
    print("Time bounds: ", t_bounds)
    print("Domain bounds: x", x_bounds, "y", y_bounds, "z", z_bounds, "\n")

    interface, normal, tangent1, tangent2 = compute_normals_3d(X, Y, Z, levelset, dx, dy, dz)
    print("Generating points for A")
    data_A = get_points_a_3d(NOP_A, times, X, Y, Z, cell_size, interface, normal, tangent1, tangent2, levelset)
    print("Generating points for PDE")
    data_PDE = get_points_pde_3d(NOP_PDE, times, X, Y, Z, cell_size, interface, normal, tangent1, tangent2, levelset)
    print("Generating points for TOP/BOTTOM/XPER/YPER")
    data_top = get_top(NOP_top, x_bounds, y_bounds, z_bounds[1], t_bounds)
    data_bottom = get_bottom(NOP_bottom, x_bounds, y_bounds, z_bounds[0], t_bounds)
    data_xper = get_x_periodic(NOP_xper, x_bounds, y_bounds, z_bounds, t_bounds)
    data_yper = get_y_periodic(NOP_yper, x_bounds, y_bounds, z_bounds, t_bounds)

    # NONDIMENSIONALIZE SPACE AND TIME (assumes U_ref = 1.0, so T_ref = L_ref)
    data_A[:, :4] /= L_ref
    data_PDE[:, :4] /= L_ref
    data_top[:, :4] /= L_ref
    data_bottom[:, :4] /= L_ref
    data_xper /= L_ref
    data_yper /= L_ref

    # Extra PDE collocation points at the boundaries, mirroring the 2D pipeline's
    # inclusion of the NSEW coordinates in data_PDE.
    boundary_coords = np.vstack([
        data_top[:, :4],
        data_bottom[:, :4],
        data_xper[:, [0, 1, 2, 6]], data_xper[:, [3, 4, 5, 6]],
        data_yper[:, [0, 1, 2, 6]], data_yper[:, [3, 4, 5, 6]],
    ])
    data_PDE = np.vstack([data_PDE, np.hstack([boundary_coords, np.zeros((len(boundary_coords), 1))])])

    data_noslip = np.vstack([data_top[:, [0, 1, 2, 3, 4, 5, 6]], data_bottom])

    print("Assembling data frames\n")
    data_A = pd.DataFrame(data=data_A, columns=["x_A", "y_A", "z_A", "t_A", "a_A"])
    data_PDE = pd.DataFrame(data=data_PDE, columns=["x_PDE", "y_PDE", "z_PDE", "t_PDE", "f_PDE"])
    data_TOP = pd.DataFrame(data=data_top,
                             columns=["x_TOP", "y_TOP", "z_TOP", "t_TOP", "u_TOP", "v_TOP", "w_TOP", "p_TOP"])
    data_NOSLIP = pd.DataFrame(data=data_noslip,
                                columns=["x_NS", "y_NS", "z_NS", "t_NS", "u_NS", "v_NS", "w_NS"])
    data_XPER = pd.DataFrame(data=data_xper,
                              columns=["x_Xp", "y_Xp", "z_Xp", "x_Xm", "y_Xm", "z_Xm", "t_XPER"])
    data_YPER = pd.DataFrame(data=data_yper,
                              columns=["x_Yp", "y_Yp", "z_Yp", "x_Ym", "y_Ym", "z_Ym", "t_YPER"])

    return dict(A=data_A, PDE=data_PDE, TOP=data_TOP, NOSLIP=data_NOSLIP, XPER=data_XPER, YPER=data_YPER)
