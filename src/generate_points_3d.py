import numpy as np
import math
import pandas as pd
import h5py

''' 3D counterpart of generate_points.py, kept equivalent to it in every respect except
the added dimension. It reads H5_PATH: either "rising_bubble_3d.h5" (written by
revolve_2d_to_3d.py from the 2D "rising_bubble.h5": the 2D bubble revolved about its
vertical axis into a sphere, levelset only) or "rising_bubble_3d_basilisk.h5" (a real 3D
Basilisk simulation of the same case, cfd/basilisk_rising_bubble_3d, via basilisk_to_h5.py)
and produces the same training-point dataframes as get_training_data, extended with a
z coordinate and a w velocity.

Axis convention (shared with revolve_2d_to_3d.py and lift_2d_initial_weights):
    2D x (horizontal)  -> 3D x and y   (the box is 1 x 1 x 2, bubble on the z axis)
    2D y (vertical)    -> 3D z         gravity acts along z, so it sits in the w equation
Fields are stored as (time, Z, Y, X) on the 2D grid spacing, so the cell-unit levelset
thresholds below mean exactly what they do in 2D.

Boundary conditions mirror the 2D case face for face, with the y faces treated exactly
like the 2D east/west faces:
    north (z max)        u = v = w = 0 and p = 1
    south (z min)        u = v = w = 0
    east/west (x)        u = v = w = 0 (first snapshot only, as in 2D), plus periodic
    front/back (y)       pairing of u, v, w, p
Space and time are nondimensionalised by L_ref = 0.25 (U_ref = 1), as in 2D. '''

H5_PATH = "../cfd_data/rising_bubble_3d.h5"
L_REF = 0.25
# The 2D selection, indices [0, 15] + [30:100:5] + [100::5] + [1, 2] of the 0.02-spaced
# snapshots, as times: 0, 0.02, 0.04, 0.3, then every 0.1 from 0.6 to 3.0.
TRAIN_TIMES = np.round(np.arange(151)[np.sort(np.concatenate(
    [np.arange(0, 30, 15), np.arange(30, 100, 5), np.arange(100, 151, 5), [1, 2]]))] * 0.02, 2)


def get_points_for_interface_3d(NOP, t_value, interface, normal, tangent1, tangent2,
                                 cell_size, refine_start, refine_end):
    ''' 3D analogue of generate_points.get_points_for_interface. Samples points along
    the interface normal (inward = bubble side, a = 1; outward = liquid side, a = 0),
    staggered across a small stencil of tangential offsets (built from tangent1/tangent2)
    so that points are spread across the interface surface between grid nodes. The 2D
    routine uses offsets 0, 1/3, 2/3 of a cell along its single tangent; this adds the
    same offsets along the second tangent.

    A 3D interface carries far more cells than the few hundred points wanted, so a random
    subset of interface cells is drawn up front instead of building every candidate and
    throwing almost all of them away. '''

    stencil = np.array([(0.0, 0.0), (1 / 3, 0.0), (2 / 3, 0.0), (0.0, 1 / 3), (0.0, 2 / 3)])

    n_cells_needed = max(1, math.ceil(NOP / (2 * len(stencil))))
    if len(interface) > n_cells_needed:
        pick = np.random.choice(len(interface), n_cells_needed, replace=False)
        interface, normal = interface[pick], normal[pick]
        tangent1, tangent2 = tangent1[pick], tangent2[pick]

    # (cells, stencil, 3) tangential offsets around each interface cell
    offsets = (cell_size * stencil[None, :, 0, None] * tangent1[:, None, :] +
               cell_size * stencil[None, :, 1, None] * tangent2[:, None, :])
    base = interface[:, None, :] + offsets
    n = normal[:, None, :]

    shape = base.shape[:2] + (1,)
    inward = base + np.random.uniform(refine_start, refine_end, shape) * n
    outward = base - np.random.uniform(refine_start, refine_end, shape) * n

    inward = inward.reshape(-1, 3)
    outward = outward.reshape(-1, 3)
    interface_inward = np.hstack([inward, t_value * np.ones((len(inward), 1)),
                                  np.ones((len(inward), 1))])
    interface_outward = np.hstack([outward, t_value * np.ones((len(outward), 1)),
                                   np.zeros((len(outward), 1))])
    interface_data = np.vstack([interface_inward, interface_outward])
    indices_to_keep = np.random.choice(len(interface_data), NOP, replace=False)
    return interface_data[indices_to_keep, :]


def get_points_for_domain_3d(NOP, t_value, X, Y, Z, levelset, max_levelset):
    ''' 3D analogue of generate_points.get_points_for_domain. Stratified by x column
    exactly as in 2D -- ceil(NOP / len(X)) points per column, drawn without replacement
    and kept only far from the interface -- except that each column is now a (y, z)
    plane instead of a y line. '''

    no_yz = math.ceil(NOP / len(X))
    n_yz = len(Y) * len(Z)
    domain_data = np.empty((0, 5), float)
    for index_x, x_value in enumerate(X):
        t_domain = t_value * np.ones((no_yz, 1))
        x_domain = x_value * np.ones((no_yz, 1))
        y_domain, z_domain, a_domain = [], [], []
        remaining_points = no_yz
        while remaining_points != 0:
            flat = np.random.choice(n_yz, remaining_points, replace=False)
            indices_z, indices_y = np.divmod(flat, len(Y))
            vals = levelset[indices_z, indices_y, index_x]
            indices_to_keep = np.where((np.abs(vals) >= max_levelset) | (np.abs(vals) == 8))[0]
            y_domain.append(Y[indices_y[indices_to_keep]])
            z_domain.append(Z[indices_z[indices_to_keep]])
            a_domain.append((vals[indices_to_keep] > 0).astype(float))
            remaining_points = remaining_points - len(indices_to_keep)
        y_domain = np.hstack(y_domain).reshape(no_yz, 1)
        z_domain = np.hstack(z_domain).reshape(no_yz, 1)
        a_domain = np.hstack(a_domain).reshape(no_yz, 1)
        domain_data = np.vstack([domain_data, np.hstack([x_domain, y_domain, z_domain, t_domain, a_domain])])
    indices_to_keep = np.random.choice(len(domain_data), NOP, replace=False)
    return domain_data[indices_to_keep, :]


def get_points_a_3d(NOP, times, X, Y, Z, cell_size, interface_all, normal_all,
                     tangent1_all, tangent2_all, levelset_all):
    ''' Generates the points for the volume fraction. Same refinement widths as 2D (the
    coordinates are the same physical ones). '''

    # PARAMETERS FOR INTERFACE REFINEMENT
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
    ''' Generates the residual points. Same refinement widths as 2D. '''

    # PARAMETERS FOR INTERFACE AND NEARFIELD REFINEMENT
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


def _random_times(NOP_t, t):
    ''' t[0], NOP_t - 2 uniform random times, t[1] -- the north/south time sampling of 2D. '''
    low_t = t[0] + np.finfo(float).eps
    return np.hstack([0.0, np.random.uniform(low_t, t[1], NOP_t - 2), t[1]])


def get_north(NOP, x, y, z, t):
    ''' Top face: u = v = w = 0 and p = 1. Columns x, y, z, t, u, v, w, p. '''
    boundary_north = np.empty((0, 8), float)
    for time_value in _random_times(NOP[1], t):
        bx = np.random.uniform(x[0], x[1], (NOP[0], 1))
        by = np.random.uniform(y[0], y[1], (NOP[0], 1))
        bz = z[1] * np.ones((NOP[0], 1))
        zeros = np.zeros((NOP[0], 1))
        ones = np.ones((NOP[0], 1))
        boundary_north = np.vstack([boundary_north, np.hstack([bx, by, bz, time_value * ones,
                                                               zeros, zeros, zeros, ones])])
    return boundary_north


def get_south(NOP, x, y, z, t):
    ''' Bottom face: u = v = w = 0. Columns x, y, z, t, u, v, w. '''
    boundary_south = np.empty((0, 7), float)
    for time_value in _random_times(NOP[1], t):
        bx = np.random.uniform(x[0], x[1], (NOP[0], 1))
        by = np.random.uniform(y[0], y[1], (NOP[0], 1))
        bz = z[0] * np.ones((NOP[0], 1))
        zeros = np.zeros((NOP[0], 1))
        boundary_south = np.vstack([boundary_south, np.hstack([bx, by, bz, time_value * np.ones((NOP[0], 1)),
                                                               zeros, zeros, zeros])])
    return boundary_south


def get_east_west(NOP, x, y, z, t):
    ''' x faces. As in 2D, z is a linspace over the face and t a linspace over the run;
    y is drawn at random but SHARED by the two faces, so each east row and the west row
    with the same index form a periodic pair at equal (y, z, t), exactly like the 2D
    pairing at equal (y, t). Columns x, y, z, t, u, v, w. '''
    boundary_east = np.empty((0, 7), float)
    boundary_west = np.empty((0, 7), float)
    for time_value in np.linspace(t[0], t[1], NOP[1]):
        bz = np.linspace(z[0], z[1], NOP[0]).reshape(-1, 1)
        by = np.random.uniform(y[0], y[1], (NOP[0], 1))
        zeros = np.zeros((NOP[0], 1))
        tt = time_value * np.ones((NOP[0], 1))
        boundary_east = np.vstack([boundary_east, np.hstack([x[1] * np.ones((NOP[0], 1)), by, bz, tt,
                                                             zeros, zeros, zeros])])
        boundary_west = np.vstack([boundary_west, np.hstack([x[0] * np.ones((NOP[0], 1)), by, bz, tt,
                                                             zeros, zeros, zeros])])
    return boundary_east, boundary_west


def get_front_back(NOP, x, y, z, t):
    ''' y faces, sampled exactly like east/west with the roles of x and y swapped:
    z linspace, t linspace, x random but shared by the pair. Columns x, y, z, t, u, v, w. '''
    boundary_front = np.empty((0, 7), float)
    boundary_back = np.empty((0, 7), float)
    for time_value in np.linspace(t[0], t[1], NOP[1]):
        bz = np.linspace(z[0], z[1], NOP[0]).reshape(-1, 1)
        bx = np.random.uniform(x[0], x[1], (NOP[0], 1))
        zeros = np.zeros((NOP[0], 1))
        tt = time_value * np.ones((NOP[0], 1))
        boundary_front = np.vstack([boundary_front, np.hstack([bx, y[1] * np.ones((NOP[0], 1)), bz, tt,
                                                               zeros, zeros, zeros])])
        boundary_back = np.vstack([boundary_back, np.hstack([bx, y[0] * np.ones((NOP[0], 1)), bz, tt,
                                                             zeros, zeros, zeros])])
    return boundary_front, boundary_back


def compute_normals_3d(X, Y, Z, levelset, cell_size):
    ''' 3D analogue of generate_points.compute_normals. For each interface cell
    (|levelset| <= 0.75) computes the surface normal from the central-difference levelset
    gradient and two tangent vectors spanning the tangent plane (via Gram-Schmidt against
    a helper axis). In the y = 0 plane tangent1 reduces to the 2D tangent (-n_y, n_x). '''

    interface, normal, tangent1, tangent2 = [], [], [], []

    for snap in levelset:
        nz, ny, nx = snap.shape
        idx_z, idx_y, idx_x = np.where(np.abs(snap) <= 0.75)

        # The 2D routine indexes i+1 / i-1 without a bounds check; drop the domain-edge
        # cells here instead, since an interface cell on the edge has no centred stencil.
        valid = ((idx_x > 0) & (idx_x < nx - 1) & (idx_y > 0) & (idx_y < ny - 1) &
                 (idx_z > 0) & (idx_z < nz - 1))
        idx_x, idx_y, idx_z = idx_x[valid], idx_y[valid], idx_z[valid]

        gx = (snap[idx_z, idx_y, idx_x + 1] - snap[idx_z, idx_y, idx_x - 1]) / (2 * cell_size)
        gy = (snap[idx_z, idx_y + 1, idx_x] - snap[idx_z, idx_y - 1, idx_x]) / (2 * cell_size)
        gz = (snap[idx_z + 1, idx_y, idx_x] - snap[idx_z - 1, idx_y, idx_x]) / (2 * cell_size)

        grad_abs = np.sqrt(gx ** 2 + gy ** 2 + gz ** 2)
        grad_abs = np.where(grad_abs == 0, np.finfo(float).eps, grad_abs)
        n = np.stack([gx, gy, gz], axis=1) / grad_abs[:, None]

        helper = np.tile(np.array([0.0, 1.0, 0.0]), (len(n), 1))
        near_parallel = np.abs(n[:, 1]) > 0.9
        helper[near_parallel] = np.array([1.0, 0.0, 0.0])

        t1 = np.cross(n, helper)
        t1 = t1 / np.linalg.norm(t1, axis=1, keepdims=True)
        t2 = np.cross(n, t1)

        interface.append(np.stack([X[idx_x], Y[idx_y], Z[idx_z]], axis=1))
        normal.append(n)
        tangent1.append(t1)
        tangent2.append(t2)

    return interface, normal, tangent1, tangent2


def get_training_data_3d(NOP_A, NOP_PDE, NOP_north, NOP_south, NOP_east, NOP_west, NOP_front_back):

    ''' Generates the training points for all losses -- 3D version of
    generate_points.get_training_data, with the same snapshot selection, refinement and
    boundary sampling.

    Args:
        NOP_A: Points for the volume fraction. Tuple containig amount of training points for the interface refinement (index 0) and the rest of the domain (index 1) per time snapshot
        NOP_PDE: Points for the PDEs. Tuple containig amount of training points for the interface refinement (index 0) the nearfield refinement (index 1) and the rest of the domain (index 2) per time snapshot
        NOP_boundaries: Tuple containig amount of points for the boundary conditions. (index 0) amount spatial points (index 1) amount of time snapshots
        NOP_front_back: as NOP_boundaries, for the y faces (treated like east/west) '''

    with h5py.File(H5_PATH, "r") as data:
        X = np.array(data["X"], dtype=float)
        Y = np.array(data["Y"], dtype=float)
        Z = np.array(data["Z"], dtype=float)
        times = np.array(data["time"])

        # TIME SNAPSHOT SELECTION (identical to 2D). Selected by time value, not position,
        # so files holding only these snapshots (rising_bubble_3d_basilisk.h5: 29 training
        # + 10 validation times) give the same set as the full 151-snapshot one. Read only
        # the selected snapshots -- the full 3D levelset (~20 GB) does not need to be in memory.
        indices = np.array([np.argmin(np.abs(times - t)) for t in TRAIN_TIMES])
        off = np.abs(times[indices] - TRAIN_TIMES)
        assert off.max() < 0.005, f"no snapshot near t = {TRAIN_TIMES[off.argmax()]}"
        times = times[indices]
        levelset = -np.array(data["levelset"][indices])

    # DOMAIN PARAMETERS PROVIDED BY CFD
    cell_size = np.diff(X)[0]
    t_bounds = [times[0], times[-1]]
    x_bounds = [X[0], X[-1]]
    y_bounds = [Y[0], Y[-1]]
    z_bounds = [Z[0], Z[-1]]

    print("\nDistributing points for time snapshots:\n", times)
    print("Number of time snapshots: ", len(times))
    print("Time bounds: ", t_bounds, "\n")

    interface, normal, tangent1, tangent2 = compute_normals_3d(X, Y, Z, levelset, cell_size)
    print("Generating points for A")
    data_A = get_points_a_3d(NOP_A, times, X, Y, Z, cell_size, interface, normal, tangent1, tangent2, levelset)
    print("Generating points for PDE")
    data_PDE = get_points_pde_3d(NOP_PDE, times, X, Y, Z, cell_size, interface, normal, tangent1, tangent2, levelset)
    print("Generating points for NSEW and front/back")
    data_north = get_north(NOP_north, x_bounds, y_bounds, z_bounds, t_bounds)
    data_south = get_south(NOP_south, x_bounds, y_bounds, z_bounds, t_bounds)
    assert NOP_east == NOP_west, "east/west are sampled as periodic pairs and must match"
    data_east, data_west = get_east_west(NOP_east, x_bounds, y_bounds, z_bounds, t_bounds)
    data_front, data_back = get_front_back(NOP_front_back, x_bounds, y_bounds, z_bounds, t_bounds)
    data_nsew = np.vstack([data_north[:, 0:4], data_south[:, 0:4], data_east[:, 0:4], data_west[:, 0:4],
                           data_front[:, 0:4], data_back[:, 0:4]])

    # NONDIMENSIONALIZE SPACE AND TIME (U_ref = 1, so T_ref = L_ref)
    for arr in (data_A, data_PDE, data_north, data_south, data_east, data_west,
                data_front, data_back, data_nsew):
        arr[:, :4] /= L_REF

    print("Assembling data frames\n")
    data_EW = pd.DataFrame(data=np.hstack([data_east[:, 0:3], data_west[:, 0:4]]),
                           columns=["x_E", "y_E", "z_E", "x_W", "y_W", "z_W", "t_EW"])
    data_FB = pd.DataFrame(data=np.hstack([data_front[:, 0:3], data_back[:, 0:4]]),
                           columns=["x_F", "y_F", "z_F", "x_B", "y_B", "z_B", "t_FB"])
    data_N = pd.DataFrame(data=data_north[:, [0, 1, 2, 3, 7]], columns=["x_N", "y_N", "z_N", "t_N", "p_N"])
    # As in 2D, only the FIRST time snapshot of the side faces enters the no-slip loss
    # ([:NOP_east[0]]); the later ones are used only by the periodic pairing.
    data_NSEW = pd.DataFrame(data=np.vstack([data_north[:, 0:7], data_south, data_east[:NOP_east[0]], data_west[:NOP_west[0]],
                                             data_front[:NOP_front_back[0]], data_back[:NOP_front_back[0]]]),
                             columns=["x_NSEW", "y_NSEW", "z_NSEW", "t_NSEW", "u_NSEW", "v_NSEW", "w_NSEW"])
    data_A = pd.DataFrame(data=data_A, columns=["x_A", "y_A", "z_A", "t_A", "a_A"])
    data_PDE = pd.DataFrame(data=np.vstack([data_PDE, np.hstack([data_nsew, np.zeros((len(data_nsew), 1))])]),
                            columns=["x_PDE", "y_PDE", "z_PDE", "t_PDE", "f_PDE"])
    return dict(A=data_A, PDE=data_PDE, N=data_N, EW=data_EW, FB=data_FB, NSEW=data_NSEW)


def lift_2d_initial_weights(path="initial_weights.h5"):
    ''' Lifts the 2D starting weights (the Keras checkpoint every 2D run loads) to the 3D
    network, so the 3D run starts from the same function the 2D run does:
        u3(x, y, z, t) = u2(x, z, t),   w3 = v2,   p3 = p2,   a3 = a2,   v3 = 0.
    The first-layer kernel gets a zero row for the new y input and the 2D y row moves to
    z; the new v output starts at zero. Returns Keras-convention arrays -- kernels are
    (in, out) -- as {layer_name: (kernel, bias)}, with hidden layers named "dense",
    "dense_1", ... and outputs "output_u", "output_v", "output_w", "output_p", "output_a". '''

    weights = {}
    with h5py.File(path, "r") as f:
        for name in f:
            if name.startswith("dense") or name.startswith("output_"):
                group = f[name][name]
                kernel = next(np.array(group[k]) for k in group if k.startswith("kernel"))
                bias = next(np.array(group[k]) for k in group if k.startswith("bias"))
                weights[name] = (kernel, bias)

    k2, b2 = weights["dense"]
    assert k2.shape[0] == 3, "expected a 2D (x, y, t) input layer, got %s" % (k2.shape,)
    weights["dense"] = (np.stack([k2[0], np.zeros_like(k2[0]), k2[1], k2[2]]), b2)
    weights["output_w"] = weights["output_v"]
    weights["output_v"] = (np.zeros_like(weights["output_w"][0]), np.zeros_like(weights["output_w"][1]))
    return weights
