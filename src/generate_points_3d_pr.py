import math
import numpy as np
import pandas as pd
import h5py


''' Training-point generator for the Peng-Robinson (eos_type = "cp") PINN,
pytorch_3d_pr_train.py. Same LBPM run family as generate_points_3d.py, but it reads the
density field the h5 now carries and supervises the network on DENSITY (plus the
levelset-derived phase indicator a) rather than on a binary volume fraction alone.

What the LBPM cp model integrates (LBPM-UNC-4, cuda/LIBB.cu, all in lattice units):

    mu      = k^2 * mu_PR(rho) - kappa * lap(rho)              dvc_update_chemical_potential
    F       = -rho grad(mu) + (1/3) grad(rho) + rho g          dvc_update_force_cp
    p (h5)  = P_PR(rho)                                         dvc_update_pressure

The +(1/3) grad(rho) cancels the LBM's own ideal-gas pressure, and Gibbs-Duhem
(rho grad(mu_PR) = grad(P_PR)) turns the rest into

    rho Du/Dt = -k^2 grad(P_PR(rho)) + (kappa rho + beta) grad(lap(rho)) + rho g + div(tau)

beta (LATTICE_BETA) is not in the continuum model above; it is measured from the data.
Checked against lbpm_bubble_dataset_newrun_nowrap.h5 (diagnose_rho_residuals.py):
  - The h5 pressure IS P_PR(density) to 2e-8, so p is not independent of rho. The
    training script substitutes p = P_PR(rho) exactly rather than learning p.
  - At the interface the CFD satisfies k^2 grad(P_PR) = (kappa rho + beta) grad(lap(rho))
    with beta = 0.085 ~= 1/12 (lattice units) to 2%; without beta the two sides differ by
    2.3x and the trainer's momentum residual on the TRUTH is ~1.5e4 per point, which the
    network could only lower by bending rho away from the data. Origin of beta (presumably
    an O(dx^2) lattice term) is unverified.
  - The h5 velocity is the bare moment sum(f c) / rho. The physical (Guo) velocity is
    u = u_h5 + F / (2 rho), F the force density below; without it the interface carries a
    frozen "parasitic" shell and continuity fails ~2e4x worse. get_training_data_3d_pr
    applies the correction before supervising u, v, w.

Body force. LBPM ramps it (ColorModel.cpp, lfz += flux_increment every 20000 steps) over
the whole periodic domain, so g(t) = FORCE_INCREMENT * floor(t / FORCE_RAMP_INTERVAL):
zero at the t = 19600 snapshot, 2e-6 from t = 20000 on. The stored velocities do NOT
accelerate at that rate (~0.74 of it, even in pure gas; see measure_bulk_acceleration),
which is why the training script can learn g(t) instead.

Grid conventions (levelset raw -8 = drop, +8 = carrier; fields stored (time, Z, Y, X);
per-axis [0, 1] h5 normalisation undone to cubic lattice cells; fully periodic) are as in
generate_points_3d.py. '''


# Point-sampling helpers, frozen from generate_points_3d.py as of 703d53e. That module
# has since become the Hysing 2D-in-3D generator (absolute refinement widths, a single
# cell_size), which is wrong for the LBPM lattice-unit data this module reads.


def get_points_for_interface_3d(NOP, t_value, interface, normal, tangent1, tangent2,
                                 cell_size, refine_start, refine_end):
    ''' 3D analogue of generate_points.get_points_for_interface. Samples points along
    the interface normal (inward = drop side, outward = carrier side), staggered
    across a small stencil of tangential offsets (built from tangent1/tangent2) so that
    points are spread across the interface surface between grid nodes, not just at the
    interface nodes themselves.

    A 3D interface carries ~12700 cells per snapshot against the few hundred points
    actually wanted, so a random subset of interface cells is drawn up front instead
    of building ~127k candidates and throwing almost all of them away. '''

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
    ''' 3D analogue of generate_points.get_points_for_domain, sampling points far from
    the interface (|levelset| >= max_levelset).

    The 2D routine stratifies by x column so every column is represented. That is not
    reachable in 3D -- there are 8100 (x, y) columns against a few hundred to a few
    thousand requested points -- so this draws a uniform random sample over the whole
    grid by rejection instead. '''

    kept = []
    n_kept = 0
    while n_kept < NOP:
        draw = max(NOP - n_kept, 1024)
        iz = np.random.randint(0, len(Z), draw)
        iy = np.random.randint(0, len(Y), draw)
        ix = np.random.randint(0, len(X), draw)
        vals = levelset[iz, iy, ix]
        keep = (np.abs(vals) >= max_levelset) | (np.abs(vals) == 8)
        iz, iy, ix, vals = iz[keep], iy[keep], ix[keep], vals[keep]
        kept.append(np.column_stack([X[ix], Y[iy], Z[iz],
                                     t_value * np.ones(len(ix)),
                                     (vals > 0).astype(float)]))
        n_kept += len(ix)
    return np.vstack(kept)[:NOP, :]


def get_points_pde_3d(NOP, times, X, Y, Z, cell_size, interface_all, normal_all,
                       tangent1_all, tangent2_all, levelset_all):
    ''' Generates the residual (collocation) points for the PDEs (3D). '''

    # Offsets in CELLS, see the note in get_points_a_3d. The 2D literals 0.001 and 0.1
    # were 0.26 cells and 25.6 cells, the latter being 0.4 * L_ref; the nearfield band is
    # kept at 0.4 * L_ref here (L_ref ~ 28.8 cells) so it scales with the drop, not the grid.
    refine_interface_start = 0.0
    refine_interface_end = 0.25 * cell_size
    refine_nearfield_end = 12.0 * cell_size
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


# Physical parameters of the LBPM run, from t14_forced_sphere.db. KEEP IN SYNC BY HAND --
# none of these can be recovered from the h5.
RHO_LIQ = 5.9082346452       # Color/rhoA = Chemicalpotential/rho_1 (drop)
RHO_GAS = 0.5799723133       # Color/rhoB = Chemicalpotential/rho_2 (carrier)
NU = 0.1666666667            # niu_1 = niu_2 = (tau - 0.5)/3, tauA = tauB = 1.0
PR_A = 0.0408163265          # pr_a = 2/49
PR_B = 0.0952380952          # pr_b = 2/21
PR_OMEGA = 0.344             # pr_omega
PR_KAPPA = 0.02              # pr_kappa (Korteweg coefficient)
T_TC = 0.9                   # Chemicalpotential/T_Tc
K_FACTOR = 0.5               # Domain/k_factor; enters the momentum equation as k^2
# Body-force ramp, hardcoded in ColorModel.cpp (flux_increment = 2e-6, ramped every
# 20000 steps); the .db's F and body_force_z are both 0.
FORCE_INCREMENT = 2e-6
FORCE_RAMP_INTERVAL = 20000
# Extra rho-independent Korteweg-like coefficient the CFD obeys (lattice units, see the
# module docstring): F_kort = (kappa rho + LATTICE_BETA) grad(lap(rho)). Fit: 0.0847.
LATTICE_BETA = 1.0 / 12.0


def pr_constants():
    ''' T and a*alpha(T) exactly as LBPM computes them (Tc = 0.17014 a / b, R = 1). '''
    Tc = 0.17014 * PR_A / PR_B
    m = 0.37464 + 1.54226 * PR_OMEGA - 0.26992 * PR_OMEGA ** 2
    alpha_T = (1.0 + m * (1.0 - np.sqrt(T_TC))) ** 2
    return T_TC * Tc, PR_A * alpha_T


def pressure_pr(rho):
    ''' P_PR(rho), lattice units. Works on numpy arrays and torch tensors alike. '''
    T, aT = pr_constants()
    b = PR_B
    return T * rho / (1.0 - b * rho) - aT * rho * rho / (1.0 + 2.0 * b * rho - (b * rho) ** 2)


def periodic_d1(f, axis):
    ''' Central first derivative on the periodic lattice, unit spacing. '''
    return 0.5 * (np.roll(f, -1, axis=axis) - np.roll(f, 1, axis=axis))


def periodic_lap(f):
    ''' 7-point Laplacian on the periodic lattice, unit spacing. '''
    return sum(np.roll(f, -1, axis=a) - 2.0 * f + np.roll(f, 1, axis=a) for a in range(f.ndim))


def lbpm_force(rho, t_lattice):
    ''' LBPM's cp-model force density (dvc_update_force_cp), lattice units, for a (Z, Y, X)
    density snapshot: F = -rho grad(mu) + (1/3) grad(rho) + rho g, written through
    Gibbs-Duhem as -k^2 grad(P_PR) + kappa rho grad(lap(rho)) + (1/3) grad(rho) + rho g.
    Returns [Fx, Fy, Fz]. '''
    P = pressure_pr(rho)
    lap_rho = periodic_lap(rho)
    F = []
    for c, axis in (("x", 2), ("y", 1), ("z", 0)):
        f = -K_FACTOR ** 2 * periodic_d1(P, axis) + PR_KAPPA * rho * periodic_d1(lap_rho, axis) + \
            periodic_d1(rho, axis) / 3.0
        if c == "z":
            f = f + rho * body_force(t_lattice)
        F.append(f)
    return F


def body_force(t_lattice):
    ''' g(t) of LBPM's ramp, numpy version (the training script has a torch twin). '''
    return FORCE_INCREMENT * np.floor(t_lattice / FORCE_RAMP_INTERVAL)


def measure_bulk_acceleration(h5_path):
    ''' Effective uniform body force from the domain momentum budget. In a periodic box
    every internal force (pressure, Korteweg, viscous) integrates to zero, so
    d<rho w>/dt / <rho> is exactly the net uniform force per unit mass the flow actually
    felt. It is the reference the learned g(t) is compared against -- on this dataset it
    sits at ~0.74 of the nominal 2e-6 and falls off after t ~ 24000.

    Returns (t_mid, accel), both in lattice units, one entry per snapshot interval. '''
    with h5py.File(h5_path, "r") as data:
        times = np.array(data["time"], dtype=float)
        momentum, mass = [], []
        for i in range(len(times)):
            rho = np.array(data["density"][i], dtype=float)
            # Half-force corrected momentum: the gradient parts of F sum to zero over the
            # periodic box, so sum(rho u) = sum(rho u_h5) + mass g(t) / 2.
            momentum.append((rho * np.array(data["velocityZ"][i], dtype=float)).sum() +
                            0.5 * rho.sum() * body_force(times[i]))
            mass.append(rho.sum())
    momentum, mass = np.array(momentum), np.array(mass)
    accel = np.diff(momentum) / np.diff(times) / mass[:-1]
    return 0.5 * (times[1:] + times[:-1]), accel


def _phase_indicator(levelset_raw):
    ''' a = 1 in the drop, 0 in the carrier, linear across the +-8 levelset band. '''
    return np.clip((8.0 - levelset_raw) / 16.0, 0.0, 1.0)


def get_points_data_3d(NOP, t_value, X, Y, Z, fields, levelset_raw):
    ''' Supervised points ON GRID NODES, so every target is the exact stored value and no
    interpolation enters. NOP = (band, bulk): `band` nodes inside the diffuse interface
    (|levelset| < 8), where rho changes by 10x over a few cells and the network needs the
    most help, and `bulk` nodes drawn uniformly from the rest. '''

    band = np.argwhere(np.abs(levelset_raw) < 8.0)
    pick = np.random.choice(len(band), min(NOP[0], len(band)), replace=False)
    iz, iy, ix = band[pick].T

    bz = np.random.randint(0, len(Z), 4 * NOP[1])
    by = np.random.randint(0, len(Y), 4 * NOP[1])
    bx = np.random.randint(0, len(X), 4 * NOP[1])
    keep = np.abs(levelset_raw[bz, by, bx]) >= 8.0
    bz, by, bx = bz[keep][:NOP[1]], by[keep][:NOP[1]], bx[keep][:NOP[1]]

    iz, iy, ix = np.concatenate([iz, bz]), np.concatenate([iy, by]), np.concatenate([ix, bx])
    cols = [X[ix], Y[iy], Z[iz], t_value * np.ones(len(ix))]
    cols += [f[iz, iy, ix] for f in fields]
    cols.append(_phase_indicator(levelset_raw[iz, iy, ix]))
    return np.column_stack(cols)


def get_training_data_3d_pr(NOP_DATA, NOP_PDE, supervised=("rho",),
                             h5_path="../cfd_data/lbpm_bubble_dataset_newrun_nowrap.h5",
                             time_indices=None):
    ''' Args:
        NOP_DATA: (interface band, bulk) supervised grid nodes per snapshot
        NOP_PDE: (interface, nearfield, domain) collocation points per snapshot
        supervised: which of "rho", "u", "v", "w", "p" to supervise besides a. Density
            alone is the physics-inferred setup (velocities and pressure come from the
            PDEs); adding "u", "v", "w" gives a fully data-assisted run.
        time_indices: snapshots to use; None takes all of them. Every second one (800
            steps) is too coarse for rho_t once the drop is moving: continuity on the truth
            is ~10x worse at the interface late in the run than with 400 steps.

    Returns (frames, scales). frames["DATA"] has x, y, z, t, then one column per
    supervised field (nondimensional), then a. '''

    h5_names = dict(rho="density", u="velocityX", v="velocityY", w="velocityZ", p="pressure")
    with h5py.File(h5_path, "r") as data:
        X = np.array(data["X"], dtype=float)
        Y = np.array(data["Y"], dtype=float)
        Z = np.array(data["Z"], dtype=float)
        times_all = np.array(data["time"], dtype=float)
        if time_indices is None:
            time_indices = np.arange(len(times_all))
        time_indices = np.sort(np.asarray(time_indices))
        times = times_all[time_indices]
        levelset_raw = np.array(data["levelset"][time_indices], dtype=float)
        density = np.array(data["density"][time_indices], dtype=float)
        raw = {k: density if k == "rho" else np.array(data[h5_names[k]][time_indices], dtype=float)
               for k in supervised}

    # u_h5 -> physical velocity. Needs all three force components only for the
    # supervised ones, but lbpm_force is cheap enough to compute whole per snapshot.
    vel_keys = [k for k in ("u", "v", "w") if k in supervised]
    if vel_keys:
        for i, t in enumerate(times):
            F = lbpm_force(density[i], t)
            for k in vel_keys:
                raw[k][i] += 0.5 * F["uvw".index(k)] / density[i]

    # Cubic lattice cells, centred on the origin (see generate_points_3d.py).
    X, Y, Z = X / np.diff(X)[0], Y / np.diff(Y)[0], Z / np.diff(Z)[0]
    X, Y, Z = X - X.mean(), Y - Y.mean(), Z - Z.mean()
    dx = dy = dz = cell_size = 1.0

    # REFERENCE SCALES. L_ref is the drop radius from its volume (rho above the midpoint
    # density); U_ref the buoyant scale sqrt(g' D) at the force the run actually applies
    # over this window (the drop never reaches a terminal velocity in a periodic box, see
    # generate_points_3d._measure_scales). rho_ref is the carrier density; p_scale is the
    # coexistence pressure, which puts p* at O(1).
    rho_mid = 0.5 * (RHO_LIQ + RHO_GAS)
    volume = np.median([(snap > rho_mid).sum() for snap in density])
    L_ref = float((3.0 * volume / (4.0 * np.pi)) ** (1.0 / 3.0))
    g_active = FORCE_INCREMENT
    U_ref = float(np.sqrt(g_active * abs(RHO_LIQ - RHO_GAS) / RHO_GAS * 2.0 * L_ref))
    T_ref = L_ref / U_ref
    rho_ref = RHO_GAS
    p_scale = float(pressure_pr(RHO_GAS))

    t_origin = times[0]
    t_rel = times - t_origin
    print("Snapshots (LBM steps):", times.astype(int))
    print("Reference scales: L_ref = %.4f cells, U_ref = %.6g cells/step, T_ref = %.1f steps, "
          "rho_ref = %.5f, p_scale = %.5g" % (L_ref, U_ref, T_ref, rho_ref, p_scale))

    # Collocation points: reuse the interface-refined sampler. It wants the levelset with
    # the drop positive (only |levelset| and the gradient direction are used).
    levelset = -levelset_raw
    interface, normal, tangent1, tangent2 = compute_normals_3d(X, Y, Z, levelset, dx, dy, dz)
    print("Generating PDE points")
    data_PDE = get_points_pde_3d(NOP_PDE, t_rel, X, Y, Z, cell_size,
                                 interface, normal, tangent1, tangent2, levelset)

    print("Generating supervised points (%s + a)" % ", ".join(supervised))
    field_scale = dict(rho=rho_ref, u=U_ref, v=U_ref, w=U_ref, p=p_scale)
    data_DATA = np.vstack([
        get_points_data_3d(NOP_DATA, t_value, X, Y, Z,
                           [raw[k][i] / field_scale[k] for k in supervised], levelset_raw[i])
        for i, t_value in enumerate(t_rel)])

    for arr in (data_DATA, data_PDE):
        arr[:, :3] /= L_ref
        arr[:, 3] /= T_ref

    period = [len(X) * dx / L_ref, len(Y) * dy / L_ref, len(Z) * dz / L_ref]

    frames = dict(
        DATA=pd.DataFrame(data_DATA, columns=["x", "y", "z", "t"] + list(supervised) + ["a"]),
        PDE=pd.DataFrame(data_PDE[:, :4], columns=["x", "y", "z", "t"]),
    )

    k2 = K_FACTOR ** 2
    print("Nondimensional groups: 1/Re = %.4f (times rho*), Eu = k^2 p_scale/(rho_ref U^2) = %.4f, "
          "Ko = %.4f, 1/Fr = %.4f"
          % (NU / (U_ref * L_ref), k2 * p_scale / (rho_ref * U_ref ** 2),
             PR_KAPPA * rho_ref / (L_ref ** 2 * U_ref ** 2), g_active * L_ref / U_ref ** 2))
    print("Nondimensional bounds: x +-%.3f, y +-%.3f, z +-%.3f, t [0, %.3f]\n"
          % (period[0] / 2, period[1] / 2, period[2] / 2, t_rel[-1] / T_ref))

    # Knots for a learned g(t): every stored snapshot in the training window, not just the
    # sampled ones, nondimensionalised like t.
    in_window = (times_all >= times[0]) & (times_all <= times[-1])
    t_knots = (times_all[in_window] - t_origin) / T_ref
    t_budget, g_budget = measure_bulk_acceleration(h5_path)

    scales = dict(L_ref=L_ref, U_ref=U_ref, T_ref=T_ref, t_origin=float(t_origin),
                  rho_ref=rho_ref, p_scale=p_scale, period=period, t_knots=t_knots,
                  t_budget=t_budget, g_budget=g_budget)
    return frames, scales
