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
  - The LBPM lattice is CUBIC, with twice as many cells in Z as in X/Y
    (Domain/nproc * Domain/n in t14_forced_sphere.db). Earlier data had
    Domain/L = 1,1,1, which made the h5 writer normalise every axis
    independently to [0, 1], squashing Z by a factor of two and turning the
    spherical drop into a 2:1 oblate ellipsoid. The .db now declares
    L = 1,1,2, but this module stays robust either way: each axis is divided
    by its own stored spacing, recovering cubic lattice cells regardless.
  - Body force / gravity is applied along +Z (Color/F = 0, 0, 2e-6). Because the
    box is triply periodic and the force acts PER UNIT MASS, both phases feel
    the same acceleration and the whole domain accelerates together: the
    measured bulk acceleration is 1.98e-6 cells/step^2 against the 2e-6 body
    force, and drop-to-carrier slip is only 0.2% of sqrt(g'D). The drop
    therefore never reaches a terminal velocity and barely rises relative to
    its surroundings. Two consequences, both deliberate: U_ref is the buoyant
    scale rather than the drop's own speed (see _measure_scales), and the
    momentum equation uses FULL gravity rather than a reduced (rho - rho_mean)
    one (see PDE_caller in 3d_rising_bubble_train.py).
  - The .db sets Domain/BC = 0, i.e. FULLY PERIODIC in x, y and z, with no
    solid geometry (nspheres = 0, NumberOfPlateSets = 0). The end planes are
    not walls: <w> on both z faces equals the bulk drift speed, and the two
    end planes match each other three orders of magnitude more closely than
    an unrelated plane pair. Periodicity is built into the network itself
    (a sin/cos embedding of x, y, z with the domain lengths as periods, see
    utilities.NNCreator.get_model_dnn), so no face-pair points are generated;
    scales["period"] carries those lengths. The otherwise-undetermined
    pressure gauge is pinned at a single corner point. If the .db is switched
    to a pressure BC in z, use the 2D script's north-outlet structure instead.
  - T_MAX optionally discards trailing snapshots on load, before any scale is
    measured. It is None for the current dataset, which is used whole.
  - Raw levelset convention is +8 = carrier phase, -8 = drop phase; it is
    negated below (as the 2D pipeline does) so that a > 0 => drop (a=1),
    matching the a=1 branch of mu/rho in 3d_rising_bubble_train.py. Note that
    here the a=1 phase is the DENSE one (rhoA = 5.908 vs rhoB = 0.580), the
    opposite of the 2D light-bubble-in-water case.

Nondimensionalisation. The h5 "time" values are raw LBM timestep indices, not
seconds. Dividing them by L_ref the way the 2D pipeline does put t in the tens
of thousands, which saturates every tanh in the first layer and drives all
autograd derivatives to ~1e-11. Space and time are therefore scaled by
references measured from the dataset itself (see _measure_scales):

    L_ref = drop radius from its equilibrium volume, in lattice cells
    U_ref = buoyant velocity scale sqrt(g' D), in cells per timestep, where
            g' = |F_z| * |rhoA - rhoB| / rhoB (see _measure_scales -- the drop never
            reaches a terminal velocity in this triply periodic box, so its own speed
            is not a usable reference)
    T_ref = L_ref / U_ref

which reproduces the O(1) input ranges the working 2D script feeds its network
(2D: x in [-2,2], y in [-4,4], t in [0,12]). The measured values are printed at
generation time -- check them against those ranges after regenerating the CFD
data. sigma is measured too, so no LBPM-run-specific number is hardcoded except
the densities, viscosity and body force, which cannot be recovered from the h5
and must be synced with the .db by hand. '''


# Physical parameters of the LBPM run. THESE MUST BE KEPT IN SYNC BY HAND with the
# source-of-truth config, twophasePINN/t14_forced_sphere.db (the repo root, NOT the
# stale copy in the home directory) -- unlike L_ref/U_ref/sigma below, they cannot be
# recovered from the h5 output, so changing the .db without changing them here
# silently trains against the wrong physics.
# rho1/mu1 are the a=1 (drop) phase, rho2/mu2 the a=0 (carrier) phase.
RHO_DROP = 5.9082346452      # Color/rhoA  (Peng-Robinson coexistence density at T_Tc=0.9)
RHO_GAS = 0.5799723133       # Color/rhoB
NU = 0.1666666667            # Chemicalpotential/niu_1 = niu_2 = (tauA-0.5)/3, tauA = tauB = 1.0
MU_DROP = RHO_DROP * NU      # 0.984706
MU_GAS = RHO_GAS * NU        # 0.096662
# The .db carries Color/F = 0,0,0 and body_force_z = 0: the acceleration comes from a
# flux increment of 2e-6 added in the chemical-potential model, which is what makes the
# domain gently accelerate along +Z over the run (measured bulk accel 1.97e-6 per step).
BODY_FORCE_Z = 2e-6
# Color/pr_kappa. In the Peng-Robinson free-energy model this is the density-gradient
# (Korteweg) coefficient, and it is what actually produces surface tension in the run.
# The momentum equation uses it directly -- kappa*rho*grad(lap(rho)) -- instead of a
# sharp-interface sigma*curvature term, so no sigma calibration enters training. SIGMA
# below is kept only as a diagnostic.
PR_KAPPA = 0.02
# Fallback surface tension, used only if it cannot be measured from the data.
# Surface tension is emergent from the Peng-Robinson chemical-potential model
# (set by pr_kappa and T_Tc) rather than given in the .db, so _measure_sigma
# recovers it from the pressure field instead of trusting this number.
SIGMA_FALLBACK = 0.0030722
# Optional cap on the LBM timestep kept from the h5; None keeps every snapshot.
# The current dataset spans t = 19600 .. 30000 and is used whole. Set this only to cut a
# trailing transient -- a value below the FIRST stored timestep silently empties the
# dataset, so get_training_data_3d raises instead of failing later on an empty array.
T_MAX = None


def _measure_sigma(levelset, pressure, R):
    ''' Recovers the surface tension from the data by Laplace's law, sigma = dp * R / 2.

    The inside/outside pressures are both sampled on the plane through the drop centroid
    so that the hydrostatic gradient cancels -- comparing a drop-interior average against
    a whole-domain carrier average instead biases the result by the body force acting over
    the drop radius. Only deep-phase cells (|levelset| >= 7.5) are used, keeping the
    diffuse interface and its Peng-Robinson pressure spike out of both averages, and only
    settled snapshots count (the drop is still equilibrating in the first few). '''

    if pressure is None:
        print("  WARNING: no 'pressure' dataset in the h5, using SIGMA_FALLBACK = %g" % SIGMA_FALLBACK)
        return SIGMA_FALLBACK, float("nan")

    values = []
    for snap, pres in zip(levelset[len(levelset) // 3:], pressure[len(pressure) // 3:]):
        drop = np.argwhere(snap > 0)
        if len(drop) == 0:
            continue
        z_c = int(round(drop[:, 0].mean()))
        plane_ls, plane_p = snap[z_c], pres[z_c]
        inside = plane_p[plane_ls >= 7.5]
        outside = plane_p[plane_ls <= -7.5]
        if len(inside) < 10 or len(outside) < 10:
            continue
        values.append((inside.mean() - outside.mean()) * R / 2.0)

    if not values:
        print("  WARNING: could not measure sigma from the data, using SIGMA_FALLBACK")
        return SIGMA_FALLBACK, float("nan")
    return float(np.median(values)), float(np.std(values))


def _measure_scales(levelset, pressure, times):
    ''' Measures the reference length, velocity and surface tension from the dataset
    itself, so that the generator and the training script cannot drift apart -- and so
    that regenerating the CFD data with different LBPM parameters does not silently leave
    stale constants behind.

    L_ref is the drop radius implied by its equilibrium volume; early snapshots are
    excluded from the median because the drop may still be equilibrating there.

    U_ref is the BUOYANT velocity scale sqrt(g' D), not the drop's own speed. Earlier
    revisions used the mean centroid speed, which presumes the drop has reached a
    terminal velocity. It has not, and cannot: the box is triply periodic and the body
    force acts per unit mass, so both phases feel the same acceleration and the whole
    domain drifts together. Measured on this dataset, drop-to-carrier slip is 1.8e-5
    against a rise speed of ~1e-2 (0.2% of sqrt(g' D)), while the centroid speed ramps
    linearly from 1e-5 to 1.95e-2 cells/step across the window. Averaging that ramp
    yields a number set by where the window happens to start and stop, and it put 1/We
    at 1.5 -- an order above the 2D benchmark's 0.098.

    sqrt(g' D) is instead a property of the configuration, and reproduces the 2D
    benchmark's groups closely: 1/Re(carrier) 0.050, 1/We 0.191, 1/Fr 0.054 against 2D's
    0.040, 0.098, 0.245, with t spanning [0, 11.8] against 2D's [0, 12]. '''

    volumes = np.array([(snap > 0).sum() for snap in levelset], dtype=float)
    settled = volumes[len(volumes) // 3:]
    L_ref = (3.0 * np.median(settled) / (4.0 * np.pi)) ** (1.0 / 3.0)

    g_prime = abs(BODY_FORCE_Z) * abs(RHO_DROP - RHO_GAS) / RHO_GAS
    U_ref = float(np.sqrt(g_prime * 2.0 * L_ref))

    # Diagnostics, printed rather than used: if a future dataset DOES reach a terminal
    # velocity, the centroid speed will flatten and the bulk acceleration will fall well
    # below the body force, and the choice above should be revisited.
    z_centroid = np.array([np.argwhere(snap > 0)[:, 0].mean() for snap in levelset])
    speed = np.gradient(z_centroid, times)
    bulk_accel = float(np.polyfit(times, speed, 1)[0])
    print("  centroid speed %.4g -> %.4g cells/step (buoyant scale sqrt(g'D) = %.4g)"
          % (speed[0], speed[-1], U_ref))
    print("  bulk acceleration %.3g per step vs body force %.3g (ratio %.2f): %s"
          % (bulk_accel, BODY_FORCE_Z, bulk_accel / BODY_FORCE_Z,
             "still accelerating, no terminal velocity"
             if abs(bulk_accel / BODY_FORCE_Z) > 0.5 else "approaching terminal velocity"))

    # Volume-averaged drop fraction, for the reduced-gravity correction in the momentum
    # equation (the domain is triply periodic, so the mean body force must be removed).
    alpha_frac = np.array([(snap > 0).mean() for snap in levelset])
    alpha_mean = float(np.mean(alpha_frac[len(alpha_frac) // 3:]))

    sigma, sigma_spread = _measure_sigma(levelset, pressure, L_ref)

    return float(L_ref), U_ref, alpha_mean, sigma, sigma_spread


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


def get_points_a_3d(NOP, times, X, Y, Z, cell_size, interface_all, normal_all,
                     tangent1_all, tangent2_all, levelset_all):
    ''' Generates the points for the volume fraction loss (3D). '''

    # Offsets are expressed in CELLS. The 2D pipeline hard-coded 0.004/0.008 because its
    # coordinates were physical and its cell was 0.0039 wide, i.e. it stepped 1 to 2 cells
    # off the interface. Here coordinates are in lattice cells, so the same intent has to
    # be written as a multiple of cell_size -- using the 2D literals directly would place
    # the inward (a=1) and outward (a=0) points 0.4% of a cell apart, i.e. on top of each
    # other with contradictory labels.
    refine_start = 1.0 * cell_size
    refine_end = 2.0 * cell_size
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


def get_pressure_gauge(NOP_t, x, y, z, t, p_value=1.0):
    ''' With every face periodic there is no Dirichlet pressure anywhere, leaving the
    pressure determined only up to an additive constant. This pins the gauge at a
    single spatial point -- the (x_min, y_min, z_min) corner, which sits ~64 cells off
    the drop's rise axis and so stays in the carrier phase -- across NOP_t times.
    Only one point is used so the gauge constrains the offset and nothing else. '''
    t_vals = np.linspace(t[0], t[1], NOP_t).reshape(-1, 1)
    ones = np.ones_like(t_vals)
    return np.hstack([x[0] * ones, y[0] * ones, z[0] * ones, t_vals, p_value * ones])


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


def get_training_data_3d(NOP_A, NOP_PDE, NOP_gauge=40,
                          h5_path="../cfd_data/lbpm_bubble_dataset.h5", time_indices=None):
    ''' Generates the training points for all losses from the LBPM 3D drop dataset.

    Args:
        NOP_A: (interface, domain) point counts for the volume fraction loss, per snapshot
        NOP_PDE: (interface, nearfield, domain) point counts for the PDE residual, per snapshot
        NOP_gauge: number of times at which the single-point pressure gauge is imposed
        h5_path: path to lbpm_bubble_dataset.h5
        time_indices: optional explicit array of time-snapshot indices to use, counted
            AFTER the T_MAX cut; if None, every second snapshot is taken. Earlier
            revisions also skipped index 0 because the drop was still equilibrating
            there (81k cells against a settled 99.5k); in the current dataset it is
            settled from the first snapshot (99717 cells against 99602 at the end), so
            it is kept. All 27 snapshots are usable -- the stride of 2 is a cost choice,
            not a correctness one.

    Returns:
        (data, gauge, scales) where data is the dict of dataframes and scales carries the
        measured L_ref / U_ref / T_ref, the nondimensional periodic lengths, and the derived
        physical parameters, so that the training script uses exactly the same
        nondimensionalisation. '''

    with h5py.File(h5_path, "r") as data:
        X = np.array(data["X"], dtype=float)
        Y = np.array(data["Y"], dtype=float)
        Z = np.array(data["Z"], dtype=float)
        times = np.array(data["time"], dtype=float)
        # DROP EVERYTHING PAST T_MAX. Sliced at read time so the discarded snapshots are
        # never loaded and cannot leak into the measured scales.
        assert np.all(np.diff(times) > 0), "h5 time axis is not increasing"
        n_total = len(times)
        n_keep = n_total if T_MAX is None else int(np.searchsorted(times, T_MAX, side="right"))
        if n_keep == 0:
            raise ValueError(
                "T_MAX = %g discards every snapshot: %s stores t = %g .. %g. Raise T_MAX "
                "above the first stored timestep, or set it to None to keep them all."
                % (T_MAX, h5_path, times[0], times[-1]))
        times = times[:n_keep]
        levelset = -np.array(data["levelset"][:n_keep])
        # Only needed to measure the surface tension; fall back if the writer omits it.
        pressure = np.array(data["pressure"][:n_keep], dtype=float) if "pressure" in data else None
    print("Kept %d of %d snapshots (t = %g .. %g, T_MAX = %s)"
          % (n_keep, n_total, times[0], times[-1], T_MAX))

    # UNDO THE PER-AXIS [0,1] NORMALISATION IN THE H5 WRITER.
    # Each axis is divided by its own stored spacing, putting all three back in cubic
    # lattice cells (dx = dy = dz = 1). Without this, Z is compressed 2:1 relative to
    # X and Y and the spherical drop is an oblate ellipsoid.
    X = X / np.diff(X)[0]
    Y = Y / np.diff(Y)[0]
    Z = Z / np.diff(Z)[0]
    dx = dy = dz = 1.0
    cell_size = 1.0

    if time_indices is None:
        time_indices = np.arange(0, len(times), 2)
    L_ref, U_ref, alpha_mean, SIGMA, sigma_spread = _measure_scales(levelset, pressure, times)
    T_ref = L_ref / U_ref

    times = times[time_indices]
    levelset = levelset[time_indices]

    # CENTRE THE DOMAIN ON THE ORIGIN (the 2D case is centred; this one ran 0..90/180)
    # and take the time origin at the first retained snapshot.
    x_mid, y_mid, z_mid = X.mean(), Y.mean(), Z.mean()
    X, Y, Z = X - x_mid, Y - y_mid, Z - z_mid
    t_origin = times[0]
    times = times - t_origin

    t_bounds = [times[0], times[-1]]
    x_bounds = [X[0], X[-1]]
    y_bounds = [Y[0], Y[-1]]
    z_bounds = [Z[0], Z[-1]]

    print("\nDistributing points for time snapshots (LBM steps, relative to %g):\n" % t_origin, times)
    print("Number of time snapshots: ", len(times))
    print("Reference scales: L_ref = %.4f cells, U_ref = %.6g cells/step, T_ref = %.1f steps"
          % (L_ref, U_ref, T_ref))
    print("Measured surface tension: sigma = %.6g (snapshot spread +/- %.2g)" % (SIGMA, sigma_spread))
    print("Nondimensional bounds: x", [b / L_ref for b in x_bounds],
          "y", [b / L_ref for b in y_bounds],
          "z", [b / L_ref for b in z_bounds],
          "t", [b / T_ref for b in t_bounds], "\n")

    interface, normal, tangent1, tangent2 = compute_normals_3d(X, Y, Z, levelset, dx, dy, dz)
    print("Generating points for A")
    data_A = get_points_a_3d(NOP_A, times, X, Y, Z, cell_size, interface, normal, tangent1, tangent2, levelset)
    print("Generating points for PDE")
    data_PDE = get_points_pde_3d(NOP_PDE, times, X, Y, Z, cell_size, interface, normal, tangent1, tangent2, levelset)
    print("Generating points for GAUGE")
    data_gauge = get_pressure_gauge(NOP_gauge, x_bounds, y_bounds, z_bounds, t_bounds)

    # NONDIMENSIONALIZE SPACE BY L_ref AND TIME BY T_ref (they differ here, unlike the
    # 2D pipeline where U_ref = 1 made T_ref = L_ref)
    for arr in (data_A, data_PDE, data_gauge):
        arr[:, :3] /= L_ref
        arr[:, 3] /= T_ref

    # PERIODIC LENGTHS. The period is the full cell count along each axis (N cells of
    # size 1), not X[-1] - X[0], which is one cell short: cell centres 0.5 and N - 0.5
    # are neighbours across the periodic seam, N - 1 cells apart in the interior.
    period = [len(X) * dx / L_ref, len(Y) * dy / L_ref, len(Z) * dz / L_ref]

    print("Assembling data frames\n")
    frames = dict(
        A=pd.DataFrame(data=data_A, columns=["x_A", "y_A", "z_A", "t_A", "a_A"]),
        PDE=pd.DataFrame(data=data_PDE, columns=["x_PDE", "y_PDE", "z_PDE", "t_PDE", "f_PDE"]),
    )
    gauge = pd.DataFrame(data=data_gauge, columns=["x_G", "y_G", "z_G", "t_G", "p_G"])

    rho_mean = RHO_GAS + (RHO_DROP - RHO_GAS) * alpha_mean
    scales = dict(
        L_ref=L_ref, U_ref=U_ref, T_ref=T_ref, t_origin=float(t_origin),
        mu=[MU_DROP, MU_GAS], rho=[RHO_DROP, RHO_GAS], sigma=SIGMA, g=BODY_FORCE_Z,
        alpha_mean=alpha_mean, rho_mean=rho_mean, period=period, kappa=PR_KAPPA,
    )
    rho_ref = RHO_GAS
    print("Volume-averaged drop fraction <a> = %.5f, rho_mean = %.5f (%.4f rho_ref)"
          % (alpha_mean, rho_mean, rho_mean / rho_ref))
    print("Nondimensional groups: 1/Re(carrier) = %.4f, 1/Re(drop) = %.4f, Ko = %.4f, 1/Fr = %.4f"
          % (MU_GAS / (rho_ref * U_ref * L_ref), MU_DROP / (rho_ref * U_ref * L_ref),
             PR_KAPPA * rho_ref / (L_ref ** 2 * U_ref ** 2), BODY_FORCE_Z * L_ref / U_ref ** 2))
    print("  (Ko is the Korteweg coefficient now used in the momentum equation; the measured"
          " sigma gives 1/We = %.4f, diagnostic only)" % (SIGMA / (rho_ref * U_ref ** 2 * L_ref)))

    return frames, gauge, scales
