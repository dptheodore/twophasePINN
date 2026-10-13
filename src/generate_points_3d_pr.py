import numpy as np
import pandas as pd
import h5py

from generate_points_3d import compute_normals_3d, get_points_pde_3d

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
    training script keeps p as an output but ties it to rho through an EOS residual.
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
