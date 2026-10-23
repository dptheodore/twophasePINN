import numpy as np
import math
import pandas as pd
import h5py

from generate_points_3d import (get_points_for_interface_3d, get_points_for_domain_3d,
                                compute_normals_3d, get_north, get_east_west, get_front_back,
                                _random_times)
import generate_points_3d_pr as pr

''' generate_points_3d.py (the Basilisk sharp-interface pipeline) pointed at the LBPM
Peng-Robinson run "lbpm_bubble_dataset_newrun_nowrap.h5" (t14_forced_sphere.db, 27
snapshots t = 19600..30000, 90 x 90 x 180 cells, triply periodic). The trainer,
pytorch_3d_lbpm_train.py, keeps the incompressible two-phase equations of
pytorch_3d_rising_bubble_train.py unchanged; only the parameters and the boundary
conditions differ. LBPM is a diffuse-interface Peng-Robinson model, so the sharp-interface
parameters are approximations, set as follows (lattice units throughout):

    rho1, rho2   MEASURED medians of the drop core / far carrier (levelset at -8 / +8):
                 ~5.920 and ~0.456. The .db coexistence values (5.908, 0.580) are not what
                 the run sits at -- the carrier is 21% lighter.
    mu1, mu2     rho_i * NU, NU = (tau - 0.5)/3 = 1/6 from the .db (tauA = tauB = 1).
    sigma        MEASURED Laplace jump of the mechanical pressure k^2 P_PR, core (r < R/3)
                 minus far field (r > R + 15 cells), times L_ref / 2: ~0.0255. The
                 Korteweg integral kappa * int(rho_r^2) gives ~0.030 instead; the 15-20%
                 gap is the radius ambiguity of a 14-cell interface on a 29-cell drop.
    g            MEASURED mean bulk acceleration over the window from the periodic
                 momentum budget (pr.measure_bulk_acceleration): ~1.13e-6 along +z. The
                 nominal ramp is 2e-6, but the budget shows 0 before t = 20000, ~1.48e-6
                 to t ~ 24000, then a fall to ~0.54e-6. A constant g cannot match all of it.
    p            mechanical pressure k^2 P_PR(rho). The h5 "pressure" IS P_PR(rho) (checked
                 to 1e-10); LBPM's +(1/3) grad(rho) force cancels the lattice rho c_s^2.
    u, v, w      h5 velocity + F/(2 rho) (Guo half-force), validation only.
    a            (rho - rho2) / (rho1 - rho2), clipped to [0, 1] and interpolated
                 trilinearly at the sample points. NOT the levelset sign as in the Basilisk
                 generator: the interface is ~14 cells wide (10-90%), so a sign label even
                 4 cells off the midline would be ~0.2 wrong. With a linear in rho, the
                 trainer's rho(a) reproduces the CFD density exactly.

Reference scales: L_ref = drop radius from the volume above the mid density (~28.8
cells); U_ref = sqrt(g' D) at the NOMINAL .db values (g = 2e-6, coexistence densities), the
same ~0.0325 cells/step as generate_points_3d_pr.py, so the two pipelines share T_ref
(~885 steps); rho_ref = rho2. Coordinates are cells centred on the origin, then / L_ref; t is
(t - 19600) / T_ref.

Boundary conditions. The box is periodic on all six faces, so Basilisk's no-slip walls
go and the z faces get the same periodic pairing as x and y:
    north/south (z)      periodic pairing of u, v, w, p; plus p = 1 on north (gauge only)
    east/west (x)        periodic pairing of u, v, w, p
    front/back (y)       periodic pairing of u, v, w, p
    t = 0                u = v = w = 0 everywhere (the CFD is at rest to 0.004 U_ref),
                         which takes over the walls' job of fixing the Galilean frame
The pressure gauge mirrors the Basilisk file's: the CFD p is shifted per snapshot so its
mean over the top cell layer is p* = 1.

Snapshots: VAL_INDICES (every third from the third, 8 of 27) are held out for validation;
the other 19, including the first and last, are training snapshots. '''

H5_PATH = "../cfd_data/lbpm_bubble_dataset_newrun_nowrap.h5"
VAL_INDICES = np.arange(2, 26, 3)
BULK_LEVELSET = 8.0          # |levelset| is clipped at 8 cells; == 8 means outside the band

_SCALES = None


def _cell_axes(data):
    ''' Cell-centre coordinates in cells, centred on the origin (the h5 stores [0, 1] per
    axis, so dz = dx / 2 there), and the box size in cells. '''
    axes = []
    for k in ("X", "Y", "Z"):
        c = np.array(data[k], dtype=float)
        c = c / np.diff(c)[0]
        axes.append(c - c.mean())
    return axes, [len(a) for a in axes]


def mechanical_pressure(rho):
    ''' k^2 P_PR(rho): the pressure the LBPM momentum equation sees, lattice units. '''
    return pr.K_FACTOR ** 2 * pr.pressure_pr(rho)


def get_scales():
    ''' Measures the phase densities, sigma and the bulk acceleration from the h5 and
    returns them with the reference scales (lattice units). Cached. '''
    global _SCALES
    if _SCALES is not None:
        return _SCALES

    with h5py.File(H5_PATH, "r") as data:
        times = np.array(data["time"], dtype=float)
        n_t = len(times)
        rho_core, rho_far, volume, dp = [], [], [], []
        for i in range(n_t):
            rho = np.array(data["density"][i], dtype=float)
            levelset = np.array(data["levelset"][i], dtype=float)
            rho_core.append(np.median(rho[levelset <= -BULK_LEVELSET]))
            rho_far.append(np.median(rho[levelset >= BULK_LEVELSET]))
            volume.append((rho > 0.5 * (pr.RHO_LIQ + pr.RHO_GAS)).sum())
        rho1, rho2 = float(np.median(rho_core)), float(np.median(rho_far))
        L_ref = float((3.0 * np.median(volume) / (4.0 * np.pi)) ** (1.0 / 3.0))

        # Laplace jump: drop core against the far carrier, measured from the a-weighted
        # centroid (the drop never wraps in this trimmed file).
        for i in range(n_t):
            rho = np.array(data["density"][i], dtype=float)
            a = np.clip((rho - rho2) / (rho1 - rho2), 0.0, 1.0)
            idx = np.indices(rho.shape)
            r = np.sqrt(sum((q - (q * a).sum() / a.sum()) ** 2 for q in idx))
            P = mechanical_pressure(rho)
            dp.append(np.median(P[r < L_ref / 3]) - np.median(P[r > L_ref + 15]))
    sigma = float(np.median(dp) * L_ref / 2.0)

    _, g_budget = pr.measure_bulk_acceleration(H5_PATH)
    g = float(np.mean(g_budget))         # equal snapshot spacing: = net momentum change

    U_ref = float(np.sqrt(pr.FORCE_INCREMENT * abs(pr.RHO_LIQ - pr.RHO_GAS) / pr.RHO_GAS * 2.0 * L_ref))
    _SCALES = dict(rho1=rho1, rho2=rho2, mu1=rho1 * pr.NU, mu2=rho2 * pr.NU, sigma=sigma, g=g,
                   L_ref=L_ref, U_ref=U_ref, T_ref=L_ref / U_ref, rho_ref=rho2, t_origin=times[0])
    s = _SCALES
    print("LBPM sharp-interface parameters (lattice units): rho1 = %.5f, rho2 = %.5f, "
          "mu1 = %.5g, mu2 = %.5g, sigma = %.5g, g = %.4g" %
          (rho1, rho2, s["mu1"], s["mu2"], sigma, g))
    print("Reference scales: L_ref = %.4f cells, U_ref = %.6g cells/step, T_ref = %.1f steps, "
          "rho_ref = %.5f" % (L_ref, U_ref, s["T_ref"], rho2))
    print("Groups: 1/Re (carrier) = %.4f, 1/We = %.4f, 1/Fr = %.4f\n" %
          (pr.NU / (U_ref * L_ref), sigma / (rho2 * U_ref ** 2 * L_ref), g * L_ref / U_ref ** 2))
    return _SCALES


def phase_fraction(rho, scales):
    return np.clip((rho - scales["rho2"]) / (scales["rho1"] - scales["rho2"]), 0.0, 1.0)


def interpolate_periodic(field, X, Y, Z, pts):
    ''' Trilinear interpolation of a (Z, Y, X) cell-centred field at pts (n, 3) = (x, y, z)
    in the same cell coordinates, wrapping across the periodic faces. '''
    out = 0.0
    frac, base = [], []
    for c, axis in zip(pts.T, (X, Y, Z)):
        s = c - axis[0]
        i0 = np.floor(s).astype(int)
        frac.append(s - i0)
        base.append(i0)
    shape = field.shape                 # (nz, ny, nx)
    for dz in (0, 1):
        for dy in (0, 1):
            for dx in (0, 1):
                w = (np.where(dx, frac[0], 1 - frac[0]) * np.where(dy, frac[1], 1 - frac[1]) *
                     np.where(dz, frac[2], 1 - frac[2]))
                out = out + w * field[(base[2] + dz) % shape[0], (base[1] + dy) % shape[1],
                                      (base[0] + dx) % shape[2]]
    return out


def get_points_a_3d(NOP, times, X, Y, Z, interface_all, normal_all, tangent1_all,
                     tangent2_all, levelset_all, a_all):
    ''' Volume-fraction points: NOP[0] spread across the diffuse interface (0-8 cells
    along the normal, the whole |levelset| < 8 band) and NOP[1] in the domain, labelled
    with the CFD a interpolated at each point. '''
    data = np.empty((0, 5), float)
    for t_value, interface, normal, t1, t2, levelset, a in zip(
            times, interface_all, normal_all, tangent1_all, tangent2_all, levelset_all, a_all):
        interface_data = get_points_for_interface_3d(NOP[0], t_value, interface, normal, t1, t2,
                                                       1.0, 0.0, BULK_LEVELSET)
        domain_data = get_points_for_domain_3d(NOP[1], t_value, X, Y, Z, levelset, 4)
        block = np.vstack([interface_data, domain_data])
        block[:, 4] = interpolate_periodic(a, X, Y, Z, block[:, :3])
        data = np.vstack([data, block])
    return data


def get_points_pde_3d(NOP, times, X, Y, Z, interface_all, normal_all, tangent1_all,
                       tangent2_all, levelset_all):
    ''' Residual points: interface (0-2 cells), nearfield (2 cells to 0.4 L_ref, the band
    generate_points_3d_pr.py uses) and domain. '''
    nearfield_end = 0.4 * get_scales()["L_ref"]
    data = np.empty((0, 5), float)
    for t_value, interface, normal, t1, t2, levelset in zip(
            times, interface_all, normal_all, tangent1_all, tangent2_all, levelset_all):
        interface_temp = get_points_for_interface_3d(NOP[0], t_value, interface, normal, t1, t2,
                                                       1.0, 0.0, 2.0)
        nearfield_temp = get_points_for_interface_3d(NOP[1], t_value, interface, normal, t1, t2,
                                                       1.0, 2.0, nearfield_end)
        domain_temp = get_points_for_domain_3d(NOP[2], t_value, X, Y, Z, levelset, 4)
        data = np.vstack([data, interface_temp, nearfield_temp, domain_temp])
    data[:, 4] = 0.0
    return data


def get_north_south(NOP, x, y, z, t):
    ''' z faces as a periodic pair, sampled like east/west with z and x swapped: x
    linspace, t linspace, y random but shared by the pair. Columns x, y, z, t. '''
    north, south = [], []
    for time_value in np.linspace(t[0], t[1], NOP[1]):
        bx = np.linspace(x[0], x[1], NOP[0]).reshape(-1, 1)
        by = np.random.uniform(y[0], y[1], (NOP[0], 1))
        tt = time_value * np.ones((NOP[0], 1))
        north.append(np.hstack([bx, by, z[1] * np.ones((NOP[0], 1)), tt]))
        south.append(np.hstack([bx, by, z[0] * np.ones((NOP[0], 1)), tt]))
    return np.vstack(north), np.vstack(south)


def get_initial(NOP, x, y, z):
    ''' u = v = w = 0 at t = 0, uniform over the box. Columns x, y, z, t, u, v, w. '''
    pts = np.column_stack([np.random.uniform(lo, hi, NOP) for lo, hi in (x, y, z)])
    return np.hstack([pts, np.zeros((NOP, 4))])


def get_training_data_3d(NOP_A, NOP_PDE, NOP_north, NOP_NS, NOP_east, NOP_west,
                         NOP_front_back, NOP_initial):
    ''' As generate_points_3d.get_training_data_3d, for the LBPM file. NOP_north is the
    p = 1 gauge (spatial points, times); NOP_NS the z periodic pairs; NOP_initial the
    number of t = 0 rest points. Returns dataframes A, PDE, N, EW, FB, NS, IC. '''

    scales = get_scales()
    L_ref, T_ref = scales["L_ref"], scales["T_ref"]
    with h5py.File(H5_PATH, "r") as data:
        (X, Y, Z), (nx, ny, nz) = _cell_axes(data)
        times_all = np.array(data["time"], dtype=float)
        indices = np.setdiff1d(np.arange(len(times_all)), VAL_INDICES)
        levelset = -np.array(data["levelset"][indices], dtype=float)
        a_all = [phase_fraction(np.array(data["density"][i], dtype=float), scales) for i in indices]
    times = times_all[indices] - scales["t_origin"]

    # Faces of the periodic box (cell centres sit half a cell inside them)
    x_bounds, y_bounds, z_bounds = [-nx / 2, nx / 2], [-ny / 2, ny / 2], [-nz / 2, nz / 2]
    t_bounds = [times[0], times[-1]]

    print("\nDistributing points for time snapshots (LBM steps):\n", (times + scales["t_origin"]).astype(int))
    print("Number of time snapshots: ", len(times))
    print("Box (cells): x %s, y %s, z %s\n" % (x_bounds, y_bounds, z_bounds))

    interface, normal, tangent1, tangent2 = compute_normals_3d(X, Y, Z, levelset, 1.0)
    print("Generating points for A")
    data_A = get_points_a_3d(NOP_A, times, X, Y, Z, interface, normal, tangent1, tangent2, levelset, a_all)
    print("Generating points for PDE")
    data_PDE = get_points_pde_3d(NOP_PDE, times, X, Y, Z, interface, normal, tangent1, tangent2, levelset)
    print("Generating boundary and initial points")
    data_north = get_north(NOP_north, x_bounds, y_bounds, z_bounds, t_bounds)
    data_n, data_s = get_north_south(NOP_NS, x_bounds, y_bounds, z_bounds, t_bounds)
    assert NOP_east == NOP_west, "east/west are sampled as periodic pairs and must match"
    data_east, data_west = get_east_west(NOP_east, x_bounds, y_bounds, z_bounds, t_bounds)
    data_front, data_back = get_front_back(NOP_front_back, x_bounds, y_bounds, z_bounds, t_bounds)
    data_IC = get_initial(NOP_initial, x_bounds, y_bounds, z_bounds)
    data_faces = np.vstack([data_north[:, 0:4], data_n, data_s, data_east[:, 0:4], data_west[:, 0:4],
                            data_front[:, 0:4], data_back[:, 0:4]])

    # NONDIMENSIONALIZE SPACE BY L_ref AND TIME BY T_ref
    for arr in (data_A, data_PDE, data_north, data_n, data_s, data_east, data_west,
                data_front, data_back, data_IC, data_faces):
        arr[:, :3] /= L_ref
        arr[:, 3] /= T_ref

    print("Assembling data frames\n")
    data_EW = pd.DataFrame(data=np.hstack([data_east[:, 0:3], data_west[:, 0:4]]),
                           columns=["x_E", "y_E", "z_E", "x_W", "y_W", "z_W", "t_EW"])
    data_FB = pd.DataFrame(data=np.hstack([data_front[:, 0:3], data_back[:, 0:4]]),
                           columns=["x_F", "y_F", "z_F", "x_B", "y_B", "z_B", "t_FB"])
    data_NS = pd.DataFrame(data=np.hstack([data_n[:, 0:3], data_s[:, 0:4]]),
                           columns=["x_Nz", "y_Nz", "z_Nz", "x_Sz", "y_Sz", "z_Sz", "t_NS"])
    data_N = pd.DataFrame(data=data_north[:, [0, 1, 2, 3, 7]], columns=["x_N", "y_N", "z_N", "t_N", "p_N"])
    data_IC = pd.DataFrame(data=data_IC, columns=["x_IC", "y_IC", "z_IC", "t_IC", "u_IC", "v_IC", "w_IC"])
    data_A = pd.DataFrame(data=data_A, columns=["x_A", "y_A", "z_A", "t_A", "a_A"])
    data_PDE = pd.DataFrame(data=np.vstack([data_PDE, np.hstack([data_faces, np.zeros((len(data_faces), 1))])]),
                            columns=["x_PDE", "y_PDE", "z_PDE", "t_PDE", "f_PDE"])
    return dict(A=data_A, PDE=data_PDE, N=data_N, EW=data_EW, FB=data_FB, NS=data_NS, IC=data_IC)


def get_validation_data_3d(NOP, seed=4321):
    ''' Samples the VAL_INDICES snapshots: per snapshot NOP[0] cells within 4 cells of the
    interface and NOP[1] cells anywhere. Columns x, y, z, t, a, u, v, w, p, bulk,
    nondimensionalised like the training points: u by U_ref after the F/(2 rho) correction,
    p = (k^2 P_PR - its top-layer mean) / (rho_ref U_ref^2) + 1, a as in training. bulk = 1
    where |levelset| = 8, the only place the diffuse-interface P_PR (which dips far
    negative inside the interface) is comparable with a sharp-interface pressure. '''

    scales = get_scales()
    L_ref, U_ref, T_ref = scales["L_ref"], scales["U_ref"], scales["T_ref"]
    p_ref = scales["rho_ref"] * U_ref ** 2
    rng = np.random.default_rng(seed)
    rows = []
    with h5py.File(H5_PATH, "r") as data:
        (X, Y, Z), _ = _cell_axes(data)
        times = np.array(data["time"], dtype=float)
        for i in VAL_INDICES:
            levelset = np.array(data["levelset"][i], dtype=float)
            rho = np.array(data["density"][i], dtype=float)
            near = np.flatnonzero(np.abs(levelset) <= 4)
            flat = np.concatenate([rng.choice(near, NOP[0], replace=False),
                                   rng.choice(levelset.size, NOP[1], replace=False)])
            iz, iy, ix = np.unravel_index(flat, levelset.shape)
            F = pr.lbpm_force(rho, times[i])
            vel = [(np.array(data[name][i], dtype=float) + 0.5 * F[k] / rho).ravel()[flat] / U_ref
                   for k, name in enumerate(("velocityX", "velocityY", "velocityZ"))]
            P = mechanical_pressure(rho)
            p = (P.ravel()[flat] - P[-1].mean()) / p_ref + 1.0
            rows.append(np.column_stack([X[ix] / L_ref, Y[iy] / L_ref, Z[iz] / L_ref,
                                         np.full(len(flat), (times[i] - scales["t_origin"]) / T_ref),
                                         phase_fraction(rho, scales).ravel()[flat], *vel, p,
                                         (np.abs(levelset.ravel()[flat]) >= BULK_LEVELSET).astype(float)]))
    print(f"Validation points: {sum(NOP)} per snapshot at t = {times[VAL_INDICES].astype(int)}")
    return pd.DataFrame(np.vstack(rows), columns=["x", "y", "z", "t", "a", "u", "v", "w", "p", "bulk"])
