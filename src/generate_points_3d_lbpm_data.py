import numpy as np
import pandas as pd
import h5py

from generate_points_3d import (get_points_for_interface_3d, get_points_for_domain_3d,
                                compute_normals_3d, get_east_west, get_front_back)
from generate_points_3d_lbpm import (H5_PATH, VAL_INDICES, BULK_LEVELSET, get_scales,
                                     interpolate_periodic, get_north_south, _cell_axes)

''' Data-driven points for pytorch_3d_lbpm_datadriven_geom_train.py on the LBPM file
"lbpm_bubble_dataset_newrun_nowrap.h5": no PDE points, the CFD fields themselves as
targets, plus the geometric regions of compute_regions_3d_lbpm.py.

Network outputs and their targets (all O(1)):

    rho_hat   (rho - rho2) / (rho1 - rho2), NOT clipped: the CFD overshoots the measured
              phase densities slightly (rho_hat in about [-0.001, 1.006]). The phase
              fraction a used by the geometric loss is this, clipped, exactly as the
              regions were computed. rho1, rho2 from generate_points_3d_lbpm.get_scales.
    phi_hat   h5 levelset / 8: signed distance to the a = 0.5 surface in cells, NEGATIVE in
              the drop (the h5 sign), clipped at |phi_hat| = 1 by the h5 itself.
    p_hat     (p - p_min) / (p_max - p_min), p = h5 "pressure" = P_PR(rho) (to 2e-8),
              p_min/p_max over the training snapshots. No gauge: p is absolute.

Coordinates as generate_points_3d_lbpm: cells centred on the box, / L_ref; t is
(t - 19600) / T_ref. Training snapshots are those not in VAL_INDICES (19 of 27).

Frames returned by get_training_data_3d:
    DATA   x, y, z, t, rho_hat, phi_hat, p_hat   NOP[0] points 0-8 cells off the a = 0.5
                                                 surface (the whole levelset band), NOP[1]
                                                 cells 4+ cells off it, per snapshot. The
                                                 trainer passes NOP[1] = 0 and draws those
                                                 every epoch from BulkSampler instead
    EW, FB, NS                                   periodic face pairs, as the LBPM trainer
    GEOM   x_geom, y_geom, z_geom, t_geom,       region centres, nondimensional;
           grad_x, grad_y, grad_z                grad_vec / h^2 = face-MEAN differences
                                                 (minus - plus) of a, dimensionless '''

PHI_SCALE = BULK_LEVELSET
_DATA_SCALES = None


def get_data_scales():
    ''' Pressure normalisation over the training snapshots. Cached. '''
    global _DATA_SCALES
    if _DATA_SCALES is None:
        with h5py.File(H5_PATH, "r") as data:
            indices = np.setdiff1d(np.arange(len(data["time"])), VAL_INDICES)
            p = [np.array(data["pressure"][i], dtype=float) for i in indices]
        _DATA_SCALES = dict(p_min=float(min(q.min() for q in p)), p_max=float(max(q.max() for q in p)))
    return _DATA_SCALES


def normalised_fields(data, i, scales, data_scales):
    ''' (rho_hat, phi_hat, p_hat) of snapshot i, each (Z, Y, X). '''
    rho = np.array(data["density"][i], dtype=float)
    p = np.array(data["pressure"][i], dtype=float)
    return ((rho - scales["rho2"]) / (scales["rho1"] - scales["rho2"]),
            np.array(data["levelset"][i], dtype=float) / PHI_SCALE,
            (p - data_scales["p_min"]) / (data_scales["p_max"] - data_scales["p_min"]))


def geom_half_size():
    ''' Half the region side, nondimensional (x / L_ref). '''
    with h5py.File(H5_PATH, "r") as data:
        return float(data["computation_regions"].attrs["region_half_size"]) / get_scales()["L_ref"]


def periodic_lengths():
    ''' Box size per axis, nondimensional: the periods of the network's sin/cos embedding. '''
    with h5py.File(H5_PATH, "r") as data:
        _, sizes = _cell_axes(data)
    return [n / get_scales()["L_ref"] for n in sizes]


class BulkSampler:
    ''' Bulk DATA points drawn afresh on every call: NOP per training snapshot, uniform in
    the box (not only at cell centres) and at least max_levelset cells off the a = 0.5
    surface, with the CFD fields interpolated periodically as targets. Replaces a fixed
    get_points_for_domain_3d set, between whose points the network grew a sub-cell plane
    with rho < 0 that no training point ever saw (runs Oct-26-2026_02-00-01, 06-37-38).
    sample() draws everything from the rng it is given, so ranks seeded alike agree. '''

    def __init__(self, NOP, max_levelset=4.0):
        scales, data_scales = get_scales(), get_data_scales()
        with h5py.File(H5_PATH, "r") as data:
            (self.X, self.Y, self.Z), sizes = _cell_axes(data)
            times = np.array(data["time"], dtype=float)
            indices = np.setdiff1d(np.arange(len(times)), VAL_INDICES)
            # (snapshot, field, Z, Y, X), fields rho_hat, phi_hat, p_hat
            self.fields = np.stack([np.stack(normalised_fields(data, i, scales, data_scales)).astype(np.float32)
                                    for i in indices])
        self.t = (times[indices] - scales["t_origin"]) / scales["T_ref"]
        self.size = np.array(sizes, dtype=float)
        self.lo = -self.size / 2
        self.NOP, self.L_ref = NOP, scales["L_ref"]
        self.phi_cut = max_levelset / PHI_SCALE

    def _interpolate(self, k, pts):
        """(n, 3) rho_hat, phi_hat, p_hat of snapshot k at pts: interpolate_periodic with the
        trilinear weights shared by the three fields."""
        s = (pts - np.array([self.X[0], self.Y[0], self.Z[0]])).T
        base = np.floor(s).astype(int)
        frac = (s - base).astype(np.float32)
        nz, ny, nx = self.fields.shape[2:]
        out = 0.0
        for dz in (0, 1):
            for dy in (0, 1):
                for dx in (0, 1):
                    w = ((frac[0] if dx else 1 - frac[0]) * (frac[1] if dy else 1 - frac[1]) *
                         (frac[2] if dz else 1 - frac[2]))
                    # the slice between the index arrays puts the points first: (n, 3)
                    out = out + w[:, None] * self.fields[k, :, (base[2] + dz) % nz, (base[1] + dy) % ny, (base[0] + dx) % nx]
        return out

    def sample(self, rng):
        blocks = []
        for k, t in enumerate(self.t):
            kept, n = [], 0
            while n < self.NOP:
                pts = self.lo + rng.random((int(1.1 * (self.NOP - n)) + 64, 3)) * self.size
                vals = self._interpolate(k, pts)
                far = np.abs(vals[:, 1]) >= self.phi_cut
                kept.append(np.column_stack([pts[far], vals[far]]))
                n += far.sum()
            rows = np.vstack(kept)[:self.NOP]
            blocks.append(np.column_stack([rows[:, :3] / self.L_ref, np.full(self.NOP, t), rows[:, 3:]]))
        return pd.DataFrame(np.vstack(blocks), columns=["x", "y", "z", "t", "rho_hat", "phi_hat", "p_hat"])


def get_geom_data_3d():
    scales = get_scales()
    with h5py.File(H5_PATH, "r") as data:
        assert "computation_regions" in data, \
            f"{H5_PATH} has no computation_regions; run compute_regions_3d_lbpm.py first"
        comp = data["computation_regions"]
        h = 2.0 * float(comp.attrs["region_half_size"])
        x, y, z = (np.array(comp[k]) / scales["L_ref"] for k in ("x_center", "y_center", "z_center"))
        t = (np.array(comp["time_value"]) - scales["t_origin"]) / scales["T_ref"]
        grad_vec = np.array(comp["grad_vec"]) / h ** 2
    print(f"Geometric regions: {len(x)}, cube side {h:g} cells = {h / scales['L_ref']:.4f} (nondim)")
    return pd.DataFrame({"x_geom": x, "y_geom": y, "z_geom": z, "t_geom": t,
                         "grad_x": grad_vec[:, 0], "grad_y": grad_vec[:, 1], "grad_z": grad_vec[:, 2]})


def get_training_data_3d(NOP_data, NOP_NS, NOP_east, NOP_west, NOP_front_back):
    scales = get_scales()
    data_scales = get_data_scales()
    L_ref, T_ref = scales["L_ref"], scales["T_ref"]
    with h5py.File(H5_PATH, "r") as data:
        (X, Y, Z), (nx, ny, nz) = _cell_axes(data)
        times_all = np.array(data["time"], dtype=float)
        indices = np.setdiff1d(np.arange(len(times_all)), VAL_INDICES)
        levelset = -np.array(data["levelset"][indices], dtype=float)      # drop positive
        fields = [normalised_fields(data, i, scales, data_scales) for i in indices]
    times = times_all[indices] - scales["t_origin"]

    x_bounds, y_bounds, z_bounds = [-nx / 2, nx / 2], [-ny / 2, ny / 2], [-nz / 2, nz / 2]
    lo_box = np.array([x_bounds[0], y_bounds[0], z_bounds[0]])
    size = np.array([nx, ny, nz], dtype=float)
    t_bounds = [times[0], times[-1]]

    print("\nDistributing points for time snapshots (LBM steps):\n", (times + scales["t_origin"]).astype(int))
    print("Number of time snapshots: ", len(times))
    print("p normalisation: p_min = %.6g, p_max = %.6g" % (data_scales["p_min"], data_scales["p_max"]))

    interface, normal, tangent1, tangent2 = compute_normals_3d(X, Y, Z, levelset, 1.0)
    print("Generating data points")
    blocks = []
    for k, t_value in enumerate(times):
        near = get_points_for_interface_3d(NOP_data[0], t_value, interface[k], normal[k],
                                           tangent1[k], tangent2[k], 1.0, 0.0, BULK_LEVELSET)
        if NOP_data[1]:
            near = np.vstack([near, get_points_for_domain_3d(NOP_data[1], t_value, X, Y, Z, levelset[k], 4)])
        pts = near[:, :4]
        pts[:, :3] = np.mod(pts[:, :3] - lo_box, size) + lo_box
        targets = [interpolate_periodic(f, X, Y, Z, pts[:, :3]) for f in fields[k]]
        blocks.append(np.column_stack([pts, *targets]))
    data_D = np.vstack(blocks)

    print("Generating periodic face pairs")
    data_n, data_s = get_north_south(NOP_NS, x_bounds, y_bounds, z_bounds, t_bounds)
    assert NOP_east == NOP_west, "east/west are sampled as periodic pairs and must match"
    data_east, data_west = get_east_west(NOP_east, x_bounds, y_bounds, z_bounds, t_bounds)
    data_front, data_back = get_front_back(NOP_front_back, x_bounds, y_bounds, z_bounds, t_bounds)

    # NONDIMENSIONALIZE SPACE BY L_ref AND TIME BY T_ref
    for arr in (data_D, data_n, data_s, data_east, data_west, data_front, data_back):
        arr[:, :3] /= L_ref
        arr[:, 3] /= T_ref

    print("Assembling data frames\n")
    frames = dict(
        DATA=pd.DataFrame(data_D, columns=["x", "y", "z", "t", "rho_hat", "phi_hat", "p_hat"]),
        EW=pd.DataFrame(np.hstack([data_east[:, 0:3], data_west[:, 0:4]]),
                        columns=["x_E", "y_E", "z_E", "x_W", "y_W", "z_W", "t_EW"]),
        FB=pd.DataFrame(np.hstack([data_front[:, 0:3], data_back[:, 0:4]]),
                        columns=["x_F", "y_F", "z_F", "x_B", "y_B", "z_B", "t_FB"]),
        NS=pd.DataFrame(np.hstack([data_n[:, 0:3], data_s[:, 0:4]]),
                        columns=["x_Nz", "y_Nz", "z_Nz", "x_Sz", "y_Sz", "z_Sz", "t_NS"]),
        GEOM=get_geom_data_3d(),
    )
    t_train = np.unique(np.round(frames["DATA"]["t"].to_numpy(), 6))
    t_geom = np.unique(np.round(frames["GEOM"]["t_geom"].to_numpy(), 6))
    assert np.allclose(t_train, t_geom), "computation_regions snapshots differ from the training snapshots"
    return frames


def get_validation_data_3d(NOP, seed=4321):
    ''' Samples the VAL_INDICES snapshots at cell centres: per snapshot NOP[0] cells within
    4 cells of the interface and NOP[1] cells anywhere. Columns x, y, z, t, rho_hat,
    phi_hat, p_hat, a, band; band = 1 inside the levelset band (|levelset| < 8), where
    phi is a distance rather than the clip value. '''
    scales = get_scales()
    data_scales = get_data_scales()
    L_ref, T_ref = scales["L_ref"], scales["T_ref"]
    rng = np.random.default_rng(seed)
    rows = []
    with h5py.File(H5_PATH, "r") as data:
        (X, Y, Z), _ = _cell_axes(data)
        times = np.array(data["time"], dtype=float)
        for i in VAL_INDICES:
            rho_hat, phi_hat, p_hat = normalised_fields(data, i, scales, data_scales)
            levelset = phi_hat * PHI_SCALE
            near = np.flatnonzero(np.abs(levelset) <= 4)
            flat = np.concatenate([rng.choice(near, NOP[0], replace=False),
                                   rng.choice(levelset.size, NOP[1], replace=False)])
            iz, iy, ix = np.unravel_index(flat, levelset.shape)
            rows.append(np.column_stack([
                X[ix] / L_ref, Y[iy] / L_ref, Z[iz] / L_ref,
                np.full(len(flat), (times[i] - scales["t_origin"]) / T_ref),
                rho_hat.ravel()[flat], phi_hat.ravel()[flat], p_hat.ravel()[flat],
                np.clip(rho_hat.ravel()[flat], 0.0, 1.0),
                (np.abs(levelset.ravel()[flat]) < BULK_LEVELSET).astype(float)]))
    print(f"Validation points: {sum(NOP)} per snapshot at t = {times[VAL_INDICES].astype(int)}")
    return pd.DataFrame(np.vstack(rows), columns=["x", "y", "z", "t", "rho_hat", "phi_hat", "p_hat", "a", "band"])
