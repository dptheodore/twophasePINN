import numpy as np
import pandas as pd
import h5py

import generate_points_3d
from generate_points_3d import H5_PATH, L_REF, get_validation_data_3d  # noqa: F401 (re-export)

''' generate_points_3d.py plus the geometric-constraint regions, as generate_points_geom.py
is generate_points.py plus them in 2D. The regions are precomputed by
compute_regions_3d.py into H5_PATH:/computation_regions (marching cubes; see there).

get_training_data_3d returns the generate_points_3d frames unchanged -- the GEOM frame is
read after them and draws no random numbers, so A, PDE and the boundary points are the
same as a run of pytorch_3d_rising_bubble_train.py -- plus

    GEOM   x_geom, y_geom, z_geom, t_geom   region centre, / L_ref like every other point
           grad_x, grad_y, grad_z           (f_minus - f_plus) per axis: face-fraction
                                            differences, i.e. grad_vec / h^2 (dimensionless)
           normal_x, normal_y, normal_z     integrated_normal / h^2 (out of the bubble)

The 2D file stored grad_vec as raw wetted LENGTHS while the trainer compared them with
differences of a (dimensionless); here both sides are face fractions. '''


def geom_half_size():
    ''' Half the region side, nondimensional (x / L_ref). '''
    with h5py.File(H5_PATH, "r") as data:
        return float(data["computation_regions"].attrs["region_half_size"]) / L_REF


def get_geom_data_3d():
    with h5py.File(H5_PATH, "r") as data:
        assert "computation_regions" in data, \
            f"{H5_PATH} has no computation_regions; run compute_regions_3d.py first"
        comp = data["computation_regions"]
        h = 2.0 * float(comp.attrs["region_half_size"])
        centre = [np.array(comp[k]) for k in ("x_center", "y_center", "z_center", "time_value")]
        grad_vec = np.array(comp["grad_vec"]) / h ** 2
        int_normal = np.array(comp["integrated_normal"]) / h ** 2

    # time_value is physical here; U_ref = 1, so t / L_ref as for space
    x_geom, y_geom, z_geom, t_geom = (c / L_REF for c in centre)
    print(f"Geometric regions: {len(x_geom)}, cube side {h / L_REF:.4f} (nondim)")
    return pd.DataFrame({
        "x_geom": x_geom, "y_geom": y_geom, "z_geom": z_geom, "t_geom": t_geom,
        "grad_x": grad_vec[:, 0], "grad_y": grad_vec[:, 1], "grad_z": grad_vec[:, 2],
        "normal_x": int_normal[:, 0], "normal_y": int_normal[:, 1], "normal_z": int_normal[:, 2],
    })


def get_training_data_3d(*args, **kwargs):
    frames = generate_points_3d.get_training_data_3d(*args, **kwargs)
    frames["GEOM"] = get_geom_data_3d()
    t_train = np.unique(np.round(frames["A"]["t_A"].to_numpy(), 6))
    t_geom = np.unique(np.round(frames["GEOM"]["t_geom"].to_numpy(), 6))
    assert np.allclose(t_train, t_geom), "computation_regions snapshots differ from TRAIN_TIMES"
    return frames
