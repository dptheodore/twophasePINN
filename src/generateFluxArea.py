import h5py
import numpy as np
from tqdm import tqdm
from scipy.ndimage import sobel
from scipy.interpolate import interpn
from marching_squares import Grid, march

HDF5_DATA_PATH = "../cfd_data/rising_bubble.h5"

def clip_segment(p1, p2, x_min, x_max, y_min, y_max):
    x1, y1 = p1; x2, y2 = p2; dx, dy = x2 - x1, y2 - y1
    t0, t1 = 0.0, 1.0
    for edge in range(4):
        if edge == 0: p, q = -dx, -(x_min - x1)
        if edge == 1: p, q = dx, (x_max - x1)
        if edge == 2: p, q = -dy, -(y_min - y1)
        if edge == 3: p, q = dy, (y_max - y1)
        if p == 0:
            if q < 0: return None
        else:
            r = q * (1.0 / p)
            if p < 0:
                if r > t1: return None
                t0 = max(t0, r)
            else:
                if r < t0: return None
                t1 = min(t1, r)
    if t0 > t1: return None
    return ((x1 + t0 * dx, y1 + t0 * dy), (x1 + t1 * dx, y1 + t1 * dy))


def compute_grad_normals_region_bounded(levelset_t, X_patch, Y_patch, grid_scale, region_bounds):
    ny, nx = levelset_t.shape
    dphi_dy = sobel(levelset_t, axis=0, mode='constant') / grid_scale
    dphi_dx = sobel(levelset_t, axis=1, mode='constant') / grid_scale

    ms_grid = Grid(scale=grid_scale, x_count=nx - 1, y_count=ny - 1)
    ms_grid.values = levelset_t.astype(np.float32)
    edges = march(ms_grid, iso=0.0, interpolated=True)

    integrated_normal_sum = np.zeros(2, dtype=np.float64)
    intersection_points_x_bottom = []
    intersection_points_x_top = []
    intersection_points_y_left = []
    intersection_points_y_right = []

    x_min_bound, x_max_bound = region_bounds[0], region_bounds[1]
    y_bottom, y_top = region_bounds[2], region_bounds[3]

    for p1, p2 in edges:
        p1_phys = (p1[1] + X_patch[0], p1[0] + Y_patch[0])
        p2_phys = (p2[1] + X_patch[0], p2[0] + Y_patch[0])

        # intersections with patch boundaries (collect points)
        y1, y2 = p1_phys[1], p2_phys[1]
        if (y1 - y_bottom) * (y2 - y_bottom) <= 0:
            x1, x2 = p1_phys[0], p2_phys[0]
            if abs(y2 - y1) < 1e-12:
                intersection_points_x_bottom.extend([x1, x2])
            else:
                x_int = x1 + (x2 - x1) * (y_bottom - y1) / (y2 - y1)
                intersection_points_x_bottom.append(x_int)
        if (y1 - y_top) * (y2 - y_top) <= 0:
            x1, x2 = p1_phys[0], p2_phys[0]
            if abs(y2 - y1) < 1e-12:
                intersection_points_x_top.extend([x1, x2])
            else:
                x_int = x1 + (x2 - x1) * (y_top - y1) / (y2 - y1)
                intersection_points_x_top.append(x_int)

        x1, x2 = p1_phys[0], p2_phys[0]
        if (x1 - x_min_bound) * (x2 - x_min_bound) <= 0:
            y1, y2 = p1_phys[1], p2_phys[1]
            if abs(x2 - x1) < 1e-12:
                intersection_points_y_left.extend([y1, y2])
            else:
                y_int = y1 + (y2 - y1) * (x_min_bound - x1) / (x2 - x1)
                intersection_points_y_left.append(y_int)
        if (x1 - x_max_bound) * (x2 - x_max_bound) <= 0:
            y1, y2 = p1_phys[1], p2_phys[1]
            if abs(x2 - x1) < 1e-12:
                intersection_points_y_right.extend([y1, y2])
            else:
                y_int = y1 + (y2 - y1) * (x_max_bound - x1) / (x2 - x1)
                intersection_points_y_right.append(y_int)

        # clip to the region bounds (returns physical points)
        clipped = clip_segment(p1_phys, p2_phys, x_min_bound, x_max_bound, y_bottom, y_top)
        if clipped is None:
            continue
        p1_c, p2_c = clipped
        seg_length = np.linalg.norm(np.array(p2_c) - np.array(p1_c))
        if seg_length < 1e-12:
            continue

        points_to_interp = [[p1_c[1], p1_c[0]], [p2_c[1], p2_c[0]]]
        grads_dy_dx = interpn((Y_patch, X_patch), np.stack([dphi_dy, dphi_dx], axis=-1),
                              points_to_interp, method='linear', bounds_error=False, fill_value=0)
        grads_dx_dy = grads_dy_dx[:, ::-1]
        magnitudes = np.linalg.norm(grads_dx_dy, axis=1, keepdims=True) + 1e-12
        normals = grads_dx_dy / magnitudes
        avg_normal = np.mean(normals, axis=0)
        integrated_normal_sum += avg_normal * seg_length

    grad_vec = np.zeros(2, dtype=np.float64)
    if intersection_points_x_bottom:
        min_x = max(min(intersection_points_x_bottom), x_min_bound)
        max_x = min(max(intersection_points_x_bottom), x_max_bound)
        grad_vec[1] = max(0.0, max_x - min_x)
    if intersection_points_x_top:
        min_x = max(min(intersection_points_x_top), x_min_bound)
        max_x = min(max(intersection_points_x_top), x_max_bound)
        grad_vec[1] -= max(0.0, max_x - min_x)
    if intersection_points_y_left:
        min_y = max(min(intersection_points_y_left), y_bottom)
        max_y = min(max(intersection_points_y_left), y_top)
        grad_vec[0] = max(0.0, max_y - min_y)
    if intersection_points_y_right:
        min_y = max(min(intersection_points_y_right), y_bottom)
        max_y = min(max(intersection_points_y_right), y_top)
        grad_vec[0] -= max(0.0, max_y - min_y)

    return grad_vec, integrated_normal_sum


def make_region_grid(nx_total, ny_total, patch_w, patch_h, stride_x=1, stride_y=1):
    xs = list(range(0, nx_total - patch_w + 1, stride_x))
    ys = list(range(0, ny_total - patch_h + 1, stride_y))
    regions = []
    for y0 in ys:
        for x0 in xs:
            regions.append((x0, y0, patch_w, patch_h))
    return regions


# ---------- configuration ----------
patch_w = 256           # patch width (cells)
patch_h = 256           # patch height (cells)
stride_x = 128           # stride across x (reduce to 1 for every pixel, higher to reduce samples)
stride_y = 256           # stride across y
use_inner_tqdm = False # set True to show inner per-time-region progress
# -----------------------------------

with h5py.File(HDF5_DATA_PATH, "a") as f:
    X = f["X"][:]                    # len == nx
    Y = f["Y"][:]                    # len == ny
    levelset_full = f["levelset"][:] / 8.0   # shape (nt, ny, nx)
    times = f["time"][:]             # shape (nt,)
    nt, ny, nx = levelset_full.shape
    computation_regions = f["computation_regions"]
    regions = make_region_grid(nx, ny, patch_w, patch_h, stride_x, stride_y)
    n_regions = len(regions)
    print(n_regions)
    # preallocate arrays
    phi_patches = np.zeros((nt, n_regions, patch_h, patch_w), dtype=np.float32)
    grad_vecs = np.zeros((nt, n_regions, 2), dtype=np.float64)
    integrated_normals = np.zeros((nt, n_regions, 2), dtype=np.float64)
    centers = np.zeros((nt, n_regions, 2), dtype=np.float64)
    has_interface = np.zeros((nt, n_regions), dtype=np.uint8)

    comp_grp = f.require_group("computation_regions")
    # if replacing, remove old datasets first to avoid conflicts
    for name in ("phi_patches", "grad_vec", "integrated_normal", "centers", "has_interface", "times"):
        if name in comp_grp:
            del comp_grp[name]

    # Process timesteps with progress bar
    for t_idx in tqdm(range(nt), desc="Processing timesteps"):
        phi_t = levelset_full[t_idx]  # shape (ny, nx)

        if use_inner_tqdm:
            iterator = enumerate(tqdm(regions, desc=f"t={t_idx}", leave=False))
        else:
            iterator = enumerate(regions)

        for r_idx, (x0, y0, w, h) in iterator:
            # patch indices (x runs along nx direction)
            x1 = x0 + w
            y1 = y0 + h
            phi_patch = phi_t[y0:y1, x0:x1].astype(np.float32)

            # physical X/Y slices for the patch (pass these to compute function)
            X_patch = X[x0:x1]
            Y_patch = Y[y0:y1]

            # region bounds (physical)
            # note X_patch[0] is left edge; X[x0 + w] is the next node after patch if available.
            # we use endpoints as patch physical bounds; safe because X is monotonically spaced
            x_min = X[x0]
            x_max = X[x0 + w - 1] if (x0 + w - 1) < len(X) else X[-1]
            # to get edge-to-edge, attempt to extend by grid spacing if possible
            if (x0 + w) < len(X):
                x_max = X[x0 + w]
            y_min = Y[y0]
            if (y0 + h) < len(Y):
                y_max = Y[y0 + h]
            else:
                y_max = Y[-1]
            region_bounds = [x_min, x_max, y_min, y_max]

            # compute interface presence (zero crossing)
            interface_present = (phi_patch.min() < 0.0) and (phi_patch.max() > 0.0)
            has_interface[t_idx, r_idx] = 1 if interface_present else 0

            # compute grad_vec and integrated_normal using marching squares based routine
            gv, inn = compute_grad_normals_region_bounded(phi_patch, X_patch, Y_patch, grid_scale=(X[1]-X[0]), region_bounds=region_bounds)

            # store
            phi_patches[t_idx, r_idx] = phi_patch
            grad_vecs[t_idx, r_idx] = gv
            integrated_normals[t_idx, r_idx] = inn
            centers[t_idx, r_idx] = [ X[x0 + w//2], Y[y0 + h//2] ]

    valid_regions = []

    # write datasets
    comp_grp.create_dataset("phi_patches", data=phi_patches, compression="gzip")
    comp_grp.create_dataset("grad_vec", data=grad_vecs, compression="gzip")
    comp_grp.create_dataset("integrated_normal", data=integrated_normals, compression="gzip")
    comp_grp.create_dataset("centers", data=centers, compression="gzip")
    comp_grp.create_dataset("has_interface", data=has_interface, compression="gzip")
    # store times (copied from root time for convenience)
    comp_grp.create_dataset("times", data=times)

    # summary
    total_interface_patches = int(has_interface.sum())
    print(f"Done. total regions per timestep = {n_regions}, nt = {nt}")
    print(f"Total interface-containing (t,region) patches = {total_interface_patches} out of {nt * n_regions}")
