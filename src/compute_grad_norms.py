from scipy.ndimage import sobel
from marching_squares import Grid, march
import numpy as np
from scipy.interpolate import interpn


def process_line(X, Y, grid_scale, levelset_t, intersections, orientation, fixed_coord, lo_bound, hi_bound):
    ISO_VALUE = 0.5
    # Trim to region bounds
    if len(intersections) == 0:
        return 0.0, (None, None), {}

    # keep only finite floats
    pts = np.array([float(x) for x in intersections if np.isfinite(x)])
    if pts.size == 0:
        return 0.0, (None, None), {}

    # clamp to [lo_bound, hi_bound]
    pts = np.clip(pts, lo_bound, hi_bound)

    # sort and unique near-duplicates
    pts = np.unique(np.round(pts, decimals=12))

    # assemble endpoints including region boundaries
    endpoints = np.concatenate(([lo_bound], pts, [hi_bound]))

    total_len = 0.0
    intersection_side_info = {}
    inside_intervals = []  # store (a,b) for intervals that are inside

    # small offset to check sides of intersection points
    delta = max(grid_scale * 1e-2, 1e-8)

    # iterate intervals between consecutive endpoints
    for i in range(len(endpoints) - 1):
        a = float(endpoints[i])
        b = float(endpoints[i + 1])
        seg_len = max(0.0, b - a)
        if seg_len <= 0:
            continue

        mid = 0.5 * (a + b)
        if orientation == 'horizontal':
            sample_point = (fixed_coord, mid)   # (y, x)
        else:
            sample_point = (mid, fixed_coord)   # (y, x)

        val_mid = interpn((Y, X), levelset_t, np.array([sample_point]),
                          method='linear', bounds_error=False, fill_value=1.0)[0]
        if val_mid < ISO_VALUE:
            total_len += seg_len
            inside_intervals.append((a, b))

    # For each intersection point, sample slightly left/right (or lower/upper)
    for x in pts:
        if orientation == 'horizontal':
            left_x = x - delta
            right_x = x + delta
            left_val = interpn((Y, X), levelset_t, np.array([(fixed_coord, left_x)]),
                               method='linear', bounds_error=False, fill_value=1.0)[0]
            right_val = interpn((Y, X), levelset_t, np.array([(fixed_coord, right_x)]),
                                method='linear', bounds_error=False, fill_value=1.0)[0]
            intersection_side_info[float(x)] = (left_val < ISO_VALUE, right_val < ISO_VALUE)
        else:
            lower_y = x - delta
            upper_y = x + delta
            lower_val = interpn((Y, X), levelset_t, np.array([(lower_y, fixed_coord)]),
                                method='linear', bounds_error=False, fill_value=1.0)[0]
            upper_val = interpn((Y, X), levelset_t, np.array([(upper_y, fixed_coord)]),
                                method='linear', bounds_error=False, fill_value=1.0)[0]
            intersection_side_info[float(x)] = (lower_val < ISO_VALUE, upper_val < ISO_VALUE)

    # compute span as bounding box of inside intervals (if any)
    if inside_intervals:
        starts, ends = zip(*inside_intervals)
        span_min = max(min(starts), lo_bound)
        span_max = min(max(ends), hi_bound)
        span = (float(span_min), float(span_max))
    else:
        span = (None, None)

    return total_len, span, intersection_side_info

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
            r = q * (1/p)
            if p < 0:
                if r > t1: return None
                t0 = max(t0, r)
            else:
                if r < t0: return None
                t1 = min(t1, r)
    if t0 > t1: return None
    return ((x1 + t0 * dx, y1 + t0 * dy), (x1 + t1 * dx, y1 + t1 * dy))


def compute_grad_normals_region_bounded(levelset_t, X, Y, grid_scale, region_bounds):
    ny, nx = levelset_t.shape
    dphi_dy = sobel(levelset_t, axis=0, mode='constant') / grid_scale
    dphi_dx = sobel(levelset_t, axis=1, mode='constant') / grid_scale
    
    # Marching squares
    ms_grid = Grid(scale=grid_scale, x_count=nx - 1, y_count=ny - 1)
    ms_grid.values = levelset_t.astype(np.float32)
    edges = march(ms_grid, iso=0.5, interpolated=True)

    edges_info_for_plot = []
    integrated_normal_sum = np.zeros(2, dtype=np.float64)
    intersection_points_x_bottom = []
    intersection_points_x_top = []
    intersection_points_y_left = []
    intersection_points_y_right = []

    # bounds
    x_min_bound, x_max_bound = region_bounds[0], region_bounds[1]
    y_bottom, y_top = region_bounds[2], region_bounds[3]

    for p1, p2 in edges:
        # use march coords directly, add domain offset
        p1_phys = (p1[1] + X[0], p1[0] + Y[0])
        p2_phys = (p2[1] + X[0], p2[0] + Y[0])

        # intersection with bottom boundary
        y1, y2 = p1_phys[1], p2_phys[1]
        if (y1 - y_bottom) * (y2 - y_bottom) <= 0:
            x1, x2 = p1_phys[0], p2_phys[0]
            if abs(y2 - y1) < 1e-16:
                intersection_points_x_bottom.extend([x1, x2])
            else:
                x_int = x1 + (x2 - x1) * (y_bottom - y1) / (y2 - y1)
                intersection_points_x_bottom.append(x_int)

        # intersection with top boundary
        y1, y2 = p1_phys[1], p2_phys[1]
        if (y1 - y_top) * (y2 - y_top) <= 0:
            x1, x2 = p1_phys[0], p2_phys[0]
            if abs(y2 - y1) < 1e-16:
                intersection_points_x_top.extend([x1, x2])
            else:
                x_int = x1 + (x2 - x1) * (y_top - y1) / (y2 - y1)
                intersection_points_x_top.append(x_int)

        # intersection with left boundary
        x1, x2 = p1_phys[0], p2_phys[0]
        if (x1 - x_min_bound) * (x2 - x_min_bound) <= 0:
            y1, y2 = p1_phys[1], p2_phys[1]
            if abs(x2 - x1) < 1e-16:
                intersection_points_y_left.extend([y1, y2])
            else:
                y_int = y1 + (y2 - y1) * (x_min_bound - x1) / (x2 - x1)
                intersection_points_y_left.append(y_int)

        # intersection with right boundary
        x1, x2 = p1_phys[0], p2_phys[0]
        if (x1 - x_max_bound) * (x2 - x_max_bound) <= 0:
            y1, y2 = p1_phys[1], p2_phys[1]
            if abs(x2 - x1) < 1e-16:
                intersection_points_y_right.extend([y1, y2])
            else:
                y_int = y1 + (y2 - y1) * (x_max_bound - x1) / (x2 - x1)
                intersection_points_y_right.append(y_int)

        # normals based on the clipping region
        clipped = clip_segment(p1_phys, p2_phys,
                               region_bounds[0], region_bounds[1],
                               region_bounds[2], region_bounds[3])
        if clipped is None: #means we aren't within the region bounds we are expecting, so skip these marching squares edges
            continue
        p1_c, p2_c = clipped
        seg_length = np.linalg.norm(np.array(p2_c) - np.array(p1_c))
        if seg_length < 1e-12:
            continue

        points_to_interp = [[p1_c[1], p1_c[0]], [p2_c[1], p2_c[0]]]
        grads_dy_dx = interpn((Y, X), np.stack([dphi_dy, dphi_dx], axis=-1),
                              points_to_interp, method='linear', bounds_error=False, fill_value=0)
        grads_dx_dy = grads_dy_dx[:, ::-1]
        magnitudes = np.linalg.norm(grads_dx_dy, axis=1, keepdims=True) + 1e-12
        normals = grads_dx_dy / magnitudes
        avg_normal = np.mean(normals, axis=0)
        integrated_normal_sum += avg_normal * seg_length

        mid_x, mid_y = (p1_c[0] + p2_c[0]) / 2, (p1_c[1] + p2_c[1]) / 2
        edges_info_for_plot.append(((mid_x, mid_y), avg_normal))

    integrated_normal = integrated_normal_sum

    grad_vec = np.array([0.0, 0.0])
    # bottom: horizontal line at y_bottom, intersections are x values
    bottom_len, intersection_span_bottom, bottom_side_info = process_line(X, Y, grid_scale, levelset_t,
        intersection_points_x_bottom, 'horizontal', y_bottom, x_min_bound, x_max_bound)
    grad_vec[1] = bottom_len

    # top: horizontal line at y_top
    top_len, intersection_span_top, top_side_info = process_line(X, Y, grid_scale, levelset_t,
        intersection_points_x_top, 'horizontal', y_top, x_min_bound, x_max_bound)
    grad_vec[1] -= top_len

    # left: vertical line at x_min_bound, intersections are y values
    left_len, intersection_span_left, left_side_info = process_line(X, Y, grid_scale, levelset_t,
        intersection_points_y_left, 'vertical', x_min_bound, y_bottom, y_top)
    grad_vec[0] = left_len

    # right: vertical line at x_max_bound
    right_len, intersection_span_right, right_side_info = process_line(X, Y, grid_scale, levelset_t,
        intersection_points_y_right, 'vertical', x_max_bound, y_bottom, y_top)
    grad_vec[0] -= right_len

    # keep lengths for return in same variable names as before
    intersection_length_bottom = bottom_len
    intersection_length_top = top_len
    intersection_length_left = left_len
    intersection_length_right = right_len

    return (grad_vec, integrated_normal,
            edges, edges_info_for_plot,
            intersection_span_bottom, intersection_length_bottom, \
            intersection_span_left, intersection_length_left,\
            intersection_span_top, intersection_length_top, \
            intersection_span_right, intersection_length_right)