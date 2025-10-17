import numpy as np
from typing import Sequence, Dict, Any
from compute_grad_norms import compute_grad_normals_region_bounded
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from functools import partial
from pathlib import Path

def process_point(idx, xs, ys, ts, hx, hy, times_arr, levelsets, X, Y, grid_scale, time_tolerance):
    x_c, y_c, t_c = float(xs[idx]), float(ys[idx]), float(ts[idx])
    time_diffs = np.abs(times_arr - t_c)
    t_idx = int(np.argmin(time_diffs))
    if time_tolerance is not None and time_diffs[t_idx] > time_tolerance:
        return None, idx

    region_bounds = [x_c - hx, x_c + hx, y_c - hy, y_c + hy]
    levelset_t = levelsets[t_idx]

    try:
        (grad_vec, integrated_normal,
         edges, edges_info_for_plot,
         intersection_span_bottom, intersection_length_bottom,
         intersection_span_left, intersection_length_left,
         intersection_span_top, intersection_length_top,
         intersection_span_right, intersection_length_right) = \
            compute_grad_normals_region_bounded(levelset_t, X, Y, grid_scale, region_bounds)

        result = {
            "idx": idx,
            "center": (x_c, y_c, t_c),
            "time_index": t_idx,
            "time_value": float(times_arr[t_idx]),
            "region_bounds": region_bounds,
            "grad_vec": np.asarray(grad_vec, dtype=float),
            "integrated_normal": np.asarray(integrated_normal, dtype=float),
            "edges": edges,
            "edges_info_for_plot": edges_info_for_plot,
            "intersection_span_bottom": intersection_span_bottom,
            "intersection_length_bottom": float(intersection_length_bottom),
            "intersection_span_left": intersection_span_left,
            "intersection_length_left": float(intersection_length_left),
            "intersection_span_top": intersection_span_top,
            "intersection_length_top": float(intersection_length_top),
            "intersection_span_right": intersection_span_right,
            "intersection_length_right": float(intersection_length_right),
        }
        return result, None
    except Exception as e:
        return {
            "idx": idx,
            "center": (x_c, y_c, t_c),
            "time_index": t_idx,
            "time_value": float(times_arr[t_idx]),
            "region_bounds": region_bounds,
            "error": str(e)
        }, idx


def make_json_safe(obj):
    """Recursively convert any numpy / non-serializable objects to JSON-safe types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.generic,)):  # np.float32, np.int64, etc.
        return obj.item()
    elif isinstance(obj, (list, tuple, set)):
        return [make_json_safe(v) for v in obj]
    elif isinstance(obj, dict):
        return {str(k): make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, (float, int, str, bool)) or obj is None:
        return obj
    else:
        # Fallback: try string conversion for anything unexpected
        return str(obj)


def compute_patches_for_points(
    points: np.ndarray,
    times: np.ndarray,
    levelsets: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    grid_scale: float,
    region_half_size: Sequence[float],
    time_tolerance: float|None = None,
) -> Dict[str, Any]:

    path = Path('grad_patches.json')
    if path.exists():
        with path.open('r') as f:
            data = json.load(f)
            return data

    pts = np.asarray(points)
    if pts.ndim != 2 or pts.shape[1] < 3:
        raise ValueError("points must be an (N,>=3) array with columns [x, y, t, ...]")

    L_ref = 0.25
    xs = pts[:, 0].astype(float) * L_ref
    ys = pts[:, 1].astype(float) * L_ref
    ts = pts[:, 2].astype(float)
    hx, hy = float(region_half_size[0]) * L_ref, float(region_half_size[1]) * L_ref
    x_min_domain, x_max_domain = float(X[0]), float(X[-1])
    y_min_domain, y_max_domain = float(Y[0]), float(Y[-1])

    viable_mask = (xs - hx >= x_min_domain) & (xs + hx <= x_max_domain)
    viable_mask &= (ys - hy >= y_min_domain) & (ys + hy <= y_max_domain)
    viable_indices = np.nonzero(viable_mask)[0].tolist()
    skipped_indices = np.nonzero(~viable_mask)[0].tolist()
    times_arr = np.asarray(times)
    if times_arr.ndim != 1:
        raise ValueError("times must be 1D array corresponding to first axis of levelsets")
    if levelsets.ndim < 3:
        raise ValueError("levelsets must be shape (NT, ny, nx)")

    results = []
    func = partial(process_point, xs=xs, ys=ys, ts=ts, hx=hx, hy=hy,
                   times_arr=times_arr, levelsets=levelsets, X=X, Y=Y,
                   grid_scale=grid_scale, time_tolerance=time_tolerance)

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(func, idx) for idx in viable_indices]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Processing Points"):
            res, skipped = fut.result()
            if res is not None:
                results.append(res)
            if skipped is not None:
                skipped_indices.append(skipped)

    skipped_indices = sorted(set(skipped_indices) - set([r["idx"] for r in results if "idx" in r]))

    dict_to_return = {
        "viable_indices": [r["idx"] for r in results],
        "results": results
    }

    # Convert deeply nested numpy objects to JSON-safe form
    dict_to_save = make_json_safe(dict_to_return)

    with open(path, 'w') as f:
        json.dump(dict_to_save, f)

    return dict_to_return
