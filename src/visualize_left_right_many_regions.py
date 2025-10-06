import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import imageio
from tqdm import tqdm
from scipy.ndimage import sobel
from marching_squares import Grid, march
from scipy.interpolate import interpn
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
# -----------------------------
# Utility: generate rectangular patches covering the domain
# -----------------------------
def generate_patches(X, Y, num_patches_x, num_patches_y):
    """Split the domain into num_patches_x × num_patches_y rectangular patches."""
    x_min, x_max = X[0], X[-1]
    y_min, y_max = Y[0], Y[-1]

    x_edges = np.linspace(x_min, x_max, num_patches_x + 1)
    y_edges = np.linspace(y_min, y_max, num_patches_y + 1)

    patches_list = []
    for i in range(num_patches_x):
        for j in range(num_patches_y):
            patches_list.append((x_edges[i], x_edges[i+1],
                                 y_edges[j], y_edges[j+1]))
    return patches_list

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

# -----------------------------
# Placeholder: your region computation function
# -----------------------------
def compute_grad_normals_region_bounded(levelset_t, X, Y, grid_scale, region_bounds):
    ny, nx = levelset_t.shape
    dphi_dy = sobel(levelset_t, axis=0, mode='constant') / grid_scale
    dphi_dx = sobel(levelset_t, axis=1, mode='constant') / grid_scale
    
    # Marching squares
    ms_grid = Grid(scale=grid_scale, x_count=nx - 1, y_count=ny - 1)
    ms_grid.values = levelset_t.astype(np.float32)
    edges = march(ms_grid, iso=0.0, interpolated=True)

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

    # span & length this is mainly debugging to ensure marching squares intersection is working as expected
    grad_vec = np.array([0.0, 0.0])
    intersection_points_x_bottom = [x for x in intersection_points_x_bottom if x_min_bound <= x <= x_max_bound]
    if intersection_points_x_bottom and len(intersection_points_x_bottom) != 1:
        min_x = max(min(intersection_points_x_bottom), x_min_bound)
        max_x = min(max(intersection_points_x_bottom), x_max_bound)
        intersection_span_bottom = (min_x, max_x)
        intersection_length_bottom = max(0.0, max_x - min_x)
        grad_vec[1] = intersection_length_bottom
    elif intersection_points_x_bottom and len(intersection_points_x_bottom) == 1:
        probe_x = x_min_bound + 1e-8  # tiny offset to stay inside
        probe_y = y_bottom
        phi_probe = interpn((Y, X), levelset_t, [[probe_y, probe_x]],
                            method="linear", bounds_error=False, fill_value=np.nan)[0]

        if phi_probe < 0:
            # left side is inside → span from left bound to intercept
            intersection_span_bottom = (x_min_bound, intersection_points_x_bottom[0])
            intersection_length_bottom = (intersection_points_x_bottom[0] - x_min_bound)
        else:
            # left side is outside → span from intercept to right bound
            intersection_span_bottom = (intersection_points_x_bottom[0], x_max_bound)
            intersection_length_bottom = (x_max_bound - intersection_points_x_bottom[0])
    else:
        intersection_span_bottom = (None, None)
        intersection_length_bottom = 0.0

    if intersection_points_x_top:
        min_x = max(min(intersection_points_x_top), x_min_bound)
        max_x = min(max(intersection_points_x_top), x_max_bound)
        intersection_span_top = (min_x, max_x)
        intersection_length_top = max(0.0, max_x - min_x)
        grad_vec[1] -= intersection_length_top
    else:
        intersection_span_top = (None, None)
        intersection_length_top = 0.0

    if intersection_points_y_left:
        min_y = max(min(intersection_points_y_left), y_bottom)
        max_y = min(max(intersection_points_y_left), y_top)
        intersection_span_left = (min_y, max_y)
        intersection_length_left = max(0.0, max_y - min_y)
        grad_vec[0] = intersection_length_left
    else:
        intersection_span_left = (None, None)
        intersection_length_left = 0.0

    if intersection_points_y_right:
        min_y = max(min(intersection_points_y_right), y_bottom)
        max_y = min(max(intersection_points_y_right), y_top)
        intersection_span_right = (min_y, max_y)
        intersection_length_right = max(0.0, max_y - min_y)
        grad_vec[0] -= intersection_length_right
    else:
        intersection_span_right = (None, None)
        intersection_length_right = 0.0
    grad_vec[1] = intersection_length_bottom - intersection_length_top
    return (grad_vec, integrated_normal,
            edges, edges_info_for_plot,
            intersection_span_bottom, intersection_length_bottom, \
            intersection_span_left, intersection_length_left,\
            intersection_span_top, intersection_length_top, \
            intersection_span_right, intersection_length_right)
# -----------------------------
# Animation creator
# -----------------------------
def create_animation(h5_path, num_frames, time_range, num_patches=(8, 8)):
    """Animate comparison across patches, with mean error history."""
    print(f"Loading data from: {h5_path}")
    with h5py.File(h5_path, "r") as data:
        X, Y = np.array(data["X"]), np.array(data["Y"])
        times, levelset_data = np.array(data["time"]), np.array(data["levelset"])

    levelset_data /= 8.0

    start_idx = np.searchsorted(times, time_range[0])
    end_idx = np.searchsorted(times, time_range[1], side='right')
    indices = np.linspace(start_idx, end_idx, num_frames, dtype=int)

    frame_dir, filenames = "multi_region_frames", []
    os.makedirs(frame_dir, exist_ok=True)
    grid_scale = X[1] - X[0]

    # --- patches
    patches_list = generate_patches(X, Y, num_patches[0], num_patches[1])
    #patches_list = [(-0.12451171875, 0.0, -0.24951171875, -0.124755859375)]
    fig, ax = plt.subplots(figsize=(8, 8))

    error_history = []

    for i, time_idx in enumerate(tqdm(indices, desc="Processing Frames")):
        time_t = times[time_idx]
        levelset_t = levelset_data[time_idx, :, :]

        ax.clear()
        ax.imshow(levelset_t, extent=[X[0], X[-1], Y[0], Y[-1]],
                  origin='lower', cmap='coolwarm', vmin=-1.0, vmax=1.0, alpha=0.5)

        patch_errors = []

        for region_bounds in patches_list:
            grad_vec, integrated_normal, \
            edges, edges_info_for_plot, \
            intersection_span_bottom, intersection_length_bottom, \
            intersection_span_left, intersection_length_left,\
            intersection_span_top, intersection_length_top, \
            intersection_span_right, intersection_length_right = \
                compute_grad_normals_region_bounded(levelset_t, X, Y, grid_scale, region_bounds)
            y_bottom, y_top = region_bounds[2],region_bounds[3]
            x_left, x_right = region_bounds[0], region_bounds[1]
            x_min, x_max, y_min, y_max = X[0], X[-1], Y[0], Y[-1]
            for p1, p2 in edges:
                ax.plot([p1[1] + x_min, p2[1] + x_min], [p1[0] + y_min, p2[0] + y_min], color='black', linewidth=1.5, zorder=3)

            x_start, x_end = intersection_span_bottom
            if x_start is not None and x_end is not None:
                ax.plot([x_start, x_end], [y_bottom, y_bottom],
                        color='purple', linewidth=1, zorder=7, solid_capstyle='butt')

            x_start, x_end = intersection_span_top
            if x_start is not None and x_end is not None:
                ax.plot([x_start, x_end], [y_top, y_top],
                        color='purple', linewidth=1, zorder=7, solid_capstyle='butt')
            
            y_start, y_end = intersection_span_left
            if y_start is not None and y_end is not None:
                ax.plot([x_left, x_left], [y_start, y_end],
                        color='purple', linewidth=1, zorder=7, solid_capstyle='butt')

            y_start, y_end = intersection_span_right
            if y_start is not None and y_end is not None:
                ax.plot([x_right, x_right], [y_start, y_end],
                        color='purple', linewidth=1, zorder=7, solid_capstyle='butt')

            # Compute error for this patch
            patch_error = np.linalg.norm(grad_vec - integrated_normal)
            patch_errors.append(patch_error)


            # Draw region box
            rect = patches.Rectangle((region_bounds[0], region_bounds[2]),
                                     region_bounds[1] - region_bounds[0],
                                     region_bounds[3] - region_bounds[2],
                                     linewidth=0.3, edgecolor='gray',
                                     linestyle='--', facecolor='none', alpha=0.5)
            ax.add_patch(rect)

            # Draw vectors at region center
            cx = (region_bounds[0] + region_bounds[1]) / 2
            cy = (region_bounds[2] + region_bounds[3]) / 2
            ax.arrow(cx, cy, grad_vec[0]*0.05, grad_vec[1]*0.05,
                     color='deeppink', head_width=0.01, alpha=0.7)
            ax.arrow(cx, cy, integrated_normal[0]*0.05, integrated_normal[1]*0.05,
                     color='blue', head_width=0.01, alpha=0.7)
        mean_error = np.mean(patch_errors)
        error_history.append(mean_error)

        ax.set_title(f"Multi-Patch Comparison, t={time_t:.3f}s\nMean error={mean_error:.3e}")
        ax.set_aspect('equal')

        filename = f"{frame_dir}/frame_{i:04d}.png"
        filenames.append(filename)
        # axins = inset_axes(ax, width="40%", height="40%", loc="upper right")
        # axins.imshow(levelset_t, extent=[X[0], X[-1], Y[0], Y[-1]],
        #              origin='lower', cmap='coolwarm', vmin=-1.0, vmax=1.0, alpha=0.5)

        # x_min, x_max, y_min, y_max = region_bounds
        # pad = 0.1 * max(x_max - x_min, y_max - y_min)  # add 1% padding
        # ax.set_xlim(x_min - pad, x_max + pad)
        # ax.set_ylim(y_min - pad, y_max + pad)
        plt.savefig(filename, dpi=120)
    plt.close(fig)

    # Save GIF
    output_gif = "multi_region_comparison.gif"
    print(f"Assembling frames into {output_gif}...")
    with imageio.get_writer(output_gif, mode='I', duration=0.1) as writer:
        for filename in tqdm(filenames, desc="Assembling GIF"):
            writer.append_data(imageio.v2.imread(filename))
    print(f"✅ Animation saved to {output_gif}")

    # Plot error history
    plt.figure(figsize=(8, 4))
    plt.plot(np.linspace(time_range[0], time_range[1], num_frames),
             error_history, marker='o')
    plt.xlabel("Time")
    plt.ylabel("Mean Patch Error")
    plt.title("Totalized Error History (mean over patches)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("mean_error_history.png", dpi=150)
    plt.close()
    print("✅ Error history plot saved to mean_error_history.png")


# -----------------------------
# Run example
# -----------------------------
if __name__ == "__main__":
    HDF5_DATA_PATH = "../cfd_data/rising_bubble.h5"
    create_animation(HDF5_DATA_PATH,
                     num_frames=1,
                     time_range=(2.020, 2.020),
                     num_patches=(8, 16))
