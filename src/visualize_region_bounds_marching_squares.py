import h5py
import imageio
import numpy as np
import os
from matplotlib import pyplot as plt, patches
from scipy.interpolate import interpn
from scipy.ndimage import sobel
from skimage.metrics import mean_squared_error
from tqdm import tqdm
from marching_squares import Grid, march

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

# --- MAIN COMPUTATION FUNCTION ---
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
    intersection_points_x = []

    # bounds
    x_min_bound, x_max_bound = region_bounds[0], region_bounds[1]
    y_bottom = region_bounds[2]

    for p1, p2 in edges:
        # use march coords directly, add domain offset
        p1_phys = (p1[1] + X[0], p1[0] + Y[0])
        p2_phys = (p2[1] + X[0], p2[0] + Y[0])

        # intersection with bottom boundary
        y1, y2 = p1_phys[1], p2_phys[1]
        if (y1 - y_bottom) * (y2 - y_bottom) <= 0:
            x1, x2 = p1_phys[0], p2_phys[0]
            if abs(y2 - y1) < 1e-12:
                intersection_points_x.extend([x1, x2])
            else:
                x_int = x1 + (x2 - x1) * (y_bottom - y1) / (y2 - y1)
                intersection_points_x.append(x_int)

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
    if intersection_points_x:
        min_x = max(min(intersection_points_x), x_min_bound)
        max_x = min(max(intersection_points_x), x_max_bound)
        intersection_span = (min_x, max_x)
        intersection_length = max(0.0, max_x - min_x)
        grad_vec = np.array([0.0, intersection_length])
    else:
        intersection_span = (None, None)
        intersection_length = 0.0

    return (grad_vec, integrated_normal,
            edges, edges_info_for_plot,
            intersection_span, intersection_length)


def plot_frame(ax, X, Y, levelset_t, edges, edges_info, region_bounds,
               grad_vec, integrated_normal, track_point, time_to_plot, 
               intersection_span, intersection_length, arrow_skip_rate=5):
    """Plots a single frame of the animation."""
    ax.clear()
    x_min, x_max, y_min, y_max = X[0], X[-1], Y[0], Y[-1]
    y_bottom = region_bounds[2]
    ax.imshow(levelset_t, extent=[x_min, x_max, y_min, y_max], origin='lower',
              cmap='coolwarm', vmin=-1.0, vmax=1.0, alpha=0.5, zorder=1)

    for p1, p2 in edges:
        ax.plot([p1[1] + x_min, p2[1] + x_min], [p1[0] + y_min, p2[0] + y_min], color='black', linewidth=1.5, zorder=3)

    for (mx, my), n_out in edges_info[::arrow_skip_rate]:
        ax.arrow(mx, my, n_out[0]*0.05, n_out[1]*0.05, head_width=0.015, color='cyan', zorder=4)

    rect = patches.Rectangle((region_bounds[0], region_bounds[2]), region_bounds[1] - region_bounds[0], region_bounds[3] - region_bounds[2],
                             linewidth=2, edgecolor='red', linestyle='--', facecolor='none', zorder=5)
    ax.add_patch(rect)
    
    x_start, x_end = intersection_span
    if x_start is not None and x_end is not None:
        ax.plot([x_start, x_end], [y_bottom, y_bottom],
                color='purple', linewidth=1, zorder=7, solid_capstyle='butt')
    
    ax.arrow(track_point[0], track_point[1], grad_vec[0], grad_vec[1], width=0.005, head_width=0.02, fc='deeppink', ec='deeppink', zorder=6)
    ax.arrow(track_point[0], track_point[1], integrated_normal[0], integrated_normal[1], width=0.005, head_width=0.02, fc='blue', ec='blue', zorder=5, alpha=0.7)
    
    squared_error = mean_squared_error(grad_vec, integrated_normal)
    text_str = (f"Area Grad (pink): ({grad_vec[0]:.6f}, {grad_vec[1]:.6f})\n"
                f"Avg Norm (blue): ({integrated_normal[0]:.6f}, {integrated_normal[1]:.6f})\n"
                f"MSE             : {squared_error:.3e}\n")
    ax.text(0.05, 0.95, text_str, transform=ax.transAxes, fontsize=9, verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.8), fontfamily='monospace')

    ax.set_title(f"Field View at t={time_to_plot:.3f}s")
    ax.set_aspect('equal')
    half_win = 1
    ax.set_xlim(track_point[0] - half_win, track_point[0] + half_win)
    ax.set_ylim(track_point[1] - half_win, track_point[1] + half_win)


def plot_timeseries(ax, time_history, area_history, flux_history, current_time):
    """Plots the vector components as a function of time."""
    ax.clear()
    area_np = np.array(area_history)
    flux_np = np.array(flux_history)
    
    ax.plot(time_history, area_np[:, 0], color='red', linestyle='-', label='Area Term (X)')
    ax.plot(time_history, area_np[:, 1], color='green', linestyle='-', label='Area Term (Y)')
    ax.plot(time_history, flux_np[:, 0], color='red', linestyle='--', label='Avg Norm Term (X)')
    ax.plot(time_history, flux_np[:, 1], color='green', linestyle='--', label='Avg Norm Term (Y)')
    
    ax.axvline(current_time, color='black', linestyle=':', lw=1.5)
    ax.legend(fontsize='small')
    ax.set_title('Vector Components vs. Time')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Component Value')
    ax.grid(True, linestyle='--', alpha=0.6)

def plot_error(ax, time_history, error_history, current_time):
    """Plots the squared error between the vectors as a function of time."""
    ax.clear()
    ax.plot(time_history, error_history, 'b-')
    ax.set_yscale('log')
    
    ax.axvline(current_time, color='black', linestyle=':', lw=1.5)
    ax.set_title('MSE vs. Time')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Squared Error $||V_A - V_N||^2$ (log scale)')
    ax.grid(True, linestyle='--', alpha=0.6)

def create_animation(h5_path, num_frames, time_range, region_bounds):
    """Main animation loop."""
    print(f"Loading data from: {h5_path}")
    with h5py.File(h5_path, "r") as data:
        X, Y = np.array(data["X"]), np.array(data["Y"])
        times, levelset_data = np.array(data["time"]), np.array(data["levelset"])
    
    levelset_data /= 8.0 

    start_idx = np.searchsorted(times, time_range[0])
    end_idx = np.searchsorted(times, time_range[1], side='right')
    indices = np.linspace(start_idx, end_idx, num_frames, dtype=int)
    
    frame_dir, filenames = "definitive_computation_frames", []
    os.makedirs(frame_dir, exist_ok=True)
    grid_scale = X[1] - X[0]
        # Create a multi-panel figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 14))
    ax_main = axes[0, 0]
    ax_timeseries = axes[0, 1]
    ax_error = axes[1, 0]
    axes[1, 1].axis('off') # Hide the unused subplot

    # Lists to store data for time-series plots
    time_history, area_history, flux_history, error_history = [], [], [], []

    for i, time_idx in enumerate(tqdm(indices, desc="Processing Frames")):
        time_t = times[time_idx]
        levelset_t = levelset_data[time_idx, :, :]
        
        grad_vec, integrated_normal, edges, edges_info, intersection_span, intersection_length = \
            compute_grad_normals_region_bounded(levelset_t, X, Y, grid_scale, region_bounds)

        bubble_points_y, bubble_points_x = np.where(levelset_t < 0)
        track_point = (np.mean(X), np.mean(Y))
        if len(bubble_points_y) > 0:
            top_idx = np.argmax(bubble_points_y)
            track_point = (X[bubble_points_x[top_idx]], Y[bubble_points_y[top_idx]])

        # Update history lists
        time_history.append(time_t)
        area_history.append(grad_vec)
        flux_history.append(integrated_normal)
        squared_error = mean_squared_error(abs(grad_vec), abs(integrated_normal))
        error_history.append(squared_error)

        plot_frame(ax_main, X, Y, levelset_t, edges, edges_info,
                    region_bounds, grad_vec, integrated_normal, track_point, time_t,
                   intersection_span, intersection_length, arrow_skip_rate=1)

        plot_timeseries(ax_timeseries, time_history, area_history, flux_history, time_t)
        plot_error(ax_error, time_history, error_history, time_t)
        
        filename = f"{frame_dir}/frame_{i:04d}.png"
        filenames.append(filename)
        plt.savefig(filename, dpi=150)

    plt.close(fig)
    print("All frames generated.")
    
    output_gif = "definitive_comparison.gif"
    print(f"Assembling frames into {output_gif}...")
    with imageio.get_writer(output_gif, mode='I', duration=0.1) as writer:
        for filename in tqdm(filenames, desc="Assembling GIF"):
            writer.append_data(imageio.v2.imread(filename))
    print(f"✅ Animation saved to {output_gif}")

if __name__ == '__main__':
    HDF5_DATA_PATH = "../cfd_data/rising_bubble.h5"
    COMPUTATION_REGION = (-0.45, 0.45, -0.2, 0.5)
    
    create_animation(
        h5_path=HDF5_DATA_PATH,
        num_frames=100,
        time_range=(0.0, 3.0),
        region_bounds=COMPUTATION_REGION
    )