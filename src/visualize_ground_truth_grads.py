import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.interpolate import interpn
import imageio
import os
from tqdm import tqdm
from marching_squares import Grid, march

def clip_segment(p1, p2, x_min, x_max, y_min, y_max):
    """Clips a line segment to a rectangular box."""
    x1, y1 = p1; x2, y2 = p2
    dx, dy = x2 - x1, y2 - y1
    t0, t1 = 0.0, 1.0
    for edge in range(4):
        if edge == 0: p, q = -dx, -(x_min - x1)
        if edge == 1: p, q = dx, (x_max - x1)
        if edge == 2: p, q = -dy, -(y_min - y1)
        if edge == 3: p, q = dy, (y_max - y1)
        if p == 0:
            if q < 0: return None
        else:
            r = q / p
            if p < 0:
                if r > t1: return None
                t0 = max(t0, r)
            else:
                if r < t0: return None
                t1 = min(t1, r)
    if t0 > t1: return None
    return ((x1 + t0 * dx, y1 + t0 * dy), (x1 + t1 * dx, y1 + t1 * dy))

def calculate_normal_and_length_for_region(edges, dx, dy, X, Y, region_bounds):
    """
    Calculates the integrated normal and sums the geometric length of the
    interface segments within a specific region.
    """
    x_min_bound, x_max_bound, y_min_bound, y_max_bound = region_bounds
    x_min_phys, y_min_phys = X[0], Y[0]
    grid_points = (Y, X)
    integrated_normal = np.array([0.0, 0.0])
    total_geometric_length = 0.0
    
    for p1_idx, p2_idx in edges:
        p1_phys = (p1_idx[1] + x_min_phys, p1_idx[0] + y_min_phys)
        p2_phys = (p2_idx[1] + x_min_phys, p2_idx[0] + y_min_phys)
        
        clipped = clip_segment(p1_phys, p2_phys, x_min_bound, x_max_bound, y_min_bound, y_max_bound)
        if clipped:
            p1_c, p2_c = clipped
            mid_point = ((p1_c[0] + p2_c[0]) / 2, (p1_c[1] + p2_c[1]) / 2)
            seg_length = np.linalg.norm(np.array(p2_c) - np.array(p1_c))
            
            if seg_length < 1e-9: continue
            
            total_geometric_length += seg_length
            
            normal_x = interpn(grid_points, dx, (mid_point[1], mid_point[0]), method='linear', bounds_error=False, fill_value=0)[0]
            normal_y = interpn(grid_points, dy, (mid_point[1], mid_point[0]), method='linear', bounds_error=False, fill_value=0)[0]
            magnitude = np.hypot(normal_x, normal_y) + 1e-9
            n_hat = np.array([normal_x / magnitude, normal_y / magnitude])
            integrated_normal += n_hat * seg_length
            
    return -integrated_normal, total_geometric_length

def calculate_cell_by_cell_data(alpha_t, X, Y, grid_scale, dx, dy, all_edges):
    """
    Compares predicted normal (from ∇α) with actual normal (from geometry).
    Returns alignment measures per interface cell.
    """
    y_dim, x_dim = alpha_t.shape
    results = []
    for j in range(y_dim - 1):
        for i in range(x_dim - 1):
            # Skip non-interface cells
            corners = [alpha_t[j,i], alpha_t[j,i+1], alpha_t[j+1,i], alpha_t[j+1,i+1]]
            if all(c < 0.5 for c in corners) or all(c > 0.5 for c in corners):
                continue

            # --- Gradient at cell center ---
            grad_x_corners = (dx[j, i], dx[j, i+1], dx[j+1, i], dx[j+1, i+1])
            grad_y_corners = (dy[j, i], dy[j, i+1], dy[j+1, i], dy[j+1, i+1])
            avg_grad_x = np.mean(grad_x_corners)
            avg_grad_y = np.mean(grad_y_corners)
            grad_vec = np.array([avg_grad_x, avg_grad_y])
            grad_mag = np.linalg.norm(grad_vec)

            if grad_mag > 1e-9:
                predicted_normal = -grad_vec / grad_mag  # theoretical
            else:
                predicted_normal = np.array([0.0, 0.0])

            # --- Geometric normal ---
            cell_x_min, cell_x_max = X[i], X[i+1]
            cell_y_min, cell_y_max = Y[j], Y[j+1]
            cell_bounds = (cell_x_min, cell_x_max, cell_y_min, cell_y_max)

            integrated_normal, length_geom = calculate_normal_and_length_for_region(
                all_edges, dx, dy, X, Y, cell_bounds
            )

            if length_geom > 1e-9:
                actual_normal = integrated_normal / length_geom
            else:
                actual_normal = np.array([0.0, 0.0])

            # --- Compare via dot product (cosine of angle) ---
            alignment = np.dot(predicted_normal, actual_normal)

            results.append({
                "alignment": alignment,
                "grad_mag": grad_mag,
                "length_geom": length_geom
            })

    return results


def plot_alignment_comparison_frame(ax, X, Y, alpha_t, all_edges, time_to_plot, mean_align, std_align, num_interface_cells, num_alignment_cells):
    """Plots a frame showing alignment between gradient-based and geometric normals."""
    ax.clear()
    x_min, x_max, y_min, y_max = X[0], X[-1], Y[0], Y[-1]

    ax.imshow(alpha_t, extent=[x_min, x_max, y_min, y_max],
              origin='lower', cmap='coolwarm', alpha=0.5, zorder=2)
    
    for p1, p2 in all_edges:
        x1, y1 = p1[1] + x_min, p1[0] + y_min
        x2, y2 = p2[1] + x_min, p2[0] + y_min
        ax.plot([x1, x2], [y1, y2], color='black', linewidth=1.5, zorder=10)

    text_str = (
        f'Normal Vector Alignment (t={time_to_plot:.3f}s)\n'
        '--------------------------------------------\n'
        f'Num Interface Cells: {num_interface_cells}\n'
        f'Num Alignment Cells: {num_alignment_cells}\n'
        '--------------------------------------------\n'
        'Alignment = dot( -∇α/|∇α| , geometric normal )\n'
        f'Mean Alignment: {mean_align:6.6f}\n'
        f'Std Dev:        {std_align:6.6f}\n'
    )

    ax.text(0.05, 0.95, text_str, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.8),
            fontfamily='monospace')

    ax.set_title(f"Normal Vector Comparison at t={time_to_plot:.3f}s")
    ax.set_xlabel("X coordinate"); ax.set_ylabel("Y coordinate")
    ax.set_aspect('equal', adjustable='box')



def create_length_comparison_animation(h5_path, num_frames, time_range, zoom_window_size):
    """Creates an animation showing the interface length comparison."""
    print(f"Loading data from: {h5_path}")
    with h5py.File(h5_path, "r") as data:
        X, Y = np.array(data["X"]), np.array(data["Y"])
        times, levelset_data_raw = np.array(data["time"]), np.array(data["levelset"])
    
    start_idx, end_idx = np.searchsorted(times, time_range[0]), np.searchsorted(times, time_range[1], side='right')
    indices = np.linspace(start_idx, end_idx, num_frames, dtype=int)
    
    frame_dir, filenames = "length_comparison_frames", []
    os.makedirs(frame_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 10))
    grid_scale = X[1] - X[0]
    
    XX, YY = np.meshgrid(X, Y)

    for i, time_idx in enumerate(tqdm(indices, desc="Processing Frames")):
        time_t = times[time_idx]
        levelset_t = levelset_data_raw[time_idx, :, :]
        alpha_t = (levelset_t + 8.0) / 16.0
        
        dy, dx = np.gradient(alpha_t, grid_scale)
        ms_grid = Grid(scale=grid_scale, x_count=alpha_t.shape[1] - 1, y_count=alpha_t.shape[0] - 1)
        ms_grid.values = alpha_t.astype(np.float32)
        all_edges = march(ms_grid, iso=0.5, interpolated=True)
        
        cell_data = calculate_cell_by_cell_data(alpha_t, X, Y, grid_scale, dx, dy, all_edges)
        
        alignments = [
            d["alignment"] 
            for d in cell_data 
            if d["alignment"] is not None and not np.isnan(d["alignment"]) and abs(d["alignment"]) != 0
        ]
        num_interface_cells = len(cell_data)
        num_alignment_cells = len(alignments)
        if alignments:
            mean_align = np.mean(alignments)
            std_align = np.std(alignments)
        else:
            mean_align = std_align = 0
        
        sum_alpha = np.sum(alpha_t)
        if sum_alpha > 1e-9:
            track_x = np.sum(XX * alpha_t) / sum_alpha
            track_y = np.sum(YY * alpha_t) / sum_alpha
            track_point = (track_x, track_y)
        else:
             track_point = (np.mean(X), np.mean(Y))

        plot_alignment_comparison_frame(ax, X, Y, alpha_t, all_edges,
                                        time_t, mean_align, std_align, num_interface_cells, num_alignment_cells)

        
        half_win = zoom_window_size / 2
        ax.set_xlim(track_point[0] - half_win, track_point[0] + half_win)
        ax.set_ylim(track_point[1] - half_win, track_point[1] + half_win)
        
        filename = f"{frame_dir}/frame_{i:03d}.png"
        filenames.append(filename)
        plt.savefig(filename, dpi=120, bbox_inches='tight')

    plt.close(fig)
    print("All frames generated.")
    
    output_gif = "interface_length_comparison.gif"
    print(f"Assembling frames into {output_gif}...")
    with imageio.get_writer(output_gif, mode='I', duration=0.1) as writer:
        for filename in tqdm(filenames, desc="  Assembling GIF"):
            writer.append_data(imageio.v2.imread(filename))
    print(f"✅ Animation saved to {output_gif}")


if __name__ == '__main__':
    HDF5_DATA_PATH = "../cfd_data/rising_bubble.h5"
    
    create_length_comparison_animation(
        h5_path=HDF5_DATA_PATH,
        num_frames=50,
        time_range=(0.0, 3.0),
        zoom_window_size=1.0
    )