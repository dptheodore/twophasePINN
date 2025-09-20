from math import nan
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.interpolate import interpn
import imageio
import os
from tqdm import tqdm
from marching_squares import Grid, march
from skimage import measure

def clip_segment(p1, p2, x_min, x_max, y_min, y_max):
    """Clips a line segment to a rectangular box."""
    x1, y1 = p1
    x2, y2 = p2
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
            r = q * (1/p)
            if p < 0:
                if r > t1: return None
                t0 = max(t0, r)
            else:
                if r < t0: return None
                t1 = min(t1, r)
    if t0 > t1: return None
    return ((x1 + t0 * dx, y1 + t0 * dy), (x1 + t1 * dx, y1 + t1 * dy))

def plot_computation_frame(ax, X, Y, alpha_t, edges, edges_info, combined_mask,
                           region_bounds, total_flux_term, integrated_norm, 
                           absolute_error, track_point, time_to_plot):
    x_min, x_max, y_min, y_max = X[0], X[-1], Y[0], Y[-1]
    ax.clear()

    # base alpha field
    ax.imshow(alpha_t, extent=[x_min, x_max, y_min, y_max],
              origin='lower', cmap='viridis', alpha=0.3, zorder=1)

    contours = measure.find_contours(combined_mask.astype(float), 0.5)
    for contour in contours:
        ax.plot(
            np.interp(contour[:,1], np.arange(combined_mask.shape[1]), np.linspace(x_min, x_max, combined_mask.shape[1])),
            np.interp(contour[:,0], np.arange(combined_mask.shape[0]), np.linspace(y_min, y_max, combined_mask.shape[0])),
            color="green", linewidth=2, linestyle="--", zorder=5
        )

    # marching squares edges
    for p1, p2 in edges:
        x1, y1 = p1[1] + x_min, p1[0] + y_min
        x2, y2 = p2[1] + x_min, p2[0] + y_min
        ax.plot([x1, x2], [y1, y2], color='black', linewidth=1.5, zorder=3)

    # outward normals at midpoints
    for (mx, my), n_out in edges_info:
        ax.arrow(mx, my, n_out[0]*0.05, n_out[1]*0.05,
                 head_width=0.015, color='cyan', zorder=4)
        #ax.text(mx, my, f"({n_out[0]:.2f},{n_out[1]:.2f})", fontsize=6, color='cyan')

    # region bounds rectangle
    rect_x, rect_y = region_bounds[0], region_bounds[2]
    rect_w, rect_h = region_bounds[1] - rect_x, region_bounds[3] - rect_y
    rect = patches.Rectangle((rect_x, rect_y), rect_w, rect_h,
                             linewidth=2, edgecolor='red', linestyle='--',
                             facecolor='none', zorder=5)
    ax.add_patch(rect)

    # integrated normal arrow (deeppink)
    vec_x, vec_y = -integrated_norm
    ax.arrow(track_point[0], track_point[1], vec_x*0.2, vec_y*0.2,
             width=0.005, head_width=0.02, fc='deeppink', ec='deeppink', zorder=6)

    # info textbox
    flux_x, flux_y = total_flux_term
    norm_x, norm_y = -integrated_norm
    text_str = (
        f"Flux (Va): ({flux_x:.4f}, {flux_y:.4f})\n"
        f"Norm (Vn): ({norm_x:.4f}, {norm_y:.4f})\n"
        f"Abs Error ||Va-Vn||: {absolute_error:.3e}"
    )
    ax.text(0.05, 0.95, text_str, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.8),
            fontfamily='monospace')

    ax.set_title(f"Computation at t={time_to_plot:.3f}s")
    ax.set_aspect('equal')


import numpy as np
from scipy.interpolate import interpn
from marching_squares import Grid, march

def compute_transport_terms(alpha_t, X, Y, grid_scale, region_bounds):
    """
    Computes and compares the key terms from the 2D Reynolds Transport Theorem.
    This version correctly masks the boundary flux calculation.
    """
    x_min_bound, x_max_bound, y_min_bound, y_max_bound = region_bounds
    cell_area = grid_scale ** 2
    x_min_phys, y_min_phys = X[0], Y[0]
    ny, nx = alpha_t.shape

    # --- 1. Area Term: ∫_A(bubble) ∇α dA ---
    dady, dadx = np.gradient(alpha_t, grid_scale)
    
    # Define the integration domain: bubble area inside the region
    bubble_mask = alpha_t >= 0.5
    i_min, i_max = np.searchsorted(X, (x_min_bound, x_max_bound))
    j_min, j_max = np.searchsorted(Y, (y_min_bound, y_max_bound))
    region_mask = np.zeros_like(alpha_t, dtype=bool)
    region_mask[j_min:j_max, i_min:i_max] = True
    combined_mask = bubble_mask & region_mask

    # Sum gradients over the masked area
    area_integral_x = np.sum(dadx[combined_mask]) * cell_area
    area_integral_y = np.sum(dady[combined_mask]) * cell_area
    area_term = np.array([area_integral_x, area_integral_y])

    # --- 2. Curve Term: ∫_C(interface) α n dS ---
    ms_grid = Grid(scale=grid_scale, x_count=nx - 1, y_count=ny - 1)
    ms_grid.values = alpha_t.astype(np.float32)
    edges = march(ms_grid, iso=0.5, interpolated=True)
    edges_info = []
    curve_integral = np.zeros(2, dtype=np.float64)
    for p1_idx, p2_idx in edges:
        p1_phys = (p1_idx[1] + x_min_phys, p1_idx[0] + y_min_phys)
        p2_phys = (p2_idx[1] + x_min_phys, p2_idx[0] + y_min_phys)
        
        clipped = clip_segment(p1_phys, p2_phys, x_min_bound, x_max_bound, y_min_bound, y_max_bound)
        if clipped is None: continue
        
        p1_c, p2_c = clipped
        seg_length = np.linalg.norm(np.array(p2_c) - np.array(p1_c))
        if seg_length < 1e-12: continue

        mid_x, mid_y = (p1_c[0] + p2_c[0])/2, (p1_c[1] + p2_c[1])/2
        
        grad_x = interpn((Y, X), dadx, [[mid_y, mid_x]], method='linear', bounds_error=False, fill_value=0)[0]
        grad_y = interpn((Y, X), dady, [[mid_y, mid_x]], method='linear', bounds_error=False, fill_value=0)[0]
        magnitude = np.hypot(grad_x, grad_y) + 1e-12
        normal = np.array([grad_x / magnitude, grad_y / magnitude])
        
        f_val = 0.5
        curve_integral += f_val * normal * seg_length
        edges_info.append(((mid_x, mid_y), -normal))
    
    curve_term = -curve_integral

    # --- 3. Boundary Flux Term: ∫_C(bounds) α n dS (Corrected) ---
    boundary_flux = np.zeros(2, dtype=np.float64)
    
    # Clamp indices to be valid for direct array access
    i_min_idx, i_max_idx = np.clip([i_min, i_max], 0, nx - 1)
    j_min_idx, j_max_idx = np.clip([j_min, j_max], 0, ny - 1)

    if i_min < i_max and j_min < j_max:
        # Get alpha values on each face
        right_face_alpha  = alpha_t[j_min:j_max, i_max_idx]
        left_face_alpha   = alpha_t[j_min:j_max, i_min_idx]
        top_face_alpha    = alpha_t[j_max_idx, i_min:i_max]
        bottom_face_alpha = alpha_t[j_min_idx, i_min:i_max]

        # Get the bubble mask (alpha >= 0.5) for each face
        right_face_mask   = bubble_mask[j_min:j_max, i_max_idx]
        left_face_mask    = bubble_mask[j_min:j_max, i_min_idx]
        top_face_mask     = bubble_mask[j_max_idx, i_min:i_max]
        bottom_face_mask  = bubble_mask[j_min_idx, i_min:i_max]

        # Calculate flux ONLY over the part of the boundary where alpha >= 0.5
        if i_max_idx > i_min_idx:
            flux_x = (np.sum(right_face_alpha[right_face_mask]) - 
                      np.sum(left_face_alpha[left_face_mask]))
            boundary_flux[0] = flux_x * grid_scale
        
        if j_max_idx > j_min_idx:
            flux_y = (np.sum(top_face_alpha[top_face_mask]) - 
                      np.sum(bottom_face_alpha[bottom_face_mask]))
            boundary_flux[1] = flux_y * grid_scale
    
    # --- Final Comparison ---
    total_flux_term = curve_term + boundary_flux
    
    return total_flux_term, area_term, edges, edges_info, combined_mask
# --- MODIFIED: Added error calculation in the main loop ---
def create_computation_animation(h5_path, num_frames, time_range, region_bounds, zoom_window_size):
    """Creates a zoomed-in animation showing the comparison and error."""
    print(f"Loading data from: {h5_path}")
    with h5py.File(h5_path, "r") as data:
        X, Y = np.array(data["X"]), np.array(data["Y"])
        times, levelset_data = np.array(data["time"]), -np.array(data["levelset"])

    start_idx = np.searchsorted(times, time_range[0])
    end_idx = np.searchsorted(times, time_range[1], side='right')
    indices = np.linspace(start_idx, end_idx, num_frames, dtype=int)
    
    frame_dir, filenames = "computation_animation_frames", []
    os.makedirs(frame_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 8))
    grid_scale = X[1] - X[0]

    for i, time_idx in enumerate(tqdm(indices, desc="Processing Frames")):
        time_t = times[time_idx]
        levelset_t = levelset_data[time_idx, :, :]

        min_val, max_val = np.min(levelset_t), np.max(levelset_t)
        alpha_t = (levelset_t - min_val) / (max_val - min_val)
        
        total_flux_term, area_term, edges, edges_info, combined_mask = compute_transport_terms(alpha_t, X, Y, grid_scale, region_bounds)

        diff_vec = area_term - total_flux_term
        absolute_error = np.linalg.norm(diff_vec)
        #avg_mag = (np.linalg.norm(total_flux_term) + np.linalg.norm(scaled_gradient)) / 2.0
        #relative_error = absolute_error / avg_mag if avg_mag > 1e-9 else 0.0

        #print(f"Abs E:{absolute_error:.2e} Curve: {total_flux_term} area_term: {area_term}")

        bubble_points = np.where(alpha_t > 0.5)
        track_point = (np.mean(X), np.mean(Y))
        if len(bubble_points[0]) > 0:
            top_idx = np.argmax(bubble_points[0])
            track_point = (X[bubble_points[1][top_idx]], Y[bubble_points[0][top_idx]])

        plot_computation_frame(ax, X, Y, alpha_t, edges, edges_info, combined_mask, region_bounds,
                               total_flux_term, area_term,
                               absolute_error,
                               track_point, time_t)

        half_win = zoom_window_size / 2
        ax.set_xlim(track_point[0] - half_win, track_point[0] + half_win)
        ax.set_ylim(track_point[1] - half_win, track_point[1] + half_win)
        
        filename = f"{frame_dir}/frame_{i:03d}.png"
        filenames.append(filename)
        plt.savefig(filename, dpi=150)

    plt.close(fig)
    print("All frames generated.")
    
    output_gif = "computation_comparison_animation.gif"
    print(f"Assembling frames into {output_gif}...")
    with imageio.get_writer(output_gif, mode='I', duration=0.15) as writer:
        for filename in tqdm(filenames, desc="Assembling GIF"):
            writer.append_data(imageio.v2.imread(filename))
    print(f"✅ Animation saved to {output_gif}")


if __name__ == '__main__':
    HDF5_DATA_PATH = "../cfd_data/rising_bubble.h5"
    COMPUTATION_REGION = (-0.45, 0.45, -0.2, 0.5)
    
    create_computation_animation(
        h5_path=HDF5_DATA_PATH,
        num_frames=50,
        time_range=(0.0, 3.0),
        region_bounds=COMPUTATION_REGION,
        zoom_window_size=0.5
    )