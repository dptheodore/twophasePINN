import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interpn
import imageio
import os
from marching_squares import Grid, march

# The plot_frame function from the previous step is unchanged and still needed.
def plot_frame(ax, X, Y, alpha_grid, edges, time_to_plot, annotation_skip=5):
    """
    Plots a single frame of the animation with cleaner axes and sparse annotations.
    """
    x_min, x_max = X[0], X[-1]
    y_min, y_max = Y[0], Y[-1]
    y_dim, x_dim = alpha_grid.shape
    
    ax.clear()

    # Draw faint grid lines manually
    for x_val in X:
        ax.axvline(x_val, color='gray', linestyle='--', linewidth=0.3, zorder=1)
    for y_val in Y:
        ax.axhline(y_val, color='gray', linestyle='--', linewidth=0.3, zorder=1)
    
    # Annotate with a skip rate to reduce clutter
    annotated_points = set()
    for i in range(0, y_dim - 1, annotation_skip):
        for j in range(0, x_dim - 1, annotation_skip):
            corners = [
                alpha_grid[i, j], alpha_grid[i+1, j],
                alpha_grid[i, j+1], alpha_grid[i+1, j+1]
            ]
            if 0 < sum(corners) < 4:
                points_to_annotate = [(i,j), (i+1,j), (i,j+1), (i+1,j+1)]
                for p_i, p_j in points_to_annotate:
                    if (p_i, p_j) not in annotated_points:
                        # ax.text(X[p_j], Y[p_i], str(int(alpha_grid[p_i, p_j])),
                        #         color='black', fontsize=6, alpha=0.7, 
                        #         ha='center', va='center', zorder=3)
                        annotated_points.add((p_i, p_j))

    # Plot the raw marching squares edges
    for p1, p2 in edges:
        x1, y1 = p1[1] + x_min, p1[0] + y_min
        x2, y2 = p2[1] + x_min, p2[0] + y_min
        ax.plot([x1, x2], [y1, y2], color='black', linewidth=1.5, linestyle=':', zorder=10)

    # Plot the nearly transparent background field
    ax.imshow(alpha_grid, extent=[x_min, x_max, y_min, y_max], 
              origin='lower', cmap='RdBu_r', alpha=0.2, zorder=2)

    ax.set_title(f"Marching Squares at t={time_to_plot:.3f}s")
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")
    ax.set_aspect('equal', adjustable='box')


# --- NEW FUNCTION FOR THE ZOOMED GIF ---
def create_zoomed_animation(h5_path, num_frames=20, time_range=(0.0, 3.0), zoom_window_size=0.15, annotation_skip=2):
    """
    Creates a zoomed-in animation that tracks the top of the bubble.
    """
    print(f"Loading data from: {h5_path}")
    with h5py.File(h5_path, "r") as data:
        X = np.array(data["X"])
        Y = np.array(data["Y"])
        times = np.array(data["time"])
        levelset_data = -np.array(data["levelset"])

    start_index = np.searchsorted(times, time_range[0], side='left')
    end_index = np.searchsorted(times, time_range[1], side='right')
    indices_to_plot = np.linspace(start_index, end_index, num_frames, dtype=int)
    print(f"Will generate {num_frames} zoomed frames from index {start_index} to {end_index}.")

    frame_dir = "zoomed_animation_frames"
    if not os.path.exists(frame_dir):
        os.makedirs(frame_dir)
    
    filenames = []
    fig, ax = plt.subplots(figsize=(8, 8))

    for i, time_index in enumerate(indices_to_plot):
        time_to_plot = times[time_index]
        print(f"Processing zoomed frame {i+1}/{num_frames} (t={time_to_plot:.3f}s)...")
        
        levelset_t = levelset_data[time_index, :, :]
        alpha_grid = levelset_t.astype(np.float32)
        
        bubble_points = np.where(alpha_grid == 1)
        if len(bubble_points[0]) > 0:
            # Find the index of the highest bubble point
            top_point_idx = np.argmax(bubble_points[0])
            track_y_idx, track_x_idx = bubble_points[0][top_point_idx], bubble_points[1][top_point_idx]
            
            # Convert grid indices to physical coordinates
            track_y_coord = Y[track_y_idx]
            track_x_coord = X[track_x_idx]
        else:
            # Fallback if no bubble is found
            track_y_coord, track_x_coord = np.mean(Y), np.mean(X)
        
        # Run Marching Squares
        grid_scale = X[1] - X[0]
        y_dim, x_dim = alpha_grid.shape
        ms_grid = Grid(scale=grid_scale, x_count=x_dim - 1, y_count=y_dim - 1)
        ms_grid.values = alpha_grid
        edges = march(ms_grid, iso=0.5, interpolated=True)
        
        # Plot the frame using the existing function
        plot_frame(ax, X, Y, alpha_grid, edges, time_to_plot, annotation_skip=annotation_skip)

        half_win = zoom_window_size / 2
        ax.set_xlim(track_x_coord - half_win, track_x_coord + half_win)
        ax.set_ylim(track_y_coord - half_win, track_y_coord + half_win)

        # Save the frame
        filename = f"{frame_dir}/frame_{i:03d}.png"
        filenames.append(filename)
        plt.savefig(filename, dpi=150)

    plt.close(fig)
    print("All zoomed frames generated.")

    # Assemble the GIF
    output_gif = "rising_bubble_zoom_animation.gif"
    print(f"Assembling frames into {output_gif}...")
    with imageio.get_writer(output_gif, mode='I', duration=0.2) as writer:
        for filename in filenames:
            image = imageio.v2.imread(filename)
            writer.append_data(image)
    
    print(f"✅ Zoomed animation saved to {output_gif}")


if __name__ == '__main__':
    HDF5_DATA_PATH = "../cfd_data/rising_bubble.h5"
    
    # Call the new zoomed animation function
    create_zoomed_animation(
        h5_path=HDF5_DATA_PATH,
        num_frames=30,          
        time_range=(0.0, 3.0),   
        zoom_window_size=1,  # How "wide" the zoom window is. Bigger value = zoomed out more
        annotation_skip=1       # Use a small skip value for high detail in the zoom.
    )