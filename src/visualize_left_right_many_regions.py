import os
import h5py
from matplotlib import patches
import numpy as np
import matplotlib.pyplot as plt
import imageio
from tqdm import tqdm
from compute_patches import compute_patches_for_points
from generate_points import get_training_data
import tensorflow as tf
from matplotlib.animation import FuncAnimation
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
# -----------------------------
# Animation creator
# -----------------------------
np.random.seed(1234)
tf.random.set_seed(1234)
NOP_a = (500, 400)
NOP_PDE = (400, 2000, 3000)
NOP_north = (20, 20)
NOP_south = (20, 20)
NOP_east = (20, 20)
NOP_west = (20, 20)

#every point not in boundary, compute small marching squares window around it (i.e. 1 voxel)
#plot is bunch of little squares around each little point in domain, overlapping squares everywhere, espec. near interface
#once have meshes can compute theorems
#theorem is compute fraction of top and bottom and fraction of left and right that are immersed inside the red or blue fields
#gradient, deriv of volume fraciton field, interp on mesh, --> gets normal vec
#4 points vert 5 points horiz entire domain, compute theorems in each region
#eventually nice: for any arb point in domain compute region around it


def create_animation(h5_path, num_frames, time_range, num_patches=(8, 8)):
    """Animate comparison across patches, with mean error history."""
    print(f"Loading data from: {h5_path}")
    with h5py.File(h5_path, "r") as data:
        X, Y = np.array(data["X"]), np.array(data["Y"])
        times, levelset_data = np.array(data["time"]), np.array(data["levelset"])

    levelset_data /= 8.0
    levelset_data = -levelset_data #might need to flip this back based on the data

    levelset_data = (levelset_data - (np.min(levelset_data))) / (np.max(levelset_data) - np.min(levelset_data))
    start_idx = np.searchsorted(times, time_range[0])
    end_idx = np.searchsorted(times, time_range[1], side='right')
    indices = np.linspace(start_idx, end_idx-1, num_frames, dtype=int)

    frame_dir, filenames = "multi_region_frames", []
    os.makedirs(frame_dir, exist_ok=True)
    grid_scale = X[1] - X[0]
    training_data = get_training_data(NOP_a, NOP_PDE, NOP_north, NOP_south, NOP_east, NOP_west)

    # --- patches
    data_A = training_data['A'].to_numpy()
    region_half_size = (10*grid_scale,10*grid_scale)
    trainingDataTimeList = np.array(list(set(training_data['A']['t_A'])), dtype=float)

    patches_list_a = compute_patches_for_points(data_A[:,:3], trainingDataTimeList,levelset_data, X, Y, grid_scale, region_half_size)
    fig, ax = plt.subplots(figsize=(8, 8))

    error_history = []

    testSet = set()
    for p in patches_list_a['results']:
        testSet.add(p['time_value'])
    timeList = sorted(list(testSet))
    indicesUsed = [i for (i, time) in enumerate(times) for timeSnapshot in timeList if np.isclose(time, timeSnapshot/4)]

    for i, time_idx in enumerate(tqdm(indicesUsed, desc="Processing Frames")):
        time_t = times[time_idx]
        levelset_t = levelset_data[time_idx, :, :]

        ax.clear()
        ax.imshow(levelset_t, extent=[X[0], X[-1], Y[0], Y[-1]],
                  origin='lower', cmap='coolwarm', vmin=0, vmax=1.0, alpha=0.5)

        patch_errors = []

        time_patches = [p for p in patches_list_a['results'] if np.isclose(p['time_value']/4, time_t)]

        for patch in tqdm(time_patches, desc="Processing Patches"):
            region_bounds = patch["region_bounds"]
            grad_vec = patch["grad_vec"]
            integrated_normal = patch["integrated_normal"]
            edges = patch["edges_info_for_plot"]
            intersection_span_bottom = patch["intersection_span_bottom"]
            intersection_span_left = patch["intersection_span_left"]
            intersection_span_right = patch["intersection_span_right"]
            intersection_span_top = patch["intersection_span_top"]
            y_bottom, y_top = region_bounds[2],region_bounds[3]
            x_left, x_right = region_bounds[0], region_bounds[1]
            x_min, x_max, y_min, y_max = X[0], X[-1], Y[0], Y[-1]
            # for p1, p2 in edges:
            #     ax.plot([p1[1] + x_min, p2[1] + x_min], [p1[0] + y_min, p2[0] + y_min], color='black', linewidth=1.5, zorder=3)

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
            grad_vec = np.array(patch["grad_vec"], dtype=float)
            integrated_normal = np.array(patch["integrated_normal"], dtype=float)
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
                     num_frames=29,
                     time_range=(0, 3.00008416),
                     num_patches=(8, 16))
