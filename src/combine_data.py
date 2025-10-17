import h5py
import json
import numpy as np

H5_PATH = "../cfd_data/rising_bubble.h5"
JSON_PATH = "grad_patches.json"

with open(JSON_PATH, 'r') as f:
    data = json.load(f)

results = data["results"]

n_regions = len(results)
x_center = np.zeros(n_regions)
y_center = np.zeros(n_regions)
time_value = np.zeros(n_regions)
grad_vec = np.zeros((n_regions, 2))
integrated_normal = np.zeros((n_regions, 2))

for i, r in enumerate(results):
    x_center[i], y_center[i], time_value[i] = r["center"]
    grad_vec[i, :] = np.array(r["grad_vec"], dtype=float)
    integrated_normal[i, :] = np.array(r["integrated_normal"], dtype=float)

# Write to HDF5
with h5py.File(H5_PATH, "a") as f:
    if "computation_regions" in f:
        del f["computation_regions"]  # overwrite cleanly
    grp = f.create_group("computation_regions")
    grp.create_dataset("x_center", data=x_center)
    grp.create_dataset("y_center", data=y_center)
    grp.create_dataset("time_value", data=time_value)
    grp.create_dataset("grad_vec", data=grad_vec)
    grp.create_dataset("integrated_normal", data=integrated_normal)

print(f"✅ Added {n_regions} computation regions to {H5_PATH}")