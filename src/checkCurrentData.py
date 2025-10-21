import h5py
import json
import numpy as np


H5_PATH = "../cfd_data/rising_bubble.h5"

with h5py.File(H5_PATH, "a") as f:
    grad_vec = np.array(f['computation_regions']['grad_vec'])
    integrated_normal = np.array(f['computation_regions']['integrated_normal'])

for (g_v, i_n) in zip(grad_vec, integrated_normal):
    val = np.linalg.norm(g_v - i_n)
    if val > 1e-4:
        print(np.linalg.norm(g_v- i_n))