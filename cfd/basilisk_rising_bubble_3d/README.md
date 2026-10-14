# 3D rising bubble (Basilisk)

The 2D rising bubble of `src/paper2d.pdf` (section 3.2.3, Hysing case 1) run in 3D.
Its output is the h5 the 3D trainer reads, `src/generate_points_3d.py`.

| | 2D (paper) | 3D (this case) |
|---|---|---|
| domain | x in [-0.5, 0.5], y in [-1, 1] | x, y in [-0.5, 0.5], z in [-1, 1] (vertical) |
| bubble | r = 0.25 at (0, -0.5), at rest | sphere r = 0.25 at (0, 0, -0.5), at rest |
| liquid / bubble | rho 1000 / 100, mu 10 / 1 | same |
| sigma, g | 24.5, 0.98 along -y | 24.5, 0.98 along -z |
| boundaries | no-slip N/S, periodic E/W | no-slip z = +-1, periodic x and y |
| pressure | p* = 1 at north | full (hydrostatic) pressure, gauged to 1000 (p* = 1) on the top layer |
| grid | 256 x 512, dx = 1/256 | 256 x 256 x 512 at LEVEL 8 (uniform, dx = 1/256) |
| time | t in [0, 3], dt_out = 0.02 | t in [0, 3]; 29 training + 10 validation snapshots |

The snapshots stored are the 29 times the generator samples (0, 0.02, 0.04, 0.3, then
every 0.1 from 0.6 to 3.0). There are also 10 held-out validation times (0.16, 0.46, ...,
2.86, every 0.3). The h5 has a `split` dataset that labels each one.

Numerics: Basilisk's centered Navier-Stokes solver with two-phase VOF
(`two-phase.h`) and height-function surface tension (`tension.h`). It runs on a
uniform multigrid with MPI. Uniform rather than adaptive, because only the uniform
grid supports the non-cubic periodic 1 x 1 x 2 box. The timestep is adaptive and set
by the capillary limit, about 6.5e-4 at LEVEL 8.

GPUs: not supported for this case. Basilisk's GPU grids are single-GPU, its
surface-tension curvature does not run on them, and the GPU alternative (CLSVOF +
integral tension) failed in 3D. See the comment at the top of `rising3d.c`.

## Validation

Run in benchmark mode (`make rising3d_walls_mpi`: no-slip on all six faces, the setup
of Adelsberger et al. 2014, test case 1), the solver reproduces the published numbers
already at LEVEL 6 (dx = 1/64):

| | LEVEL 6, walls | benchmark | LEVEL 6, periodic (this case) |
|---|---|---|---|
| max rise velocity | 0.3585 at t = 0.89 | 0.352-0.358 at t ~ 0.9 | 0.392 at t = 0.99 |
| rise velocity, t = 3 | 0.348 | 0.33-0.35 | 0.383 |
| diameters, t = 3 (horiz. / vert.) | 0.56 / 0.375 | ~0.58 / 0.355-0.37 | 0.59 / 0.34 |
| sphericity, t = 3 | 0.958 | 0.955-0.96 | 0.938 |
| volume drift | -0.03% | | -0.04% |

The periodic case (the 2D paper's boundary conditions, used for the training data) rises
about 9% faster and flattens more, because no side walls hold it back. That difference
comes from the boundary conditions, not the resolution: the periodic rise velocity is
0.392 / 0.392 / 0.388 at LEVEL 5 / 6 / 7. Diameters are quantised to one cell (1/64
at LEVEL 6).

Check any run with `python check_run.py run_L8/log.txt ../../cfd_data/rising_bubble_3d_basilisk_L8.h5`.

## Install (Linux)

```bash
# compilers, MPI (Debian/Ubuntu shown; Fedora: sudo dnf install gcc make openmpi-devel)
sudo apt install build-essential openmpi-bin libopenmpi-dev curl
# on Fedora, MPI is not on PATH by default:
#   export PATH=$PATH:/usr/lib64/openmpi/bin LD_LIBRARY_PATH=/usr/lib64/openmpi/lib

# Basilisk
cd ~ && curl -sSfL http://basilisk.fr/basilisk/basilisk.tar.gz | tar xz
cd ~/basilisk/src
ln -sf config.gcc config
make -k            # errors in the gpu/cuda/hip/opencl backends are expected and harmless
ls -la qcc         # this is what matters

# add to ~/.bashrc
export BASILISK=$HOME/basilisk/src
export PATH=$PATH:$BASILISK
```

Python (for the h5 conversion): the repo venv needs `numpy h5py scipy scikit-image`.
`run.sh` uses `../../.venv/bin/python`; to use another interpreter, run
`PYTHON=/path/to/python ./run.sh 8 16`.

## Two DGX Sparks

Run `setup_spark.sh` once on **each** Spark, from the repo checkout (it uses sudo for apt).
It installs gcc, OpenMPI and tmux, Basilisk in `~/basilisk`, and a conversion venv in
`~/.venvs/basilisk-cfd`, then builds the case and smoke-tests it:

```bash
cfd/basilisk_rising_bubble_3d/setup_spark.sh
```

Then launch from Spark A. This needs passwordless ssh to B and the repo at the same path
on both:

```bash
SPARK_B=<spark-b-hostname> cfd/basilisk_rising_bubble_3d/spark_run.sh             # cluster (default)
SPARK_B=<spark-b-hostname> cfd/basilisk_rising_bubble_3d/spark_run.sh parallel
```

- `cluster` (default) runs a single LEVEL 8 job across both Sparks, 128 ranks, 64 per
  Spark. Basilisk cannot mix MPI with OpenMP, and this domain only splits into 2, 16 or
  128 ranks, so this is the only way to put both machines on one run. It is about 3x
  oversubscribed.
  - **Validation gate:** before LEVEL 8, it runs the benchmark-mode validation
    (`rising3d_walls_mpi`, LEVEL 6) on the same 128 ranks, hostfile and network settings,
    and checks it with `check_run.py --strict`. LEVEL 8 starts only if the peak rise
    velocity is within 0.345-0.365 at t = 0.8-1.0 and the volume drifts by less than 0.5%.
  - Follow it in `run_cluster.log`, then `run_L8/log.txt`.
  - If OpenMPI picks the wrong network, set `MPI_IFACE=<ConnectX interface>`.
- `parallel` runs LEVEL 8 on A and LEVEL 7 on B, 16 ranks each. Use it if `cluster`
  turns out slower: compare step rates in `run_L8/log.txt`.

Jobs run in detached tmux sessions (`basilisk_L8`, `basilisk_L7`). Each h5 lands in
`cfd_data/` on the Spark that ran the job.

GPUs are not used. The Spark's 128 GB of memory is what makes LEVEL 8 comfortable.

## Run at LEVEL 8 and extract the data (any single Linux machine)

```bash
cd twophasePINN/cfd/basilisk_rising_bubble_3d
tmux new -s basilisk           # takes hours: keep it alive if you disconnect
./run.sh 8 16                  # LEVEL 8, 16 MPI ranks
```

`run.sh` builds `rising3d_mpi`, runs it into `run_L8/`, and then converts the output to
`cfd_data/rising_bubble_3d_basilisk_L8.h5`. It deletes the raw `.bin` snapshots once
they are in the h5.

- **Ranks** must be **2, 16 or 128** (2 x 8^i). The box is two cubes stacked in z, and
  Basilisk's MPI multigrid splits each cube into 8^i blocks. Use 16 on a single machine.
- **Memory**: about 30 GB of RAM at LEVEL 8 (5 GB at LEVEL 7).
- **Disk**: about 26 GB of raw snapshots during the run, plus the compressed h5.
- **Progress**: `tail -f run_L8/log.txt`. The columns are step, t, dt, bubble volume,
  centroid z and rise velocity. The run ends at t = 3.
- If `run.sh` stops after the simulation, convert by hand:
  `cd ../../src && ../.venv/bin/python basilisk_to_h5.py ../cfd/basilisk_rising_bubble_3d/run_L8 --dst ../cfd_data/rising_bubble_3d_basilisk_L8.h5`

To train on it, point `H5_PATH` in `src/generate_points_3d.py` at the new file.
The generator selects snapshots by time, so the 39-snapshot file works as it is.

Quick test first: `make rising3d_serial && ./rising3d_serial 5 run_L5` takes about 30 s.
