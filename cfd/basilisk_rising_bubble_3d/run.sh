#!/usr/bin/env bash
# Runs the 3D rising bubble and converts it to the h5 the 3D trainer reads.
#
#   ./run.sh [LEVEL] [NP]        default LEVEL 8 (dx = 1/256, the 2D grid), NP 16
#
# NP must be 2 * k^3 (2, 16, 54, 128, ...): the domain is two cubes stacked in z and
# Basilisk's MPI multigrid splits each cube into k^3 equal blocks. LEVEL 8 needs roughly
# 30 GB of RAM; LEVEL 7 about 5 GB.
set -euo pipefail
cd "$(dirname "$0")"

LEVEL=${1:-8}
NP=${2:-16}
RUN=run_L${LEVEL}
DST=../../cfd_data/rising_bubble_3d_basilisk_L${LEVEL}.h5

make rising3d_mpi
mkdir -p "$RUN"
# --use-hwthread-cpus lets NP exceed the physical core count (e.g. 16 ranks on 8 cores).
mpirun --use-hwthread-cpus -np "$NP" ./rising3d_mpi "$LEVEL" "$RUN" 2>&1 | tee "$RUN/basilisk.log"

PYTHON=${PYTHON:-../../.venv/bin/python}
(cd ../../src && "$PYTHON" basilisk_to_h5.py "../cfd/basilisk_rising_bubble_3d/$RUN" --dst "$DST" --delete-raw)
echo "done: $DST"
