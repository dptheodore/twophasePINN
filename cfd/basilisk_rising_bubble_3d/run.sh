#!/usr/bin/env bash
# Runs the 3D rising bubble and converts it to the h5 the 3D trainer reads.
#
#   ./run.sh [LEVEL] [NP]        default LEVEL 8 (dx = 1/256, the 2D grid), NP 16
#
# NP must be 2 * 8^i (2, 16, 128): the domain is two cubes stacked in z, and Basilisk's
# MPI multigrid splits each cube into 8^i equal blocks. LEVEL 8 needs roughly 30 GB of
# RAM; LEVEL 7 about 5 GB.
#
# Environment:
#   MPIRUN_ARGS  extra mpirun arguments, e.g. "--hostfile hosts" for a multi-node run
#                (default: --use-hwthread-cpus, letting NP exceed the physical cores)
#   PYTHON       interpreter for the h5 conversion (default: the repo's .venv)
set -euo pipefail
cd "$(dirname "$0")"

LEVEL=${1:-8}
NP=${2:-16}
RUN=run_L${LEVEL}
DST=$(cd ../.. && pwd)/cfd_data/rising_bubble_3d_basilisk_L${LEVEL}.h5

make rising3d_mpi
mkdir -p "$RUN"
# shellcheck disable=SC2086
mpirun ${MPIRUN_ARGS:---use-hwthread-cpus} -np "$NP" ./rising3d_mpi "$LEVEL" "$RUN" 2>&1 | tee "$RUN/basilisk.log"

PYTHON=${PYTHON:-$(cd ../.. && pwd)/.venv/bin/python}
mkdir -p ../../cfd_data
(cd ../../src && "$PYTHON" basilisk_to_h5.py "../cfd/basilisk_rising_bubble_3d/$RUN" --dst "$DST" --delete-raw)
echo "done: $DST"
