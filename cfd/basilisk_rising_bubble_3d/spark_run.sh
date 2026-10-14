#!/usr/bin/env bash
# Launches the 3D rising bubble on two DGX Sparks. Run it on Spark A, from the repo
# checkout, after setup_spark.sh has been run on BOTH Sparks:
#
#   SPARK_B=<hostname of Spark B> cfd/basilisk_rising_bubble_3d/spark_run.sh [parallel|cluster]
#
# Requirements: passwordless ssh from A to B, and the repo checked out at the SAME path
# on both. Jobs run in detached tmux sessions, so you can log out.
#
# Modes
#   parallel (default)  LEVEL 8 on Spark A (16 ranks) = the training data, and LEVEL 7 on
#                       Spark B (16 ranks) = the grid-convergence check. Each Spark's
#                       20 cores carry 16 ranks, no oversubscription.
#   cluster             ONE LEVEL 8 run across both Sparks, 128 ranks (64 per Spark on
#                       20 cores). Basilisk's multigrid only splits this domain into
#                       2, 16 or 128 ranks and cannot mix MPI with OpenMP, so this is the
#                       only way to put both Sparks on one run. It is ~3x oversubscribed:
#                       it may or may not beat parallel mode's LEVEL 8, so compare the
#                       step rate in run_L8/log.txt after ~10 minutes.
#                       Set MPI_IFACE to the ConnectX interface (e.g. enp1s0f0np0) if
#                       OpenMPI picks the wrong network.
#
# Results land in cfd_data/ on the Spark that ran each job:
#   rising_bubble_3d_basilisk_L8.h5 (A), rising_bubble_3d_basilisk_L7.h5 (B, parallel mode)
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
MODE=${1:-parallel}
SPARK_B=${SPARK_B:?set SPARK_B to the hostname of the second Spark}
BAS=${BASILISK:-$HOME/basilisk/src}
PY=${PYTHON:-$HOME/.venvs/basilisk-cfd/bin/python}
ENV="BASILISK=$BAS PATH=\$PATH:$BAS PYTHON=$PY"

ssh -o BatchMode=yes "$SPARK_B" true || { echo "passwordless ssh to $SPARK_B failed"; exit 1; }
ssh "$SPARK_B" "test -d '$HERE'" || { echo "$HERE does not exist on $SPARK_B (same path needed)"; exit 1; }

case "$MODE" in
parallel)
    ssh "$SPARK_B" "cd '$HERE' && tmux new-session -d -s basilisk_L7 \"$ENV ./run.sh 7 16\""
    (cd "$HERE" && tmux new-session -d -s basilisk_L8 "$ENV ./run.sh 8 16")
    echo "started: LEVEL 8 on $(hostname) (tmux basilisk_L8), LEVEL 7 on $SPARK_B (tmux basilisk_L7)"
    echo "progress: tail -f $HERE/run_L8/log.txt   |   ssh $SPARK_B tail -f $HERE/run_L7/log.txt"
    ;;
cluster)
    # same binary at the same path on both nodes
    ssh "$SPARK_B" "cd '$HERE' && $ENV make rising3d_mpi"
    (cd "$HERE" && env BASILISK="$BAS" PATH="$PATH:$BAS" make rising3d_mpi)
    NC_A=$(nproc --all); NC_B=$(ssh "$SPARK_B" nproc --all)
    printf '%s slots=%s\n%s slots=%s\n' "$(hostname)" "$NC_A" "$SPARK_B" "$NC_B" > "$HERE/hosts.spark"
    # 64 ranks per node, ranks 0-63 on A (rank 0 writes the snapshots, so they land on A).
    # --oversubscribe tells OpenMPI the cores are shared, so idle ranks yield.
    ARGS="--hostfile hosts.spark --map-by ppr:64:node --oversubscribe -x PATH -x LD_LIBRARY_PATH"
    [ -n "${MPI_IFACE:-}" ] && ARGS="$ARGS --mca btl_tcp_if_include $MPI_IFACE --mca oob_tcp_if_include $MPI_IFACE"
    (cd "$HERE" && tmux new-session -d -s basilisk_L8 "$ENV MPIRUN_ARGS='$ARGS' ./run.sh 8 128")
    echo "started: LEVEL 8 on $(hostname) + $SPARK_B, 128 ranks (tmux basilisk_L8)"
    echo "progress: tail -f $HERE/run_L8/log.txt"
    ;;
*)
    echo "unknown mode $MODE (parallel | cluster)"; exit 1 ;;
esac
