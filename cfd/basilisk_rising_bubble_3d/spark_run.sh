#!/usr/bin/env bash
# Launches the 3D rising bubble on two DGX Sparks. Run it on Spark A, from the repo
# checkout, after setup_spark.sh has been run on BOTH Sparks:
#
#   SPARK_B=<hostname of Spark B> cfd/basilisk_rising_bubble_3d/spark_run.sh [cluster|parallel]
#
# Requirements: passwordless ssh from A to B, and the repo checked out at the SAME path
# on both. Jobs run in detached tmux sessions, so you can log out.
#
# Modes
#   cluster (default)   ONE LEVEL 8 run across both Sparks, 128 ranks (64 per Spark).
#                       Basilisk's multigrid only splits this domain into 2, 16 or 128
#                       ranks and cannot mix MPI with OpenMP, so this is the only way to put
#                       both Sparks on one run. It is ~3x oversubscribed on 2 x 20 cores.
#                       Before LEVEL 8 it runs the benchmark-mode validation (WALLS=1, no-slip
#                       on all faces) at LEVEL 6 with the SAME 128 ranks, hostfile and network
#                       settings, and checks it against Adelsberger et al. 2014 (check_run.py
#                       --strict). LEVEL 8 starts only if that passes. Log: run_cluster.log.
#                       Set MPI_IFACE to the ConnectX interface (e.g. enp1s0f0np0) if
#                       OpenMPI picks the wrong network.
#   parallel            LEVEL 8 on Spark A and LEVEL 7 on Spark B, 16 ranks each (no
#                       oversubscription). Use it if cluster mode turns out slower.
#
# The h5 lands in cfd_data/ on Spark A (and, for parallel mode, the LEVEL 7 one on B).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
MODE=${1:-cluster}
BAS=${BASILISK:-$HOME/basilisk/src}
PY=${PYTHON:-$HOME/.venvs/basilisk-cfd/bin/python}
ENV="BASILISK=$BAS PATH=\$PATH:$BAS PYTHON=$PY"
NP=128

# ---------------------------------------------------------------- cluster job (in tmux)
if [ "$MODE" = _cluster_job ]; then
    cd "$HERE"
    export BASILISK=$BAS PATH=$PATH:$BAS PYTHON=$PY
    ARGS=$(cat mpirun_args.spark)
    echo "== $(date) benchmark-mode validation, LEVEL 6, $NP ranks: $ARGS"
    rm -rf run_walls_L6
    # shellcheck disable=SC2086
    mpirun $ARGS -np $NP ./rising3d_walls_mpi 6 run_walls_L6
    if ! "$PY" check_run.py --strict run_walls_L6/log.txt; then
        echo "== validation FAILED: LEVEL 8 not started (run_walls_L6/ kept for inspection)"
        exit 1
    fi
    rm -f run_walls_L6/*.bin
    echo "== $(date) validation passed, starting LEVEL 8"
    MPIRUN_ARGS=$ARGS ./run.sh 8 $NP
    echo "== $(date) done"
    exit 0
fi

SPARK_B=${SPARK_B:?set SPARK_B to the hostname of the second Spark}
ssh -o BatchMode=yes "$SPARK_B" true || { echo "passwordless ssh to $SPARK_B failed"; exit 1; }
ssh "$SPARK_B" "test -d '$HERE'" || { echo "$HERE does not exist on $SPARK_B (same path needed)"; exit 1; }

case "$MODE" in
cluster)
    # identical binaries at the same path on both nodes
    for host in localhost "$SPARK_B"; do
        if [ "$host" = localhost ]; then
            (cd "$HERE" && env BASILISK="$BAS" PATH="$PATH:$BAS" make rising3d_mpi rising3d_walls_mpi)
        else
            ssh "$host" "cd '$HERE' && $ENV make rising3d_mpi rising3d_walls_mpi"
        fi
    done
    NC_A=$(nproc --all); NC_B=$(ssh "$SPARK_B" nproc --all)
    printf '%s slots=%s\n%s slots=%s\n' "$(hostname)" "$NC_A" "$SPARK_B" "$NC_B" > "$HERE/hosts.spark"
    # 64 ranks per node, ranks 0-63 on A (rank 0 writes the snapshots, so they land on A).
    # --oversubscribe tells OpenMPI the cores are shared, so idle ranks yield.
    ARGS="--hostfile hosts.spark --map-by ppr:$((NP / 2)):node --oversubscribe -x PATH -x LD_LIBRARY_PATH"
    [ -n "${MPI_IFACE:-}" ] && ARGS="$ARGS --mca btl_tcp_if_include $MPI_IFACE --mca oob_tcp_if_include $MPI_IFACE"
    echo "$ARGS" > "$HERE/mpirun_args.spark"
    tmux new-session -d -s basilisk_L8 "'$HERE/spark_run.sh' _cluster_job 2>&1 | tee '$HERE/run_cluster.log'"
    echo "started on $(hostname) + $SPARK_B, $NP ranks (tmux basilisk_L8):"
    echo "  1. LEVEL 6 benchmark-mode validation (a few minutes), 2. LEVEL 8 if it passes"
    echo "progress: tail -f $HERE/run_cluster.log   then   tail -f $HERE/run_L8/log.txt"
    ;;
parallel)
    ssh "$SPARK_B" "cd '$HERE' && tmux new-session -d -s basilisk_L7 \"$ENV ./run.sh 7 16\""
    (cd "$HERE" && tmux new-session -d -s basilisk_L8 "$ENV ./run.sh 8 16")
    echo "started: LEVEL 8 on $(hostname) (tmux basilisk_L8), LEVEL 7 on $SPARK_B (tmux basilisk_L7)"
    echo "progress: tail -f $HERE/run_L8/log.txt   |   ssh $SPARK_B tail -f $HERE/run_L7/log.txt"
    ;;
*)
    echo "unknown mode $MODE (cluster | parallel)"; exit 1 ;;
esac
