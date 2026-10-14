#!/usr/bin/env bash
# One-time setup of a DGX Spark (or any Ubuntu/Debian box, arm64 or x86) for the 3D
# rising-bubble case. Run it on EACH Spark, from inside the repo checkout:
#
#   cfd/basilisk_rising_bubble_3d/setup_spark.sh
#
# Installs: compilers, OpenMPI, tmux (apt, needs sudo); Basilisk into ~/basilisk;
# a Python venv for the h5 conversion in ~/.venvs/basilisk-cfd. It then builds
# rising3d_mpi and runs a 30 s serial smoke test. Safe to rerun.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
BASILISK_DIR=${BASILISK_DIR:-$HOME/basilisk}
VENV=${VENV:-$HOME/.venvs/basilisk-cfd}

echo "== system packages"
sudo apt-get update
sudo apt-get install -y build-essential openmpi-bin libopenmpi-dev curl tmux \
     python3-venv python3-dev

echo "== Basilisk -> $BASILISK_DIR"
if [ ! -x "$BASILISK_DIR/src/qcc" ]; then
    mkdir -p "$(dirname "$BASILISK_DIR")"
    (cd "$(dirname "$BASILISK_DIR")" && curl -sSfL http://basilisk.fr/basilisk/basilisk.tar.gz | tar xz)
    [ "$(basename "$BASILISK_DIR")" = basilisk ] || mv "$(dirname "$BASILISK_DIR")/basilisk" "$BASILISK_DIR"
    cd "$BASILISK_DIR/src"
    ln -sf config.gcc config
    # The gpu/cuda/hip/opencl backends fail to build without their SDKs; they are not
    # used. Only qcc and the CPU grids matter.
    make -k > "$BASILISK_DIR/build.log" 2>&1 || true
fi
[ -x "$BASILISK_DIR/src/qcc" ] || { echo "qcc did not build, see $BASILISK_DIR/build.log"; exit 1; }
export BASILISK=$BASILISK_DIR/src
export PATH=$PATH:$BASILISK
grep -q "BASILISK=$BASILISK" ~/.bashrc 2>/dev/null || {
    echo "export BASILISK=$BASILISK" >> ~/.bashrc
    echo 'export PATH=$PATH:$BASILISK' >> ~/.bashrc
}

echo "== Python venv -> $VENV"
[ -x "$VENV/bin/python" ] || python3 -m venv "$VENV"
"$VENV/bin/pip" install -q --upgrade pip
"$VENV/bin/pip" install -q numpy h5py scipy scikit-image pandas

echo "== build + smoke test"
cd "$HERE"
make rising3d_mpi rising3d_serial
rm -rf run_smoke
./rising3d_serial 5 run_smoke > /dev/null 2>&1
tail -1 run_smoke/log.txt | awk '{ if ($2 + 0 < 2.999) { print "smoke test did not reach t = 3"; exit 1 }
    printf "smoke test OK: t = %s, bubble volume %s, rise velocity %s\n", $2, $4, $6 }'
rm -rf run_smoke
echo "== setup done on $(hostname), $(nproc) threads, $(free -g | awk '/Mem:/ {print $2}') GB RAM"
