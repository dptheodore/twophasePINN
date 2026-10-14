# Builds the 3D rising-bubble case. Needs $BASILISK (path to basilisk/src, with qcc
# built) and, for the MPI build, mpicc on PATH. See README.md.

BASILISK ?= $(HOME)/basilisk/src
QCC = $(BASILISK)/qcc
CFLAGS = -O2 -Wall

all: rising3d_mpi

# MPI build (the production one): mpirun -np 16 ./rising3d_mpi 8 run_L8
rising3d_mpi: rising3d.c
	CC99='mpicc -std=c99' $(QCC) $(CFLAGS) -D_MPI=1 rising3d.c -o $@ -lm

# Serial build, for quick tests at low levels: ./rising3d_serial 5 run_L5
rising3d_serial: rising3d.c
	$(QCC) $(CFLAGS) rising3d.c -o $@ -lm

clean:
	rm -f rising3d_mpi rising3d_serial _rising3d.c

.PHONY: all clean
