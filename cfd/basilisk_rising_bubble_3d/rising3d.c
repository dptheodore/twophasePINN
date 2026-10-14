/**
# 3D rising bubble: the 2D case of src/paper2d.pdf (section 3.2.3), in 3D

Hysing case 1 parameters, with the paper's boundary conditions carried over:

  domain     x, y in [-0.5, 0.5] (periodic), z in [-1, 1] (vertical)
  walls      no-slip on z = -1 and z = 1 (the 2D north/south)
  bubble     sphere r = 0.25 at (0, 0, -0.5), fluid at rest
  liquid     rho1 = 1000, mu1 = 10   (f = 1)
  bubble     rho2 = 100,  mu2 = 1    (f = 0)
  sigma      24.5
  g          0.98 along -z, applied as a body force, so p is the FULL pressure
             (hydrostatic included), as in the 2D data

Uniform grid: N = 2^LEVEL cells across x and y, 2N along z. LEVEL 8 gives
dx = 1/256, the 2D data's spacing, and the grid IS the output grid.

Snapshots are written at the times listed in SNAP_TIMES as raw float32 arrays
in (z, y, x) order: f, u, v, w, p. src/basilisk_to_h5.py turns them into the h5
the 3D trainer reads.

    qcc -O2 -Wall -D_MPI=1 rising3d.c -o rising3d -lm      # see Makefile
    mpirun -np 16 ./rising3d 8 out_dir
*/

/* CPU only. Basilisk's GPU grids (tarball of 2026-10-14) cannot run this case: they are
   single-GPU, tension.h's curvature is not GPU-portable, and the CLSVOF + integral.h
   alternative failed in 3D (reduction assert in redistance on the 1x1x2 domain; constant
   field error in integral.h and half the initial volume on a cube). */
#include "grid/multigrid3D.h"
#include "navier-stokes/centered.h"
#include "two-phase.h"
#include "tension.h"

int LEVEL = 8;
char OUT[256] = "out";

/* 29 training times (the snapshots generate_points.py samples) + 10 held-out
   validation times, every 0.3 from 0.16. All are multiples of 0.02. */
static const double SNAP_TIMES[] = {
  0.00, 0.02, 0.04, 0.16, 0.30, 0.46, 0.60, 0.70, 0.76, 0.80, 0.90, 1.00,
  1.06, 1.10, 1.20, 1.30, 1.36, 1.40, 1.50, 1.60, 1.66, 1.70, 1.80, 1.90,
  1.96, 2.00, 2.10, 2.20, 2.26, 2.30, 2.40, 2.50, 2.56, 2.60, 2.70, 2.80,
  2.86, 2.90, 3.00
};
#define NSNAP (sizeof (SNAP_TIMES)/sizeof (SNAP_TIMES[0]))

/* No-slip on the z walls (Basilisk: back = z min, front = z max). x and y are
   made periodic in main(). */
u.n[front] = dirichlet(0);
u.t[front] = dirichlet(0);
u.r[front] = dirichlet(0);
u.n[back]  = dirichlet(0);
u.t[back]  = dirichlet(0);
u.r[back]  = dirichlet(0);

int main (int argc, char * argv[])
{
  if (argc > 1) LEVEL = atoi (argv[1]);
  if (argc > 2) strcpy (OUT, argv[2]);

  dimensions (nx = 1, ny = 1, nz = 2);
  size (1.);
  origin (-0.5, -0.5, -1.);
  periodic (right);
  periodic (top);

  rho1 = 1000., mu1 = 10.;
  rho2 = 100.,  mu2 = 1.;
  f.sigma = 24.5;

  TOLERANCE = 1e-4;
  DT = 0.02;
  init_grid (1 << LEVEL);

  char cmd[300];
  sprintf (cmd, "mkdir -p %s", OUT);
  if (pid() == 0) system (cmd);
  run();
}

event init (t = 0)
{
  fraction (f, sq(x) + sq(y) + sq(z + 0.5) - sq(0.25));
}

event acceleration (i++)
{
  face vector av = a;
  foreach_face(z)
    av.z[] -= 0.98;
}

/* Bubble volume, centroid and rise velocity, every step. */
event logfile (i++)
{
  double vb = 0., zb = 0., wb = 0.;
  foreach (reduction(+:vb) reduction(+:zb) reduction(+:wb)) {
    double dv = (1. - f[])*dv();
    vb += dv;
    zb += z*dv;
    wb += u.z[]*dv;
  }
  if (pid() == 0) {
    static FILE * fp = NULL;
    if (!fp) {
      char name[300];
      sprintf (name, "%s/log.txt", OUT);
      fp = fopen (name, "w");
      fprintf (fp, "# i t dt volume z_centroid w_rise\n");
    }
    fprintf (fp, "%d %.8f %.6e %.10e %.10e %.10e\n", i, t, dt, vb, zb/vb, wb/vb);
    fflush (fp);
  }
}

/* Write every field on the full grid as float32 (z, y, x), assembled on rank 0. */
static void write_snapshot (int n)
{
  int N = 1 << LEVEL;
  long nx = N, ny = N, nz = 2*N, ncell = nx*ny*nz;
  float * buf = calloc (ncell, sizeof (float));
  scalar * fields = {f, u.x, u.y, u.z, p};
  const char * names[5] = {"f", "u", "v", "w", "p"};
  int k = 0;
  for (scalar s in fields) {
    memset (buf, 0, ncell*sizeof (float));
    foreach (serial) {
      long ix = (long) floor ((x - X0)/Delta);
      long iy = (long) floor ((y - Y0)/Delta);
      long iz = (long) floor ((z - Z0)/Delta);
      buf[(iz*ny + iy)*nx + ix] = s[];
    }
@if _MPI
    if (pid() == 0)
      MPI_Reduce (MPI_IN_PLACE, buf, ncell, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
    else
      MPI_Reduce (buf, NULL, ncell, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
@endif
    if (pid() == 0) {
      char name[300];
      sprintf (name, "%s/snap_%03d_%s.bin", OUT, n, names[k]);
      FILE * fp = fopen (name, "wb");
      fwrite (buf, sizeof (float), ncell, fp);
      fclose (fp);
    }
    k++;
  }
  free (buf);
  if (pid() == 0) {
    char name[300];
    sprintf (name, "%s/snapshots.txt", OUT);
    FILE * fp = fopen (name, n == 0 ? "w" : "a");
    fprintf (fp, "%d %.10f %d %d %d\n", n, t, (int) nx, (int) ny, (int) nz);
    fclose (fp);
  }
}

/* Every 0.02 (Basilisk lands on these times exactly); only the listed ones are written. */
event snapshots (t = 0; t += 0.02; t <= 3.0 + 1e-9)
{
  static int n = 0;
  for (int k = 0; k < NSNAP; k++)
    if (fabs (t - SNAP_TIMES[k]) < 1e-6) {
      write_snapshot (n++);
      break;
    }
}

event end (t = 3.0)
{
}
