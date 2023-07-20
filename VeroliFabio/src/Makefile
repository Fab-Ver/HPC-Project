# Makefile for the High Performance Computing programming project,
# Academic Year 2022/2023.
# Fabio Veroli 0000970669 fabio.veroli@studio.unibo.it
#
# Available targets:
#
# - omp: builds the OpenMP parallel version
#
# - mpi: biulds the MPI parallel version
#
# - all: builds both the OpenMP and MPI versions
#
# - clean: clean up
#

CFLAGS=-std=c99 -Wall -Wpedantic
LDLIBS=-lm

all: omp mpi

omp: omp-sph.c
	gcc ${CFLAGS} -o omp-sph omp-sph.c -fopenmp ${LDLIBS}

mpi: mpi-sph.c
	mpicc ${CFLAGS} -o mpi-sph mpi-sph.c ${LDLIBS}

.PHONY: clean

clean:
	-rm -f mpi-sph omp-sph
