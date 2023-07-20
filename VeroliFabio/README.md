# Progetto HPC 2022/2023
## Istruzioni per la compilazione e l'esecuzione
Dentro la cartella `src` si trovano i codici sorgenti delle versioni parallele OpenMP (`omp-sph`) e MPI (`mpi-sph`), il file `hpc.h` (utilizzato per la misurazione del tempo) e il `Makefile`.

Per compilare i programmi, all'interno della cartella `src`, eseguire uno dei seguenti comandi:
- `make`: per compilare entrambe le versioni.
- `make omp`: per compilare solo la versione OpenMP.
- `make mpi`: per compilare solo la versione MPI.

Una volta compilati i programmi possono essere eseguiti utilizzando i comandi: 
- `OMP_NUM_THREADS=NT ./omp-sph [N [S]]`: versione OpenMP.
- `mpirun -n NT ./mpi-sph [N [S]]`: versione MPI.

dove NT è il numero di thread/processi, N è il numero di particelle e S è il numero di passi da simulare (S, N > 0).

Utilizzare il comando `make clean` per eliminare gli eseguibili di entrambe le versioni proposte. 

[comment]: # (Fabio Veroli 0000970669 fabio.veroli@studio.unibo.it)
