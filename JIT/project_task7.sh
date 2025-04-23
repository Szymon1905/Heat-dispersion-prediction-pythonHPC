#!/bin/bash
#BSUB -J project
#BSUB -q hpc
#BSUB -W 24:00
#BSUB -R "rusage[mem=1GB]"
#BSUB -R "select[model == XeonGold6226R]"
#BSUB -n 32
#BSUB -R "span[hosts=1]"
#BSUB -o project_%J.out
#BSUB -e project_%J.err


source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613


# numba parameters for parallelization
export OMP_NUM_THREADS=$LSB_DJOB_NUMPROC
export NUMBA_NUM_THREADS=$OMP_NUM_THREADS
export NUMBA_THREADING_LAYER=omp
export MKL_NUM_THREADS=1

python jitversion.py 10 > task7resultsNumba.txt



python numpycompare.py 10 > task7resultsNumpy.txt



