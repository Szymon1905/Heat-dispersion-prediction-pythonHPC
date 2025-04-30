#!/bin/bash
#BSUB -J jitpool256
#BSUB -q hpc
#BSUB -W 2:00
#BSUB -R "rusage[mem=2GB]"
#BSUB -R "select[model == XeonGold6226R]"
#BSUB -n 16
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

python jitversion_parallel.py 128 > JIT_pool_results128.csv

python jitversion_parallel.py 256 > JIT_pool_results256.csv



