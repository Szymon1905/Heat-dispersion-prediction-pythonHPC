#!/bin/bash
#BSUB -J projecttest1
#BSUB -q hpc
#BSUB -W 10
#BSUB -R "rusage[mem=2GB]"
#BSUB -o oooo_%J.out
#BSUB -e eeee_%J.err                  
#BSUB -n 4                  # 4 cores  
#BSUB -R "span[hosts=1]"      # 1 host 
#BSUB -R "select[model == XeonGold6226R]"

source /dtu/projects/02613_2025/conda/conda_init.sh

conda activate 02613

time python simulate_original.py 4
