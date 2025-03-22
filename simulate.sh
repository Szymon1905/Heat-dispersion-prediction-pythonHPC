#!/bin/bash
#BSUB -J python
#BSUB -q hpc
#BSUB -n 3
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=2048MB]"
#BSUB -R "select[model == XeonGold6126]"
#BSUB -o python_%J.out
#BSUB -e python_%J.er

# InitializePythonenvironment
source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613

# Run Pythonscript
time python simulate.py 15