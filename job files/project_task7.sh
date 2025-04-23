#!/bin/bash
#BSUB -J project
#BSUB -q hpc
#BSUB -W 24:00
#BSUB -R "rusage[mem=4GB]"
#BSUB -R "select[model == XeonGold6226R]"
#BSUB -n 32
#BSUB -R "span[hosts=1]"
#BSUB -o project_%J.out
#BSUB -e project_%J.err


source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613



python jitversion.py 50 >> task7resultsNumba.txt

python numpycompare.py 50 >> task7resultsNumpy.txt



