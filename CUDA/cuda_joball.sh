#!/bin/bash
#BSUB -J gpuALL
#BSUB -q gpua100
#BSUB -W 12:00
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "select[gpu80gb]"
#BSUB -n 4
#BSUB -o gpuALL_%J.out
#BSUB -e gpuALL_%J.err


source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613

python cudaversion.py 4571 > CUDAresultsALL.csv






