#!/bin/bash
#BSUB -J cuda_gpu_quick_test
#BSUB -q gpua100
#BSUB -W 0:10
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=1GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "select[gpu80gb]"
#BSUB -n 4
#BSUB -o batch_output/gpujob_%J.out
#BSUB -e batch_output/gpujob_%J.err


source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613

python cudaversion.py 10 > CUDAresults10.csv






