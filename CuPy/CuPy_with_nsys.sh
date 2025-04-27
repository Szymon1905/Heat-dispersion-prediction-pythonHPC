
#!/bin/bash
#BSUB -J CuPy_test
#BSUB -q c02613
#BSUB -W 0:30
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=1GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -o gpujob_%J.out
#BSUB -e gpujob_%J.err

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613

nsys profile -o profile_of_CuPy_reference python CuPy_changed.py 10

