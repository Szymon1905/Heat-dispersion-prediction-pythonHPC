#!/bin/bash
#BSUB -J python
#BSUB -q hpc
#BSUB -n 16
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=1024MB]"
#BSUB -R "select[model == XeonGold6126]"
#BSUB -o python_%J.out
#BSUB -e python_%J.er

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613

for n in 1 2 4 8 16
do
    echo "Running with $n workers..."
    /usr/bin/time -f "$n %e" python simulate_parallelized.py 50 $n 2>> timings_static_scheduling.txt
done

python plot_speedup.py