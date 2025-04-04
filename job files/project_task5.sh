#!/bin/bash
#BSUB -J project
#BSUB -q hpc
#BSUB -W 24:00
#BSUB -R "rusage[mem=4GB]"
#BSUB -R "select[model == XeonGold6226R]"
#BSUB -n 16
#BSUB -R "span[hosts=1]"
#BSUB -o project_%J.out
#BSUB -e project_%J.err

#InitializePythonenvironment
source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613

# Run Python script
#python -m cProfile -o script.prof program.py input.csv
#kernprof -l simulate.py 10
#python -m cProfile -s cumulative simulate.py 10
TASK="task5.txt"
for n in 1 2 4 8 16
do
    echo "Now using $n threads"
    /usr/bin/time -f "$n %e" python simulate_parallel_static.py 50 $n 2>> $TASK
done

python plot_speedup.py $TASK