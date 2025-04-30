#!/bin/bash
#BSUB -q gpua100
#BSUB -W 10:00
#BSUB -J t10_job
#BSUB -o t10.txt
#BSUB -e t10_error.txt
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "rusage[mem=8GB]"
#BSUB -n 16
#BSUB -R "span[hosts=1]"

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613

# Number of floorplans to process
N=10

# Run with profiling
python -m cProfile -o t10_profile.prof t10.py $N

# To visualize the profile later:
# python -c "import pstats; p = pstats.Stats('t10_profile.prof'); p.sort_stats('cumtime').print_stats(20)"