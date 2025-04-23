#!/bin/bash
#BSUB -q hpc
#BSUB -W 00:05
#BSUB -J t10_job
#BSUB -o t10.txt
#BSUB -e t10_error.txt
#BSUB -R "select[model==XeonGold6126]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -n 16
#BSUB -R "span[hosts=1]"

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613

# Number of floorplans to process
N=10

# Run the profiler
nsys profile -o cupy_profile python t9.py $N