#!/bin/bash
#BSUB -q hpc
#BSUB -u s240396@dtu.dk  
#BSUB -W 00:05
#BSUB -J t7_job
#BSUB -o t7.txt
#BSUB -e t7_error.txt
#BSUB -R "select[model==XeonGold6126]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -n 16
#BSUB -R "span[hosts=1]"

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613

python t7_simulate_jit.py