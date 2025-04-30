#!/bin/bash
#BSUB -q hpc
#BSUB -W 00:05
#BSUB -J time_ref_job
#BSUB -o time.txt
#BSUB -e time_error.txt
#BSUB -R "select[model==XeonGold6126]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -n 16
#BSUB -R "span[hosts=1]"

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613

python t2_time_reference.py