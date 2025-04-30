#!/bin/sh
#BSUB -q gpua100
#BSUB -J task8_job
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 02:00
#BSUB -o batch_output/task8_%J.out
#BSUB -e batch_output/task8_%J.err

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613

python t8.py 100