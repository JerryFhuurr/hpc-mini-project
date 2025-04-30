#BSUB -N                  
#BSUB -u s240396@dtu.dk   
#BSUB -n 1
#BSUB -R "span[hosts=1]"    
#BSUB -R "rusage[mem=1024]"
#BSUB -W 15
#BSUB -J task6
#BSUB -o simulate_parallelization_output.log
#BSUB -e simulate_parallelization_error.log
#BSUB -q hpc

# 初始化conda环境
source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613

# 运行Python程序
/usr/bin/time -v python t6_simulate_parallelization.py
   