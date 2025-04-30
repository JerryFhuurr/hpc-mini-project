#BSUB -N                  
#BSUB -u s240396@dtu.dk   
#BSUB -n 16
#BSUB -R "span[hosts=1]"    
#BSUB -R "rusage[mem=2048]"
#BSUB -W 120
#BSUB -J task1
#BSUB -o task1.log
#BSUB -e task1_error.log
#BSUB -q hpc

# 初始化conda环境
source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613 

# 运行Python程序
/usr/bin/time -v python t1_visualize_input.py
   