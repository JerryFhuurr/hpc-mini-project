#BSUB -N                  
#BSUB -u s240396@dtu.dk   
#BSUB -n 16
#BSUB -R "span[hosts=1]"    
#BSUB -R "rusage[mem=2048]"
#BSUB -W 120
#BSUB -J simulate_parallelization_static
#BSUB -o simulate_parallelization_static_output.log
#BSUB -e simulate_parallelization_static_error.log
#BSUB -q hpc

# 初始化conda环境
module load conda
conda activate 02613env 

# 运行Python程序
/usr/bin/time -v python simulate_parallelization_static.py
   