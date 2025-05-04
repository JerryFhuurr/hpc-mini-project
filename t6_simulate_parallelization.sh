#BSUB -N                  
#BSUB -u s242608@dtu.dk   
#BSUB -n 1
#BSUB -R "span[hosts=1]"    
#BSUB -R "rusage[mem=1024]"
#BSUB -W 15
#BSUB -J week5_2_1
#BSUB -o simulate_parallelization_output.log
#BSUB -e simulate_parallelization_error.log
#BSUB -q hpc

# 初始化conda环境
module load conda
conda activate 02613env 

# 运行Python程序
/usr/bin/time -v python simulate_parallelization.py
   