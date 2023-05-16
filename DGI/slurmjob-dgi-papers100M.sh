#!/bin/bash -l
#SBATCH --job-name="dgi-papers100M"
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --time=00:10:00

conda activate s3gc
srun python3 /home/stud68/s3gc/DGI/dgi_papers100M.py
