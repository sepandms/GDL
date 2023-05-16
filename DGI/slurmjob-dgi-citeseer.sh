#!/bin/bash -l
#SBATCH --job-name="dgi-citseer"
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --time=00:10:00

conda activate s3gc
srun python3 /home/stud68/s3gc/DGI/dgi_citeseer.py

