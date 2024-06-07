#!/bin/bash
#SBATCH -p gpu_requeue # partition (queue)
#SBATCH -N 1 # number of nodes
#SBATCH -n 2 # number of cores
#SBATCH --mem 96000 # memory pool for all cores
#SBATCH --gres=gpu:1
#SBATCH -t 0-20:00 # time (D-HH:MM)
#SBATCH -o ./slurm/%j.out # STDOUT
#SBATCH -e ./slurm/%j.err # STDERR

source activate spem
python train_FPI.py -n $1 -m $2 -e $3