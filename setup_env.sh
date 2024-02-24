#!/bin/bash
# FILENAME: job.sh
#SBATCH --output=myjob.out
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --job-name cifar-resnet

module load anaconda
module load use.own
conda env remove --name d22env
conda create --name d22env python=3.11 jupyter pytorch torchvision matplotlib pandas -y
source activate d22env
conda info --envs
echo -e "module loaded"



