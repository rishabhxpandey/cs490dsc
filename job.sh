#!/bin/bash
# FILENAME: job.sh
#SBATCH --output=/home/dixit22/cs490dsc/joboutput/myjob.out
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --job-name cifar-resnet

module load anaconda
module load use.own
source activate d22env
echo -e "module loaded"

cd /home/dixit22/cs490dsc
jupyter execute cifar10_resnet18.ipynb
