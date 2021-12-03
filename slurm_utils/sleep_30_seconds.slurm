#!/bin/bash
#SBATCH -J sleep_30_seconds_
#SBATCH -o "slurm_logs/sleep_30_seconds_%j"
#SBATCH --mail-user=mbaradad@mit.edu
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:2
#SBATCH --gpus-per-node=2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=65
#SBATCH --mem=64GB
#SBATCH --time=23:59:59

## User python environment
HOME2=/nobackup/users/$(whoami)
CONDA_VIRTUAL_ENVIRONMENT=default_env37
CONDA_ROOT=$HOME2/anaconda3

## Activate virtual environment
source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate $CONDA_VIRTUAL_ENVIRONMENT
# Unlimit number of files
ulimit -s unlimited

echo " Running on single nodes with 2 GPU"
echo ""
echo " Run started at:- "
date

cd /home/mbaradad/projects/learning_with_noise_private
sleep 30s

echo "Run completed at:- "
date