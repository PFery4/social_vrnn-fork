#!/bin/sh

#SBATCH --job-name="social_vrnn_first_run"
#SBATCH --partition=gpu
#SBATCH --account=Education-3mE-MSc-RO
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=2
#SBATCH --mem-per-cpu=6GB

# This script's purpose is to enable training of the Social-VRNN model using TU Delft's HPC; Delftblue
# If you have access to the DHPC, you can refer to its documentation here: https://doc.dhpc.tudelft.nl/delftblue/

module load 2022r2
module load cuda/11.6

unset CONDA_SHLVL
source "$(conda info --base)/etc/profile.d/conda.sh"

conda activate social_vrnn

srun ./train.sh
