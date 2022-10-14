#!/bin/sh

#SBATCH --job-name="social_vrnn_flags_tests"
#SBATCH --partition=compute
#SBATCH --account=Education-3mE-MSc-RO
#SBATCH --time=05:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8GB

# This script's purpose is to enable training of the Social-VRNN model using TU Delft's HPC; Delftblue
# If you have access to the DHPC, you can refer to its documentation here: https://doc.dhpc.tudelft.nl/delftblue/

module load miniconda3
module load 2022r2

unset CONDA_SHLVL
source "$(conda info --base)/etc/profile.d/conda.sh"

conda activate social_vrnn

srun python3 train_WIP.py --model_name SocialVRNN_LSTM_ED --exp_num 1010 --n_mixtures 3 --others_info relative --total_training_steps 20000 --print_freq 40 --warm_start_convnet true --freeze_grid_cnn true --warm_start_query_agent_module false --freeze_query_agent_module false --scenario 'real_world/ewap_dataset/seq_eth' --pretrained_query_agent_module_path "../trained_models/LSTM_ED_module/21" --rotated_grid true > output_1010.log;
srun python3 train_WIP.py --model_name SocialVRNN_LSTM_ED --exp_num 1011 --n_mixtures 3 --others_info relative --total_training_steps 20000 --print_freq 40 --warm_start_convnet true --freeze_grid_cnn true --warm_start_query_agent_module false --freeze_query_agent_module false --scenario 'real_world/ewap_dataset/seq_eth' --pretrained_query_agent_module_path "../trained_models/LSTM_ED_module/21" --normalize_data true > output_1011.log;
srun python3 train_WIP.py --model_name SocialVRNN_LSTM_ED --exp_num 1012 --n_mixtures 3 --others_info relative --total_training_steps 20000 --print_freq 40 --warm_start_convnet true --freeze_grid_cnn true --warm_start_query_agent_module false --freeze_query_agent_module false --scenario 'real_world/ewap_dataset/seq_eth' --pretrained_query_agent_module_path "../trained_models/LSTM_ED_module/21" --diversity_update true > output_1012.log;


