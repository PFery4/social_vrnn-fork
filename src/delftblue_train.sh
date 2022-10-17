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

# srun python3 train_WIP.py --model_name SocialVRNN_LSTM_ED --exp_num 1010 --n_mixtures 3 --others_info relative --total_training_steps 20000 --print_freq 40 --warm_start_convnet true --freeze_grid_cnn true --warm_start_query_agent_module false --freeze_query_agent_module false --scenario 'real_world/ewap_dataset/seq_eth' --pretrained_query_agent_module_path "../trained_models/LSTM_ED_module/21" --rotated_grid true > output_1010.log;
# srun python3 train_WIP.py --model_name SocialVRNN_LSTM_ED --exp_num 1011 --n_mixtures 3 --others_info relative --total_training_steps 20000 --print_freq 40 --warm_start_convnet true --freeze_grid_cnn true --warm_start_query_agent_module false --freeze_query_agent_module false --scenario 'real_world/ewap_dataset/seq_eth' --pretrained_query_agent_module_path "../trained_models/LSTM_ED_module/21" --normalize_data true > output_1011.log;
# srun python3 train_WIP.py --model_name SocialVRNN_LSTM_ED --exp_num 1012 --n_mixtures 3 --others_info relative --total_training_steps 20000 --print_freq 40 --warm_start_convnet true --freeze_grid_cnn true --warm_start_query_agent_module false --freeze_query_agent_module false --scenario 'real_world/ewap_dataset/seq_eth' --pretrained_query_agent_module_path "../trained_models/LSTM_ED_module/21" --diversity_update true > output_1012.log;

# special parameters (compare with exp_num 10101)
srun python3 train_WIP.py --exp_num 1010101 --model_name SocialVRNN_LSTM_ED --prev_horizon 8 --prediction_horizon 12 --truncated_backprop_length 8 --print_freq 100 --warm_start_convnet true --freeze_grid_cnn true --warm_start_query_agent_module false --freeze_query_agent_module false --scenario 'real_world/ewap_dataset/seq_eth' --pretrained_query_agent_module_path "../trained_models/LSTM_ED_module/71" --rotated_grid true > output_1010101.log;
srun python3 train_WIP.py --exp_num 2010101 --model_name SocialVRNN_LSTM_ED --prev_horizon 8 --prediction_horizon 12 --truncated_backprop_length 8 --print_freq 100 --warm_start_convnet true --freeze_grid_cnn true --warm_start_query_agent_module false --freeze_query_agent_module false --scenario 'real_world/ewap_dataset/seq_eth' --pretrained_query_agent_module_path "../trained_models/LSTM_ED_module/71" --normalize_date true > output_2010101.log;
srun python3 train_WIP.py --exp_num 3010101 --model_name SocialVRNN_LSTM_ED --prev_horizon 8 --prediction_horizon 12 --truncated_backprop_length 8 --print_freq 100 --warm_start_convnet true --freeze_grid_cnn true --warm_start_query_agent_module false --freeze_query_agent_module false --scenario 'real_world/ewap_dataset/seq_eth' --pretrained_query_agent_module_path "../trained_models/LSTM_ED_module/71" --diversity_update true > output_3010101.log;
srun python3 train_WIP.py --exp_num 4010101 --model_name SocialVRNN_LSTM_ED --prev_horizon 8 --prediction_horizon 12 --truncated_backprop_length 8 --print_freq 100 --warm_start_convnet true --freeze_grid_cnn true --warm_start_query_agent_module false --freeze_query_agent_module false --scenario 'real_world/ewap_dataset/seq_eth' --pretrained_query_agent_module_path "../trained_models/LSTM_ED_module/71" --correction_div_loss_in_total_loss true > output_4010101.log;
srun python3 train_WIP.py --exp_num 5010101 --model_name SocialVRNN_LSTM_ED --prev_horizon 8 --prediction_horizon 12 --truncated_backprop_length 8 --print_freq 100 --warm_start_convnet true --freeze_grid_cnn true --warm_start_query_agent_module false --freeze_query_agent_module false --scenario 'real_world/ewap_dataset/seq_eth' --pretrained_query_agent_module_path "../trained_models/LSTM_ED_module/71" --correction_annealing_kl_loss "article" > output_5010101.log;

# duplicate runs of 10101
srun python3 train_WIP.py --exp_num 110101 --model_name SocialVRNN_LSTM_ED --prev_horizon 8 --prediction_horizon 12 --truncated_backprop_length 8 --print_freq 100 --warm_start_convnet true --freeze_grid_cnn true --warm_start_query_agent_module false --freeze_query_agent_module false --scenario 'real_world/ewap_dataset/seq_eth' --pretrained_query_agent_module_path "../trained_models/LSTM_ED_module/71" > output_110101.log;
srun python3 train_WIP.py --exp_num 210101 --model_name SocialVRNN_LSTM_ED --prev_horizon 8 --prediction_horizon 12 --truncated_backprop_length 8 --print_freq 100 --warm_start_convnet true --freeze_grid_cnn true --warm_start_query_agent_module false --freeze_query_agent_module false --scenario 'real_world/ewap_dataset/seq_eth' --pretrained_query_agent_module_path "../trained_models/LSTM_ED_module/71" > output_210101.log;
srun python3 train_WIP.py --exp_num 310101 --model_name SocialVRNN_LSTM_ED --prev_horizon 8 --prediction_horizon 12 --truncated_backprop_length 8 --print_freq 100 --warm_start_convnet true --freeze_grid_cnn true --warm_start_query_agent_module false --freeze_query_agent_module false --scenario 'real_world/ewap_dataset/seq_eth' --pretrained_query_agent_module_path "../trained_models/LSTM_ED_module/71" > output_310101.log;


