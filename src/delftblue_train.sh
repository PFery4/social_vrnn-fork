#!/bin/sh

#SBATCH --job-name="proper_defaults_reruns_pt3"
#SBATCH --partition=compute
#SBATCH --account=Education-3mE-MSc-RO
#SBATCH --time=18:00:00
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

# srun python3 train_WIP.py --model_name SocialVRNN_LSTM_ED --exp_num 1010 --n_mixtures 3 --others_info relative --total_training_steps 20000 --print_freq 40 --warm_start_convnet true --freeze_grid_cnn true --warm_start_query_agent_module false --freeze_query_agent_module false --scenario 'real_world/ewap_dataset/seq_eth' --pretrained_query_agent_module_path "../trained_models/LSTM_ED_module/21" --rotated_grid true --lstmed_consistent_time_signal false > output_1010.log;
# srun python3 train_WIP.py --model_name SocialVRNN_LSTM_ED --exp_num 1011 --n_mixtures 3 --others_info relative --total_training_steps 20000 --print_freq 40 --warm_start_convnet true --freeze_grid_cnn true --warm_start_query_agent_module false --freeze_query_agent_module false --scenario 'real_world/ewap_dataset/seq_eth' --pretrained_query_agent_module_path "../trained_models/LSTM_ED_module/21" --normalize_data true > output_1011.log;
# srun python3 train_WIP.py --model_name SocialVRNN_LSTM_ED --exp_num 1012 --n_mixtures 3 --others_info relative --total_training_steps 20000 --print_freq 40 --warm_start_convnet true --freeze_grid_cnn true --warm_start_query_agent_module false --freeze_query_agent_module false --scenario 'real_world/ewap_dataset/seq_eth' --pretrained_query_agent_module_path "../trained_models/LSTM_ED_module/21" --diversity_update true > output_1012.log;

# special parameters (compare with exp_num 10101)
# srun python3 train_WIP.py --exp_num 1010101 --model_name SocialVRNN_LSTM_ED --prev_horizon 8 --prediction_horizon 12 --truncated_backprop_length 8 --print_freq 100 --warm_start_convnet true --freeze_grid_cnn true --warm_start_query_agent_module false --freeze_query_agent_module false --scenario 'real_world/ewap_dataset/seq_eth' --pretrained_query_agent_module_path "../trained_models/LSTM_ED_module/71" --rotated_grid true > output_1010101.log;
# srun python3 train_WIP.py --exp_num 2010101 --model_name SocialVRNN_LSTM_ED --prev_horizon 8 --prediction_horizon 12 --truncated_backprop_length 8 --print_freq 100 --warm_start_convnet true --freeze_grid_cnn true --warm_start_query_agent_module false --freeze_query_agent_module false --scenario 'real_world/ewap_dataset/seq_eth' --pretrained_query_agent_module_path "../trained_models/LSTM_ED_module/71" --normalize_data true > output_2010101.log;
# srun python3 train_WIP.py --exp_num 3010101 --model_name SocialVRNN_LSTM_ED --prev_horizon 8 --prediction_horizon 12 --truncated_backprop_length 8 --print_freq 100 --warm_start_convnet true --freeze_grid_cnn true --warm_start_query_agent_module false --freeze_query_agent_module false --scenario 'real_world/ewap_dataset/seq_eth' --pretrained_query_agent_module_path "../trained_models/LSTM_ED_module/71" --diversity_update true > output_3010101.log;
# srun python3 train_WIP.py --exp_num 4010101 --model_name SocialVRNN_LSTM_ED --prev_horizon 8 --prediction_horizon 12 --truncated_backprop_length 8 --print_freq 100 --warm_start_convnet true --freeze_grid_cnn true --warm_start_query_agent_module false --freeze_query_agent_module false --scenario 'real_world/ewap_dataset/seq_eth' --pretrained_query_agent_module_path "../trained_models/LSTM_ED_module/71" --correction_div_loss_in_total_loss true > output_4010101.log;
##NEEDS A RERUN# srun python3 train_WIP.py --exp_num 5010101 --model_name SocialVRNN_LSTM_ED --prev_horizon 8 --prediction_horizon 12 --truncated_backprop_length 8 --print_freq 100 --warm_start_convnet true --freeze_grid_cnn true --warm_start_query_agent_module false --freeze_query_agent_module false --scenario 'real_world/ewap_dataset/seq_eth' --pretrained_query_agent_module_path "../trained_models/LSTM_ED_module/71" --correction_annealing_kl_loss "article" > output_5010101.log;

# duplicate runs of 10101
# srun python3 train_WIP.py --exp_num 110101 --model_name SocialVRNN_LSTM_ED --prev_horizon 8 --prediction_horizon 12 --truncated_backprop_length 8 --print_freq 100 --warm_start_convnet true --freeze_grid_cnn true --warm_start_query_agent_module false --freeze_query_agent_module false --scenario 'real_world/ewap_dataset/seq_eth' --pretrained_query_agent_module_path "../trained_models/LSTM_ED_module/71" > output_110101.log;
# srun python3 train_WIP.py --exp_num 210101 --model_name SocialVRNN_LSTM_ED --prev_horizon 8 --prediction_horizon 12 --truncated_backprop_length 8 --print_freq 100 --warm_start_convnet true --freeze_grid_cnn true --warm_start_query_agent_module false --freeze_query_agent_module false --scenario 'real_world/ewap_dataset/seq_eth' --pretrained_query_agent_module_path "../trained_models/LSTM_ED_module/71" > output_210101.log;
# srun python3 train_WIP.py --exp_num 310101 --model_name SocialVRNN_LSTM_ED --prev_horizon 8 --prediction_horizon 12 --truncated_backprop_length 8 --print_freq 100 --warm_start_convnet true --freeze_grid_cnn true --warm_start_query_agent_module false --freeze_query_agent_module false --scenario 'real_world/ewap_dataset/seq_eth' --pretrained_query_agent_module_path "../trained_models/LSTM_ED_module/71" > output_310101.log;

# srun python3 train_WIP.py --exp_num 10301 --model_name SocialVRNN_LSTM_ED --prev_horizon 8 --prediction_horizon 12 --truncated_backprop_length 8 --print_freq 100 --warm_start_convnet false --freeze_grid_cnn false --warm_start_query_agent_module false --freeze_query_agent_module false --scenario 'real_world/ewap_dataset/seq_eth' --pretrained_query_agent_module_path "../trained_models/LSTM_ED_module/71" > slurm_jobs/output-logs/output_10301.log;
# srun python3 train_WIP.py --exp_num 10401 --model_name SocialVRNN_LSTM_ED --prev_horizon 8 --prediction_horizon 12 --truncated_backprop_length 8 --print_freq 100 --warm_start_convnet true --freeze_grid_cnn true --warm_start_query_agent_module true --freeze_query_agent_module true --scenario 'real_world/ewap_dataset/seq_eth' --pretrained_query_agent_module_path "../trained_models/LSTM_ED_module/71" > slurm_jobs/output-logs/output_10401.log;


# srun python3 train_WIP.py --exp_num 10102 --model_name SocialVRNN_LSTM_ED --prev_horizon 8 --prediction_horizon 12 --truncated_backprop_length 8 --print_freq 100 --warm_start_convnet true --freeze_grid_cnn true --warm_start_query_agent_module false --freeze_query_agent_module false --scenario 'real_world/st' --pretrained_query_agent_module_path "../trained_models/LSTM_ED_module/72" > slurm_jobs/output-logs/output_10102.log;
# srun python3 train_WIP.py --exp_num 10202 --model_name SocialVRNN_LSTM_ED --prev_horizon 8 --prediction_horizon 12 --truncated_backprop_length 8 --print_freq 100 --warm_start_convnet false --freeze_grid_cnn false --warm_start_query_agent_module true --freeze_query_agent_module true --scenario 'real_world/st' --pretrained_query_agent_module_path "../trained_models/LSTM_ED_module/72" > slurm_jobs/output-logs/output_10202.log;

# srun python3 train_WIP.py --exp_num 10101003 --model_name SocialVRNN_LSTM_ED --prev_horizon 8 --prediction_horizon 12 --truncated_backprop_length 3 --print_freq 100 --warm_start_convnet true --freeze_grid_cnn true --warm_start_query_agent_module false --freeze_query_agent_module false --scenario 'real_world/ewap_dataset/seq_eth' --pretrained_query_agent_module_path "../trained_models/LSTM_ED_module/71" > output_10101003.log;
# srun python3 train_WIP.py --exp_num 10101005 --model_name SocialVRNN_LSTM_ED --prev_horizon 8 --prediction_horizon 12 --truncated_backprop_length 5 --print_freq 100 --warm_start_convnet true --freeze_grid_cnn true --warm_start_query_agent_module false --freeze_query_agent_module false --scenario 'real_world/ewap_dataset/seq_eth' --pretrained_query_agent_module_path "../trained_models/LSTM_ED_module/71" > output_10101005.log;
# srun python3 train_WIP.py --exp_num 10101008 --model_name SocialVRNN_LSTM_ED --prev_horizon 8 --prediction_horizon 12 --truncated_backprop_length 8 --print_freq 100 --warm_start_convnet true --freeze_grid_cnn true --warm_start_query_agent_module false --freeze_query_agent_module false --scenario 'real_world/ewap_dataset/seq_eth' --pretrained_query_agent_module_path "../trained_models/LSTM_ED_module/71" > output_10101008.log;


#srun python3 train_WIP.py --exp_num 10010101 --model_name SocialVRNN_LSTM_ED --prev_horizon 8 --prediction_horizon 12 --truncated_backprop_length 3 --print_freq 100 --warm_start_convnet true --freeze_grid_cnn true --warm_start_query_agent_module false --freeze_query_agent_module false --scenario 'real_world/ewap_dataset/seq_eth' --pretrained_query_agent_module_path "../trained_models/LSTM_ED_module/71" --rotated_grid true > output_10010101.log;
#srun python3 train_WIP.py --exp_num 20010101 --model_name SocialVRNN_LSTM_ED --prev_horizon 8 --prediction_horizon 12 --truncated_backprop_length 3 --print_freq 100 --warm_start_convnet true --freeze_grid_cnn true --warm_start_query_agent_module false --freeze_query_agent_module false --scenario 'real_world/ewap_dataset/seq_eth' --pretrained_query_agent_module_path "../trained_models/LSTM_ED_module/71" --normalize_data true > output_20010101.log;
#srun python3 train_WIP.py --exp_num 30010101 --model_name SocialVRNN_LSTM_ED --prev_horizon 8 --prediction_horizon 12 --truncated_backprop_length 3 --print_freq 100 --warm_start_convnet true --freeze_grid_cnn true --warm_start_query_agent_module false --freeze_query_agent_module false --scenario 'real_world/ewap_dataset/seq_eth' --pretrained_query_agent_module_path "../trained_models/LSTM_ED_module/71" --diversity_update true > output_30010101.log;
#srun python3 train_WIP.py --exp_num 40010101 --model_name SocialVRNN_LSTM_ED --prev_horizon 8 --prediction_horizon 12 --truncated_backprop_length 3 --print_freq 100 --warm_start_convnet true --freeze_grid_cnn true --warm_start_query_agent_module false --freeze_query_agent_module false --scenario 'real_world/ewap_dataset/seq_eth' --pretrained_query_agent_module_path "../trained_models/LSTM_ED_module/71" --correction_div_loss_in_total_loss true > output_40010101.log;
#srun python3 train_WIP.py --exp_num 60010101 --model_name SocialVRNN_LSTM_ED --prev_horizon 8 --prediction_horizon 12 --truncated_backprop_length 3 --print_freq 100 --warm_start_convnet true --freeze_grid_cnn true --warm_start_query_agent_module false --freeze_query_agent_module false --scenario 'real_world/ewap_dataset/seq_eth' --pretrained_query_agent_module_path "../trained_models/LSTM_ED_module/71" --correction_div_loss_in_total_loss true --diversity_update true > output_60010101.log;

# srun models/LSTM_ED_module.py --lstmed_exp_num 80 --truncated_backprop_length 3 --prev_horizon 8 --prediction_horizon 12 --total_training_steps 20000 --scenario "real_world/ewap_dataset/seq_hotel" > output_LSTMED_80.log;
# srun models/LSTM_ED_module.py --lstmed_exp_num 81 --truncated_backprop_length 3 --prev_horizon 8 --prediction_horizon 12 --total_training_steps 20000 --scenario "real_world/ewap_dataset/seq_eth" > output_LSTMED_81.log;
# srun models/LSTM_ED_module.py --lstmed_exp_num 82 --truncated_backprop_length 3 --prev_horizon 8 --prediction_horizon 12 --total_training_steps 20000 --scenario "real_world/st" > output_LSTMED_82.log;
# srun models/LSTM_ED_module.py --lstmed_exp_num 83 --truncated_backprop_length 3 --prev_horizon 8 --prediction_horizon 12 --total_training_steps 20000 --scenario "real_world/zara_01" > output_LSTMED_83.log;
# srun models/LSTM_ED_module.py --lstmed_exp_num 84 --truncated_backprop_length 3 --prev_horizon 8 --prediction_horizon 12 --total_training_steps 20000 --scenario "real_world/zara_02" > output_LSTMED_84.log;

# srun models/LSTM_ED_module.py --lstmed_exp_num 90 --truncated_backprop_length 3 --prev_horizon 8 --prediction_horizon 12 --total_training_steps 20000 --lstmed_reverse_time_prediction true --scenario "real_world/ewap_dataset/seq_hotel" > output_LSTMED_90.log;
# srun models/LSTM_ED_module.py --lstmed_exp_num 91 --truncated_backprop_length 3 --prev_horizon 8 --prediction_horizon 12 --total_training_steps 20000 --lstmed_reverse_time_prediction true --scenario "real_world/ewap_dataset/seq_eth" > output_LSTMED_91.log;
# srun models/LSTM_ED_module.py --lstmed_exp_num 92 --truncated_backprop_length 3 --prev_horizon 8 --prediction_horizon 12 --total_training_steps 20000 --lstmed_reverse_time_prediction true --scenario "real_world/st" > output_LSTMED_92.log;
# srun models/LSTM_ED_module.py --lstmed_exp_num 93 --truncated_backprop_length 3 --prev_horizon 8 --prediction_horizon 12 --total_training_steps 20000 --lstmed_reverse_time_prediction true --scenario "real_world/zara_01" > output_LSTMED_93.log;
# srun models/LSTM_ED_module.py --lstmed_exp_num 94 --truncated_backprop_length 3 --prev_horizon 8 --prediction_horizon 12 --total_training_steps 20000 --lstmed_reverse_time_prediction true --scenario "real_world/zara_02" > output_LSTMED_94.log;




# srun python3 train_WIP.py --exp_num 80000000100 --model_name SocialVRNN_LSTM_ED --total_training_steps 20000 --print_freq 100 --warm_start_convnet true --freeze_grid_cnn true --warm_start_query_agent_module false --freeze_query_agent_module false --scenario "real_world/ewap_dataset/seq_hotel" --pretrained_query_agent_module_path "../trained_models/LSTM_ED_module/100" > output_80000000100.log;
# srun python3 train_WIP.py --exp_num 80000000101 --model_name SocialVRNN_LSTM_ED --total_training_steps 20000 --print_freq 100 --warm_start_convnet true --freeze_grid_cnn true --warm_start_query_agent_module false --freeze_query_agent_module false --scenario "real_world/ewap_dataset/seq_eth" --pretrained_query_agent_module_path "../trained_models/LSTM_ED_module/101" > output_80000000101.log;
# srun python3 train_WIP.py --exp_num 80000000102 --model_name SocialVRNN_LSTM_ED --total_training_steps 20000 --print_freq 100 --warm_start_convnet true --freeze_grid_cnn true --warm_start_query_agent_module false --freeze_query_agent_module false --scenario "real_world/st" --pretrained_query_agent_module_path "../trained_models/LSTM_ED_module/102" > output_80000000102.log;
# srun python3 train_WIP.py --exp_num 80000000103 --model_name SocialVRNN_LSTM_ED --total_training_steps 20000 --print_freq 100 --warm_start_convnet true --freeze_grid_cnn true --warm_start_query_agent_module false --freeze_query_agent_module false --scenario "real_world/zara_01" --pretrained_query_agent_module_path "../trained_models/LSTM_ED_module/103" > output_80000000103.log;
# srun python3 train_WIP.py --exp_num 80000000104 --model_name SocialVRNN_LSTM_ED --total_training_steps 20000 --print_freq 100 --warm_start_convnet true --freeze_grid_cnn true --warm_start_query_agent_module false --freeze_query_agent_module false --scenario "real_world/zara_02" --pretrained_query_agent_module_path "../trained_models/LSTM_ED_module/104" > output_80000000104.log;

# srun python3 train_WIP.py --exp_num 80000000110 --model_name SocialVRNN_LSTM_ED --total_training_steps 20000 --print_freq 100 --warm_start_convnet false --freeze_grid_cnn false --warm_start_query_agent_module true --freeze_query_agent_module true --scenario "real_world/ewap_dataset/seq_hotel" --pretrained_query_agent_module_path "../trained_models/LSTM_ED_module/100" > output_80000000110.log;
# srun python3 train_WIP.py --exp_num 80000000111 --model_name SocialVRNN_LSTM_ED --total_training_steps 20000 --print_freq 100 --warm_start_convnet false --freeze_grid_cnn false --warm_start_query_agent_module true --freeze_query_agent_module true --scenario "real_world/ewap_dataset/seq_eth" --pretrained_query_agent_module_path "../trained_models/LSTM_ED_module/101" > output_80000000111.log;
# srun python3 train_WIP.py --exp_num 80000000112 --model_name SocialVRNN_LSTM_ED --total_training_steps 20000 --print_freq 100 --warm_start_convnet false --freeze_grid_cnn false --warm_start_query_agent_module true --freeze_query_agent_module true --scenario "real_world/st" --pretrained_query_agent_module_path "../trained_models/LSTM_ED_module/102" > output_80000000112.log;
# srun python3 train_WIP.py --exp_num 80000000113 --model_name SocialVRNN_LSTM_ED --total_training_steps 20000 --print_freq 100 --warm_start_convnet false --freeze_grid_cnn false --warm_start_query_agent_module true --freeze_query_agent_module true --scenario "real_world/zara_01" --pretrained_query_agent_module_path "../trained_models/LSTM_ED_module/103" > output_80000000113.log;
# srun python3 train_WIP.py --exp_num 80000000114 --model_name SocialVRNN_LSTM_ED --total_training_steps 20000 --print_freq 100 --warm_start_convnet false --freeze_grid_cnn false --warm_start_query_agent_module true --freeze_query_agent_module true --scenario "real_world/zara_02" --pretrained_query_agent_module_path "../trained_models/LSTM_ED_module/104" > output_80000000114.log;

# srun python3 train_WIP.py --exp_num 80000000120 --model_name SocialVRNN_LSTM_ED --total_training_steps 20000 --print_freq 100 --warm_start_convnet true --freeze_grid_cnn true --warm_start_query_agent_module true --freeze_query_agent_module true --scenario "real_world/ewap_dataset/seq_hotel" --pretrained_query_agent_module_path "../trained_models/LSTM_ED_module/100" > output_80000000120.log;
# srun python3 train_WIP.py --exp_num 80000000121 --model_name SocialVRNN_LSTM_ED --total_training_steps 20000 --print_freq 100 --warm_start_convnet true --freeze_grid_cnn true --warm_start_query_agent_module true --freeze_query_agent_module true --scenario "real_world/ewap_dataset/seq_eth" --pretrained_query_agent_module_path "../trained_models/LSTM_ED_module/101" > output_80000000121.log;
# srun python3 train_WIP.py --exp_num 80000000122 --model_name SocialVRNN_LSTM_ED --total_training_steps 20000 --print_freq 100 --warm_start_convnet true --freeze_grid_cnn true --warm_start_query_agent_module true --freeze_query_agent_module true --scenario "real_world/st" --pretrained_query_agent_module_path "../trained_models/LSTM_ED_module/102" > output_80000000122.log;
# srun python3 train_WIP.py --exp_num 80000000123 --model_name SocialVRNN_LSTM_ED --total_training_steps 20000 --print_freq 100 --warm_start_convnet true --freeze_grid_cnn true --warm_start_query_agent_module true --freeze_query_agent_module true --scenario "real_world/zara_01" --pretrained_query_agent_module_path "../trained_models/LSTM_ED_module/103" > output_80000000123.log;
# srun python3 train_WIP.py --exp_num 80000000124 --model_name SocialVRNN_LSTM_ED --total_training_steps 20000 --print_freq 100 --warm_start_convnet true --freeze_grid_cnn true --warm_start_query_agent_module true --freeze_query_agent_module true --scenario "real_world/zara_02" --pretrained_query_agent_module_path "../trained_models/LSTM_ED_module/104" > output_80000000124.log;

# srun python3 train_WIP.py --exp_num 80000000130 --model_name SocialVRNN_LSTM_ED --total_training_steps 20000 --print_freq 100 --warm_start_convnet false --freeze_grid_cnn false --warm_start_query_agent_module false --freeze_query_agent_module false --scenario "real_world/ewap_dataset/seq_hotel" --pretrained_query_agent_module_path "../trained_models/LSTM_ED_module/100" > output_80000000130.log;
# srun python3 train_WIP.py --exp_num 80000000131 --model_name SocialVRNN_LSTM_ED --total_training_steps 20000 --print_freq 100 --warm_start_convnet false --freeze_grid_cnn false --warm_start_query_agent_module false --freeze_query_agent_module false --scenario "real_world/ewap_dataset/seq_eth" --pretrained_query_agent_module_path "../trained_models/LSTM_ED_module/101" > output_80000000131.log;
# srun python3 train_WIP.py --exp_num 80000000132 --model_name SocialVRNN_LSTM_ED --total_training_steps 20000 --print_freq 100 --warm_start_convnet false --freeze_grid_cnn false --warm_start_query_agent_module false --freeze_query_agent_module false --scenario "real_world/st" --pretrained_query_agent_module_path "../trained_models/LSTM_ED_module/102" > output_80000000132.log;
# srun python3 train_WIP.py --exp_num 80000000133 --model_name SocialVRNN_LSTM_ED --total_training_steps 20000 --print_freq 100 --warm_start_convnet false --freeze_grid_cnn false --warm_start_query_agent_module false --freeze_query_agent_module false --scenario "real_world/zara_01" --pretrained_query_agent_module_path "../trained_models/LSTM_ED_module/103" > output_80000000133.log;
# srun python3 train_WIP.py --exp_num 80000000134 --model_name SocialVRNN_LSTM_ED --total_training_steps 20000 --print_freq 100 --warm_start_convnet false --freeze_grid_cnn false --warm_start_query_agent_module false --freeze_query_agent_module false --scenario "real_world/zara_02" --pretrained_query_agent_module_path "../trained_models/LSTM_ED_module/104" > output_80000000134.log;

# srun python3 train_WIP.py --exp_num 80000000000 --model_name SocialVRNN_LSTM_ED --total_training_steps 20000 --print_freq 100 --warm_start_convnet true --freeze_grid_cnn true --warm_start_query_agent_module false --freeze_query_agent_module false --scenario "real_world/ewap_dataset/seq_hotel" --pretrained_query_agent_module_path "../trained_models/LSTM_ED_module/120" > output_80000000000.log;
# srun python3 train_WIP.py --exp_num 80000000001 --model_name SocialVRNN_LSTM_ED --total_training_steps 20000 --print_freq 100 --warm_start_convnet true --freeze_grid_cnn true --warm_start_query_agent_module false --freeze_query_agent_module false --scenario "real_world/ewap_dataset/seq_eth" --pretrained_query_agent_module_path "../trained_models/LSTM_ED_module/121" > output_80000000001.log;
# srun python3 train_WIP.py --exp_num 80000000002 --model_name SocialVRNN_LSTM_ED --total_training_steps 20000 --print_freq 100 --warm_start_convnet true --freeze_grid_cnn true --warm_start_query_agent_module false --freeze_query_agent_module false --scenario "real_world/st" --pretrained_query_agent_module_path "../trained_models/LSTM_ED_module/122" > output_80000000002.log;
# srun python3 train_WIP.py --exp_num 80000000003 --model_name SocialVRNN_LSTM_ED --total_training_steps 20000 --print_freq 100 --warm_start_convnet true --freeze_grid_cnn true --warm_start_query_agent_module false --freeze_query_agent_module false --scenario "real_world/zara_01" --pretrained_query_agent_module_path "../trained_models/LSTM_ED_module/123" > output_80000000003.log;
# srun python3 train_WIP.py --exp_num 80000000004 --model_name SocialVRNN_LSTM_ED --total_training_steps 20000 --print_freq 100 --warm_start_convnet true --freeze_grid_cnn true --warm_start_query_agent_module false --freeze_query_agent_module false --scenario "real_world/zara_02" --pretrained_query_agent_module_path "../trained_models/LSTM_ED_module/124" > output_80000000004.log;

# srun python3 train_WIP.py --exp_num 80000000010 --model_name SocialVRNN_LSTM_ED --total_training_steps 20000 --print_freq 100 --warm_start_convnet false --freeze_grid_cnn false --warm_start_query_agent_module true --freeze_query_agent_module true --scenario "real_world/ewap_dataset/seq_hotel" --pretrained_query_agent_module_path "../trained_models/LSTM_ED_module/120" > output_80000000010.log;
# srun python3 train_WIP.py --exp_num 80000000011 --model_name SocialVRNN_LSTM_ED --total_training_steps 20000 --print_freq 100 --warm_start_convnet false --freeze_grid_cnn false --warm_start_query_agent_module true --freeze_query_agent_module true --scenario "real_world/ewap_dataset/seq_eth" --pretrained_query_agent_module_path "../trained_models/LSTM_ED_module/121" > output_80000000011.log;
# srun python3 train_WIP.py --exp_num 80000000012 --model_name SocialVRNN_LSTM_ED --total_training_steps 20000 --print_freq 100 --warm_start_convnet false --freeze_grid_cnn false --warm_start_query_agent_module true --freeze_query_agent_module true --scenario "real_world/st" --pretrained_query_agent_module_path "../trained_models/LSTM_ED_module/122" > output_80000000012.log;
srun python3 train_WIP.py --exp_num 80000000013 --model_name SocialVRNN_LSTM_ED --total_training_steps 20000 --print_freq 100 --warm_start_convnet false --freeze_grid_cnn false --warm_start_query_agent_module true --freeze_query_agent_module true --scenario "real_world/zara_01" --pretrained_query_agent_module_path "../trained_models/LSTM_ED_module/123" > output_80000000013.log;
srun python3 train_WIP.py --exp_num 80000000014 --model_name SocialVRNN_LSTM_ED --total_training_steps 20000 --print_freq 100 --warm_start_convnet false --freeze_grid_cnn false --warm_start_query_agent_module true --freeze_query_agent_module true --scenario "real_world/zara_02" --pretrained_query_agent_module_path "../trained_models/LSTM_ED_module/124" > output_80000000014.log;

srun python3 train_WIP.py --exp_num 80000000020 --model_name SocialVRNN_LSTM_ED --total_training_steps 20000 --print_freq 100 --warm_start_convnet true --freeze_grid_cnn true --warm_start_query_agent_module true --freeze_query_agent_module true --scenario "real_world/ewap_dataset/seq_hotel" --pretrained_query_agent_module_path "../trained_models/LSTM_ED_module/120" > output_80000000020.log;
srun python3 train_WIP.py --exp_num 80000000021 --model_name SocialVRNN_LSTM_ED --total_training_steps 20000 --print_freq 100 --warm_start_convnet true --freeze_grid_cnn true --warm_start_query_agent_module true --freeze_query_agent_module true --scenario "real_world/ewap_dataset/seq_eth" --pretrained_query_agent_module_path "../trained_models/LSTM_ED_module/121" > output_80000000021.log;
srun python3 train_WIP.py --exp_num 80000000022 --model_name SocialVRNN_LSTM_ED --total_training_steps 20000 --print_freq 100 --warm_start_convnet true --freeze_grid_cnn true --warm_start_query_agent_module true --freeze_query_agent_module true --scenario "real_world/st" --pretrained_query_agent_module_path "../trained_models/LSTM_ED_module/122" > output_80000000022.log;
srun python3 train_WIP.py --exp_num 80000000023 --model_name SocialVRNN_LSTM_ED --total_training_steps 20000 --print_freq 100 --warm_start_convnet true --freeze_grid_cnn true --warm_start_query_agent_module true --freeze_query_agent_module true --scenario "real_world/zara_01" --pretrained_query_agent_module_path "../trained_models/LSTM_ED_module/123" > output_80000000023.log;
srun python3 train_WIP.py --exp_num 80000000024 --model_name SocialVRNN_LSTM_ED --total_training_steps 20000 --print_freq 100 --warm_start_convnet true --freeze_grid_cnn true --warm_start_query_agent_module true --freeze_query_agent_module true --scenario "real_world/zara_02" --pretrained_query_agent_module_path "../trained_models/LSTM_ED_module/124" > output_80000000024.log;

srun python3 train_WIP.py --exp_num 80000000030 --model_name SocialVRNN_LSTM_ED --total_training_steps 20000 --print_freq 100 --warm_start_convnet false --freeze_grid_cnn false --warm_start_query_agent_module false --freeze_query_agent_module false --scenario "real_world/ewap_dataset/seq_hotel" --pretrained_query_agent_module_path "../trained_models/LSTM_ED_module/120" > output_80000000030.log;
srun python3 train_WIP.py --exp_num 80000000031 --model_name SocialVRNN_LSTM_ED --total_training_steps 20000 --print_freq 100 --warm_start_convnet false --freeze_grid_cnn false --warm_start_query_agent_module false --freeze_query_agent_module false --scenario "real_world/ewap_dataset/seq_eth" --pretrained_query_agent_module_path "../trained_models/LSTM_ED_module/121" > output_80000000031.log;
srun python3 train_WIP.py --exp_num 80000000032 --model_name SocialVRNN_LSTM_ED --total_training_steps 20000 --print_freq 100 --warm_start_convnet false --freeze_grid_cnn false --warm_start_query_agent_module false --freeze_query_agent_module false --scenario "real_world/st" --pretrained_query_agent_module_path "../trained_models/LSTM_ED_module/122" > output_80000000032.log;
srun python3 train_WIP.py --exp_num 80000000033 --model_name SocialVRNN_LSTM_ED --total_training_steps 20000 --print_freq 100 --warm_start_convnet false --freeze_grid_cnn false --warm_start_query_agent_module false --freeze_query_agent_module false --scenario "real_world/zara_01" --pretrained_query_agent_module_path "../trained_models/LSTM_ED_module/123" > output_80000000033.log;
srun python3 train_WIP.py --exp_num 80000000034 --model_name SocialVRNN_LSTM_ED --total_training_steps 20000 --print_freq 100 --warm_start_convnet false --freeze_grid_cnn false --warm_start_query_agent_module false --freeze_query_agent_module false --scenario "real_world/zara_02" --pretrained_query_agent_module_path "../trained_models/LSTM_ED_module/124" > output_80000000034.log;

# flags
srun python3 train_WIP.py --exp_num 80000010102 --model_name SocialVRNN_LSTM_ED --total_training_steps 20000 --print_freq 100 --warm_start_convnet true --freeze_grid_cnn true --warm_start_query_agent_module false --freeze_query_agent_module false --scenario "real_world/st" --pretrained_query_agent_module_path "../trained_models/LSTM_ED_module/102" --rotated_grid true > output_80000010102.log;
srun python3 train_WIP.py --exp_num 80000020102 --model_name SocialVRNN_LSTM_ED --total_training_steps 20000 --print_freq 100 --warm_start_convnet true --freeze_grid_cnn true --warm_start_query_agent_module false --freeze_query_agent_module false --scenario "real_world/st" --pretrained_query_agent_module_path "../trained_models/LSTM_ED_module/102" --normalize_data true > output_80000020102.log;
srun python3 train_WIP.py --exp_num 80000030102 --model_name SocialVRNN_LSTM_ED --total_training_steps 20000 --print_freq 100 --warm_start_convnet true --freeze_grid_cnn true --warm_start_query_agent_module false --freeze_query_agent_module false --scenario "real_world/st" --pretrained_query_agent_module_path "../trained_models/LSTM_ED_module/102" --diversity_update true > output_80000030102.log;
srun python3 train_WIP.py --exp_num 80000040102 --model_name SocialVRNN_LSTM_ED --total_training_steps 20000 --print_freq 100 --warm_start_convnet true --freeze_grid_cnn true --warm_start_query_agent_module false --freeze_query_agent_module false --scenario "real_world/st" --pretrained_query_agent_module_path "../trained_models/LSTM_ED_module/102" --correction_div_loss_in_total_loss true > output_80000040102.log;
srun python3 train_WIP.py --exp_num 80000050102 --model_name SocialVRNN_LSTM_ED --total_training_steps 20000 --print_freq 100 --warm_start_convnet true --freeze_grid_cnn true --warm_start_query_agent_module false --freeze_query_agent_module false --scenario "real_world/st" --pretrained_query_agent_module_path "../trained_models/LSTM_ED_module/102" --diversity_update true --correction_div_loss_in_total_loss true > output_80000050102.log;



