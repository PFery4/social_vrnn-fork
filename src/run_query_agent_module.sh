# BE CAREFUL TO CHECK THE MAIN FUNCTION OF THE LSTM_ED_MODULE:
# src/models/LSTM_ED_module.py

# python3 models/LSTM_ED_module.py --lstmed_exp_num 60 --truncated_backprop_length 3 --prev_horizon 8 --prediction_horizon 12 --scenario "real_world/ewap_dataset/seq_hotel";
# python3 models/LSTM_ED_module.py --lstmed_exp_num 61 --truncated_backprop_length 3 --prev_horizon 8 --prediction_horizon 12 --scenario "real_world/ewap_dataset/seq_eth";
# python3 models/LSTM_ED_module.py --lstmed_exp_num 62 --truncated_backprop_length 3 --prev_horizon 8 --prediction_horizon 12 --scenario "real_world/st";
# python3 models/LSTM_ED_module.py --lstmed_exp_num 63 --truncated_backprop_length 3 --prev_horizon 8 --prediction_horizon 12 --scenario "real_world/zara_01";
# python3 models/LSTM_ED_module.py --lstmed_exp_num 64 --truncated_backprop_length 3 --prev_horizon 8 --prediction_horizon 12 --scenario "real_world/zara_02";

# python3 models/LSTM_ED_module.py --lstmed_exp_num 70 --truncated_backprop_length 8 --prev_horizon 8 --prediction_horizon 12 --scenario "real_world/ewap_dataset/seq_hotel";
# python3 models/LSTM_ED_module.py --lstmed_exp_num 71 --truncated_backprop_length 8 --prev_horizon 8 --prediction_horizon 12 --scenario "real_world/ewap_dataset/seq_eth";
# python3 models/LSTM_ED_module.py --lstmed_exp_num 72 --truncated_backprop_length 8 --prev_horizon 8 --prediction_horizon 12 --scenario "real_world/st";
# python3 models/LSTM_ED_module.py --lstmed_exp_num 73 --truncated_backprop_length 8 --prev_horizon 8 --prediction_horizon 12 --scenario "real_world/zara_01";
# python3 models/LSTM_ED_module.py --lstmed_exp_num 74 --truncated_backprop_length 8 --prev_horizon 8 --prediction_horizon 12 --scenario "real_world/zara_02";

# python3 models/LSTM_ED_module.py --lstmed_exp_num 80 --truncated_backprop_length 3 --prev_horizon 8 --prediction_horizon 12 --total_training_steps 20000 --scenario "real_world/ewap_dataset/seq_hotel";
# python3 models/LSTM_ED_module.py --lstmed_exp_num 81 --truncated_backprop_length 3 --prev_horizon 8 --prediction_horizon 12 --total_training_steps 20000 --scenario "real_world/ewap_dataset/seq_eth";
# python3 models/LSTM_ED_module.py --lstmed_exp_num 82 --truncated_backprop_length 3 --prev_horizon 8 --prediction_horizon 12 --total_training_steps 20000 --scenario "real_world/st";
# python3 models/LSTM_ED_module.py --lstmed_exp_num 83 --truncated_backprop_length 3 --prev_horizon 8 --prediction_horizon 12 --total_training_steps 20000 --scenario "real_world/zara_01";
# python3 models/LSTM_ED_module.py --lstmed_exp_num 84 --truncated_backprop_length 3 --prev_horizon 8 --prediction_horizon 12 --total_training_steps 20000 --scenario "real_world/zara_02";

# python3 models/LSTM_ED_module.py --lstmed_exp_num 90 --truncated_backprop_length 3 --prev_horizon 8 --prediction_horizon 12 --total_training_steps 20000 --lstmed_reverse_time_prediction true --scenario "real_world/ewap_dataset/seq_hotel";
# python3 models/LSTM_ED_module.py --lstmed_exp_num 91 --truncated_backprop_length 3 --prev_horizon 8 --prediction_horizon 12 --total_training_steps 20000 --lstmed_reverse_time_prediction true --scenario "real_world/ewap_dataset/seq_eth";
# python3 models/LSTM_ED_module.py --lstmed_exp_num 92 --truncated_backprop_length 3 --prev_horizon 8 --prediction_horizon 12 --total_training_steps 20000 --lstmed_reverse_time_prediction true --scenario "real_world/st";
# python3 models/LSTM_ED_module.py --lstmed_exp_num 93 --truncated_backprop_length 3 --prev_horizon 8 --prediction_horizon 12 --total_training_steps 20000 --lstmed_reverse_time_prediction true --scenario "real_world/zara_01";
# python3 models/LSTM_ED_module.py --lstmed_exp_num 94 --truncated_backprop_length 3 --prev_horizon 8 --prediction_horizon 12 --total_training_steps 20000 --lstmed_reverse_time_prediction true --scenario "real_world/zara_02";

# python3 models/LSTM_ED_module.py --lstmed_exp_num 100 --truncated_backprop_length 3 --prev_horizon 8 --prediction_horizon 12 --total_training_steps 20000 --scenario "real_world/ewap_dataset/seq_hotel";
# python3 models/LSTM_ED_module.py --lstmed_exp_num 101 --truncated_backprop_length 3 --prev_horizon 8 --prediction_horizon 12 --total_training_steps 20000 --scenario "real_world/ewap_dataset/seq_eth";
# python3 models/LSTM_ED_module.py --lstmed_exp_num 102 --truncated_backprop_length 3 --prev_horizon 8 --prediction_horizon 12 --total_training_steps 20000 --scenario "real_world/st";
# python3 models/LSTM_ED_module.py --lstmed_exp_num 103 --truncated_backprop_length 3 --prev_horizon 8 --prediction_horizon 12 --total_training_steps 20000 --scenario "real_world/zara_01";
# python3 models/LSTM_ED_module.py --lstmed_exp_num 104 --truncated_backprop_length 3 --prev_horizon 8 --prediction_horizon 12 --total_training_steps 20000 --scenario "real_world/zara_02";

# python3 models/LSTM_ED_module.py --lstmed_exp_num 110 --truncated_backprop_length 3 --prev_horizon 8 --prediction_horizon 12 --total_training_steps 20000 --lstmed_reverse_time_prediction true --scenario "real_world/ewap_dataset/seq_hotel";
# python3 models/LSTM_ED_module.py --lstmed_exp_num 111 --truncated_backprop_length 3 --prev_horizon 8 --prediction_horizon 12 --total_training_steps 20000 --lstmed_reverse_time_prediction true --scenario "real_world/ewap_dataset/seq_eth";
# python3 models/LSTM_ED_module.py --lstmed_exp_num 112 --truncated_backprop_length 3 --prev_horizon 8 --prediction_horizon 12 --total_training_steps 20000 --lstmed_reverse_time_prediction true --scenario "real_world/st";
# python3 models/LSTM_ED_module.py --lstmed_exp_num 113 --truncated_backprop_length 3 --prev_horizon 8 --prediction_horizon 12 --total_training_steps 20000 --lstmed_reverse_time_prediction true --scenario "real_world/zara_01";
# python3 models/LSTM_ED_module.py --lstmed_exp_num 114 --truncated_backprop_length 3 --prev_horizon 8 --prediction_horizon 12 --total_training_steps 20000 --lstmed_reverse_time_prediction true --scenario "real_world/zara_02";

