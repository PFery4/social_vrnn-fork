# BE CAREFUL TO CHECK THE MAIN FUNCTION OF THE LSTME_D_MODULE:
# src/models/LSTM_ED_module.py

python3 models/LSTM_ED_module.py --lstmed_exp_num 60 --truncated_backprop_length 3 --prev_horizon 8 --prediction_horizon 12 --scenario "real_world/ewap_dataset/seq_hotel";
python3 models/LSTM_ED_module.py --lstmed_exp_num 61 --truncated_backprop_length 3 --prev_horizon 8 --prediction_horizon 12 --scenario "real_world/ewap_dataset/seq_eth";
python3 models/LSTM_ED_module.py --lstmed_exp_num 62 --truncated_backprop_length 3 --prev_horizon 8 --prediction_horizon 12 --scenario "real_world/st";
python3 models/LSTM_ED_module.py --lstmed_exp_num 63 --truncated_backprop_length 3 --prev_horizon 8 --prediction_horizon 12 --scenario "real_world/zara_01";
python3 models/LSTM_ED_module.py --lstmed_exp_num 64 --truncated_backprop_length 3 --prev_horizon 8 --prediction_horizon 12 --scenario "real_world/zara_02";

# python3 models/LSTM_ED_module.py --lstmed_exp_num 70 --truncated_backprop_length 8 --prev_horizon 8 --prediction_horizon 12 --scenario "real_world/ewap_dataset/seq_hotel";
# python3 models/LSTM_ED_module.py --lstmed_exp_num 71 --truncated_backprop_length 8 --prev_horizon 8 --prediction_horizon 12 --scenario "real_world/ewap_dataset/seq_eth";
# python3 models/LSTM_ED_module.py --lstmed_exp_num 72 --truncated_backprop_length 8 --prev_horizon 8 --prediction_horizon 12 --scenario "real_world/st";
# python3 models/LSTM_ED_module.py --lstmed_exp_num 73 --truncated_backprop_length 8 --prev_horizon 8 --prediction_horizon 12 --scenario "real_world/zara_01";
# python3 models/LSTM_ED_module.py --lstmed_exp_num 74 --truncated_backprop_length 8 --prev_horizon 8 --prediction_horizon 12 --scenario "real_world/zara_02";


