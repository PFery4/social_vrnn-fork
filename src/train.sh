
#python3 train.py --exp_num 10 --model_name SocialVRNN --n_mixtures 3 --output_pred_state_dim 4 --scenario real_world/ewap_dataset/seq_hotel --gpu false --prev_horizon 8 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 1 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 ;
#python3 train.py --exp_num 11 --model_name SocialVRNN --n_mixtures 3 --output_pred_state_dim 4 --scenario real_world/zara_01 --gpu false --prev_horizon 8 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 12 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 ;
#python3 train.py --exp_num 12 --model_name SocialVRNN --n_mixtures 3 --output_pred_state_dim 4 --scenario real_world/st --gpu true --prev_horizon 8 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 12 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 ;
#python3 train.py --exp_num 13 --model_name SocialVRNN --n_mixtures 3 --output_pred_state_dim 4 --scenario real_world/zara_02 --gpu true --prev_horizon 8 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 12 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 ;
#python3 train.py --exp_num 14 --model_name SocialVRNN --n_mixtures 3 --output_pred_state_dim 4 --scenario real_world/ewap_dataset/seq_eth --gpu true --prev_horizon 8 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 12 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 ;

# small tests
# python3 train.py --exp_num 444003 --model_name SocialVRNN --n_mixtures 3 --output_pred_state_dim 4 --scenario real_world/ewap_dataset/seq_hotel --gpu false --prev_horizon 7 --prediction_horizon 12 --dt 0.4 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --total_training_steps 800 --debug_plotting true;
python3 train_WIP.py --exp_num 444004 --model_name WIP_SocialVRNN --n_mixtures 3 --output_pred_state_dim 4 --scenario real_world/ewap_dataset/seq_hotel --gpu false --prev_horizon 7 --prediction_horizon 12 --dt 0.4 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --total_training_steps 800;

# python3 train_WIP.py;
