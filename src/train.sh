
#python3 train.py --exp_num 10 --model_name SocialVRNN --n_mixtures 3 --output_pred_state_dim 4 --scenario real_world/ewap_dataset/seq_hotel --gpu false --prev_horizon 8 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 1 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 ;
#python3 train.py --exp_num 11 --model_name SocialVRNN --n_mixtures 3 --output_pred_state_dim 4 --scenario real_world/zara_01 --gpu false --prev_horizon 8 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 12 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 ;
#python3 train.py --exp_num 12 --model_name SocialVRNN --n_mixtures 3 --output_pred_state_dim 4 --scenario real_world/st --gpu true --prev_horizon 8 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 12 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 ;
#python3 train.py --exp_num 13 --model_name SocialVRNN --n_mixtures 3 --output_pred_state_dim 4 --scenario real_world/zara_02 --gpu true --prev_horizon 8 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 12 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 ;
#python3 train.py --exp_num 14 --model_name SocialVRNN --n_mixtures 3 --output_pred_state_dim 4 --scenario real_world/ewap_dataset/seq_eth --gpu true --prev_horizon 8 --prediction_horizon 12 --dt 0.4 --truncated_backprop_length 12 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 ;

# small tests
# python3 train.py --exp_num 444003 --model_name SocialVRNN --n_mixtures 3 --output_pred_state_dim 4 --scenario real_world/ewap_dataset/seq_hotel --gpu false --prev_horizon 7 --prediction_horizon 12 --dt 0.4 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --total_training_steps 800 --debug_plotting true;
#python3 train_WIP.py --exp_num 444005 --model_name SocialVRNN_AE --n_mixtures 3 --output_pred_state_dim 4 --scenario real_world/ewap_dataset/seq_hotel --gpu false --prev_horizon 7 --prediction_horizon 12 --dt 0.4 --others_info relative --pedestrian_vector_dim 4 --batch_size 16 --total_training_steps 800;

# Pretraining of the Past Trajectory Autoencoder;
# python3 train_try_2.py --exp_num 444005 --scenario 'real_world/ewap_dataset/seq_hotel' --query_agent_ae_optimizer 'Adam' --query_agent_ae_encoding_layers 12 --query_agent_ae_latent_space_dim 8;
# python3 train_try_2.py --exp_num 444006 --scenario 'real_world/ewap_dataset/seq_eth' --query_agent_ae_optimizer 'Adam' --query_agent_ae_encoding_layers 12 --query_agent_ae_latent_space_dim 8;
# python3 train_try_2.py --exp_num 444007 --scenario 'real_world/st' --query_agent_ae_optimizer 'Adam' --query_agent_ae_encoding_layers 12 --query_agent_ae_latent_space_dim 8;
# python3 train_try_2.py --exp_num 444008 --scenario 'real_world/zara_01' --query_agent_ae_optimizer 'Adam' --query_agent_ae_encoding_layers 12 --query_agent_ae_latent_space_dim 8;
# python3 train_try_2.py --exp_num 444009 --scenario 'real_world/zara_02' --query_agent_ae_optimizer 'Adam' --query_agent_ae_encoding_layers 12 --query_agent_ae_latent_space_dim 8;
# python3 train_try_2.py --exp_num 444010 --scenario 'real_world/ewap_dataset/seq_hotel' --query_agent_ae_optimizer 'RMSProp' --query_agent_ae_encoding_layers 12 --query_agent_ae_latent_space_dim 8;
# python3 train_try_2.py --exp_num 444011 --scenario 'real_world/ewap_dataset/seq_eth' --query_agent_ae_optimizer 'RMSProp' --query_agent_ae_encoding_layers 12 --query_agent_ae_latent_space_dim 8;
# python3 train_try_2.py --exp_num 444012 --scenario 'real_world/st' --query_agent_ae_optimizer 'RMSProp' --query_agent_ae_encoding_layers 12 --query_agent_ae_latent_space_dim 8;
# python3 train_try_2.py --exp_num 444013 --scenario 'real_world/zara_01' --query_agent_ae_optimizer 'RMSProp' --query_agent_ae_encoding_layers 12 --query_agent_ae_latent_space_dim 8;
# python3 train_try_2.py --exp_num 444014 --scenario 'real_world/zara_02' --query_agent_ae_optimizer 'RMSProp' --query_agent_ae_encoding_layers 12 --query_agent_ae_latent_space_dim 8;
# python3 train_try_2.py --exp_num 444015 --scenario 'real_world/ewap_dataset/seq_hotel' --query_agent_ae_optimizer 'Adam' --query_agent_ae_encoding_layers 10 6 --query_agent_ae_latent_space_dim 2;
# python3 train_try_2.py --exp_num 444016 --scenario 'real_world/ewap_dataset/seq_eth' --query_agent_ae_optimizer 'Adam' --query_agent_ae_encoding_layers 10 6 --query_agent_ae_latent_space_dim 2;
# python3 train_try_2.py --exp_num 444017 --scenario 'real_world/st' --query_agent_ae_optimizer 'Adam' --query_agent_ae_encoding_layers 10 6 --query_agent_ae_latent_space_dim 2;
# python3 train_try_2.py --exp_num 444018 --scenario 'real_world/zara_01' --query_agent_ae_optimizer 'Adam' --query_agent_ae_encoding_layers 10 6 --query_agent_ae_latent_space_dim 2;
# python3 train_try_2.py --exp_num 444019 --scenario 'real_world/zara_02' --query_agent_ae_optimizer 'Adam' --query_agent_ae_encoding_layers 10 6 --query_agent_ae_latent_space_dim 2;
# python3 train_try_2.py --exp_num 444020 --scenario 'real_world/ewap_dataset/seq_hotel' --query_agent_ae_optimizer 'RMSProp' --query_agent_ae_encoding_layers 10 6 --query_agent_ae_latent_space_dim 2;
# python3 train_try_2.py --exp_num 444021 --scenario 'real_world/ewap_dataset/seq_eth' --query_agent_ae_optimizer 'RMSProp' --query_agent_ae_encoding_layers 10 6 --query_agent_ae_latent_space_dim 2;
# python3 train_try_2.py --exp_num 444022 --scenario 'real_world/st' --query_agent_ae_optimizer 'RMSProp' --query_agent_ae_encoding_layers 10 6 --query_agent_ae_latent_space_dim 2;
# python3 train_try_2.py --exp_num 444023 --scenario 'real_world/zara_01' --query_agent_ae_optimizer 'RMSProp' --query_agent_ae_encoding_layers 10 6 --query_agent_ae_latent_space_dim 2;
# python3 train_try_2.py --exp_num 444024 --scenario 'real_world/zara_02' --query_agent_ae_optimizer 'RMSProp' --query_agent_ae_encoding_layers 10 6 --query_agent_ae_latent_space_dim 2;

# python3 train_WIP.py --exp_num 4000 --others_info relative --total_training_steps 20000 --print_freq 40;
# python3 train_WIP.py --model_name SocialVRNN --exp_num 4001 --others_info relative --total_training_steps 20000 --print_freq 40 --freeze_grid_cnn false;
# python3 train_WIP.py --model_name SocialVRNN_AE --exp_num 4001 --others_info relative --total_training_steps 20000 --print_freq 40 --freeze_grid_cnn true;
# python3 train_WIP.py --model_name SocialVRNN_AE --exp_num 9000 --others_info relative --total_training_steps 20000 --print_freq 40 --warmstart_model true;
#python3 train_WIP.py --model_name SocialVRNN --scenario 'real_world/ewap_dataset/seq_hotel' --exp_num 4000 --others_info relative --total_training_steps 20000 --print_freq 40 --freeze_grid_cnn false;
#python3 train_WIP.py --model_name SocialVRNN --scenario 'real_world/ewap_dataset/seq_hotel' --exp_num 4001 --others_info relative --total_training_steps 20000 --print_freq 40 --freeze_grid_cnn true;
#python3 train_WIP.py --model_name SocialVRNN --scenario 'real_world/ewap_dataset/seq_eth' --exp_num 4002 --others_info relative --total_training_steps 20000 --print_freq 40 --freeze_grid_cnn false;
#python3 train_WIP.py --model_name SocialVRNN --scenario 'real_world/ewap_dataset/seq_eth' --exp_num 4003 --others_info relative --total_training_steps 20000 --print_freq 40 --freeze_grid_cnn true;
#python3 train_WIP.py --model_name SocialVRNN --scenario 'real_world/st' --exp_num 4004 --others_info relative --total_training_steps 20000 --print_freq 40 --freeze_grid_cnn false;
#python3 train_WIP.py --model_name SocialVRNN --scenario 'real_world/st' --exp_num 4005 --others_info relative --total_training_steps 20000 --print_freq 40 --freeze_grid_cnn true;
#python3 train_WIP.py --model_name SocialVRNN --scenario 'real_world/zara_01' --exp_num 4006 --others_info relative --total_training_steps 20000 --print_freq 40 --freeze_grid_cnn false;
#python3 train_WIP.py --model_name SocialVRNN --scenario 'real_world/zara_01' --exp_num 4007 --others_info relative --total_training_steps 20000 --print_freq 40 --freeze_grid_cnn true;
#python3 train_WIP.py --model_name SocialVRNN --scenario 'real_world/zara_02' --exp_num 4008 --others_info relative --total_training_steps 20000 --print_freq 40 --freeze_grid_cnn false;
#python3 train_WIP.py --model_name SocialVRNN --scenario 'real_world/zara_02' --exp_num 4009 --others_info relative --total_training_steps 20000 --print_freq 40 --freeze_grid_cnn true;


# python3 train_WIP.py --model_name SocialVRNN --scenario 'real_world/zara_01' --exp_num 4444 --others_info relative --total_training_steps 800 --print_freq 20;
python3 train_WIP.py --model_name SocialVRNN --scenario 'real_world/zara_01' --exp_num 5555 --others_info relative --total_training_steps 800 --print_freq 20;
