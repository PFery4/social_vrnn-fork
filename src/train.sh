
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

# python3 models/LSTM_ED_module.py --lstmed_exp_num 20 --total_training_steps 5000 --scenario 'real_world/ewap_dataset/seq_hotel';
# python3 models/LSTM_ED_module.py --lstmed_exp_num 21 --total_training_steps 5000 --scenario 'real_world/ewap_dataset/seq_eth';
# python3 models/LSTM_ED_module.py --lstmed_exp_num 22 --total_training_steps 5000 --scenario 'real_world/st';
# python3 models/LSTM_ED_module.py --lstmed_exp_num 23 --total_training_steps 5000 --scenario 'real_world/zara_01';
# python3 models/LSTM_ED_module.py --lstmed_exp_num 24 --total_training_steps 5000 --scenario 'real_world/zara_02';


# CNN autoencoder trained separately and frozen whereas the Past Trajectory autoencoder is not pretrained and frozen (this is already SocialVRNN's implementation)
# Past trajectory autoencoder trained separately and frozen whereas the CNN is not pretrained and frozen
# None of the subsections are pretrained and frozen
# Both CNN autoencoder and Past Trajectory autoencoder are pretrained and frozen.


python3 train_WIP.py --model_name SocialVRNN_LSTM_ED --exp_num 100 --n_mixtures 3 --others_info relative --total_training_steps 20000 --print_freq 40 --warm_start_convnet true --freeze_grid_cnn true --warm_start_query_agent_module false --freeze_query_agent_module false --scenario 'real_world/ewap_dataset/seq_hotel' --pretrained_query_agent_module_path "../trained_models/LSTM_ED_module/20";
python3 train_WIP.py --model_name SocialVRNN_LSTM_ED --exp_num 200 --n_mixtures 3 --others_info relative --total_training_steps 20000 --print_freq 40 --warm_start_convnet false --freeze_grid_cnn false --warm_start_query_agent_module true --freeze_query_agent_module true --scenario 'real_world/ewap_dataset/seq_hotel' --pretrained_query_agent_module_path "../trained_models/LSTM_ED_module/20";
python3 train_WIP.py --model_name SocialVRNN_LSTM_ED --exp_num 300 --n_mixtures 3 --others_info relative --total_training_steps 20000 --print_freq 40 --warm_start_convnet false --freeze_grid_cnn false --warm_start_query_agent_module false --freeze_query_agent_module false --scenario 'real_world/ewap_dataset/seq_hotel' --pretrained_query_agent_module_path "../trained_models/LSTM_ED_module/20";
python3 train_WIP.py --model_name SocialVRNN_LSTM_ED --exp_num 400 --n_mixtures 3 --others_info relative --total_training_steps 20000 --print_freq 40 --warm_start_convnet true --freeze_grid_cnn true --warm_start_query_agent_module true --freeze_query_agent_module true --scenario 'real_world/ewap_dataset/seq_hotel' --pretrained_query_agent_module_path "../trained_models/LSTM_ED_module/20";

python3 train_WIP.py --model_name SocialVRNN_LSTM_ED --exp_num 101 --n_mixtures 3 --others_info relative --total_training_steps 20000 --print_freq 40 --warm_start_convnet true --freeze_grid_cnn true --warm_start_query_agent_module false --freeze_query_agent_module false --scenario 'real_world/ewap_dataset/seq_eth' --pretrained_query_agent_module_path "../trained_models/LSTM_ED_module/21";
python3 train_WIP.py --model_name SocialVRNN_LSTM_ED --exp_num 201 --n_mixtures 3 --others_info relative --total_training_steps 20000 --print_freq 40 --warm_start_convnet false --freeze_grid_cnn false --warm_start_query_agent_module true --freeze_query_agent_module true --scenario 'real_world/ewap_dataset/seq_eth' --pretrained_query_agent_module_path "../trained_models/LSTM_ED_module/21";
python3 train_WIP.py --model_name SocialVRNN_LSTM_ED --exp_num 301 --n_mixtures 3 --others_info relative --total_training_steps 20000 --print_freq 40 --warm_start_convnet false --freeze_grid_cnn false --warm_start_query_agent_module false --freeze_query_agent_module false --scenario 'real_world/ewap_dataset/seq_eth' --pretrained_query_agent_module_path "../trained_models/LSTM_ED_module/21";
python3 train_WIP.py --model_name SocialVRNN_LSTM_ED --exp_num 401 --n_mixtures 3 --others_info relative --total_training_steps 20000 --print_freq 40 --warm_start_convnet true --freeze_grid_cnn true --warm_start_query_agent_module true --freeze_query_agent_module true --scenario 'real_world/ewap_dataset/seq_eth' --pretrained_query_agent_module_path "../trained_models/LSTM_ED_module/21";

python3 train_WIP.py --model_name SocialVRNN_LSTM_ED --exp_num 102 --n_mixtures 3 --others_info relative --total_training_steps 20000 --print_freq 40 --warm_start_convnet true --freeze_grid_cnn true --warm_start_query_agent_module false --freeze_query_agent_module false --scenario 'real_world/st' --pretrained_query_agent_module_path "../trained_models/LSTM_ED_module/22";
python3 train_WIP.py --model_name SocialVRNN_LSTM_ED --exp_num 202 --n_mixtures 3 --others_info relative --total_training_steps 20000 --print_freq 40 --warm_start_convnet false --freeze_grid_cnn false --warm_start_query_agent_module true --freeze_query_agent_module true --scenario 'real_world/st' --pretrained_query_agent_module_path "../trained_models/LSTM_ED_module/22";
python3 train_WIP.py --model_name SocialVRNN_LSTM_ED --exp_num 302 --n_mixtures 3 --others_info relative --total_training_steps 20000 --print_freq 40 --warm_start_convnet false --freeze_grid_cnn false --warm_start_query_agent_module false --freeze_query_agent_module false --scenario 'real_world/st' --pretrained_query_agent_module_path "../trained_models/LSTM_ED_module/22";
python3 train_WIP.py --model_name SocialVRNN_LSTM_ED --exp_num 402 --n_mixtures 3 --others_info relative --total_training_steps 20000 --print_freq 40 --warm_start_convnet true --freeze_grid_cnn true --warm_start_query_agent_module true --freeze_query_agent_module true --scenario 'real_world/st' --pretrained_query_agent_module_path "../trained_models/LSTM_ED_module/22";

python3 train_WIP.py --model_name SocialVRNN_LSTM_ED --exp_num 103 --n_mixtures 3 --others_info relative --total_training_steps 20000 --print_freq 40 --warm_start_convnet true --freeze_grid_cnn true --warm_start_query_agent_module false --freeze_query_agent_module false --scenario 'real_world/zara_01' --pretrained_query_agent_module_path "../trained_models/LSTM_ED_module/23";
python3 train_WIP.py --model_name SocialVRNN_LSTM_ED --exp_num 203 --n_mixtures 3 --others_info relative --total_training_steps 20000 --print_freq 40 --warm_start_convnet false --freeze_grid_cnn false --warm_start_query_agent_module true --freeze_query_agent_module true --scenario 'real_world/zara_01' --pretrained_query_agent_module_path "../trained_models/LSTM_ED_module/23";
python3 train_WIP.py --model_name SocialVRNN_LSTM_ED --exp_num 303 --n_mixtures 3 --others_info relative --total_training_steps 20000 --print_freq 40 --warm_start_convnet false --freeze_grid_cnn false --warm_start_query_agent_module false --freeze_query_agent_module false --scenario 'real_world/zara_01' --pretrained_query_agent_module_path "../trained_models/LSTM_ED_module/23";
python3 train_WIP.py --model_name SocialVRNN_LSTM_ED --exp_num 403 --n_mixtures 3 --others_info relative --total_training_steps 20000 --print_freq 40 --warm_start_convnet true --freeze_grid_cnn true --warm_start_query_agent_module true --freeze_query_agent_module true --scenario 'real_world/zara_01' --pretrained_query_agent_module_path "../trained_models/LSTM_ED_module/23";

python3 train_WIP.py --model_name SocialVRNN_LSTM_ED --exp_num 104 --n_mixtures 3 --others_info relative --total_training_steps 20000 --print_freq 40 --warm_start_convnet true --freeze_grid_cnn true --warm_start_query_agent_module false --freeze_query_agent_module false --scenario 'real_world/zara_02' --pretrained_query_agent_module_path "../trained_models/LSTM_ED_module/24";
python3 train_WIP.py --model_name SocialVRNN_LSTM_ED --exp_num 204 --n_mixtures 3 --others_info relative --total_training_steps 20000 --print_freq 40 --warm_start_convnet false --freeze_grid_cnn false --warm_start_query_agent_module true --freeze_query_agent_module true --scenario 'real_world/zara_02' --pretrained_query_agent_module_path "../trained_models/LSTM_ED_module/24";
python3 train_WIP.py --model_name SocialVRNN_LSTM_ED --exp_num 304 --n_mixtures 3 --others_info relative --total_training_steps 20000 --print_freq 40 --warm_start_convnet false --freeze_grid_cnn false --warm_start_query_agent_module false --freeze_query_agent_module false --scenario 'real_world/zara_02' --pretrained_query_agent_module_path "../trained_models/LSTM_ED_module/24";
python3 train_WIP.py --model_name SocialVRNN_LSTM_ED --exp_num 404 --n_mixtures 3 --others_info relative --total_training_steps 20000 --print_freq 40 --warm_start_convnet true --freeze_grid_cnn true --warm_start_query_agent_module true --freeze_query_agent_module true --scenario 'real_world/zara_02' --pretrained_query_agent_module_path "../trained_models/LSTM_ED_module/24";


