"""

Configuration file for handling the default arguments for running scripts within this project.

"""
import os.path
import sys
sys.path.append(os.path.abspath(os.path.join(__file__, "../../..")))
import argparse
import json
import src.data_utils.Support as sup
import src.data_utils.plot_utils
import pickle as pkl
from colorama import Fore, Style


def parse_args():
    """
    Specify the hyperparameters and settings for running the script
    """
    # Filesystem parameters
    model_name = "SocialVRNN"
    pretrained_convnet_path = "../trained_models/autoencoder_with_ped"
    pretrained_qa_ae_path = "../trained_models/PastTrajAE/0"

    data_path = '../data/'
    # scenario = 'real_world/ewap_dataset/seq_hotel'
    # scenario = 'real_world/ewap_dataset/seq_eth'
    scenario = 'real_world/st'      # this one as the default, since it is the richest in data
    # scenario = 'real_world/zara_01'
    # scenario = 'real_world/zara_02'
    exp_num = 0

    # Hyperparameters
    n_epochs = 2        # does not seem to have a purpose
    batch_size = 16
    regularization_weight = 0.0001

    # Time parameters
    truncated_backprop_length = 3
    prediction_horizon = 12
    prev_horizon = 8

    # Input / Output dimension parameters
    input_dim = 4  # [x, y, vx, vy]
    input_state_dim = 2  # [vx, vy]
    output_dim = 2  # data state dimension
    output_pred_state_dim = 4  # ux uy simgax sigmay
    pedestrian_vector_dim = 4  # used to be 36 (I do not know why, seems that 4 is correct (look at implementation of fillbatch method in DataHandlerLSTM class))
    pedestrian_vector_state_dim = 2
    cmd_vector_dim = 2
    pedestrian_radius = 0.3
    max_range_ped_grid = 5

    # Vanilla SocialVRNN hyperparameters
    rnn_state_size = 32
    rnn_state_size_lstm_grid = 256
    rnn_state_size_lstm_ped = 128
    rnn_state_ped_size = 16
    rnn_state_size_lstm_concat = 512
    prior_size = 512
    latent_space_size = 256
    x_dim = 512
    fc_hidden_unit_size = 256
    learning_rate_init = 0.001      # should be 10^-4 according to the publication
    beta_rate_init = 0.01
    keep_prob = 1.0
    dropout = False
    n_mixtures = 3  # USE ZERO FOR MSE MODEL
    grads_clip = 1.0
    n_other_agents = 18
    tensorboard_logging = True

    # Query Agent Past Trajectory mod:
    # Velocity Feature Autoencoder
    query_agent_ae_encoding_layers = [12, 8]
    query_agent_ae_latent_space_dim = 4
    query_agent_ae_optimizer = 'Adam'
    freeze_qa_ae = False

    # Query Agent Past Trajectory mod:
    # LSTM Encoder/Decoder
    lstmed_encoding_layers = [rnn_state_size]
    lstmed_exp_num = 0
    lstmed_reverse_time_prediction = False
    lstmed_consistent_time_signal = True
    lstmed_n_features = input_state_dim * (prev_horizon + 1)

    freeze_query_agent_module = False
    warm_start_query_agent_module = False
    pretrained_query_agent_module_path = "../trained_models/LSTM_ED_module/0"


    # Training process parameters
    print_freq = 200
    save_freq = 500
    total_training_steps = 20000
    dt = 0.4

    # Behaviour Flags
    warmstart_model = False
    real_world_data = False     # Seems useless
    end_to_end = False
    agents_on_grid = False      # Seems useless
    rotated_grid = False
    centered_grid = True
    noise = False
    normalize_data = False
    regulate_log_loss = False

    # Map parameters
    submap_resolution = 0.1
    submap_width = 6
    submap_height = 6

    # Miscellaneous
    diversity_update = False
    predict_positions = False
    warm_start_convnet = True
    freeze_grid_cnn = True
    debug_plotting = False

    # Dataset division
    train_set = 0.8

    # Corrections code/article
    correction_div_loss_in_total_loss = False
    correction_annealing_kl_loss = "codebase"     # can be either "codebase" or "article"

    # # Random Seed
    # rng_seed = 314159265

    parser = argparse.ArgumentParser(description='LSTM model training')

    parser.add_argument('--model_name',
                        help='Path to directory that comprises the model (default="model_name").',
                        type=str, default=model_name)
    parser.add_argument('--pretrained_convnet_path',
                        help='Path to directory that comprises the pre-trained convnet model (default=" ").',
                        type=str, default=pretrained_convnet_path)
    parser.add_argument('--scenario', help='Scenario of the dataset (default="").',
                        type=str, default=scenario)
    parser.add_argument('--real_world_data', help=f'Real world dataset.', type=sup.str2bool,
                        default=real_world_data)
    parser.add_argument('--data_path', help='Path to directory that saves pickle data (default=" ").', type=str,
                        default=data_path)
    parser.add_argument('--data_handler', help='Datahandler class needed to load the data', type=str,
                        default='LSTM')
    parser.add_argument('--warmstart_model', help='Restore from pretained model (default=False).', type=bool,
                        default=warmstart_model)
    parser.add_argument('--warm_start_convnet', help='Restore from pretained convnet model (default=False).', type=sup.str2bool,
                        default=warm_start_convnet)
    parser.add_argument('--dt', help='Data sampling time (default=0.3).', type=float,
                        default=dt)
    parser.add_argument('--n_epochs', help='Number of epochs (default=10000).', type=int, default=n_epochs)
    parser.add_argument('--total_training_steps', help='Number of training steps (default=20000).', type=int,
                        default=total_training_steps)
    parser.add_argument('--batch_size', help='Batch size for training (default=32).', type=int, default=batch_size)
    parser.add_argument('--regularization_weight', help='Weight scaling of regularizer (default=0.01).', type=float,
                        default=regularization_weight)
    parser.add_argument('--keep_prob', help='Dropout (default=0.8).', type=float,
                        default=keep_prob)
    parser.add_argument('--learning_rate_init', help='Initial learning rate (default=0.005).', type=float,
                        default=learning_rate_init)
    parser.add_argument('--beta_rate_init', help='Initial beta rate (default=0.005).', type=float,
                        default=beta_rate_init)
    parser.add_argument('--dropout', help='Enable Dropout', type=sup.str2bool,
                        default=dropout)
    parser.add_argument('--grads_clip', help='Gradient clipping (default=10.0).', type=float,
                        default=grads_clip)
    parser.add_argument('--truncated_backprop_length', help='Backpropagation length during training (default=5).',
                        type=int, default=truncated_backprop_length)
    parser.add_argument('--prediction_horizon', help='Length of predicted sequences (default=10).', type=int,
                        default=prediction_horizon)
    parser.add_argument('--prev_horizon', help='Previous seq length.', type=int,
                        default=prev_horizon)
    parser.add_argument('--rnn_state_size', help='Number of RNN / LSTM units (default=16).', type=int,
                        default=rnn_state_size)
    parser.add_argument('--rnn_state_size_lstm_ped',
                        help='Number of RNN / LSTM units of the grid lstm layer (default=32).',
                        type=int, default=rnn_state_size_lstm_ped)
    parser.add_argument('--rnn_state_ped_size',
                        help='Number of RNN / LSTM units of the grid lstm layer (default=32).',
                        type=int, default=rnn_state_ped_size)
    parser.add_argument('--rnn_state_size_lstm_grid',
                        help='Number of RNN / LSTM units of the grid lstm layer (default=32).',
                        type=int, default=rnn_state_size_lstm_grid)
    parser.add_argument('--rnn_state_size_lstm_concat',
                        help='Number of RNN / LSTM units of the concatenation lstm layer (default=32).',
                        type=int, default=rnn_state_size_lstm_concat)
    parser.add_argument('--prior_size', help='prior_size',
                        type=int, default=prior_size)
    parser.add_argument('--latent_space_size', help='latent_space_size',
                        type=int, default=latent_space_size)
    parser.add_argument('--x_dim', help='x_dim',
                        type=int, default=x_dim)
    parser.add_argument('--fc_hidden_unit_size',
                        help='Number of fully connected layer units after LSTM layer (default=64).',
                        type=int, default=fc_hidden_unit_size)
    parser.add_argument('--input_state_dim', help='Input state dimension (default=).', type=int,
                        default=input_state_dim)
    parser.add_argument('--input_dim', help='Input state dimension (default=).', type=float, default=input_dim)
    parser.add_argument('--output_dim', help='Output state dimension (default=).', type=float, default=output_dim)
    parser.add_argument('--goal_size', help='Goal dimension (default=).', type=int, default=2)
    parser.add_argument('--output_pred_state_dim', help='Output prediction state dimension (default=).', type=int,
                        default=output_pred_state_dim)
    parser.add_argument('--cmd_vector_dim', help='Command control dimension.', type=int, default=cmd_vector_dim)
    parser.add_argument('--n_mixtures', help='Number of modes (default=).', type=int, default=n_mixtures)
    parser.add_argument('--pedestrian_vector_dim', help='Number of angular grid sectors (default=72).', type=int,
                        default=pedestrian_vector_dim)
    parser.add_argument('--pedestrian_vector_state_dim', help='Number of angular grid sectors (default=2).', type=int,
                        default=pedestrian_vector_state_dim)
    parser.add_argument('--max_range_ped_grid', help='Maximum pedestrian distance (default=2).', type=float,
                        default=max_range_ped_grid)
    parser.add_argument('--pedestrian_radius', help='Pedestrian radius (default=0.3).', type=float,
                        default=pedestrian_radius)
    parser.add_argument('--n_other_agents', help='Number of other agents incorporated in the network.', type=int,
                        default=n_other_agents)
    parser.add_argument('--debug_plotting', help='Plotting for debugging (default=False).', type=sup.str2bool,
                        default=debug_plotting)
    parser.add_argument('--print_freq', help='Print frequency of training info (default=100).', type=int,
                        default=print_freq)
    parser.add_argument('--save_freq', help='Save frequency of the temporary model during training. (default=20k).',
                        type=int, default=save_freq)
    parser.add_argument('--exp_num', help='Experiment number', type=int, default=exp_num)
    parser.add_argument('--noise', help='Likelihood? (default=True).', type=sup.str2bool,
                        default=noise)
    parser.add_argument('--agents_on_grid', help='Likelihood? (default=True).', type=sup.str2bool,
                        default=agents_on_grid)
    parser.add_argument('--normalize_data', help='Normalize? (default=False).', type=sup.str2bool,
                        default=normalize_data)
    parser.add_argument('--rotated_grid', help='Rotate grid? (default=False).', type=sup.str2bool, default=rotated_grid)
    parser.add_argument('--centered_grid', help='Center grid? (default=False).', type=sup.str2bool,
                        default=centered_grid)
    parser.add_argument('--sigma_bias', help='Percentage of the dataset used for training', type=float,
                        default=0)  # incorrect help statement
    parser.add_argument('--submap_width', help='width of occupancy grid', type=int, default=submap_width)
    parser.add_argument('--submap_height', help='height of occupancy grid', type=int, default=submap_height)
    parser.add_argument('--submap_resolution', help='Map resolution.', type=float, default=submap_resolution)
    parser.add_argument('--min_buffer_size', help='Minimum buffer size (default=1000).', type=int, default=1000)
    parser.add_argument('--max_buffer_size', help='Maximum buffer size (default=100k).', type=int, default=100000)
    parser.add_argument('--max_trajectories', help='maximum number of trajectories to be recorded', type=int,
                        default=30)
    parser.add_argument('--end_to_end', help='End to end trainning.', type=sup.str2bool, default=end_to_end)
    parser.add_argument('--predict_positions', help='predict_positions.', type=sup.str2bool, default=predict_positions)
    parser.add_argument('--gpu', help='Enable GPU training.', type=sup.str2bool, default=False)
    parser.add_argument('--sequence_info', help='Use relative info for other agents.', type=sup.str2bool, default=False)
    parser.add_argument('--others_info', help='Use relative info for other agents.', type=str, default="relative")
    parser.add_argument('--regulate_log_loss', help='Enable GPU training.', type=sup.str2bool,
                        default=regulate_log_loss)
    parser.add_argument('--diversity_update', help='diversity_update', type=sup.str2bool, default=diversity_update)
    parser.add_argument('--topics_config', help='yaml file containg subscription topics (default=" ").', type=str,
                        default='../config/topics.yaml')
    parser.add_argument('--min_pos_x', help='min_pos_x', type=float, default=-1)
    parser.add_argument('--min_pos_y', help='min_pos_y', type=float, default=-1)
    parser.add_argument('--max_pos_x', help='max_pos_x', type=float, default=1)
    parser.add_argument('--max_pos_y', help='max_pos_y', type=float, default=1)
    parser.add_argument('--min_vel_x', help='min_vel_x', type=float, default=-1)
    parser.add_argument('--min_vel_y', help='min_vel_y', type=float, default=-1)
    parser.add_argument('--max_vel_x', help='max_vel_x', type=float, default=1)
    parser.add_argument('--max_vel_y', help='max_vel_y', type=float, default=1)
    parser.add_argument('--sx_vel', help='sx_vel', type=float, default=1)
    parser.add_argument('--sy_vel', help='sy_vel', type=float, default=1)
    parser.add_argument('--sx_pos', help='sx_pos', type=float, default=1)
    parser.add_argument('--sy_pos', help='sy_pos', type=float, default=1)
    parser.add_argument('--train_set', help='Percentage of the dataset used for training', type=float,
                        default=train_set)
    parser.add_argument('--tensorboard_logging', help='Whether to use tensorboard logging capability or not', type=sup.str2bool,
                        default=tensorboard_logging)

    # My added options
    parser.add_argument('--query_agent_ae_encoding_layers',
                        nargs='+',
                        help='list of integers, which specify the dimensions of the encoding layers of the Query Agent past trajectory autoencoder',
                        type=int, default=query_agent_ae_encoding_layers)
    parser.add_argument('--query_agent_ae_latent_space_dim',
                        help='dimension of the latent space of the Query Agent past trajectory autoencoder',
                        type=int, default=query_agent_ae_latent_space_dim)
    parser.add_argument('--query_agent_ae_optimizer',
                        help='can either be "Adam" of "RMSProp", specifies the optimizer for the Query Agent past trajectory autoencoder',
                        type=str, default=query_agent_ae_optimizer)
    parser.add_argument('--pretrained_qa_ae_path',
                        help=f'Path to directory that comprises the pre-trained past trajectory AE model (default: "{pretrained_qa_ae_path}").',
                        type=str, default=pretrained_qa_ae_path)
    parser.add_argument('--freeze_grid_cnn',
                        help=f'Whether the weights of the occupancy grid CNN module should be frozen while training.',
                        type=sup.str2bool, default=freeze_grid_cnn)
    parser.add_argument('--freeze_qa_ae',
                        help='whether the query agent autoencoder should have its weights frozen while training.',
                        type=sup.str2bool, default=freeze_qa_ae)

    # LSTM Encoder/Decoder
    parser.add_argument('--lstmed_encoding_layers',
                        help='a list containing the dimensions of the encoding LSTM layers of the LSTM Encoder Decoder',
                        nargs='+', type=int, default=lstmed_encoding_layers)
    parser.add_argument('--lstmed_exp_num',
                        help='experiment number for the LSTM Encoder Decoder',
                        type=int, default=lstmed_exp_num)
    parser.add_argument('--lstmed_reverse_time_prediction',
                        help='Whether the LSTM Encoder Decoder should reconstruct the input in reversed time',
                        type=sup.str2bool, default=lstmed_reverse_time_prediction)
    parser.add_argument('--lstmed_consistent_time_signal',
                        help='Whether the LSTM Encoder Decoder should keep consistent time signal values across time truncations.',
                        type=sup.str2bool, default=lstmed_consistent_time_signal)
    parser.add_argument('--lstmed_n_features',
                        help="number of features per timestep for the LSTM Encoder Decoder past trajectory module "
                             "default is equal to:\ninput_state_dim * (prev_horizon + 1)",
                        type=int, default=lstmed_n_features)
    parser.add_argument('--freeze_query_agent_module',
                        help=f'whether the query agent module should have its weights frozen while training.',
                        type=sup.str2bool, default=freeze_query_agent_module)
    parser.add_argument('--pretrained_query_agent_module_path',
                        help=f'Path to the directory containing the pre-trained LSTM ED model for the query agent (default: {pretrained_query_agent_module_path})',
                        type=str, default=pretrained_query_agent_module_path)
    parser.add_argument('--warm_start_query_agent_module',
                        help='Restore from pretrained query agent module',
                        type=sup.str2bool, default=warm_start_query_agent_module)

    # Corrections code/article
    parser.add_argument('--correction_div_loss_in_total_loss',
                        help='Correction of the total loss, as the implementation of the original codebase did not '
                             'correspond with the description made in the article. False by default '
                             '(respecting the original code implementation over the article).',
                        type=sup.str2bool, default=correction_div_loss_in_total_loss)
    parser.add_argument('--correction_annealing_kl_loss',
                        help='Correction of the Annealing KL Loss coefficient, as the implementation of the original '
                             'codebase did not correspond with the description made in the article. Can be either '
                             '"codebase" or "article". Defaults to "codebase".',
                        type=str, default=correction_annealing_kl_loss) # it seems that "codebase" is right. "article" results in failed runs

    # # RNG
    # parser.add_argument('--rng_seed',
    #                     help='Random Seed for weight initialization',
    #                     type=int, default=rng_seed)

    parsed_args = parser.parse_args()

    parsed_args.model_path = os.path.join('../trained_models/', parsed_args.model_name, str(parsed_args.exp_num))
    parsed_args.log_dir = os.path.join(parsed_args.model_path, 'log')

    # for the LSTM Encoder Decoder
    parsed_args.lstmed_n_features = parsed_args.input_state_dim * (parsed_args.prev_horizon + 1)

    return parsed_args


def prepare_model_directory(args: argparse.Namespace):
    """
    Creates the directories which will be used to store model parameters, as well as logs of the training process.
    """
    # Create Log and Model Directory to save training model
    if not os.path.exists(args.log_dir):
        print(Fore.GREEN + f"creating log directory in: {args.log_dir}" + Style.RESET_ALL)
        os.makedirs(args.log_dir)

    assert os.path.exists(args.model_path), f"The log directory should be contained within the model directory, please verify:\n" \
                                            f"args.log_dir = {args.log_dir}\n" \
                                            f"args.model_path = {args.model_path}"

    model_parameters = {"args": args}

    # Save Model Parameters
    param_file = open(args.model_path + '/model_parameters.pkl', 'wb')
    pkl.dump(model_parameters, param_file, protocol=2)  # encoding='latin1'
    param_file.close()
    with open(args.model_path + '/model_parameters.json', 'w') as f:
        json.dump(args.__dict__, f, indent=0)


def main():
    args = parse_args()
    src.data_utils.plot_utils.print_args(args)


if __name__ == '__main__':
    main()
