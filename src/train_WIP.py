import os.path
import sys
sys.path.append(os.path.abspath(os.path.join(__file__, '../..')))
import json
import importlib
# import src.data_utils.Support as sup
from src.data_utils import DataHandlerLSTM as dhlstm
from src.data_utils.plot_utils import *
from src.data_utils.utils import *
import pickle as pkl
import time
from multiprocessing.pool import ThreadPool
from colorama import Fore, Style
import tensorflow as tf
# import src.data_utils.MultiDatasetsDataHandlerLSTM as multidhlstm
from src.config.config import parse_args, prepare_model_directory


# MAIN FUNCTION ########################################################################################################


if __name__ == '__main__':
    print("Using Python " + str(sys.version_info[0]))

    # os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'

    # Parsing Arguments
    args = parse_args()
    print_args(args)

    # Enable / Disable GPU
    if not args.gpu:
        print("NO GPU: setting environment variable:\n"
              "'CUDA_VISIBLE_DEVICES'='-1'\n")
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    # preparing the directories for storing logs and model parameters
    prepare_model_directory(args)

    # Create Datahandler class
    data_prep = dhlstm.DataHandlerLSTM(args)
    # Only used to create a map from png
    # Make sure these parameters are correct otherwise it will fail training and plotting the results
    map_args = {"file_name": 'map.png',
                "resolution": 0.1,
                "map_size": np.array([30., 6.])}
    # Load dataset
    data_prep.processData(**map_args)

    # Import Deep Learning model
    module = importlib.import_module("src.models." + args.model_name)
    globals().update(module.__dict__)

    # Create Model Graph
    model = NetworkModel(args)

    # see the trainable variables
    for var in tf.trainable_variables():
        print(Fore.YELLOW + f"TRAINABLE VARIABLE OF SOCIAL_VRNN:\t{var}" + Style.RESET_ALL)

    # TensorFlow CPU and GPU configurations, see:
    # https://liyin2015.medium.com/tensorflow-cpus-and-gpus-configuration-9c223436d4ef
    config = tf.ConfigProto()

    # Start Training Session
    with tf.Session(config=config) as sess:
        # Load a pre-trained model
        if args.warmstart_model:
            model.warmstart_model(args, sess)
        else:
            # Initialize all TF variables
            sess.run(tf.global_variables_initializer())

        # Load Convnet Model
        try:
            if args.warm_start_convnet:
                model.warmstart_convnet(args, sess)
        except:
            print(Fore.RED + "Failed to initialized Convnet or Convnet does not exist" + Style.RESET_ALL)
            exit()

        # Load Query Agent Past Trajectory autoencoder
        if args.model_name == "SocialVRNN_AE" and not args.warmstart_model:
            model.warmstart_query_agent_ae(args=args, sess=sess)
        elif args.model_name == "SocialVRNN_LSTM_ED" and args.warm_start_query_agent_module and not args.warmstart_model:
            model.warmstart_query_agent_module(args=args, sess=sess)

        # if the training was interrupted load last training step index
        try:
            initial_step = int(open(args.model_path + "/tf_log", 'r').read().split('\n')[-2]) + 1
        except:
            initial_step = 1

        epoch = 0
        training_loss = []
        diversity_loss = []
        training_loss.append(0)

        # Set up multithreading for data handler
        pool = ThreadPool(1)
        res = None
        _model_prediction = []
        start_time = time.time()
        best_loss = float('inf')
        avg_training_loss = np.ones(100)


        for step in range(initial_step, args.total_training_steps):
            start_time_loop = time.time()

            # WIPCODE
            # weight_str = "lstm_encoder_decoder/rnn/basic_lstm_cell_enc_0/kernel:0"
            # print(Fore.BLUE)
            # print(f"{weight_str.ljust(70)}: " +
            #       str(get_weight_value(session=sess, weight_str=weight_str, n_weights=10)))
            # weight_str = 'auto_encoder/Conv_1/weights:0'
            # print(Fore.BLUE + f"{weight_str.ljust(70)}: " +
            #       str(get_weight_value(session=sess, weight_str=weight_str, n_weights=10)))
            # print(Style.RESET_ALL)
            # WIPCODE

            # Get Next Batch of Data
            if res == None:
                batch_x, batch_vel, batch_pos, batch_goal, batch_grid, batch_ped_grid, batch_y, batch_pos_target, other_agents_pos, new_epoch = data_prep.getBatch()
            else:
                batch = res.get(timeout=5)

            # Create dictionary to feed into the model
            dict = {"batch_x": batch_x,
                    "batch_vel": batch_vel,
                    "batch_pos": batch_pos,
                    "batch_goal": batch_goal,
                    "batch_grid": batch_grid,
                    "batch_ped_grid": batch_ped_grid,
                    "step": step,
                    "batch_y": batch_y,
                    "batch_pos_target": batch_pos_target,
                    "batch_div": batch_y,
                    "other_agents_pos": other_agents_pos
                    }

            feed_dict_train = model.feed_dic(**dict)

            # res = pool.apply_async(data_prep.getBatch)

            epoch += new_epoch

            # Initialize the new sequences with a hidden state of zeros, the continuing sequences get assigned the previous hidden state
            model.reset_cells(data_prep.sequence_reset)

            start_time_training = time.time()

            model_output = model.train_step(sess, feed_dict_train, step)

            avg_training_time = time.time() - start_time_training
            avg_loop_time = time.time() - start_time_loop

            training_loss.append(model_output["batch_loss"])

            if step == 1:
                avg_training_loss *= model_output["batch_loss"]
            else:
                avg_training_loss = np.roll(avg_training_loss, shift=1)
                avg_training_loss[0] = model_output["batch_loss"]

            # Print training info
            if step % args.print_freq == 0:

                # Get batch to compute validation loss
                validation_dict = data_prep.getTestBatch()

                model.reset_test_cells(data_prep.val_sequence_reset)

                feed_dict_validation = model.feed_val_dic(**validation_dict)

                validation_loss, validation_summary, validation_predictions = model.validation_step(sess,
                                                                                                    feed_dict_train)

                ellapsed_time = time.time() - start_time

                print(
                    Fore.BLUE +
                    "\n\nEpoch {:d}, Steps: {:d}, Train loss: {:01.2f}, Validation loss: {:01.2f}, Epoch time: {:01.2f} sec"
                    .format(epoch + 1, step, np.mean(avg_training_loss), validation_loss,
                            ellapsed_time) + Style.RESET_ALL)

                if args.tensorboard_logging:
                    model.summary_writer.add_summary(model_output["summary"], step)
                    model.summary_writer.flush()
                    model.validation_summary_writer.add_summary(validation_summary, step)
                    model.validation_summary_writer.flush()

                # Plot Global and Local Scenarios to validate datasets
                if args.debug_plotting:
                    for seq_index in range(args.batch_size):
                        for t in range(args.truncated_backprop_length):
                            data_prep.plot_global_scenario(batch_grid, batch_x, batch_y, batch_goal, other_agents_pos,
                                                           model_output["model_predictions"], t, seq_index)
                            data_prep.plot_local_scenario(batch_grid, batch_x, batch_y, batch_goal, other_agents_pos,
                                                          model_output["model_predictions"], t, seq_index)

                with open(args.model_path + "/tf_log", 'a') as f:
                    f.write(str(step) + '\n')
                curr_loss = (validation_loss + np.mean(avg_training_loss)) / 2.0
                if curr_loss < best_loss:
                    save_path = args.model_path + '/model_ckpt'
                    model.full_saver.save(sess, save_path, global_step=step)
                    best_loss = curr_loss
                    print(Fore.LIGHTCYAN_EX + 'Step {}: Saving model under {}'.format(step, save_path))

        write_summary(training_loss[-1], args)
        final_model_filename = args.model_path + '/final-model.ckpt'
        model.full_saver.save(sess, final_model_filename)
        print(Fore.LIGHTCYAN_EX +
              f"Saved final model under:\n{final_model_filename}" +
              Style.RESET_ALL)

        if args.tensorboard_logging:
            model.summary_writer.close()

    sess.close()
