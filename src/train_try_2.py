# This script can be used to try out things freely
import sys
sys.path.append('../')

import argparse
import matplotlib.pyplot as plt
import tensorflow as tf
from src.data_utils import DataHandlerLSTM as dhlstm
from src.train_WIP import parse_args, print_args
from src.models import AE_pasttraj
from src.data_utils import plot_utils
import numpy as np
import os
import pickle as pkl
import json


if __name__ == '__main__':

    # exp_num, scenario, optimizer, encoding layers, latent_space_dim

    # set number of training steps and log frequency
    num_steps = 5000
    log_freq = 1

    args = parse_args()

    print_args(args)


    # Create Datahandler class
    data_prep = dhlstm.DataHandlerLSTM(args)
    # Only used to create a map from png
    # Make sure these parameters are correct otherwise it will fail training and plotting the results
    map_args = {"file_name": 'map.png',
                "resolution": 0.1,
                "map_size": np.array([30., 6.]), }
    # Load dataset
    data_prep.processData(**map_args)

    print("Dataset: ", data_prep.args.scenario)
    print(len(data_prep.trajectory_set))
    print(len(data_prep.test_trajectory_set))

    # sess = tf.Session()
    #
    # # ae_model = AE_pasttraj.PastTrajAE(args=args)
    # ae_model = AE_pasttraj.pasttraj_ae_from_model_directory(sess=sess, model_id=os.path.basename(args.pretrained_qa_ae_path))
    #
    # weights = plot_utils.get_weight_value(session=sess, weight_str='query_agent_auto_encoder/encode0/weights:0', n_weights=5)
    # print(weights)

    # ae_model.describe()

    # sess.run(tf.global_variables_initializer())
    # ae_model.initialize_random_weights(sess)
    #
    # out_dict = AE_pasttraj.trainAE(model=ae_model,
    #                                data_prep=data_prep,
    #                                sess=sess,
    #                                num_steps=num_steps,
    #                                log_freq=log_freq)
    #
    # # preparing directory to save results
    # results_dir = os.path.join(ae_model.full_save_path, "results/")
    # if not os.path.exists(results_dir):
    #     os.makedirs(results_dir)
    #
    # # saving results
    # out_file = open(os.path.join(results_dir, "results.pkl"), "wb")
    # pkl.dump(out_dict, out_file, protocol=2)
    # with open(os.path.join(results_dir, "results.json"), "w") as f:
    #     json.dump(out_dict, f)
    #
    # lst = [444007, 444012, 444017, 444022]      # across st
    # lst = [444008, 444013, 444018, 444023]      # across zara_01
    # lst = [444009, 444014, 444019, 444024]      # across zara_02
    # lst = [444006, 444011, 444016, 444021]      # across seq_eth
    # lst = [444005, 444010, 444015, 444020]      # across seq_hotel
    # lst = [444007, 444017]      # Adam across st: narrow vs large
    # lst = [444008, 444018]      # Adam across zara_01: narrow vs large
    # lst = [444009, 444019]      # Adam across zara_02: narrow vs large
    # lst = [444006, 444016]      # Adam across seq_eth: narrow vs large
    # lst = [444005, 444015]      # Adam across seq_hotel: narrow vs large
    # lst = [444000, 444007, 444017]      # Adam across st: narrow vs large
    # lst = [444001, 444008, 444018]      # Adam across zara_01: narrow vs large
    # lst = [444002, 444009, 444019]      # Adam across zara_02: narrow vs large
    # lst = [444003, 444006, 444016]      # Adam across seq_eth: narrow vs large
    # lst = [444004, 444005, 444015]      # Adam across seq_hotel: narrow vs large
    #
    # plot_utils.compare_QA_AE_plots(lst)

    pass

