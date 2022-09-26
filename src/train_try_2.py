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

    configs = [[444005, 'real_world/ewap_dataset/seq_hotel', 'Adam', [12], 8],
               [444006, 'real_world/ewap_dataset/seq_eth', 'Adam', [12], 8],
               [444007, 'real_world/st', 'Adam', [12], 8],
               [444008, 'real_world/zara_01', 'Adam', [12], 8],
               [444009, 'real_world/zara_02', 'Adam', [12], 8],
               [444010, 'real_world/ewap_dataset/seq_hotel', 'RMSProp', [12], 8],
               [444011, 'real_world/ewap_dataset/seq_eth', 'RMSProp', [12], 8],
               [444012, 'real_world/st', 'RMSProp', [12], 8],
               [444013, 'real_world/zara_01', 'RMSProp', [12], 8],
               [444014, 'real_world/zara_02', 'RMSProp', [12], 8],
               [444015, 'real_world/ewap_dataset/seq_hotel', 'Adam', [10, 6], 2],
               [444016, 'real_world/ewap_dataset/seq_eth', 'Adam', [10, 6], 2],
               [444017, 'real_world/st', 'Adam', [10, 6], 2],
               [444018, 'real_world/zara_01', 'Adam', [10, 6], 2],
               [444019, 'real_world/zara_02', 'Adam', [10, 6], 2],
               [444020, 'real_world/ewap_dataset/seq_hotel', 'RMSProp', [10, 6], 2],
               [444021, 'real_world/ewap_dataset/seq_eth', 'RMSProp', [10, 6], 2],
               [444022, 'real_world/st', 'RMSProp', [10, 6], 2],
               [444023, 'real_world/zara_01', 'RMSProp', [10, 6], 2],
               [444024, 'real_world/zara_02', 'RMSProp', [10, 6], 2]]

    del(configs)

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

    sess = tf.Session()

    ae_model = AE_pasttraj.PastTrajAE(args=args)
    ae_model.describe()

    sess.run(tf.global_variables_initializer())
    ae_model.initialize_random_weights(sess)

    out_dict = AE_pasttraj.trainAE(model=ae_model,
                                   data_prep=data_prep,
                                   sess=sess,
                                   num_steps=num_steps,
                                   log_freq=log_freq)

    # preparing directory to save results
    results_dir = os.path.join(ae_model.full_save_path, "results/")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # saving results
    out_file = open(os.path.join(results_dir, "results.pkl"), "wb")
    pkl.dump(out_dict, out_file, protocol=2)
    with open(os.path.join(results_dir, "results.json"), "w") as f:
        json.dump(out_dict, f)
