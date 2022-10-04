"""

Implementation of an LSTM Autoencoder. It will be used to process the velocity input features of the Query Agent.

"""
import sys
sys.path.append("../")
sys.path.append("../../")

import argparse
import os.path
import tensorflow as tf


class LSTMAutoEncoder:

    def __init__(self, args: argparse.Namespace):

        # defining Filesystem relevant parameters
        self.scope_name = 'lstm_autoencoder'
        self.id = 0
        self.save_path = "../trained_models/"
        self.model_name = "LSTMAutoEncoder"
        self.model_directory = os.path.abspath(os.path.join(self.save_path, self.model_name))
        self.full_save_path = os.path.abspath(os.path.join(self.model_directory, str(self.id)))

        # setting up input dimensions
        """
        an input fed to the model is a tensor of shape [B, T, D] where:
            - B is the Batch Size:                          self.batch_size
            - T is the Truncated Backpropagation Length:    self.truncated_backprop_length
            - D is the Dimension of a Training Instance:    self.n_features
        """
        self.input_state_dim = args.input_state_dim
        self.batch_size = None
        self.truncated_backprop_length = args.truncated_backprop_length
        self.prev_horizon = args.prev_horizon
        self.n_features = self.input_state_dim * (self.prev_horizon + 1)

        # Hyperparameters

        # Architecture implementation
        self.input_placeholder = tf.placeholder(dtype=tf.float32,
                                                shape=[self.batch_size,
                                                       self.truncated_backprop_length,
                                                       self.n_features],
                                                name='input_state')

        self.input_series = tf.unstack(self.input_placeholder, axis=1)

        # WIP DEBUG CODE
        self.describe()
        # WIP DEBUG CODE

    def info_dict(self):
        info_dict = {"input_state_dim": self.input_state_dim,
                     "batch_size": self.batch_size,
                     "truncated_backprop_length": self.truncated_backprop_length,
                     "prev_horizon": self.prev_horizon,
                     "training_instance_dim": self.n_features}
        return info_dict

    def describe(self):
        """
        prints the relevant (hyper)parameters which define the architecture.
        """
        print("LSTM Autoencoder hyperparameters / info:")
        for k, v in self.info_dict().items():
            print(f"{k.ljust(45)}| {v}")
        # print("\nLayers")

    def initialize_random_weights(self):
        pass

    def feed_dic(self):
        pass

    def run_update_step(self):
        pass

    def run_val_step(self):
        pass

    def reconstruct(self):
        pass

    def encode(self):
        pass

    def save_model(self):
        pass

    def load_model(self):
        pass


if __name__ == '__main__':
    import src.train_WIP
    import src.data_utils.DataHandlerLSTM

    args = src.train_WIP.parse_args()

    data_prep = src.data_utils.DataHandlerLSTM.DataHandlerLSTM(args=args)
    data_prep.processData()

    lstm_ae_module = LSTMAutoEncoder(args=args)

    batch_x,\
    batch_vel,\
    batch_pos,\
    batch_goal,\
    batch_grid,\
    batch_ped_grid,\
    batch_y,\
    batch_pos_target,\
    other_agents_pos,\
    new_epoch = data_prep.getBatch()

    print("batch_vel.shape: ", batch_vel.shape)
    print("batch_vel[6, :2]:", batch_vel[6, :2])
