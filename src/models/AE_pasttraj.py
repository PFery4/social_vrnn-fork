import argparse
import os.path
import tensorflow as tf
import uuid
from colorama import Fore, Style
from datetime import datetime
import json

class PastTrajAE:

    """
    This is an AutoEncoder, which works with the past trajectory of the Query Agent as its input.
    The latent variable of the autoencoder can be used for further processing of the main motion prediction model.

    It is built to work with the input as formatted by the DataHandler class (see in ../data_utils/DataHandlerLSTM.py).
    """

    default_config = {'input_state_dim': 2,
                      'truncated_backprop_length': 3,
                      'prev_horizon': 7,
                      'encoding_layers_dim': [8],
                      'latent_space_dim': 4}

    def __init__(self, config: dict = None, sess: tf.Session = tf.Session()):
        """
        the config dictionary must contain:
            - 'input_state_dim': int            --> dimensions of input state (default: 2 --> [vx vy])
            - 'truncated_backprop_length': int  --> truncated backpropagation length used for BPTT*
            - 'prev_horizon': int               --> past observations used**
            - 'encoding_layers_dim': list(int)  --> the ith entry is the dimension used for the ith encoding layer
            - 'latent_space_dim': int           --> dimension used for the latent space embedding layer

        * Although the PastTrajAE does not make use of truncated backpropagation through time (as it is not a recurrent
        module), it is meant to be incorporated within an architecture which does contain recurrent modules,
        which do make use of TBPTT for training.
        """
        self.config = config
        self.scope_name = 'query_agent_auto_encoder'
        self.id = "AE--" + datetime.now().strftime("%d-%m-%Y_%H:%M:%S") + "--" + uuid.uuid4().hex
        self.save_path = "../trained_models/"
        self.model_name = "PastTrajAE"
        self.full_save_path = os.path.abspath(os.path.join(self.save_path, self.model_name, self.id))

        # Checking that the save_path of the model is not already an existing directory
        assert not os.path.exists(self.full_save_path), f"PATH ALREADY EXISTS, CANNOT SAVE PAST_TRAJ_AE MODEL:\n" \
                                                        f"{self.full_save_path}"
        print(Fore.GREEN + f"Creating model directory in:\n{self.full_save_path}" + Style.RESET_ALL)
        os.makedirs(self.full_save_path)

        # setting up input dimensions
        self.input_state_dim = self.config["input_state_dim"]
        self.batch_size = None
        self.truncated_backprop_length = self.config["truncated_backprop_length"]
        self.prev_horizon = self.config["prev_horizon"]
        # dimension of one training instance vvv
        # self.training_instance_dim = self.input_state_dim * (self.prev_horizon + 1)
        self.training_instance_dim = self.input_state_dim * (self.prev_horizon + self.truncated_backprop_length)

        # Hyperparameters
        self.encoding_layers_dim = self.config["encoding_layers_dim"]
        self.latent_space_dim = self.config["latent_space_dim"]
        self.optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
        # self.optimizer = tf.train.RMSPropOptimizer(learning_rate=1e-4)

        assert self.latent_space_dim < self.training_instance_dim, "Latent Space dimension larger than that of the input"

        # Creating the input placeholder
        self.input_placeholder = tf.placeholder(dtype=tf.float32,
                                                shape=[self.batch_size, self.truncated_backprop_length,
                                                       self.training_instance_dim],
                                                name='input_state')  # placeholder for the query agent input
        self.input_placeholder = tf.placeholder(dtype=tf.float32,
                                                shape=[self.batch_size, ])
        self.input_series = tf.unstack(self.input_placeholder, axis=1)  # list of tensors: self.truncated_backprop_length * [tf.tensor(shape=[self.batch_size, self.input_state_dim * (self.prev_horizon + 1)])]

        with tf.variable_scope(self.scope_name):

            print("input_series: ", self.input_series[0].shape)

            # creating the Encoding Layers
            prev_tensor = self.input_series[0]
            for i, dim in enumerate(self.encoding_layers_dim):
                # print(f"ENCODING LAYER: {i}, {dim}")
                layername = f"encode{i}"
                setattr(self, layername, tf.contrib.layers.fully_connected(prev_tensor, dim, activation_fn=tf.nn.relu, trainable=True, scope=layername))
                prev_tensor = getattr(self, layername)

            # creating the last Encoding Layer:
            # its output will have the dimension specified by the latent_space_dim attribute
            self.latent_layer = tf.contrib.layers.fully_connected(getattr(self, layername), self.latent_space_dim, activation_fn=tf.nn.relu, trainable=True, scope="latent_layer")

            prev_tensor = self.latent_layer
            for i, dim in enumerate(reversed(self.encoding_layers_dim)):
                # print(f"DECODING LAYER: {i}, {dim}")
                layername = f"decode{i}"
                setattr(self, layername, tf.contrib.layers.fully_connected(prev_tensor, dim, activation_fn=tf.nn.relu, trainable=True, scope=layername))
                prev_tensor = getattr(self, layername)

            # creating the Prediction Layer:
            # its output is a tensor of the same shape as the input tensor
            self.prediction = tf.contrib.layers.fully_connected(getattr(self, layername), self.training_instance_dim, activation_fn=tf.sigmoid, trainable=True, scope="prediction")

            # Loss
            reconstruction_loss = tf.squared_difference(self.input_series[0], self.prediction)      # MSE LOSS
            self.reconstruction_loss = tf.reduce_mean(reconstruction_loss)

            # Setting the training operation
            self.train_op = self.optimizer.minimize(self.reconstruction_loss)

        self.losses = {'reconstruction_loss': self.reconstruction_loss}

        self.model_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope_name)
        self.saver = tf.train.Saver(self.model_var_list)

        self.initializer = tf.initialize_variables(self.model_var_list)
        sess.run(self.initializer)

        # Save the Model Parameters
        with open(os.path.join(self.full_save_path, "parameters.json"), 'w') as f:
            json.dump(self.info_dict(), f)

    def info_dict(self):
        """
        provides a dictionary containing relevant information and hyperparameters used for the model
        """
        info_dict = {"input_state_dimensions": self.input_state_dim,
                     "batch_size": self.batch_size,
                     "truncated_backprop_length": self.truncated_backprop_length,
                     "prev_horizon": self.prev_horizon,
                     "training_instance_dimensions": self.training_instance_dim,
                     "encoding_layers": self.encoding_layers_dim,
                     "latent_space_dimensions": self.latent_space_dim}
        return info_dict

    def describe(self):
        """
        prints the relevant (hyper)parameters which define the architecture.
        """
        for k, v in self.info_dict().items():
            print(f"{k}: {v}")

        print("Layers:")
        for var in self.model_var_list:
            print(var)

    def feed_dic(self, input_data):
        return {self.input_placeholder: input_data}

    def run_update_step(self, sess: tf.Session, input_data):
        """
        forward + backward pass of the model. Model weights are updated
        """
        _, losses = sess.run(
            [self.train_op, self.losses],
            feed_dict=self.feed_dic(input_data=input_data)
        )
        return losses

    def run_val_step(self, sess: tf.Session, input_data):
        """
        Forward pass of the model. No update of the weights.
        """
        losses = sess.run(
            [self.losses],
            feed_dict=self.feed_dic(input_data=input_data)
        )
        return losses[0]

    def reconstruct(self, sess: tf.Session, input_data):
        """
        Produce the reconstruction output of the model upon being fed the input data.
        """
        pred_data = sess.run(self.prediction, feed_dict={self.input_placeholder: input_data})
        return pred_data

    def encode(self, sess: tf.Session, input_data):
        """
        Produce the latent state of the model upon being fed the input data.
        """
        z = sess.run(self.latent_layer, feed_dict={self.input_placeholder: input_data})
        return z

    def save_model(self, sess: tf.Session, step=None, filename="model_ckpt"):
        """
        saves the model parameters under self.full_save_path
        """
        # print(f"Saving Query Agent Autoencoder to: {self.full_save_path}")
        self.saver.save(sess=sess, save_path=os.path.join(self.full_save_path, filename), global_step=step)
