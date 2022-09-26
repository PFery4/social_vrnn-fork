import argparse
import os.path
from src.data_utils import DataHandlerLSTM as dhlstm
from src.data_utils import plot_utils
import numpy as np
import tensorflow as tf
import uuid
from colorama import Fore, Style
from datetime import datetime
import json
import pickle as pkl

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
                      'latent_space_dim': 4,
                      'optimizer': 'Adam'}

    def __init__(self, args: argparse.Namespace):
        """
        the args namespace must contain:
            - 'input_state_dim': int                        --> dimensions of input state (default: 2 --> [vx vy])
            - 'truncated_backprop_length': int              --> truncated backpropagation length used for BPTT*
            - 'prev_horizon': int                           --> past observations used**
            - 'query_agent_ae_encoding_layers': list(int)   --> the ith entry is the dimension used for the ith encoding layer,
                                                                decoding layers are built symmetrically to the encoding layers.
            - 'query_agent_ae_latent_space_dim': int        --> dimension used for the latent space embedding layer
            - 'query_agent_ae_optimizer': str               --> can either be 'Adam' or 'RMSProp'

        * Although the PastTrajAE does not make use of truncated backpropagation through time (as it is not a recurrent
        module), it is meant to be incorporated within an architecture which does contain recurrent modules,
        which do make use of TBPTT for training.
        """

        self.scope_name = 'query_agent_auto_encoder'
        # self.id = "AE--" + datetime.now().strftime("%d-%m-%Y_%H:%M:%S") + "--" + uuid.uuid4().hex
        self.id = args.exp_num
        self.save_path = "../trained_models/"
        self.model_name = "PastTrajAE"
        self.model_directory = os.path.abspath(os.path.join(self.save_path, self.model_name))
        self.full_save_path = os.path.abspath(os.path.join(self.model_directory, str(self.id)))

        # Checking that the save_path of the model is not already an existing directory
        # assert not os.path.exists(self.full_save_path), f"PATH ALREADY EXISTS, CANNOT SAVE PAST_TRAJ_AE MODEL:\n" \
        #                                                 f"{self.full_save_path}"
        if not os.path.exists(self.full_save_path):
            print(Fore.GREEN + f"Creating model directory in:\n{self.full_save_path}" + Style.RESET_ALL)
            os.makedirs(self.full_save_path)

        # setting up input dimensions
        self.input_state_dim = args.input_state_dim
        self.batch_size = None
        self.truncated_backprop_length = args.truncated_backprop_length
        self.prev_horizon = args.prev_horizon
        # dimension of one training instance vvv
        self.training_instance_dim = self.input_state_dim * (self.prev_horizon + 1)

        # Hyperparameters
        self.encoding_layers_dim = args.query_agent_ae_encoding_layers
        self.latent_space_dim = args.query_agent_ae_latent_space_dim
        self.optimizer_name = args.query_agent_ae_optimizer

        assert self.optimizer_name in ['Adam', 'RMSProp'], "optimizer must be specified as either 'Adam' or 'RMSProp'."
        if self.optimizer_name == 'Adam':
            self.optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
        elif self.optimizer_name == 'RMSProp':
            self.optimizer = tf.train.RMSPropOptimizer(learning_rate=1e-4)


        assert self.latent_space_dim < self.training_instance_dim, "Latent Space dimension larger than that of the input."

        # Creating the input placeholder
        self.input_placeholder = tf.placeholder(dtype=tf.float32,
                                                shape=[self.batch_size, self.truncated_backprop_length,
                                                       self.training_instance_dim],
                                                name='input_state')  # placeholder for the query agent input

        # self.input_series = tf.unstack(self.input_placeholder, axis=1)  # list of tensors: self.truncated_backprop_length * [tf.tensor(shape=[self.batch_size, self.input_state_dim * (self.prev_horizon + 1)])]
        self.input_series = tf.reshape(self.input_placeholder, [-1, self.training_instance_dim])        # reshaping [batch_size, time_trunc_BPTT, time_series] --> [batch_size * time_trunc_BPTT, time_series]
        # print(self.input_series)

        with tf.variable_scope(self.scope_name):

            # print("input_series: ", self.input_series[0].shape)

            # creating the Encoding Layers
            # prev_tensor = self.input_series[0]
            prev_tensor = self.input_series
            for i, dim in enumerate(self.encoding_layers_dim):
                # print(f"ENCODING LAYER: {i}, {dim}")
                layername = f"encode{i}"
                setattr(self, layername, tf.contrib.layers.fully_connected(prev_tensor, dim, activation_fn=tf.nn.relu, trainable=True, scope=layername))
                prev_tensor = getattr(self, layername)

            # creating the last Encoding Layer:
            # its output will have the dimension specified by the latent_space_dim attribute
            self.latent_layer = tf.contrib.layers.fully_connected(getattr(self, layername), self.latent_space_dim, activation_fn=tf.nn.relu, trainable=True, scope="latent_layer")
            # reformatting of the output so that it can be used by the main Model
            self.latent_output = tf.reshape(self.latent_layer, shape=[-1, self.truncated_backprop_length, self.latent_space_dim])     # reshaping [batch_size * time_trunc_BPTT, time_series] --> [batch_size, time_trunc_BPTT, time_series]
            self.latent_output = tf.unstack(self.latent_output, axis=1)     # unstacking: the output needs to be fed as a list of timeshifted data to account for BPTT in the next modules of the architecture

            prev_tensor = self.latent_layer
            for i, dim in enumerate(reversed(self.encoding_layers_dim)):
                # print(f"DECODING LAYER: {i}, {dim}")
                layername = f"decode{i}"
                setattr(self, layername, tf.contrib.layers.fully_connected(prev_tensor, dim, activation_fn=tf.nn.relu, trainable=True, scope=layername))
                prev_tensor = getattr(self, layername)

            # creating the Prediction Layer:
            # its output is a tensor of the same shape as the input tensor
            self.prediction = tf.contrib.layers.fully_connected(getattr(self, layername), self.training_instance_dim, activation_fn=tf.sigmoid, trainable=True, scope="prediction")
            self.output_series = tf.reshape(tensor=self.prediction, shape=tf.shape(self.input_placeholder))

            # Loss
            reconstruction_loss = tf.squared_difference(self.input_placeholder, self.output_series)      # MSE LOSS
            self.reconstruction_loss = tf.reduce_mean(reconstruction_loss)

            # Setting the training operation
            self.train_op = self.optimizer.minimize(self.reconstruction_loss)

        self.losses = {'reconstruction_loss': self.reconstruction_loss}

        # self.model_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope_name)
        self.model_var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope_name)
        self.saver = tf.train.Saver(self.model_var_list)

        self.initializer = tf.initialize_variables(self.model_var_list)

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
                     "latent_space_dimensions": self.latent_space_dim,
                     "optimizer": self.optimizer_name}
        return info_dict

    def initialize_random_weights(self, sess: tf.Session):
        """
        prompts the session to initialize the weights of the architecture
        """
        sess.run(self.initializer)


    def describe(self):
        """
        prints the relevant (hyper)parameters which define the architecture.
        """
        print("Autoencoder hyperparameters / info:")
        for k, v in self.info_dict().items():
            print(f"{k}: {v}")

        print("\nLayers")
        for var in self.model_var_list:
            print(var)

    def feed_dic(self, input_data):
        """
        input data must be of shape [self.batch_size, self.truncated_backprop_length, self.training_instance_dim]
        """
        return {self.input_placeholder: input_data}

    def run_update_step(self, sess: tf.Session, input_data):
        """
        forward + backward pass of the model. Model weights are updated

        input data must be of shape [self.batch_size, self.truncated_backprop_length, self.training_instance_dim]
        """
        _, losses = sess.run(
            [self.train_op, self.losses],
            feed_dict=self.feed_dic(input_data=input_data)
        )
        return losses

    def run_val_step(self, sess: tf.Session, input_data):
        """
        Forward pass of the model. No update of the weights.

        input data must be of shape [self.batch_size, self.truncated_backprop_length, self.training_instance_dim]
        """
        losses = sess.run(
            [self.losses],
            feed_dict=self.feed_dic(input_data=input_data)
        )
        return losses[0]

    def reconstruct(self, sess: tf.Session, input_data):
        """
        Produce the reconstruction output of the model upon being fed the input data.

        input data must be of shape [self.batch_size, self.truncated_backprop_length, self.training_instance_dim]
        """
        pred_data = sess.run(self.output_series, feed_dict={self.input_placeholder: input_data})
        return pred_data

    def encode(self, sess: tf.Session, input_data):
        """
        Produce the latent state of the model upon being fed the input data.

        input data must be of shape [self.batch_size, self.truncated_backprop_length, self.training_instance_dim]
        """
        z = sess.run(self.latent_layer, feed_dict={self.input_placeholder: input_data})
        return z

    def save_model(self, sess: tf.Session, step=None, filename="model_ckpt"):
        """
        saves the model parameters under directory specified by 'self.full_save_path'

        input data must be of shape [self.batch_size, self.truncated_backprop_length, self.training_instance_dim]
        """
        # print(f"Saving Query Agent Autoencoder to: {self.full_save_path}")
        self.saver.save(sess=sess, save_path=os.path.join(self.full_save_path, filename), global_step=step)

    def load_model(self, sess: tf.Session, load_dir: str = None):
        """
        loads the model parameters from a folder contained within self.model_directory
        """
        if load_dir is None:
            load_dir = self.id
        load_path = os.path.join(self.model_directory, str(load_dir))
        assert os.path.exists(load_path), f"ERROR WHILE LOADING QA AE MODEL:\n{load_path}\nDOES NOT EXIST"
        ckpt_ae = tf.train.get_checkpoint_state(load_path)
        print(f"Restoring PastTraj AE: {ckpt_ae.model_checkpoint_path}")
        self.saver.restore(sess, ckpt_ae.model_checkpoint_path)

def trainAE(model: PastTrajAE, data_prep: dhlstm.DataHandlerLSTM, sess: tf.Session, num_steps=20000, log_freq=100):

    train_losses = []
    val_losses = []
    val_steps = []
    best_loss = float('inf')
    for step in range(num_steps):
        batch_x, batch_vel, batch_pos, batch_goal, batch_grid, batch_ped_grid, batch_y, batch_pos_target, other_agents_pos, new_epoch = data_prep.getBatch()

        # weights = plot_utils.get_weight_value(session=sess, weight_str='query_agent_auto_encoder/encode0/weights:0', n_weights=4)
        # print('BEGINNING:'.ljust(30), weights)

        losses = model.run_update_step(sess=sess, input_data=batch_vel)
        train_losses.append(losses["reconstruction_loss"].item())

        # weights = plot_utils.get_weight_value(session=sess, weight_str='query_agent_auto_encoder/encode0/weights:0', n_weights=4)
        # print('UPDATE:'.ljust(30), weights)


        if step % log_freq == 0:
            testbatch = data_prep.getTestBatch()
            val_loss = model.run_val_step(sess=sess, input_data=testbatch["batch_vel"])
            val_losses.append(val_loss["reconstruction_loss"].item())
            val_steps.append(step)

            # weights = plot_utils.get_weight_value(session=sess, weight_str='query_agent_auto_encoder/encode0/weights:0',
            #                                       n_weights=4)
            # print('VALIDATE:'.ljust(30), weights)

            log = f"step {step}\t| Training Loss: {losses['reconstruction_loss']:.4f}\t| Validation Loss: {val_loss['reconstruction_loss']:.4f}"

            if val_loss["reconstruction_loss"] < best_loss:
                model.save_model(sess=sess, step=step)
                best_loss = val_loss["reconstruction_loss"]
                log += "\t| Saved"

            print(log)

    model.save_model(sess=sess, step=step, filename="final-model.ckpt")

    train_losses = train_losses
    val_losses = val_losses

    output_dict = {"train_losses": train_losses,
                   "val_losses": val_losses,
                   "num_steps": num_steps,
                   "log_freq": log_freq,
                   "dataset": os.path.basename(data_prep.args.scenario)}

    return output_dict
