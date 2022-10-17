"""

Implementation of an LSTM Encoder Decoder. It will be used to process the velocity input features of the Query Agent.

"""
import argparse
import os.path
import tensorflow as tf
import numpy as np
import colorama


class LSTMEncoderDecoder:

    def __init__(self, args: argparse.Namespace):
        """
        initialise an instance of the LSTMEncoderDecoder class.

        <args> must contain:
            - lstmed_exp_num                    : the experiment number for this particular training run
            - batch_size                        : the batch size with which the model will be working
            - truncated_backprop_length         : time truncation factor for unrolling the LSTM for training
            - lstmed_n_features                 : number of features contained within one training instance
            - lstmed_encoding_layers            : list of encoding LSTM dimensions for the encoder (decoder is symmetrical)
            - lstmed_reverse_time_prediction    : bool, indicates if the model is to reconstruct the input in reverse time
        """

        # defining Filesystem relevant parameters
        self.scope_name = 'lstm_encoder_decoder'
        self.id = args.lstmed_exp_num
        self.save_path = "../trained_models/"
        self.model_name = "LSTMEncoderDecoder"
        self.model_directory = os.path.join(self.save_path, self.model_name)
        self.full_save_path = os.path.join(self.model_directory, str(self.id))

        # setting up input dimensions
        """
        an input fed to the model is a tensor of shape [B, T, D] where:
            - B is the Batch Size:                          self.batch_size
            - T is the Truncated Backpropagation Length:    self.truncated_backprop_length
            - D is the Dimension of a Training Instance:    self.n_features
        """
        self.batch_size = args.batch_size
        self.truncated_backprop_length = args.truncated_backprop_length

        # when working with the ETH/UCY dataset, it is important that:
        # args.n_features = args.input_state_dim * (args.prev_horizon + 1)
        self.n_features = args.lstmed_n_features

        # input formatting
        self.input_placeholder = tf.placeholder(dtype=tf.float32,
                                                shape=[None,
                                                       self.truncated_backprop_length,
                                                       self.n_features],
                                                name='input_state')
        self.input_series = tf.unstack(self.input_placeholder, axis=1)

        # Optimizer specifiation
        self.learning_rate = 1e-3
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

        # Reconstruction specification
        self.reverse_time_prediction = args.lstmed_reverse_time_prediction

        # Creating the architecture
        self.encoding_layers_dim = args.lstmed_encoding_layers
        self.embedding_state_size = args.lstmed_encoding_layers[-1]
        for i, dim_size in enumerate(self.encoding_layers_dim):
            setattr(self, f"enc_{i}_cell_state_current", np.zeros([self.batch_size, dim_size]))
            setattr(self, f"enc_{i}_hidden_state_current", np.zeros([self.batch_size, dim_size]))
            setattr(self, f"enc_{i}_test_cell_state_current", np.zeros([self.batch_size, dim_size]))
            setattr(self, f"enc_{i}_test_hidden_state_current", np.zeros([self.batch_size, dim_size]))
            setattr(self, f"enc_{i}_cell_state", tf.placeholder(dtype=tf.float32,
                                                                shape=[None, dim_size],
                                                                name=f'enc_{i}_cell_state'))
            setattr(self, f"enc_{i}_hidden_state", tf.placeholder(dtype=tf.float32,
                                                                  shape=[None, dim_size],
                                                                  name=f'enc_{i}_hidden_state'))
            setattr(self,
                    f"enc_{i}_init_state_tuple",
                    tf.contrib.rnn.LSTMStateTuple(getattr(self, f"enc_{i}_cell_state"),
                                                  getattr(self, f"enc_{i}_hidden_state")))

        for i, dim_size in enumerate(reversed(self.encoding_layers_dim)):
            setattr(self, f"dec_{i}_cell_state_current", np.zeros([self.batch_size, dim_size]))
            setattr(self, f"dec_{i}_hidden_state_current", np.zeros([self.batch_size, dim_size]))
            setattr(self, f"dec_{i}_test_cell_state_current", np.zeros([self.batch_size, dim_size]))
            setattr(self, f"dec_{i}_test_hidden_state_current", np.zeros([self.batch_size, dim_size]))
            setattr(self, f"dec_{i}_cell_state", tf.placeholder(dtype=tf.float32,
                                                                shape=[None, dim_size],
                                                                name=f'dec_{i}_cell_state'))
            setattr(self, f"dec_{i}_hidden_state", tf.placeholder(dtype=tf.float32,
                                                                  shape=[None, dim_size],
                                                                  name=f'dec_{i}_hidden_state'))
            setattr(self,
                    f"dec_{i}_init_state_tuple",
                    tf.contrib.rnn.LSTMStateTuple(getattr(self, f"dec_{i}_cell_state"),
                                                  getattr(self, f"dec_{i}_hidden_state")))

        with tf.variable_scope(self.scope_name) as scope:
            prev_tensor = self.input_series
            for i, dim_size in enumerate(self.encoding_layers_dim):
                setattr(self, f"enc_{i}_cell", tf.nn.rnn_cell.LSTMCell(dim_size,
                                                                       name=f'basic_lstm_cell_enc_{i}',
                                                                       trainable=True))
                lstm_h, lstm_c = tf.contrib.rnn.static_rnn(getattr(self, f"enc_{i}_cell"),
                                                           prev_tensor,
                                                           dtype=tf.float32,
                                                           initial_state=getattr(self, f"enc_{i}_init_state_tuple"))
                setattr(self, f"enc_{i}_outputs_series_state", lstm_h)
                setattr(self, f"enc_{i}_current_state", lstm_c)
                prev_tensor = getattr(self, f"enc_{i}_outputs_series_state")

            self.embedding = prev_tensor

            prev_tensor = self.embedding
            for i, dim_size in enumerate(reversed(self.encoding_layers_dim)):
                setattr(self, f"dec_{i}_cell", tf.nn.rnn_cell.LSTMCell(dim_size,
                                                                       name=f'basic_lstm_cell_dec_{i}',
                                                                       trainable=True))
                lstm_h, lstm_c = tf.contrib.rnn.static_rnn(getattr(self, f"dec_{i}_cell"),
                                                           prev_tensor,
                                                           dtype=tf.float32,
                                                           initial_state=getattr(self, f"dec_{i}_init_state_tuple"))
                setattr(self, f"dec_{i}_outputs_series_state", lstm_h)
                setattr(self, f"dec_{i}_current_state", lstm_c)
                prev_tensor = getattr(self, f"dec_{i}_outputs_series_state")

            self.time_distributed_dense = tf.layers.Dense(self.n_features, name='time_distributed_dense')

            self.output_series = []
            for out in prev_tensor:
                self.output_series.append(self.time_distributed_dense(out))

            self.output_tensor = tf.stack(self.output_series, axis=1)

            # reversing the order
            # https://arxiv.org/abs/1502.04681
            if self.reverse_time_prediction:
                self.output_tensor = tf.reverse(self.output_tensor, axis=[1])

            # checking that the reconstructed tensors are of the same shape as the inputs
            assert all(self.output_series[i].shape.as_list() == self.input_series[i].shape.as_list()
                       for i in range(self.truncated_backprop_length))
            assert self.input_placeholder.shape.as_list() == self.output_tensor.shape.as_list()

            # Loss
            reconstruction_loss = tf.squared_difference(self.input_placeholder, self.output_tensor)
            self.reconstruction_loss = tf.reduce_mean(reconstruction_loss)

            # Setting the training operation
            self.train_op = self.optimizer.minimize(self.reconstruction_loss)

        # Specifying model variables, along with variable saver and initializer
        self.model_var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope_name)
        self.saver = tf.train.Saver(self.model_var_list)
        self.initializer = tf.initialize_variables(self.model_var_list)

    def info_dict(self):
        info_dict = {"batch_size": self.batch_size,
                     "truncated_backprop_length": self.truncated_backprop_length,
                     "n_features": self.n_features,
                     "encoding_layers_dim": self.encoding_layers_dim,
                     "reverse_time_prediction": self.reverse_time_prediction}
        return info_dict

    def describe(self) -> None:
        """
        prints the relevant (hyper)parameters which define the architecture.
        """
        print(colorama.Fore.YELLOW +
              "LSTM Autoencoder hyperparameters / info:")
        for k, v in self.info_dict().items():
            print(f"{k.ljust(45)}| {v}")
        print("\nLayers")
        for var in self.model_var_list:
            print(var)
        print(colorama.Style.RESET_ALL)
        return None

    def initialize_random_weights(self, sess: tf.Session):
        """
        prompts the session to initialize the weights of the module with random values
        """
        print(colorama.Fore.YELLOW +
              "Initializing LSTM Encoder Decoder with random weights" +
              colorama.Style.RESET_ALL)
        sess.run(self.initializer)

    def feed_dic(self, input_data):
        """
        input data is a np array of shape [self.batch_size,
                                           self.truncated_backprop_length,
                                           self.n_features]
        """
        feed_dictionary = {self.input_placeholder: input_data}

        for i in range(len(self.encoding_layers_dim)):
            feed_dictionary[getattr(self, f"enc_{i}_cell_state")] = getattr(self, f"enc_{i}_cell_state_current")
            feed_dictionary[getattr(self, f"enc_{i}_hidden_state")] = getattr(self, f"enc_{i}_hidden_state_current")
            feed_dictionary[getattr(self, f"dec_{i}_cell_state")] = getattr(self, f"dec_{i}_cell_state_current")
            feed_dictionary[getattr(self, f"dec_{i}_hidden_state")] = getattr(self, f"dec_{i}_hidden_state_current")
        return feed_dictionary

    def feed_test_dic(self, input_data, state_noise):
        feed_dictionary = {self.input_placeholder: input_data}
        for i in range(len(self.encoding_layers_dim)):
            feed_dictionary[getattr(self, f"enc_{i}_cell_state")] = np.random.normal(
                getattr(self, f"enc_{i}_test_cell_state_current").copy(), np.abs(np.mean(np.mean(
                    getattr(self, f"enc_{i}_test_cell_state_current")
                )) * state_noise)
            )
            feed_dictionary[getattr(self, f"enc_{i}_hidden_state")] = np.random.normal(
                getattr(self, f"enc_{i}_test_hidden_state_current").copy(), np.abs(np.mean(np.mean(
                    getattr(self, f"enc_{i}_test_cell_state_current")
                )) * state_noise)
            )
            feed_dictionary[getattr(self, f"dec_{i}_cell_state")] = np.random.normal(
                getattr(self, f"dec_{i}_test_cell_state_current").copy(), np.abs(np.mean(np.mean(
                    getattr(self, f"dec_{i}_test_cell_state_current")
                )) * state_noise)
            )
            feed_dictionary[getattr(self, f"dec_{i}_hidden_state")] = np.random.normal(
                getattr(self, f"dec_{i}_test_hidden_state_current").copy(), np.abs(np.mean(np.mean(
                    getattr(self, f"dec_{i}_test_cell_state_current")
                )) * state_noise)
            )
        return feed_dictionary

    def feed_val_dic(self, input_data, state_noise):
        feed_dictionary = {self.input_placeholder: input_data}
        for i in range(len(self.encoding_layers_dim)):
            feed_dictionary[getattr(self, f"enc_{i}_cell_state")] = np.random.normal(
                getattr(self, f"enc_{i}_test_cell_state_current").copy(),
                np.abs(getattr(self, f"enc_{i}_test_cell_state_current") * state_noise)
            )
            feed_dictionary[getattr(self, f"enc_{i}_hidden_state")] = np.random.normal(
                getattr(self, f"enc_{i}_test_hidden_state_current").copy(),
                np.abs(getattr(self, f"enc_{i}_test_cell_state_current") * state_noise)
            )
            feed_dictionary[getattr(self, f"dec_{i}_cell_state")] = np.random.normal(
                getattr(self, f"dec_{i}_test_cell_state_current").copy(),
                np.abs(getattr(self, f"dec_{i}_test_cell_state_current") * state_noise)
            )
            feed_dictionary[getattr(self, f"dec_{i}_hidden_state")] = np.random.normal(
                getattr(self, f"dec_{i}_test_hidden_state_current").copy(),
                np.abs(getattr(self, f"dec_{i}_test_cell_state_current") * state_noise)
            )
        return feed_dictionary

    def cell_states_list(self):
        """
        Provides a list of the cell states contained within each LSTM block of the module.
        The cell states are ordered in the list in the order in which the information flows through the network
        in a forward pass of the module (ie; first encoding layer, then second, [...], second to last decoding layer,
        last decoding layer).
        """
        states_list = []

        # Warning, don't merge for loops (order of the state list matters)
        for i in range(len(self.encoding_layers_dim)):
            states_list.append(getattr(self, f"enc_{i}_current_state"))
        for i in range(len(self.encoding_layers_dim)):
            states_list.append(getattr(self, f"dec_{i}_current_state"))
        return states_list

    def update_states(self, state_list, test_states=False):
        """
        out_list is the list of cell and hidden state value obtained after performing a session run. That list of cells
        corresponds to the formatting of the output of self.cell_states_list()
        """
        attr_str = "" if test_states else "test_"

        for i in range(len(self.encoding_layers_dim)):
            setattr(self, f"enc_{i}_{attr_str}cell_state_current", state_list[i][0])
            setattr(self, f"enc_{i}_{attr_str}hidden_state_current", state_list[i][1])
            setattr(self, f"dec_{i}_{attr_str}cell_state_current", state_list[i+len(state_list)//2][0])
            setattr(self, f"dec_{i}_{attr_str}hidden_state_current", state_list[i+len(state_list)//2][1])

    def run_update_step(self, sess: tf.Session, input_data: np.array):
        """
        forward + backward pass of the module architecture. Weights are updated.

        input data is a np array of shape [self.batch_size,
                                           self.truncated_backprop_length,
                                           self.n_features]
        """

        run_list = [self.train_op, self.reconstruction_loss]
        run_list.extend(self.cell_states_list())

        out_list = sess.run(
            run_list,
            feed_dict=self.feed_dic(input_data=input_data)
        )

        reconstruction_loss = out_list[1]
        states_list = out_list[2:]

        self.update_states(states_list)

        return reconstruction_loss

    def run_val_step(self, sess: tf.Session, feed_dict_validation, update=True):
        """
        produce the loss obtained from a forward pass using input data. Weights are NOT updated.
        """
        run_list = [self.reconstruction_loss]
        run_list.extend(self.cell_states_list())

        out_list = sess.run(
            run_list,
            feed_dict=feed_dict_validation
        )

        reconstruction_loss = out_list[0]
        states_list = out_list[1:]
        if update:
            self.update_states(states_list, test_states=True)

        return reconstruction_loss

    def reconstruct(self, sess: tf.Session, input_data: np.array, update_state=True):
        """
        produce the reconstruction output of the module upon being fed the input data. Weights are NOT updated.

        input data is a np array of shape [self.batch_size,
                                           self.truncated_backprop_length,
                                           self.n_features]
        """

        run_list = [self.output_series]
        run_list.extend(self.cell_states_list())

        out_list = sess.run(
            run_list,
            feed_dict=self.feed_dic(input_data=input_data)
        )
        # reshaping the output data so that it is the same shape as the numpy array fed as input_data
        output_data = out_list[0]
        output_data = np.stack(output_data, axis=1)

        states_list = out_list[1:]

        if update_state:
            self.update_states(states_list, test_states=True)

        return output_data

    def encode(self, sess: tf.Session, input_data: np.array):
        """
        produce the embedding tensor of the module upon being fed the input data. Weights are NOT updated.

        input data is a np array of shape [self.batch_size,
                                           self.truncated_backprop_length,
                                           self.n_features]
        """
        embedding = sess.run(
            self.embedding,
            feed_dict=self.feed_dic(input_data=input_data)
        )
        return embedding

    def save_model(self, sess: tf.Session, path: str, step: int = None, filename: str = "model_ckpt"):
        """
        saves the model parameters under a specific directory specified by <path>.
        """
        assert os.path.exists(path), f"Path to save the model does not exist:\n{path}"
        # print(colorama.Fore.CYAN +
        #       f"Saving LSTM Encoder Decoder to:\n{path}" +
        #       colorama.Style.RESET_ALL)
        self.saver.save(sess=sess, save_path=os.path.join(path, filename), global_step=step)

    def load_model(self, sess: tf.Session, path: str):
        """
        loads the model parameters from a file referred to by <path>.
        """
        assert os.path.exists(path), f"Path to load the model does not exist:\n{path}"

        ckpt_ae = tf.train.get_checkpoint_state(path)

        # print(ckpt_ae.model_checkpoint_path)
        # print(ckpt_ae.all_model_checkpoint_paths[-2])     <-- latest checkpoint that is NOT the final one (final one hasn't been saved because of best loss)

        print(colorama.Fore.CYAN +
              f"Loading LSTM Encoder Decoder from:\n{ckpt_ae.all_model_checkpoint_paths[-2]}" +
              colorama.Style.RESET_ALL)
        self.saver.restore(sess, ckpt_ae.all_model_checkpoint_paths[-2])

    def reset_cells(self, sequence_reset) -> None:
        """
        Initialize the new sequences with a hidden state of zeros, the continuing sequences get assigned the previous hidden state.

        *this function is adapted from the reset_cells method of the SocialVRNN class
        """
        if np.any(sequence_reset):
            for sequence_idx in range(sequence_reset.shape[0]):
                if sequence_reset[sequence_idx] == 1:
                    for i, size in enumerate(self.encoding_layers_dim):
                        enc_layer_cell = getattr(self, f"enc_{i}_cell_state_current")
                        enc_layer_cell[sequence_idx, :] = np.zeros([1, size])
                        setattr(self, f"enc_{i}_cell_state_current", enc_layer_cell)
                        enc_layer_hidden = getattr(self, f"enc_{i}_hidden_state_current")
                        enc_layer_hidden[sequence_idx, :] = np.zeros([1, size])
                        setattr(self, f"enc_{i}_hidden_state_current", enc_layer_hidden)
                    for i, size in enumerate(reversed(self.encoding_layers_dim)):
                        dec_layer_cell = getattr(self, f"dec_{i}_cell_state_current")
                        dec_layer_cell[sequence_idx, :] = np.zeros([1, size])
                        setattr(self, f"dec_{i}_cell_state_current", dec_layer_cell)
                        dec_layer_hidden = getattr(self, f"dec_{i}_hidden_state_current")
                        dec_layer_hidden[sequence_idx, :] = np.zeros([1, size])
                        setattr(self, f"dec_{i}_hidden_state_current", dec_layer_hidden)

    def reset_test_cells(self, sequence_reset) -> None:
        """
        Initialize the new sequences with a hidden state of zeros, the continuing sequences get assigned the previous hidden state.

        *this function is adapted from the reset_cells method of the SocialVRNN class
        """
        if np.any(sequence_reset):
            for sequence_idx in range(sequence_reset.shape[0]):
                if sequence_reset[sequence_idx] == 1:
                    for i, size in enumerate(self.encoding_layers_dim):
                        enc_layer_cell = getattr(self, f"enc_{i}_test_cell_state_current")
                        enc_layer_cell[sequence_idx, :] = np.zeros([1, size])
                        setattr(self, f"enc_{i}_test_cell_state_current", enc_layer_cell)
                        enc_layer_hidden = getattr(self, f"enc_{i}_test_hidden_state_current")
                        enc_layer_hidden[sequence_idx, :] = np.zeros([1, size])
                        setattr(self, f"enc_{i}_test_hidden_state_current", enc_layer_hidden)
                    for i, size in enumerate(reversed(self.encoding_layers_dim)):
                        dec_layer_cell = getattr(self, f"dec_{i}_test_cell_state_current")
                        dec_layer_cell[sequence_idx, :] = np.zeros([1, size])
                        setattr(self, f"dec_{i}_test_cell_state_current", dec_layer_cell)
                        dec_layer_hidden = getattr(self, f"dec_{i}_test_hidden_state_current")
                        dec_layer_hidden[sequence_idx, :] = np.zeros([1, size])
                        setattr(self, f"dec_{i}_test_hidden_state_current", dec_layer_hidden)

    def update_test_hidden_state(self):
        for i in range(len(self.encoding_layers_dim)):
            setattr(self, f"enc_{i}_test_cell_state_current", getattr(self, f"enc_{i}_cell_state_current").copy())
            setattr(self, f"enc_{i}_test_hidden_state_current", getattr(self, f"enc_{i}_hidden_state_current").copy())
            setattr(self, f"dec_{i}_test_cell_state_current", getattr(self, f"dec_{i}_cell_state_current").copy())
            setattr(self, f"dec_{i}_test_hidden_state_current", getattr(self, f"dec_{i}_hidden_state_current").copy())


def train_LSTM_ED_module():
    """
    A toy example to see how the LSTM_ED module can be trained using the datahandler.
    """
    import sys
    sys.path.append(os.path.abspath(os.path.join(__file__, '../../..')))
    import src.train_WIP
    import src.data_utils.DataHandlerLSTM
    import src.data_utils.plot_utils
    import src.config.config
    import matplotlib.pyplot as plt
    import pickle as pkl
    import json

    args = src.config.config.parse_args()

    num_steps = args.total_training_steps
    log_freq = 10
    plot_show = False
    save = True

    model_name = "LSTM_ED_module"

    # save_path specified relative to src folder
    save_path = f"../trained_models/{model_name}/{args.lstmed_exp_num}"
    results_path = os.path.join(save_path, "results")

    if save:
        print(colorama.Fore.GREEN +
              f"Creating folder to save model parameters in:\n{save_path}" +
              colorama.Style.RESET_ALL)
        assert not os.path.exists(save_path)
        os.makedirs(save_path)
        print(colorama.Fore.GREEN +
              f"Creating folder to save training results in:\n{results_path}" +
              colorama.Style.RESET_ALL)
        assert not os.path.exists(results_path)
        os.makedirs(results_path)

    data_prep = src.data_utils.DataHandlerLSTM.DataHandlerLSTM(args=args)
    data_prep.processData()

    session = tf.Session()
    lstm_ae_module = LSTMEncoderDecoder(args=args)

    # initializing weights
    session.run(tf.global_variables_initializer())
    lstm_ae_module.initialize_random_weights(sess=session)

    # THIS IS FOR LOADING AN ALREADY TRAINED ARCHITECTURE
    # load_path = os.path.abspath(
    #     os.path.join(os.path.dirname(__file__), f"../../trained_models/{model_name}/{args.lstmed_exp_num}")
    # )
    # lstm_ae_module.load_model(sess=session, path=load_path)

    lstm_ae_module.describe()

    train_losses = []
    val_losses = []
    best_loss = float('inf')
    train_loss_at_best_loss = float('inf')
    best_loss_idx = 0
    # _, batch_vel, _, _, _, _, _, _, _, _ = data_prep.getBatch()
    for step in range(num_steps):
        _, batch_vel, _, _, _, _, _, _, _, _ = data_prep.getBatch()

        lstm_ae_module.reset_cells(data_prep.sequence_reset)

        loss = lstm_ae_module.run_update_step(sess=session, input_data=batch_vel)
        train_losses.append(loss.item())

        if step % log_freq == 0:
            testbatch = data_prep.getTestBatch()

            lstm_ae_module.reset_test_cells(data_prep.val_sequence_reset)

            val_loss = lstm_ae_module.run_val_step(sess=session,
                                                   feed_dict_validation=lstm_ae_module.feed_val_dic(
                                                       input_data=testbatch["batch_vel"],
                                                       state_noise=testbatch["state_noise"]
                                                   ))

            val_losses.append(val_loss.item())

            log = f"step {str(step).ljust(10)} | Training Loss: {loss:.6f} | Validation Loss: {val_loss:.6f}"

            if val_loss.item() < best_loss and save:
                lstm_ae_module.save_model(sess=session, path=save_path, step=step)
                log += " | Saved"
                best_loss = val_loss.item()
                train_loss_at_best_loss = loss.item()
                best_loss_idx = step

            print(log)

    if save:
        lstm_ae_module.save_model(sess=session, path=save_path, step=step, filename="final-model.ckpt")

    yhat = lstm_ae_module.reconstruct(sess=session, input_data=batch_vel)

    # print()
    # print('---Predicted---')
    # print(np.round(yhat, 3))
    # print('---Actual---')
    # print(np.round(batch_vel, 3))

    plt.rcParams["figure.figsize"] = (16, 12)
    fig_1 = plt.figure("Reconstruction")
    plt.scatter(np.arange(yhat.size), yhat.flatten(), label="Prediction", marker="x")
    plt.scatter(np.arange(batch_vel.size), batch_vel.flatten(), label="Ground Truth", s=40, facecolors="none", edgecolors='r')
    plt.legend()
    if save:
        plt.savefig(os.path.join(results_path, "reconstruction.png"))

    output_dict = {"train_losses": train_losses,
                   "val_losses": val_losses,
                   "num_steps": num_steps,
                   "log_freq": log_freq,
                   "dataset": os.path.basename(data_prep.scenario),
                   "best_validation_loss": best_loss,
                   "train_loss_at_best_val_loss": train_loss_at_best_loss,
                   "best_val_loss_timestep": best_loss_idx}

    param_dict = lstm_ae_module.info_dict()
    param_dict.update(
        {
            "scenario": args.scenario,
            "total_training_steps": args.total_training_steps
        }
    )

    if save:
        with open(os.path.join(save_path, "parameters.json"), 'w') as file:
            json.dump(param_dict, file, indent=4)
        with open(os.path.join(results_path, "results.json"), 'w') as file:
            json.dump(output_dict, file)
        with open(os.path.join(results_path, "results.pkl"), 'wb') as file:
            pkl.dump(output_dict, file, protocol=2)

    fig_2 = plt.figure("Loss Graph")
    plt.plot(list(range(0, num_steps, log_freq)), val_losses, label="Val Loss")
    plt.plot(list(range(0, num_steps)), train_losses, label="Train Loss")
    plt.legend()
    if save:
        plt.savefig(os.path.join(results_path, "loss_graph.png"))

    if plot_show:
        plt.show()

def work_with_toy_data():
    """
    A toy example to see how the LSTM_ED works with a small batch of fake data.
    mostly following this example:
    https://towardsdatascience.com/step-by-step-understanding-lstm-autoencoder-layers-ffab055b6352
    """
    from tqdm import tqdm

    # create fake data
    timeseries = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                           [0.1 ** 3, 0.2 ** 3, 0.3 ** 3, 0.4 ** 3, 0.5 ** 3, 0.6 ** 3, 0.7 ** 3, 0.8 ** 3,
                            0.9 ** 3]]).transpose()

    timesteps = timeseries.shape[0]
    n_features = timeseries.shape[1]

    print(f"timeseries batch example:\n{timeseries}\n")
    print(f"number of timesteps:\n{timesteps}\n")
    print(f"number of features:\n{n_features}\n")

    # temporalize our data into batches
    def temporalize(X, y, lookback):
        output_X = []
        output_y = []
        for i in range(len(X) - lookback - 1):
            t = []
            for j in range(1, lookback + 1):
                # Gather past records upto the lookback period
                t.append(X[[(i + j + 1)], :])
            output_X.append(t)
            output_y.append(y[i + lookback + 1])
        return output_X, output_y

    timesteps = 3
    X, y = temporalize(X=timeseries, y=np.zeros(len(timeseries)), lookback=timesteps)

    n_features = 2
    X = np.array(X)
    X = X.reshape((X.shape[0], timesteps, n_features))

    print(f"temporalized timeseries batch:\n{X}\n")
    print(f"shape of timeseries batch:\n{X.shape}\n")

    args = argparse.Namespace()
    args.batch_size = 5
    args.truncated_backprop_length = 3
    args.lstmed_encoding_layers = [128, 64]
    args.lstmed_n_features = 2
    args.lstmed_exp_num = 0
    args.lstmed_reverse_time_prediction = False

    print(f"instantiating the LSTM Encoder Decoder using the following arguments:")
    for k, v in vars(args).items():
        print(f"{k.ljust(45)} | {v}")
    print()

    # session, and module
    session = tf.Session()
    module = LSTMEncoderDecoder(args=args)
    session.run(tf.global_variables_initializer())

    module.initialize_random_weights(sess=session)

    module.describe()

    for i in tqdm(range(3000)):
        module.run_update_step(sess=session, input_data=X)

    yhat = module.reconstruct(sess=session, input_data=X)

    print()
    print('---Predicted---')
    print(np.round(yhat, 10))
    print('---Actual---')
    print(np.round(X, 10))


if __name__ == '__main__':
    # work_with_toy_data()
    train_LSTM_ED_module()
