"""

Implementation of an LSTM Encoder Decoder. It will be used to process the velocity input features of the Query Agent.

"""
import sys
sys.path.append("../")
sys.path.append("../../")
import argparse
import os.path
import tensorflow as tf
import numpy as np
import colorama


class LSTMEncoderDecoder:

    def __init__(self, args: argparse.Namespace):

        # defining Filesystem relevant parameters
        self.scope_name = 'lstm_encoder_decoder'
        self.id = 0
        self.save_path = "../trained_models/"
        self.model_name = "LSTMEncoderDecoder"
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
        self.batch_size = args.batch_size
        self.truncated_backprop_length = args.truncated_backprop_length
        self.prev_horizon = args.prev_horizon
        self.n_features = self.input_state_dim * (self.prev_horizon + 1)

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

        # Creating the architecture
        self.encoding_layers_dim = args.encoding_layers_dim
        for i, dim_size in enumerate(self.encoding_layers_dim):
            setattr(self, f"enc_{i}_cell_state_current", np.zeros([self.batch_size, dim_size]))
            setattr(self, f"enc_{i}_hidden_state_current", np.zeros([self.batch_size, dim_size]))
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
        info_dict = {"input_state_dim": self.input_state_dim,
                     "batch_size": self.batch_size,
                     "truncated_backprop_length": self.truncated_backprop_length,
                     "prev_horizon": self.prev_horizon,
                     "n_features": self.n_features}
        return info_dict

    def describe(self) -> None:
        """
        prints the relevant (hyper)parameters which define the architecture.
        """
        print("LSTM Autoencoder hyperparameters / info:")
        for k, v in self.info_dict().items():
            print(f"{k.ljust(45)}| {v}")
        print("\nLayers")
        for var in self.model_var_list:
            print(var)
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

    def run_update_step(self, sess: tf.Session, input_data: np.array):
        """
        forward + backward pass of the module architecture. Weights are updated.

        input data is a np array of shape [self.batch_size,
                                           self.truncated_backprop_length,
                                           self.n_features]
        """
        _, reconstruction_loss = sess.run(
            [self.train_op, self.reconstruction_loss],
            feed_dict=self.feed_dic(input_data=input_data)
        )
        return reconstruction_loss

    def run_val_step(self, sess: tf.Session, input_data: np.array):
        """
        produce the loss obtained from a forward pass using input data. Weights are NOT updated.

        input data is a np array of shape [self.batch_size,
                                           self.truncated_backprop_length,
                                           self.n_features]
        """
        reconstruction_loss = sess.run(
            [self.reconstruction_loss],
            feed_dict=self.feed_dic(input_data=input_data)
        )
        return reconstruction_loss

    def reconstruct(self, sess: tf.Session, input_data: np.array):
        """
        produce the reconstruction output of the module upon being fed the input data. Weights are NOT updated.

        input data is a np array of shape [self.batch_size,
                                           self.truncated_backprop_length,
                                           self.n_features]
        """
        output_data = sess.run(
            self.output_series,
            feed_dict=self.feed_dic(input_data=input_data)
        )
        # reshaping the output data so that it is the same shape as the numpy array fed as input_data
        output_data = np.stack(output_data, axis=1)
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

        print(colorama.Fore.CYAN +
              f"Saving LSTM Encoder Decoder to:\n{path}" +
              colorama.Style.RESET_ALL)
        self.saver.save(sess=sess, save_path=os.path.join(path, filename), global_step=step)

    def load_model(self, sess: tf.Session, path: str):
        """
        loads the model parameters from a file referred to by <path>.
        """
        assert os.path.exists(path), f"Path to load the model does not exist:\n{path}"

        ckpt_ae = tf.train.get_checkpoint_state(path)
        print(colorama.Fore.CYAN +
              f"Loading LSTM Encoder Decoder from:\n{ckpt_ae.model_checkpoint_path}" +
              colorama.Style.RESET_ALL)
        self.saver.restore(sess, ckpt_ae.model_checkpoint_path)


def work_with_data_prep():
    """
    A toy example to see how the LSTM_ED module can be trained using the datahandler.
    """
    import src.train_WIP
    import src.data_utils.DataHandlerLSTM

    args = src.train_WIP.parse_args()

    data_prep = src.data_utils.DataHandlerLSTM.DataHandlerLSTM(args=args)
    data_prep.processData()

    lstm_ae_module = LSTMEncoderDecoder(args=args)

    batch_x, \
    batch_vel, \
    batch_pos, \
    batch_goal, \
    batch_grid, \
    batch_ped_grid, \
    batch_y, \
    batch_pos_target, \
    other_agents_pos, \
    new_epoch = data_prep.getBatch()


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
    args.input_state_dim = 1
    args.prev_horizon = 1       # --> such that combined with input_state_dim, n_features becomes 2
    args.batch_size = 5
    args.truncated_backprop_length = 3
    args.encoding_layers_dim = [128, 64]

    print(f"instantiating the LSTM Encoder Decoder using the following arguments:")
    for k, v in vars(args).items():
        print(f"{k.ljust(45)} | {v}")
    print()

    # session, and module
    session = tf.Session()
    module = LSTMEncoderDecoder(args=args)
    session.run(tf.global_variables_initializer())

    module.initialize_random_weights(sess=session)

    for i in tqdm(range(3000)):
        module.run_update_step(sess=session, input_data=X)

    yhat = module.reconstruct(sess=session, input_data=X)

    print()
    print('---Predicted---')
    print(np.round(yhat, 3))
    print('---Actual---')
    print(np.round(X, 3))


if __name__ == '__main__':
    # work_with_data_prep()
    work_with_toy_data()
