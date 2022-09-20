# This script can be used to try out things freely
import matplotlib.pyplot as plt
import tensorflow as tf
from src.data_utils import DataHandlerLSTM as dhlstm
from src.train import parse_args, summarise_args
from src.models import AE_pasttraj
from colorama import Fore, Style
import numpy as np


def trainAE(model: AE_pasttraj.PastTrajAE, sess: tf.Session, num_steps=1000, log_freq=10):

    train_losses = []
    val_losses = []
    val_steps = []
    best_loss = float('inf')
    for step in range(num_steps):
        batch_x, batch_vel, batch_pos, batch_goal, batch_grid, batch_ped_grid, batch_y, batch_pos_target, other_agents_pos, new_epoch = data_prep.getBatch()

        losses = model.run_update_step(sess=sess, input_data=batch_vel)
        train_losses.append(losses["reconstruction_loss"])

        if step % log_freq == 0:
            testbatch = data_prep.getTestBatch()
            test_loss = model.run_val_step(sess=sess, input_data=testbatch["batch_vel"])
            val_losses.append(test_loss["reconstruction_loss"])
            val_steps.append(step)

            log = f"step {step}\t| Training Loss: {losses['reconstruction_loss']:.4f}\t| Validation Loss: {test_loss['reconstruction_loss']:.4f}"

            if test_loss["reconstruction_loss"] < best_loss:
                model.save_model(sess=sess, step=step)
                best_loss = test_loss["reconstruction_loss"]
                log += "\t| Saved"

            print(log)

    model.save_model(sess=sess, step=step, filename="final-model.ckpt")

    output_dict = {"train_losses": train_losses,
                   "num_steps": num_steps,
                   "val_losses": val_losses,
                   "val_steps": val_steps}
    return output_dict


if __name__ == '__main__':

    args = parse_args()

    summarise_args(args)

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

    model_config = {'input_state_dim': args.input_state_dim,
                    'truncated_backprop_length': args.truncated_backprop_length,
                    'prev_horizon': args.prev_horizon,
                    'encoding_layers_dim': [8],
                    'latent_space_dim': 4}

    for k, v in model_config.items():
        print(k, v)

    # batch_x, batch_vel, batch_pos, batch_goal, batch_grid, batch_ped_grid, batch_y, batch_pos_target, other_agents_pos, new_epoch = data_prep.getBatch()
    #
    # print(batch_vel[6, :, :])
    # print()
    # print(batch_pos[6, :, :])
    # print()
    # print(batch_pos[6, :, :] - 0.4 * batch_vel[6, :, :])

    ae_model = AE_pasttraj.PastTrajAE(config=model_config, sess=sess)
    ae_model.describe()

    out_dict = trainAE(ae_model, sess=sess, num_steps=20000)

    plt.plot(out_dict["val_steps"], out_dict["val_losses"])
    plt.plot(list(range(out_dict["num_steps"])), out_dict["train_losses"])
    plt.show()

