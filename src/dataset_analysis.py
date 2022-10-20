import os.path
import sys
sys.path.append(os.path.abspath(os.path.join(__file__, '../..')))
import src.config.config as config
import src.data_utils.plot_utils as plot_utils
import src.data_utils.DataHandlerLSTM as dhlstm
import numpy as np


if __name__ == '__main__':
    # Parsing arguments and printing a summary
    args = config.parse_args()
    plot_utils.print_args(args)

    data_prep = dhlstm.DataHandlerLSTM(args)

    # Only used to create a map from png
    # Make sure these parameters are correct otherwise it will fail training and plotting the results
    map_args = {"file_name": 'map.png',
                "resolution": 0.1,
                "map_size": np.array([30., 6.])}
    # Load dataset
    data_prep.processData(**map_args)

    # printing a summary of the attributes of the DataHandler object
    data_prep.describe()

    # # To get a Batch of Data:
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

    # Loading the first trajectory
    # traj_id_list = []
    # for id, traj in data_prep.trajectory_set:
    #     traj_id_list.append(id)
    # print(sorted(traj_id_list))

    centered_batch_vel = plot_utils.centered_batch_pos_from_vel(batch_vel)
    centered_batch_pos = plot_utils.centered_batch_pos(batch_pos)
    plot_utils.plot_batch_vel_and_pos(centered_batch_vel,
                                      centered_batch_pos,
                                      block=True)
