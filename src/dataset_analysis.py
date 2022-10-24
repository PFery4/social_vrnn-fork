import os.path
import sys

import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(__file__, '../..')))
import src.config.config as config
import src.data_utils.plot_utils as plot_utils
import src.data_utils.DataHandlerLSTM as dhlstm
import src.data_utils.Support as sup
import numpy as np
import math


def centered_traj_pos(pose_vec, vel_vec):
    # TODO: FINISH TRAJECTORY ORIENTATION
    heading = math.atan2(vel_vec[0, 0], vel_vec[0, 1])
    centered_pos = pose_vec - pose_vec[0]
    return

def plot_trajectory(ax, trajectory_object):
    """
    plots the trajectory
    """
    centered_pose_vec = centered_traj_pos(trajectory_object.pose_vec)
    ax.plot(centered_pose_vec[:, 0], centered_pose_vec[:, 1], color="blue")

def describe_motion(trajectory_object):
    """
    Determines if the trajectory is a turn in a direction, a straight line, a stop, or something else
    """
    pass


if __name__ == '__main__':
    # Parsing arguments and printing a summary
    args = config.parse_args()
    plot_utils.print_args(args)

    scenarii = [
        "real_world/ewap_dataset/seq_hotel",
        "real_world/ewap_dataset/seq_eth",
        "real_world/st",
        "real_world/zara_01",
        "real_world/zara_02"
    ]
    args_list = []
    data_preps = []

    for scenario in scenarii:
        args.scenario = scenario

        data_prep = dhlstm.DataHandlerLSTM(args)

        # Only used to create a map from png
        # Make sure these parameters are correct otherwise it will fail training and plotting the results
        map_args = {"file_name": 'map.png',
                    "resolution": 0.1,
                    "map_size": np.array([30., 6.])}
        # Load dataset
        data_prep.processData(**map_args)

        # # printing a summary of the attributes of the DataHandler object
        # data_prep.describe()
        data_preps.append(data_prep)

        print(f"created datahandler with scenario: {data_prep.scenario}")

    for data_prep_idx, data_prep in enumerate(data_preps):
        print("LOOKING AT DATAPREP: ", data_prep_idx)

        # Sorting the trajectory set
        # data_prep.trajectory_set.sort(key=lambda tup: tup[0])
        # traj_id_list = []
        # for id, traj in data_prep.trajectory_set:
        #     traj_id_list.append(id)
        # print(traj_id_list)

        # # To get a Batch of Data:
        # batch_x, \
        # batch_vel, \
        # batch_pos, \
        # batch_goal, \
        # batch_grid, \
        # batch_ped_grid, \
        # batch_y, \
        # batch_pos_target, \
        # other_agents_pos, \
        # new_epoch = data_prep.getBatch()

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        sup.plot_grid(
            ax1,
            np.array([0, 0]),
            data_prep.agent_container.occupancy_grid.gridmap,
            data_prep.agent_container.occupancy_grid.resolution,
            data_prep.agent_container.occupancy_grid.map_size
        )
        ax1.set_xlim([
            -data_prep.agent_container.occupancy_grid.center[0],
            data_prep.agent_container.occupancy_grid.center[0]
        ])
        ax1.set_ylim([
            -data_prep.agent_container.occupancy_grid.center[1],
            data_prep.agent_container.occupancy_grid.center[1]
        ])
        ax1.set_aspect('equal')

        trajectory_lengths = []
        counter = 0
        for agent_id, agent_data_obj in data_prep.agent_container.agent_data.items():
            assert data_prep.agent_container.getNumberOfTrajectoriesForAgent(agent_id) == 1
            if agent_id in [item[0] for item in data_prep.trajectory_set]:
                color = "red"
            else:
                color = "grey"
            agent_data_obj.plot(ax1, color=color, x_scale=1, y_scale=1)
            trajectory_lengths.append([agent_id, len(agent_data_obj.trajectories[0])])

            print(agent_data_obj.trajectories[0].vel_vec.shape)
            print(agent_data_obj.trajectories[0].vel_vec)
            print(agent_data_obj.trajectories[0].time_vec.shape)
            print(agent_data_obj.trajectories[0].time_vec)
            print("\n\n\n")
            plot_trajectory(ax3, agent_data_obj.trajectories[0])

            if counter == 10:
                break
            counter += 1

        traj_len_indices = data_prep.agent_container.get_trajectory_length_dict()

        for key, value in traj_len_indices.items():
            if key > data_prep.min_length_trajectory:
                color = "blue"
            else:
                color = "red"
            ax2.bar(x=key, height=value, color=color)
        plt.show()

    # centered_batch_vel = plot_utils.centered_batch_pos_from_vel(batch_vel)
    # centered_batch_pos = plot_utils.centered_batch_pos(batch_pos)
    # plot_utils.plot_batch_vel_and_pos(centered_batch_vel,
    #                                   centered_batch_pos,
    #                                   block=True)
