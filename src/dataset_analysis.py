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


def centered_traj_pos(pose_vec, vel_vec, center_idx):

    centered_pos = pose_vec - pose_vec[center_idx]

    # heading = np.arctan2(vel_vec[center_idx + 1, 1], vel_vec[center_idx + 1, 0]) - np.pi/2
    heading = np.arctan2(centered_pos[center_idx + 1, 1], centered_pos[center_idx + 1, 0]) - np.pi/2

    rot_mat = np.array(
        [[np.cos(-heading), -np.sin(-heading), 0],
         [np.sin(-heading), np.cos(-heading), 0],
         [0, 0, 1]]
    )

    centered_pos = np.dot(rot_mat, centered_pos.transpose()).transpose()

    return centered_pos

def plot_trajectory(ax, centered_pose_vec, center_idx, color):
    ax.plot(centered_pose_vec[center_idx + 1:, 0], centered_pose_vec[center_idx + 1:, 1], color=color, alpha=0.5)
    # ax.plot(centered_pose_vec[:center_idx + 1, 0], centered_pose_vec[:center_idx + 1, 1], color='grey', linestyle='dashed')

def describe_motion(centered_pose_vec):
    """
    Determines if the trajectory is a turn in a direction, a straight line forward, a stop, or backward motion
    """
    idle_radius = 0.3           # [meters]
    straight_line_angle = 10        # [degrees]
    if np.linalg.norm(centered_pose_vec[-1]) < idle_radius:
        return "idle"
    elif centered_pose_vec[-1, 1] < 0:
        return "backward"
    elif np.abs(np.arctan2(centered_pose_vec[-1, 0], centered_pose_vec[-1, 1])) < straight_line_angle * np.pi/180:
        return "forward"
    elif centered_pose_vec[-1, 0] < 0:
        return "left"
    else:
        return "right"


def compute_average_curvature(centered_pose_vec):
    """
    Determines the trajectory's average curvature
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

    colors = {
        "idle": "red",
        "backward": "purple",
        "forward": "green",
        "left": "orange",
        "right": "blue"
    }

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
        # for i in range(1000):
        #     batch_x, \
        #     batch_vel, \
        #     batch_pos, \
        #     batch_goal, \
        #     batch_grid, \
        #     batch_ped_grid, \
        #     batch_y, \
        #     batch_pos_target, \
        #     other_agents_pos, \
        #     new_epoch = data_prep.getBatch()

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        fig.suptitle(f"Dataset: {data_prep.scenario}\n[ttrunc, prev_h, pred_h]: [{args.truncated_backprop_length}, {args.prev_horizon}, {args.prediction_horizon}]")
        ax1.set_title("Dataset Trajectory Overview")
        ax2.set_title("Trajectory Lengths")
        ax3.set_title("Training Instances Directions")
        sup.plot_grid(
            ax1,
            np.array([0, 0]),
            data_prep.agent_container.occupancy_grid.gridmap,
            data_prep.agent_container.occupancy_grid.resolution,
            data_prep.agent_container.occupancy_grid.map_size
        )
        ax1.set_aspect('equal')
        ax3.set_aspect('equal')

        counter = 0
        training_instances = 0
        kept_trajectories = 0
        discarded_trajectories = 0

        directions_instances = {}

        for agent_id, agent_data_obj in data_prep.agent_container.agent_data.items():
            assert data_prep.agent_container.getNumberOfTrajectoriesForAgent(agent_id) == 1
            agent_in_traj_set = agent_id in [item[0] for item in data_prep.trajectory_set]
            trajectory_long_enough = args.prev_horizon + args.truncated_backprop_length + args.prediction_horizon + 1 < len(agent_data_obj.trajectories[0])
            assert agent_in_traj_set == trajectory_long_enough

            if agent_in_traj_set:
                color = "red"
                kept_trajectories += 1
                cont = False
            else:
                color = "grey"
                discarded_trajectories += 1
                cont = True

            agent_data_obj.plot(ax1, color=color, x_scale=1, y_scale=1)

            if cont: continue

            start_idx = args.prev_horizon
            while start_idx + args.truncated_backprop_length + args.prediction_horizon + 1 < len(agent_data_obj.trajectories[0]):
                begin_idx = start_idx-args.prev_horizon
                end_idx = start_idx+args.truncated_backprop_length+args.prediction_horizon+1

                time_segment = agent_data_obj.trajectories[0].time_vec[begin_idx:end_idx]
                pose_segment = agent_data_obj.trajectories[0].pose_vec[begin_idx:end_idx]
                vel_segment = agent_data_obj.trajectories[0].vel_vec[begin_idx:end_idx]

                prev_pred_split_idx = args.prev_horizon + args.truncated_backprop_length

                prev_pose_segment = pose_segment[:prev_pred_split_idx+1]
                prev_vel_segment = vel_segment[:prev_pred_split_idx+1]

                start_idx += args.truncated_backprop_length

                centered_pose_vec = centered_traj_pos(pose_segment, vel_segment, prev_pred_split_idx)

                direction = describe_motion(centered_pose_vec)
                directions_instances.setdefault(direction, 0)
                directions_instances[direction] += 1

                color = colors[direction]

                plot_trajectory(ax3, centered_pose_vec, center_idx=prev_pred_split_idx, color=color)
                training_instances += 1

            if counter == -1:
                break

        position = ax3.get_ylim()[1] - 0.3
        ax3.text(x=ax3.get_xlim()[0] + 0.2, y=position, s=f"total: {training_instances}")
        position -= 0.3
        for dir, amount in directions_instances.items():
            ax3.text(x=ax3.get_xlim()[0] + 0.2, y=position, s=f"{dir}: {amount}", color=colors[dir])
            position -= 0.3

        traj_len_indices = data_prep.agent_container.get_trajectory_length_dict()
        for key, value in traj_len_indices.items():
            if key > data_prep.min_length_trajectory:
                color = "red"
            else:
                color = "grey"
            ax2.bar(x=key, height=value, color=color)

        total_trajectories = kept_trajectories + discarded_trajectories
        ax2.text(x=0.2*(ax2.get_xlim()[1]-ax2.get_xlim()[0]), y=ax2.get_ylim()[1]-1, s=f"Kept: {kept_trajectories} ({kept_trajectories/total_trajectories*100:.2f}% of total dataset)", color="red")
        ax2.text(x=0.2*(ax2.get_xlim()[1]-ax2.get_xlim()[0]), y=ax2.get_ylim()[1]-2, s=f"Discarded: {discarded_trajectories}", color="grey")
        ax2.text(x=0.2*(ax2.get_xlim()[1]-ax2.get_xlim()[0]), y=ax2.get_ylim()[1]-3, s=f"Total: {total_trajectories}")

    plt.show()

