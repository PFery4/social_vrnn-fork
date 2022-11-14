import os.path
import sys

import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(__file__, '../..')))
import src.config.config as config
import src.data_utils.plot_utils as plot_utils
import src.data_utils.DataHandlerLSTM as dhlstm
import src.data_utils.Support as sup
import numpy as np


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

    dir_colors = {
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
        print(f"PROCESSING DATAPREP: {data_prep.scenario}")

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

        fig1, (ax1) = plt.subplots(1, 1)
        fig2, (ax2, ax3) = plt.subplots(1, 2)
        fig3, (ax4, ax5) = plt.subplots(1, 2)
        fig4, (ax6, ax7, ax8) = plt.subplots(1, 3)
        fig5, (ax9) = plt.subplots(1, 1)

        fig1.suptitle(f"Scene - Dataset: {data_prep.scenario}\n[ttrunc, prev_h, pred_h]: [{args.truncated_backprop_length}, {args.prev_horizon}, {args.prediction_horizon}]")
        fig2.suptitle(f"Training Instances - Dataset: {data_prep.scenario}\n[ttrunc, prev_h, pred_h]: [{args.truncated_backprop_length}, {args.prev_horizon}, {args.prediction_horizon}]")
        fig3.suptitle(f"Training Instances - Dataset: {data_prep.scenario}\n[ttrunc, prev_h, pred_h]: [{args.truncated_backprop_length}, {args.prev_horizon}, {args.prediction_horizon}]")
        fig4.suptitle(f"Trajectory Summaries - Dataset: {data_prep.scenario}\n[ttrunc, prev_h, pred_h]: [{args.truncated_backprop_length}, {args.prev_horizon}, {args.prediction_horizon}]")
        fig5.suptitle(f"Trajectory Sets - Average Speeds - Dataset: {data_prep.scenario}\n[ttrunc, prev_h, pred_h]: [{args.truncated_backprop_length}, {args.prev_horizon}, {args.prediction_horizon}]")

        ax1.set_title("Scene Overview")
        sup.plot_grid(
            ax1,
            np.array([0, 0]),
            data_prep.agent_container.occupancy_grid.gridmap,
            data_prep.agent_container.occupancy_grid.resolution,
            data_prep.agent_container.occupancy_grid.map_size
        )
        ax1.set_aspect('equal')

        ax2.set_title("Training Instances Directions: Past")
        ax2.set_aspect('equal')

        ax3.set_title("Training Instances Directions: Future")
        ax3.set_aspect('equal')

        ax4.set_title("Training Instances Directions: Past")
        ax4.set_aspect('equal')

        ax5.set_title("Training Instances Directions: Future")
        ax5.set_aspect('equal')

        ax6.set_title("Trajectory Lengths")

        ax7.set_title("Average Speed [m/s]")
        ax7.set_xticks([0.5, 3.5, 6.5])
        ax7.set_xticklabels(["-Tobs:t0", "t0:Tpred", "-Tobs:Tpred"])

        ax8.set_title("Distance [m]")
        ax8.set_xticks([0.5, 3.5, 6.5])
        ax8.set_xticklabels(["-Tobs:t0", "t0:Tpred", "-Tobs:Tpred"])

        ax9.set_title("Average Speed [m/s]")
        ax9.set_xticks([0, 1])
        ax9.set_xticklabels(["Train", "Test"])
        ax9.set_xlim(-1, 2)

        counter = 0
        training_instances = 0
        train_trajectories = 0
        test_trajectories = 0
        discarded_trajectories = 0

        directions_instances = {}

        # for each trajectory, the list will be appended with its average speed
        train_trajectories_average_velocities = []
        test_trajectories_average_velocities = []

        for agent_id, agent_data_obj in data_prep.agent_container.agent_data.items():
            assert data_prep.agent_container.getNumberOfTrajectoriesForAgent(agent_id) == 1
            agent_in_traj_set = agent_id in [item[0] for item in data_prep.trajectory_set]
            train_test_split_idx = int(len(data_prep.trajectory_set) * data_prep.train_set)
            agent_in_training_set = agent_id in [item[0] for item in data_prep.trajectory_set[0:train_test_split_idx]]
            agent_in_testing_set = agent_id in [item[0] for item in data_prep.trajectory_set[train_test_split_idx:-1]]
            trajectory_long_enough = args.prev_horizon + args.truncated_backprop_length + args.prediction_horizon + 1 < len(agent_data_obj.trajectories[0])
            assert agent_in_traj_set == trajectory_long_enough

            skip = False
            if agent_in_training_set:
                color = "red"
                train_trajectories += 1
            elif agent_in_testing_set:
                color = "blue"
                test_trajectories += 1
            else:
                color = "grey"
                discarded_trajectories += 1
                skip = True

            agent_data_obj.plot(ax1, color=color, x_scale=1, y_scale=1)

            if skip:
                continue

            traj_avg_speed = np.mean(np.linalg.norm(agent_data_obj.trajectories[0].vel_vec, axis=1))
            if agent_in_training_set:
                ax9.scatter(0, traj_avg_speed, color=color, alpha=0.5)
                train_trajectories_average_velocities.append(traj_avg_speed)

            if agent_in_testing_set:
                ax9.scatter(1, traj_avg_speed, color=color, alpha=0.5)
                test_trajectories_average_velocities.append(traj_avg_speed)

            start_idx = args.prev_horizon
            while start_idx + args.truncated_backprop_length + args.prediction_horizon + 1 < len(agent_data_obj.trajectories[0]):
                begin_idx = start_idx-args.prev_horizon
                end_idx = start_idx+args.truncated_backprop_length+args.prediction_horizon + 1

                time_segment = agent_data_obj.trajectories[0].time_vec[begin_idx:end_idx]
                pose_segment = agent_data_obj.trajectories[0].pose_vec[begin_idx:end_idx]
                vel_segment = agent_data_obj.trajectories[0].vel_vec[begin_idx:end_idx]

                t0_idx = args.prev_horizon + args.truncated_backprop_length

                pose_segment_Tobs_t0 = pose_segment[:t0_idx + 1]
                vel_segment_Tobs_t0 = vel_segment[:t0_idx + 1]
                pose_segment_t0_Tpred = pose_segment[t0_idx:]
                vel_segment_t0_Tpred = vel_segment[t0_idx:]

                centered_pose_vec = plot_utils.centered_traj_pos(pose_segment, vel_segment, t0_idx)

                distance_Tobs_Tpred = np.linalg.norm(pose_segment[0] - pose_segment[-1])
                distance_Tobs_t0 = np.linalg.norm(pose_segment_Tobs_t0[0] - pose_segment_Tobs_t0[-1])
                distance_t0_Tpred = np.linalg.norm(pose_segment_t0_Tpred[0] - pose_segment_t0_Tpred[-1])

                average_velocity_Tobs_Tpred = np.mean(np.linalg.norm(vel_segment, axis=1))
                average_velocity_Tobs_t0 = np.mean(np.linalg.norm(vel_segment_Tobs_t0, axis=1))
                average_velocity_t0_Tpred = np.mean(np.linalg.norm(vel_segment_t0_Tpred, axis=1))

                ax7.scatter(0 + int(agent_in_testing_set) + np.random.normal(0, 0.1), average_velocity_Tobs_t0, color=color, alpha=0.5)
                ax8.scatter(0 + int(agent_in_testing_set) + np.random.normal(0, 0.1), distance_Tobs_t0, color=color, alpha=0.5)
                ax7.scatter(3 + int(agent_in_testing_set) + np.random.normal(0, 0.1), average_velocity_t0_Tpred, color=color, alpha=0.5)
                ax8.scatter(3 + int(agent_in_testing_set) + np.random.normal(0, 0.1), distance_t0_Tpred, color=color, alpha=0.5)
                ax7.scatter(6 + int(agent_in_testing_set) + np.random.normal(0, 0.1), average_velocity_Tobs_Tpred, color=color, alpha=0.5)
                ax8.scatter(6 + int(agent_in_testing_set) + np.random.normal(0, 0.1), distance_Tobs_Tpred, color=color, alpha=0.5)

                direction = plot_utils.describe_motion(centered_pose_vec)
                directions_instances.setdefault(direction, 0)
                directions_instances[direction] += 1

                direction_color = dir_colors[direction]

                if agent_in_training_set:
                    plot_utils.plot_centered_trajectory(
                        ax2, centered_pose_vec, center_idx=t0_idx, color=direction_color, past=True, future=False
                    )
                    plot_utils.plot_centered_trajectory(
                        ax3, centered_pose_vec, center_idx=t0_idx, color=direction_color
                    )
                    plot_utils.plot_centered_trajectory(
                        ax4, centered_pose_vec, center_idx=t0_idx, color='red', past=True, future=False
                    )
                    plot_utils.plot_centered_trajectory(
                        ax5, centered_pose_vec, center_idx=t0_idx,color='red'
                    )

                start_idx += args.truncated_backprop_length
                training_instances += 1

            if counter == -1:
                break

        print("Velocities:")
        print(f"Training set:   mean={np.mean(train_trajectories_average_velocities)}, std={np.std(train_trajectories_average_velocities)}")
        print(f"Test set:       mean={np.mean(test_trajectories_average_velocities)}, std={np.std(test_trajectories_average_velocities)}")

        position = ax3.get_ylim()[1] - 0.3
        ax3.text(x=ax3.get_xlim()[0] + 0.2, y=position, s=f"total: {training_instances}")
        position -= 0.3
        for dir, amount in directions_instances.items():
            ax3.text(x=ax3.get_xlim()[0] + 0.2, y=position, s=f"{dir}: {amount}", color=dir_colors[dir])
            position -= 0.3

        traj_len_indices = data_prep.agent_container.get_trajectory_length_dict()
        for key, value in traj_len_indices.items():
            if key > data_prep.min_length_trajectory:
                color = "red"
            else:
                color = "grey"
            ax6.bar(x=key, height=value, color=color)

        total_trajectories = train_trajectories + discarded_trajectories
        ax6.text(x=0.2 * (ax6.get_xlim()[1] - ax6.get_xlim()[0]), y=ax6.get_ylim()[1] - 1, s=f"Kept: {train_trajectories} ({train_trajectories / total_trajectories * 100:.2f}% of total dataset)", color="red")
        ax6.text(x=0.2 * (ax6.get_xlim()[1] - ax6.get_xlim()[0]), y=ax6.get_ylim()[1] - 2, s=f"Discarded: {discarded_trajectories}", color="grey")
        ax6.text(x=0.2 * (ax6.get_xlim()[1] - ax6.get_xlim()[0]), y=ax6.get_ylim()[1] - 3, s=f"Total: {total_trajectories}")

    plt.show()
