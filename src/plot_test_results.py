import json
import os.path
import sys
sys.path.append("../")
import src.data_utils.plot_utils
import matplotlib.pyplot as plt

here = os.path.abspath(__file__)

def single_vs_multi_datasets():
    datasets = ["Hotel", "ETH", "Univ", "ZARA01", "ZARA02"]
    target_ADEs = [0.35, 0.39, 0.53, 0.41, 0.51]
    target_FDEs = [0.47, 0.70, 0.65, 0.70, 0.55]
    exp_nums = [4001, 4003, 4005, 4007, 4009]
    exp_nums_multi_data = [5001, 5003, 5005, 5007, 5009]

    print(here)

    for i in range(len(datasets)):
        fig, axs = plt.subplots(1, 2, sharey=True)

        jsonpath = os.path.join(os.path.dirname(here),
                                f"../trained_models/SocialVRNN/{exp_nums[i]}/model_parameters.json")
        with open(jsonpath, "r") as file:
            json_dict = json.load(file)
        src.data_utils.plot_utils.plot_ADE_FDE_runs(ax=axs[0], model_name="SocialVRNN", exp_num=exp_nums[i])
        axs[0].scatter(1, target_ADEs[i], label="Article")
        axs[0].scatter(2, target_FDEs[i], label="Article")
        axs[0].set_title(f"Only 1 dataset\n"
                         f"freeze: {json_dict['freeze_grid_cnn']}\n"
                         f"warmstartCNN: {json_dict['warm_start_convnet']}")
        axs[0].legend()

        jsonpath = os.path.join(os.path.dirname(here),
                                f"../trained_models/SocialVRNN/{exp_nums[i]}/model_parameters.json")
        with open(jsonpath, "r") as file:
            json_dict_2 = json.load(file)
        src.data_utils.plot_utils.plot_ADE_FDE_runs(ax=axs[1], model_name="SocialVRNN", exp_num=exp_nums_multi_data[i])
        axs[1].scatter(1, target_ADEs[i], label="Article")
        axs[1].scatter(2, target_FDEs[i], label="Article")
        axs[1].set_title(f"Train 4, Test on last\n"
                         f"freeze: {json_dict_2['freeze_grid_cnn']}\n"
                         f"warmstartCNN: {json_dict_2['warm_start_convnet']}")
        axs[1].legend()

        fig.suptitle(f"Dataset:\n"
                     f"{datasets[i]}\n"
                     f"{json_dict['scenario']}\n"
                     f"{json_dict_2['scenario']}")
        plt.legend()

    plt.show()

def compare_orig_vs_own_implementation():
    datasets = ["Hotel", "ETH", "Univ", "ZARA01", "ZARA02"]
    target_ADEs = [0.35, 0.39, 0.53, 0.41, 0.51]
    target_FDEs = [0.47, 0.70, 0.65, 0.70, 0.55]
    orig_exp_nums = [4001, 4003, 4005, 4007, 4009]
    own_exp_nums = [100, 101, 102, 103, 104]

    for i in range(len(datasets)):

        fig, axs = plt.subplots(1, 2, sharey=True)

        src.data_utils.plot_utils.plot_ADE_FDE_runs(axs[0], model_name="SocialVRNN", exp_num=orig_exp_nums[i])
        axs[0].scatter(1, target_ADEs[i], label="Article")
        axs[0].scatter(2, target_FDEs[i], label="Article")
        axs[0].set_title(f"Original Implementation")
        axs[0].legend()

        src.data_utils.plot_utils.plot_ADE_FDE_runs(axs[1], model_name="SocialVRNN_LSTM_ED", exp_num=own_exp_nums[i])
        axs[1].scatter(1, target_ADEs[i], label="Article")
        axs[1].scatter(2, target_FDEs[i], label="Article")
        axs[1].set_title(f"Own Implementation")
        axs[1].legend()

        fig.suptitle(f"Comparison: Original implementation vs Own\n"
                     f"Dataset: {datasets[i]}")
        plt.legend()

    plt.show()

def ablation_study():
    plt.rcParams['font.family'] = 'monospace'

    datasets = ["Hotel", "ETH", "Univ", "ZARA01", "ZARA02"]
    target_ADEs = [0.35, 0.39, 0.53, 0.41, 0.51]
    target_FDEs = [0.47, 0.70, 0.65, 0.70, 0.55]
    experiments_idx = {
        1: "Convnet: Pretrained + Frozen\nQA LSTM: Rand init + Free",
        2: "Convnet: Rand init + Free\nQA LSTM: Pretrained + Frozen",
        3: "Convnet: Rand init + Free\nQA LSTM: Rand init + Free",
        4: "Convnet: Pretrained + Frozen\nQA LSTM: Pretrained + Frozen"
    }

    for i in range(len(datasets)):

        fig, axs = plt.subplots(1, 4, sharex=True, sharey=True)

        for k, v in experiments_idx.items():

            exp_num = int(f"{k}0{i}")
            jsonpath = os.path.join(os.path.dirname(here), f"../trained_models/SocialVRNN_LSTM_ED/{exp_num}/model_parameters.json")
            with open(jsonpath, "r") as file:
                json_dict = json.load(file)
            frame_number = k-1
            ax = axs[frame_number]
            src.data_utils.plot_utils.plot_ADE_FDE_runs(ax, model_name="SocialVRNN_LSTM_ED", exp_num=exp_num)
            ax.scatter(1, target_ADEs[i], label="Article")
            ax.scatter(2, target_FDEs[i], label="Article")

            spacing_1 = 12
            spacing_2 = 8
            ax.set_title(#f"{v}\n"
                         f"{''.ljust(spacing_1)}| {'CNN'.ljust(spacing_2)}| {'LSTM'.ljust(spacing_2)}\n"
                         f"{'Pretrained'.ljust(spacing_1)}| {str(json_dict['warm_start_convnet']).ljust(spacing_2)}| {str(json_dict['warm_start_query_agent_module']).ljust(spacing_2)}\n"
                         f"{'Frozen'.ljust(spacing_1)}| {str(json_dict['freeze_grid_cnn']).ljust(spacing_2)}| {str(json_dict['freeze_query_agent_module']).ljust(spacing_2)}", loc='left')
            ax.legend()

            fig.suptitle(f"Dataset:\n"
                         f"{datasets[i]}")

            plt.legend()
    plt.show()


if __name__ == '__main__':
    ablation_study()
