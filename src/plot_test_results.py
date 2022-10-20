import glob
import json
import os.path
import sys
sys.path.append("../")
import src.data_utils.plot_utils
import matplotlib.pyplot as plt
import src.data_utils.utils

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
    # target_ADEs = [0.35, 0.39, 0.53, 0.41, 0.51]
    # target_FDEs = [0.47, 0.70, 0.65, 0.70, 0.55]
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
            # ax.scatter(1, target_ADEs[i], c='o', label="Article")
            # ax.scatter(2, target_FDEs[i], c='o', label="Article")
            ax.set_xticks([1, 2])
            ax.set_xticklabels(["ADE", "FDE"])

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

def final_losses_LSTM_networks():
    """
    Produce the final loss values from training experiments contained in exp_num_list. This function works on
    the LSTM encoder decoder module, LSTM_ED_module.
    """
    model_name = "LSTM_ED_module"
    exp_num_list = [30, 31, 32, 33, 34]

    for exp_num in exp_num_list:
        path = os.path.abspath(os.path.join(here, f"../../trained_models/{model_name}/{exp_num}"))

        with open(os.path.join(path, "parameters.json"), 'r') as file:
            param_json = json.load(file)
        with open(os.path.join(path, "results/results.json"), 'r') as file:
            results_json = json.load(file)

        print(f"Experiment {exp_num}\n")
        print("Parameters:")
        for k, v in param_json.items():
            print(k, v)
        print('dataset', results_json['dataset'])
        print("\nLosses:")
        print("Best Validation Loss:        ", results_json["best_validation_loss"])
        print("Corresponding Training Loss: ", results_json["train_loss_at_best_val_loss"])
        print("Occuring timestep:           ", results_json["best_val_loss_timestep"])
        print("\n\n")


def compare_results(model_name, runs):
    # import src.config.config as config
    # default_args = config.parse_args()
    # src.data_utils.utils.load_model_from_parameter_file(0, default_args=default_args)
    pass


def make_defaults_csv():
    import src.data_utils.DataHandlerLSTM
    import src.train_WIP
    import csv

    filename = os.path.abspath(os.path.join(here, "../args_defaults.csv"))
    print(filename)

    args = src.train_WIP.parse_args()
    src.train_WIP.print_args(args)
    # dataprep = src.data_utils.DataHandlerLSTM.DataHandlerLSTM()
    with open(filename, "w") as f:
        writer = csv.writer(f, delimiter=";", quoting=csv.QUOTE_MINIMAL)
        for k, v in vars(args).items():
            writer.writerow([k, v])


if __name__ == '__main__':
    ablation_study()
    final_losses_LSTM_networks()
