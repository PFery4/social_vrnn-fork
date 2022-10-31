import glob
import json
import os.path
import sys
sys.path.append("../")
import src.data_utils.plot_utils
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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


def compare_ADE_FDE_results(model_name, runs, common_args, compare_args, show=False):
    # plt.rcParams['font.family'] = 'monospace'

    base_path = os.path.abspath(os.path.join(os.path.dirname(here), f"../trained_models/{model_name}"))

    col1_ratio = 0.25
    row1_ratio = 0.8

    width_ratios = [col1_ratio]
    width_ratios.extend([(1-col1_ratio)/len(runs)] * len(runs))
    height_ratios = [row1_ratio, 1-row1_ratio]

    gs = gridspec.GridSpec(nrows=2, ncols=len(runs) + 1, width_ratios=width_ratios, height_ratios=height_ratios)

    fig = plt.figure()
    fig_ax1 = fig.add_subplot(gs[0, 0])
    fig_ax1.axis('off')

    fig_ax2 = fig.add_subplot(gs[1, 0])
    fig_ax2.axis('off')
    tab2_names = fig_ax2.table(
        cellText=[[arg] for arg in compare_args],
        loc="center"
    )
    tab2_names.auto_set_font_size(False)
    tab2_names.set_fontsize(10)

    common_args_across = []

    for idx, exp_num in enumerate(runs):

        param_path = os.path.join(base_path, str(exp_num), "model_parameters.json")
        with open(param_path, "r") as f:
            param_dict = json.load(f)

        fig_ax = fig.add_subplot(gs[0, idx + 1], sharey=fig_ax1)
        src.data_utils.plot_utils.plot_ADE_FDE_runs(fig_ax, model_name=model_name, exp_num=exp_num)

        ade_fde_pos = [1, 2]
        margin = 0.5
        fig_ax.set_xlim(ade_fde_pos[0] - margin, ade_fde_pos[1] + margin)
        fig_ax.set_xticks(ade_fde_pos)
        fig_ax.set_xticklabels(["ADE", "FDE"])
        if idx != 0:
            fig_ax.tick_params(labelleft=False)

        common_arg_values = []
        for arg in common_args:
            value = param_dict.get(arg, "N/A")
            common_arg_values.append(value)
        common_args_across.append(common_arg_values)

        compare_arg_values = []
        for arg in compare_args:
            value = param_dict.get(arg, "N/A")
            compare_arg_values.append(value)

        fig_tab = fig.add_subplot(gs[1, idx + 1])
        # table_values = list(map(list, zip(important_args, arg_values)))
        table_values = [[value] for value in compare_arg_values]
        table = fig_tab.table(
            cellText=table_values,
            loc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        fig_tab.axis("off")

    # checking that the common arguments share indeed the same value across runs
    assert common_args_across[1:] == common_args_across[:-1]

    tab1_values = []
    for i in range(len(common_args)):
        tab1_values.append(common_args[i])
        tab1_values.append(common_args_across[0][i])
    tab1_values = [[value] for value in tab1_values]

    tab1 = fig_ax1.table(
        cellText=tab1_values,
        cellLoc='left',
        loc='center'
    )
    tab1.auto_set_font_size(False)
    tab1.set_fontsize(10)

    if show:
        plt.show()


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
    # ablation_study()
    # final_losses_LSTM_networks()

    # ablation_study()
    # plt.show()

    model_name = "SocialVRNN_LSTM_ED"
    common_args = ["scenario"]
    compare_args = ["exp_num", "warm_start_convnet", "freeze_grid_cnn", "warm_start_query_agent_module", "freeze_query_agent_module"]
    experiments = [
        # [100, 200, 300, 400],
        [10100, 10200, 10300, 10400],
        # [101, 201, 301, 401],
        [10101, 10201, 10301, 10401],
        # [102, 202, 302, 402],
        [10102, 10202, 10302, 10402],
        # [103, 203, 303, 403],
        [10103, 10203, 10303, 10403],
        # [104, 204, 304, 404],
        [10104, 10204, 10304, 10404]
    ]
    for runs in experiments:
        compare_ADE_FDE_results(
            model_name=model_name,
            runs=runs,
            common_args=common_args,
            compare_args=compare_args
        )
    plt.show()

    # common_args = ["scenario"]
    # compare_args = ["exp_num"]# "warm_start_convnet", "freeze_grid_cnn", "warm_start_query_agent_module", "freeze_query_agent_module", "rotated_grid", "normalize_data", "diversity_update", "correction_div_loss_in_total_loss"]
    # experiments = [[10101, 110101, 210101, 310101]]
    # for runs in experiments:
    #     compare_ADE_FDE_results(
    #         model_name=model_name,
    #         runs=runs,
    #         common_args=common_args,
    #         compare_args=compare_args
    #     )
    # plt.show()
    #
    # common_args = ["scenario"]
    # compare_args = ["exp_num", "rotated_grid", "normalize_data", "diversity_update", "correction_div_loss_in_total_loss"]
    # experiments = [[10101, 1010101, 2010101, 3010101, 4010101]]
    #
    # for runs in experiments:
    #     compare_ADE_FDE_results(
    #         model_name=model_name,
    #         runs=runs,
    #         common_args=common_args,
    #         compare_args=compare_args
    #     )
    # plt.show()
    #
    # common_args = ["scenario"]
    # compare_args = ["exp_num", "diversity_update", "correction_div_loss_in_total_loss"]
    # experiments = [[10101, 3010101, 4010101]]
    #
    # for runs in experiments:
    #     compare_ADE_FDE_results(
    #         model_name=model_name,
    #         runs=runs,
    #         common_args=common_args,
    #         compare_args=compare_args
    #     )
    # plt.show()
    #
    # common_args = ["scenario"]
    # compare_args = ["exp_num", "truncated_backprop_length"]
    # experiments = [[10101, 10101003, 10101005, 10101008]]
    # for runs in experiments:
    #     compare_ADE_FDE_results(
    #         model_name=model_name,
    #         runs=runs,
    #         common_args=common_args,
    #         compare_args=compare_args
    #     )
    # plt.show()

