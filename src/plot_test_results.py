import json
import os.path
import sys
sys.path.append("../")
import src.data_utils.plot_utils
import matplotlib.pyplot as plt

if __name__ == '__main__':

    datasets = ["ETH", "Hotel", "Univ", "ZARA01", "ZARA02"]
    target_ADEs = [0.39, 0.35, 0.53, 0.41, 0.51]
    target_FDEs = [0.70, 0.47, 0.65, 0.70, 0.55]
    exp_nums = [4003, 4001, 4005, 4007, 4009]
    exp_nums_multi_data = [5003, 5001, 5005, 5007, 5009]
    here = os.path.abspath(__file__)
    print(here)

    for i in range(len(datasets)):

        fig, axs = plt.subplots(1, 2, sharey=True)

        jsonpath = os.path.join(os.path.dirname(here), f"../trained_models/SocialVRNN/{exp_nums[i]}/model_parameters.json")
        with open(jsonpath) as file:
            json_dict = json.load(file)
        src.data_utils.plot_utils.plot_ADE_FDE_runs(ax=axs[0], model_name="SocialVRNN", exp_num=exp_nums[i])
        axs[0].scatter(1, target_ADEs[i], label="Article")
        axs[0].scatter(2, target_FDEs[i], label="Article")
        axs[0].set_title(f"Only 1 dataset\n"
                         f"freeze: {json_dict['freeze_grid_cnn']}\n"
                         f"warmstartCNN: {json_dict['warm_start_convnet']}")

        jsonpath = os.path.join(os.path.dirname(here), f"../trained_models/SocialVRNN/{exp_nums[i]}/model_parameters.json")
        with open(jsonpath) as file:
            json_dict_2 = json.load(file)
        src.data_utils.plot_utils.plot_ADE_FDE_runs(ax=axs[1], model_name="SocialVRNN", exp_num=exp_nums_multi_data[i])
        axs[1].scatter(1, target_ADEs[i], label="Article")
        axs[1].scatter(2, target_FDEs[i], label="Article")
        axs[1].set_title(f"Train 4, Test on last\n"
                         f"freeze: {json_dict_2['freeze_grid_cnn']}\n"
                         f"warmstartCNN: {json_dict_2['warm_start_convnet']}")

        fig.suptitle(f"Dataset:\n"
                     f"{datasets[i]}\n"
                     f"{json_dict['scenario']}\n"
                     f"{json_dict_2['scenario']}")

        plt.legend()

        plt.show()

