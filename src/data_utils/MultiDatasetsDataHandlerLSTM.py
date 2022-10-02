"""

This is the implementation of a class whih manages multiple DataHandlerLSTM instances.
The goal of this class is to make it possible to conduct training of a model using multiple datasets (more specifically,
to be able to properly reproduce the training procedure stated in the article, where the architecture is being trained
on 4 of the 5 ETH/UCY datasets, to be then evaluated on the last one).

"""
import os.path
import random
import sys
sys.path.append('../../')
sys.path.append('../')
import DataHandlerLSTM
import src.train_WIP
import numpy as np
import argparse


class MultiDataHandler:

    def __init__(self, args, datasets):
        """

        datasets is a list containing one or more of the following str: ['ewap_dataset/seq_hotel',
                                                                         'ewap_dataset/seq_eth',
                                                                         'st',
                                                                         'zara_01',
                                                                         'zara_02']

        """
        self.datasets = datasets
        self.dataHandlers = []
        self.datahandler_idxs = []
        for idx, dataset in enumerate(self.datasets):
            _copy_args = argparse.Namespace(**vars(args))
            _copy_args.scenario = dataset
            dataprep = DataHandlerLSTM.DataHandlerLSTM(_copy_args)
            dataprep.processData()
            self.dataHandlers.append(dataprep)
            self.datahandler_idxs.extend([idx] * len(dataprep.trajectory_set))

        random.shuffle(self.datahandler_idxs)

        for dataset in self.dataHandlers:
            print(dataset.scenario)
            print(len(dataset.trajectory_set))
            print(dataset.agent_container.getNumberOfAgents())
            # print(dataset.batch_sequences)
            print(dataset.sequence_idx)
        print(self.datahandler_idxs)


def main():
    args = src.train_WIP.parse_args()
    datasets = ["ewap_dataset/seq_hotel", "ewap_dataset/seq_eth", "st", "zara_01", "zara_02"]

    multidataprep = MultiDataHandler(args, datasets)

    # datahandlers = []
    #
    # for dataset in ["ewap_dataset/seq_hotel", "ewap_dataset/seq_eth", "st", "zara_01", "zara_02"]:
    #     args.scenario = os.path.join("real_world", dataset)
    #     # src.train_WIP.print_args(args)
    #
    #     datahandler = DataHandlerLSTM.DataHandlerLSTM(args)
    #     for k, v in datahandler.__dict__.items():
    #         print(f"{k}: {v}")
    #     print("\n\n\n\n\n\n\n")
    #
    # map_args = {"file_name": 'map.png',
    #             "resolution": 0.1,
    #             "map_size": np.array([30., 6.])}
    # # Load dataset
    # datahandler.processData(**map_args)
    #
    # batch_x, batch_vel, batch_pos, batch_goal, batch_grid, batch_ped_grid, batch_y, batch_pos_target, other_agents_pos, new_epoch = datahandler.getBatch()
    # validation_dict = datahandler.getTestBatch()




if __name__ == '__main__':
    main()
