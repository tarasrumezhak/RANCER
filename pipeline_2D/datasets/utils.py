import os

import numpy as np
import torch

from datasets.toy_2_classes import ToyDataset_2


def load_toy_dataset(config):
    X_test = np.array(np.loadtxt(os.path.join(config["dataset_path"], "X_test.out"), delimiter=","))
    y_test = np.array([np.loadtxt(os.path.join(config["dataset_path"], "y_test.out"), delimiter=",")])

    X_test = torch.from_numpy(X_test).type(torch.FloatTensor)
    y_test = torch.from_numpy(y_test).type(torch.LongTensor)[0]

    X_test = torch.unsqueeze(X_test, 1)
    X_test = torch.unsqueeze(X_test, 1)

    print("X_test.shape: ", X_test.shape)
    print("y_test.shape: ", y_test.shape)

    print("y_test[0]: ", y_test[0])
    print("y_test[1]: ", y_test[1])
    print("y_test[2]: ", y_test[2])

    return ToyDataset_2(X_test[:100], y_test[:100])
