from torch.utils.data import TensorDataset, Dataset


class ToyDataset_2(Dataset):
    def __init__(self, X_test, y_test):
        self.dataset = TensorDataset(X_test, y_test)

    def __getitem__(self, index: slice):
        data, target = self.dataset[index]
        return data, target, index

    def __len__(self):
        return len(self.dataset)
