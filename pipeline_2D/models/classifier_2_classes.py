import torch
from torch import nn


class Classifier_2_Classes(nn.Module):
    def __init__(self):
        super(Classifier_2_Classes, self).__init__()
        self.fc1 = nn.Linear(2, 3)
        self.fc2 = nn.Linear(3, 2)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.tanh(x)
        x = self.fc2(x)
        x = self.log_softmax(x)
        return x
