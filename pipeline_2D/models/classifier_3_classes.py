from torch import nn


class Classifier_3_Classes(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_classes = 3
        self.fc1 = nn.Linear(2, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, self.num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = nn.functional.relu(self.fc3(x))
        x = self.fc4(x)
        return self.softmax(x)