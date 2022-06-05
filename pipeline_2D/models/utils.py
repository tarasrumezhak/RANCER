import os

import torch

from models.classifier_2_classes import Classifier_2_Classes


def load_toy_model(config, device):
    model = Classifier_2_Classes().to(device)
    print("model: ", model)

    model.load_state_dict(torch.load(os.path.join(config["weights_path"], "model_weights.pth")))
    model.eval()

    return model
