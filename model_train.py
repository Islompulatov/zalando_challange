import torch
import torch.nn.functional as F
from torch import nn

from collections import OrderedDict

# YOUR CODE HERE


def model_check(input_size, hidden_layer1, hidden_layer2, output_size):
    model = nn.Sequential(nn.Linear(input_size, hidden_layer1),
                          nn.ReLU(),
                          nn.Linear(hidden_layer1, hidden_layer2),
                          nn.ReLU(),
                          nn.Linear(hidden_layer2, output_size),
                          nn.Softmax(dim=1))

    return model
