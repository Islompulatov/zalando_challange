import torch.nn as nn
import torch

input_size = 28*28
hiddensizes = [400, 200, 100]
output_size = 10
percent_drop = 0.20

neuralnet = nn.Sequential(
    nn.Linear(input_size, hiddensizes[0]),
    nn.Dropout(percent_drop),
    # nn.BatchNorm1d(hiddensizes[0]),
    nn.ReLU(),

    nn.Linear(hiddensizes[0], hiddensizes[1]),
    nn.Dropout(percent_drop),
    # nn.BatchNorm1d(hiddensizes[1]),
    nn.ReLU(),

    nn.Linear(hiddensizes[1], hiddensizes[2]),
    nn.Dropout(percent_drop),
    # nn.BatchNorm1d(hiddensizes[2]),
    nn.ReLU(),

    nn.Linear(hiddensizes[2], output_size),
    # nn.LogSoftmax(dim=1)

)
