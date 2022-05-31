import torch
import torch.nn as nn
import torch.nn.functional as F

class ImageClassificationBase(nn.Module):
    def __init__(self,input_size, hidden_layer1, hidden_layer2, output_size):
        super(ImageClassificationBase, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_layer1)
        self.fc2 = nn.Linear(hidden_layer1, hidden_layer2)
        self.fc3 = nn.Linear(hidden_layer2, output_size)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
    

    def forward(self, x):
        layer1 = self.fc1(x)
        act1 = self.relu(layer1)
        layer2 = self.fc2(act1)
        act2 = self.relu(layer2)
        layer3 = self.fc3(act2)
        output = self.softmax(layer3)
        return output


model_class = ImageClassificationBase(784, 128, 64, 10)      