import torch
import torch.nn as nn
import torch.nn.functional as F


# simple one hidden layer model
class Layer3Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Layer3Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        activations = torch.sigmoid(out)
        out = self.fc2(activations)
        return out, activations
