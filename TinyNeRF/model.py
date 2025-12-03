import torch
import torch.nn as nn

class TinyNerfModel(nn.Module):
    def __init__(self, filter_size=128, num_encoding_functions=6):
        super(TinyNerfModel, self).__init__()

        self.layer1 = nn.Linear(3 + 3 * 2 * num_encoding_functions, filter_size)
        self.layer2 = nn.Linear(filter_size, filter_size)
        self.layer3 = nn.Linear(filter_size, 4)
        self.relu = nn.functional.relu

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return x