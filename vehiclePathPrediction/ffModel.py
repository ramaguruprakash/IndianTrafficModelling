import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
# Neural Network Model (1 hidden layer)
class Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        #print "input ko check karo ", x.data.size()
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.fc2(out)
        return out
