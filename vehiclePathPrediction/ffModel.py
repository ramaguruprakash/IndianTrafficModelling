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
       # self.batch1 = nn.BatchNorm1d(hidden_size)
        #self.batch2 = nn.BatchNorm1d(hidden_size)
        #self.batch3 = nn.BatchNorm1d(hidden_size)
        #self.dropout = nn.Dropout()
        self.fc2 = nn.Linear(hidden_size, output_size)
        #self.fc3 = nn.Linear(hidden_size, hidden_size)
        #self.fc4 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        print "input ko check karo ", x.data.size()
        #print x.data.numpy()
        out = self.fc1(x)
        #print out.data.numpy()
        #out = self.batch1(out)
        #print("running mean1 ...\n",self.batch1.running_mean)
        #print("running var1 ...\n",self.batch1.running_var)
        #print out.data.numpy()
        out = self.relu(out)
        #print out.data.numpy()
        ##out = self.dropout(out)
        #out = self.fc3(out)
        #print out.data.numpy()
        #out = self.batch2(out)
        #print("running mean2 ...\n",self.batch2.running_mean)
        #print("running var2 ...\n",self.batch2.running_var)
        #print out.data.numpy()
        #out = self.relu(out)
        #print out.data.numpy()
        #out = self.dropout(out)
        #out = self.fc4(out)
        #print out.data.numpy()
        #out = self.batch3(out)
        #print("running mean3 ...\n",self.batch3.running_mean)
        #print("running var3 ...\n",self.batch3.running_var)
        #print out.data.numpy()
        #out = self.relu(out)
        #print out.data.numpy()
        #out = self.dropout(out)
        out = self.fc2(out)
        #print out.data.numpy()
        return out
