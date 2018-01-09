#! /users/guruprakash.r/miniconda2/bin/python
'''
	This file will have different model definitions
'''

from myimports import *
from globalHyperP import *
'''
   Its a VanillaGRUNetwork which is configurable in the following ways
   	input_size
   	hidden_size
   	output_size
   	n_layers
'''
class VanillaGRUNetwork(nn.Module):
    def __init__(self, **kw):
        super(VanillaGRUNetwork, self).__init__()
        self.input_size = kw['input_size']
        self.hidden_size = kw['hidden_size']
        self.output_size = kw['output_size']
        self.n_layers = kw['n_layers']

        self.fc_in = nn.Linear(self.input_size, self.hidden_size)
        self.rnn = nn.GRU(self.hidden_size, self.hidden_size, self.n_layers)
        self.fc_out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x, h):
        # One hot vector of single column coming in. 
        # View sorcery is to adjust to the layer's dimension requirement
        # Size(D) -> Size(1,D)

        x = self.fc_in(x.view(1, -1))

        x, h = self.rnn(x.view(1, 1, -1), h)

        x = self.fc_out(x.view(1, -1))
        return x, h

    def init_hidden(self):
		return Variable(torch.zeros(self.n_layers, 1, self.hidden_size))


if __name__ == "__main__":
        net = VanillaGRUNetwork(input_size=10, hidden_size=20, output_size=10, n_layers=1)
        print net