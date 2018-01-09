#! /users/guruprakash.r/miniconda2/bin/python
from myimports import *
from dataLoader import getTrainingData
from models import VanillaGRUNetwork


load_model = 0
starting_epoch = 0
chunk_size = 128
input_size = 2
output_size = 2
hidden_size = 100
n_layers = 1

def loadModel():
	global starting_epoch
	global net
	global optimizer
	checkpoint = torch.load("checkpoint.pth.tar")
	starting_epoch = checkpoint['epoch']
	net.load_state_dict(checkpoint['state_dict'])
	optimizer.load_state_dict(checkpoint['optimizer'])
	print "Model is loaded " + str(starting_epoch);

net = VanillaGRUNetwork(input_size=input_size, hidden_size=hidden_size, output_size=output_size, n_layers=n_layers)
criterion = nn.MSELoss()
learning_rate = 5e-3
optimizer = optim.Adam(net.parameters(), learning_rate)

if load_model:
	loadModel()

use_cuda = torch.cuda.is_available()

if use_cuda:
	print ('CUDA is available')
	net = net.cuda()
	criterion = criterion.cuda()

