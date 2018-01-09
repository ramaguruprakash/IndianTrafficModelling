#! /users/guruprakash.r/miniconda2/bin/python
from myimports import *
from dataLoader import getTrainingData
from models import VanillaGRUNetwork
sys.path.append("../../trafficSimulator/")
from vehicleTypes import vehicleTypes
vehicleTypes = VehicleTypes()

chunk_size = 128
max_num_vehicles = 30
input_size = 30*(2 + len(vehicleTypes.classes))
output_size = 30*(2 + len(vehicleTypes.classes))
hidden_size = 100
n_layers = 1

net = VanillaGRUNetwork(input_size=input_size, hidden_size=hidden_size, output_size=output_size, n_layers=n_layers)
criterion = nn.MSELoss()
learning_rate = 5e-3
optimizer = optim.Adam(net.parameters(), learning_rate)

use_cuda = torch.cuda.is_available()
if use_cuda:
    print ('CUDA is available')
    net = net.cuda()
    criterion = criterion.cuda()
