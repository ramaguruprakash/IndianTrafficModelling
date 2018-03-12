import torch
import torch.nn as nn
import argparse
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
from ffModel import Model
from sumoDataLoader import SumoDataLoader
import numpy as np
import matplotlib
matplotlib.use("tkAgg")
from matplotlib import pyplot as plt
parser = argparse.ArgumentParser(description='Hyper parameters')
parser.add_argument('--neighbours', type=int, default=5, help='The number of neighbours to consider')
parser.add_argument('--hidden_size', type=int, default=100)
parser.add_argument('--output_size', type=int, default=2)
parser.add_argument('--num_epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--learning_rate', type=float, default=0.0004)
parser.add_argument('--filename', type=str, default='./data/simulation_50000sec_20000cars.xml')
args = parser.parse_args()
# Hyper Parameters
number_of_neighbours = args.neighbours
input_size = 21
hidden_size = args.hidden_size
output_size = args.output_size
num_epochs = args.num_epochs
batch_size = args.batch_size
learning_rate = args.learning_rate

dataLoader = SumoDataLoader(args.filename, 0.05, 0.05, batch_size, number_of_neighbours, 'log_guru')
exit()
dataLoader_forLoss = SumoDataLoader(args.filename, 0.05, 0.05, batch_size, number_of_neighbours)

net = Model(input_size, hidden_size, output_size)

    
# Loss and Optimizer
criterion = nn.MSELoss() 
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)  

def getFloatTensorFromNumpy(x):
    x = torch.from_numpy(x)
    x = x.float()
    return x
def getCost(mode, net):
    dataLoader_forLoss.reset()
    dataLoader_forLoss.switchTo(mode)
    total_loss = 0.0
    count = 0
    while 1:
        X, y, br = dataLoader_forLoss.nextBatch()
        
        X = Variable(getFloatTensorFromNumpy(X))
        y = Variable(getFloatTensorFromNumpy(y))
        outputs = net(X)
        loss = criterion(outputs,y)
        total_loss += loss.data.numpy()[0]
        count += 1
        if br:
            break
    return total_loss*1.0/count


train_losses = []
validation_losses = []
# Train the Model
for epoch in range(num_epochs):
    number_of_batches_p = 0
    while 1:
        X, y, br = dataLoader.nextBatch()

        X = Variable(getFloatTensorFromNumpy(X))
        y = Variable(getFloatTensorFromNumpy(y))
        
        # Forward + Backward + Optimize
        optimizer.zero_grad()  # zero the gradient buffer
        outputs = net(X)
        #print "Batch No : ", number_of_batches_p
        #print "Batch output : ", y, outputs.data.numpy()
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        if br:
            break
        number_of_batches_p += 1
    if (epoch+1) % 10 == 0:
            print ('Epoch [%d/%d], Step [%d], Loss: %.4f' 
                   %(epoch+1, num_epochs, number_of_batches_p+1, loss.data[0]))
            ##train_losses.append(getCost('train', net))
            train_losses.append(loss.data[0])
            validation_losses.append(getCost('validation', net))

# Test the Model
dataLoader.switchTo('test')
correct = 0
total = 0

while 1:
    X, y, br = dataLoader.nextBatch()
    X = Variable(getFloatTensorFromNumpy(X))
    outputs = net(X)
    total += outputs.size(0)
    errors = 1/2.0*np.square(outputs.data.numpy()-y)[:,0] + 1/2.0*np.square(outputs.data.numpy()-y)[:,1]
    correct += (errors < 0.1).sum()
    if br:
        break

print('Accuracy of the network on the '+str(total)+' test data: %d %%' % (100 * correct / total))

# Save the Model, lets think about the best model and all later
torch.save(net.state_dict(), args.filename.split('/')[-1].split('.')[0] + "_" + str(hidden_size) + "-" + str(output_size)+ "_" + str(input_size)+ "_" + str(num_epochs) + "_" + str(batch_size) + "_" + str(learning_rate) + "_" + str(number_of_neighbours) + '_model.pkl')
print "Losses"
print train_losses, validation_losses
#plt.plot(train_losses, 'r', validation_losses, 'b')
v_l = plt.plot(validation_losses, 'b', label='validation loss')
l = plt.plot(train_losses, 'r', label='training loss')
plt.xlabel('Number of Epochs')
plt.ylabel('MSE')
plt.legend(loc='upper right', shadow=True, fontsize='x-small')
plt.show()
