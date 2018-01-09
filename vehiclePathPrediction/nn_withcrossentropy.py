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
parser.add_argument('--output_size', type=int, default=1)
parser.add_argument('--num_epochs', type=int, default=1000)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--filename', type=str, default='data/simulation10000sec_1000cars.xml')
args = parser.parse_args()
# Hyper Parameters
number_of_neighbours = args.neighbours
input_size = 21
hidden_size = args.hidden_size
output_size = args.output_size
num_epochs = args.num_epochs
batch_size = args.batch_size
learning_rate = args.learning_rate

dataLoader = SumoDataLoader(args.filename, 0.1, 0.1, batch_size, number_of_neighbours)
dataLoader_forLoss = SumoDataLoader(args.filename, 0.1, 0.1, batch_size, number_of_neighbours)
netx = Model(input_size, hidden_size, output_size)
#nety = Model(input_size, hidden_size, output_size)

# Loss and Optimizer
criterion = nn.MSELoss()
optimizerx = torch.optim.Adam(netx.parameters(), lr=learning_rate)  
#optimizery = torch.optim.Adam(nety.parameters(), lr=learning_rate)  

def getFloatTensorFromNumpy(x):
    x = torch.from_numpy(x)
    x = x.float()
    return x

def getCost(mode, net, c):
    dataLoader_forLoss.reset()
    dataLoader_forLoss.switchTo(mode)
    total_loss = 0.0
    count = 0
    while 1:
        X, y, br = dataLoader_forLoss.nextBatch()
        
        X = Variable(getFloatTensorFromNumpy(X))
        y = Variable(getFloatTensorFromNumpy(y[:,c].reshape(y.shape[0],1)))
        outputs = net(X)
        loss = criterion(outputs,y)
        total_loss += loss.data.numpy()[0]
        count += 1
        if br:
            break
    return total_loss*1.0/count

train_lossesx = []
train_lossesy = []
validation_lossesx = []
validation_lossesy = []
# Train the Model
for epoch in range(num_epochs):
    number_of_batches_p = 0
    while 1:
        X, y, br = dataLoader.nextBatch()

        X = Variable(getFloatTensorFromNumpy(X))
        yx = Variable(getFloatTensorFromNumpy(y[:,0].reshape(y.shape[0],1)))
        yy = Variable(getFloatTensorFromNumpy(y[:,1].reshape(y.shape[0],1)))
        if number_of_batches_p == 0 and epoch == 0:
            print "Data : "
            print X.data.numpy(), yx.data.numpy()
        # Forward + Backward + Optimize
        optimizerx.zero_grad()  # zero the gradient buffer
        #optimizery.zero_grad()  # zero the gradient buffer
        outputsx = netx(X)
        #outputsy = nety(X)

        lossx = criterion(outputsx, yx)
        #lossy = criterion(outputsy, yy)
        lossx.backward()
        optimizerx.step()
        #lossy.backward()
        #optimizery.step()
        if (number_of_batches_p+1) % 100 == 0:
            print ('Epoch [%d/%d], Step [%d/%d], Loss: %.4f' 
                   %(epoch+1, num_epochs, number_of_batches_p+1, batch_size, lossx.data[0]))
            ##train_losses.append(getCost('train', net))
            train_lossesx.append(getCost('train', netx, 0))
            #train_lossesy.append(lossy.data[0])
            validation_lossesx.append(getCost('validation', netx, 0))
            #plt.plot(train_lossesx, 'r', validation_lossesx, 'b')
            #plt.savefig("Train_validation.jpg")
            #validation_lossesy.append(getCost('validation', nety, 1))
        if br:
            break
        number_of_batches_p += 1

# Test the Model
dataLoader.switchTo('test')
correct = 0
total = 0

while 1:
    X, y, br = dataLoader.nextBatch()
    X = Variable(getFloatTensorFromNumpy(X))
    outputsx = netx(X)
    #outputsy = nety(X)
    total += outputsx.size(0)
    #errors = 1/2.0*np.square(outputsx.data.numpy()-y[:,0]) + 1/2.0*np.square(outputsy.data.numpy()-y[:,1])
    errors = 1/2.0*np.square(outputsx.data.numpy()-y[:,0].reshape(y.shape[0],1))
    correct += (errors < 0.1).sum()
    #print "Outputx ", outputsx.data.numpy().shape
    #print X.data.numpy(), " -> ", y,  " -> ", outputsx.data.numpy()
    #print "Outputx ", outputsx.data.numpy().shape
    #print "y ", y.shape
    #print ((errors < 0.5).sum()), errors
    if br:
        break

print('Accuracy of the network on the '+str(total)+' test data: %d %%' % (100 * correct / total))

# Save x model and the y model, lets think about the best model and all later
torch.save(netx.state_dict(), args.filename.split('/')[-1].split('.')[0] + "_" + str(hidden_size) + "-" + str(output_size)+ "_" + str(input_size)+ "_" + str(num_epochs) + "_" + str(batch_size) + "_" + str(learning_rate) + "_" + str(number_of_neighbours) + '_xmodel.pkl')

#torch.save(nety.state_dict(), args.filename.split('/')[-1].split('.')[0] + "_" + str(hidden_size) + "-" + str(output_size)+ "_" + str(input_size)+ "_" + str(num_epochs) + "_" + str(batch_size) + "_" + str(learning_rate) + "_" + str(number_of_neighbours) + '_ymodel.pkl')
#plt.plot(train_lossesx, 'r', validation_lossesx, 'b', train_lossesy, 'g', validation_lossesy, 'y')
print "TrainLosses ", train_lossesx
print "ValidationLosses ", validation_lossesx
plt.plot(train_lossesx, 'r') #, validation_lossesx, 'b')
plt.show()
