import pickle
import matplotlib
matplotlib.use("tkAgg")
from matplotlib import pyplot as plt

fig_size = plt.rcParams["figure.figsize"]

# Set figure width to 12 and height to 9
#fig_size[0] = 9
#fig_size[1] = 6
#plt.rcParams["figure.figsize"] = fig_size

#plt.rcParams['xtick.major.pad']='16'
#pylab.rcParams['ytick.major.pad']='8'

def getDistribution(train_label, text = "data"):
    dist = {}
    index = 1
    for data in train_label:
        #print "data", data
        if int(round(data[1])) not in dist.keys():    
            dist[int(round(data[1]))] = 1
        else:
            dist[int(round(data[1]))] += 1
    print dist
    ax = plt.bar(dist.keys(), dist.values())
    plt.xlabel(text)
    plt.ylabel('count')
    plt.xticks([i for i in range(-2,5)])
    #plt.xticks([i for i in range(0,10)])
    #plt.axis([-10,20])
    plt.legend(loc='upper right', shadow=True, fontsize='x-small')
    plt.show()
    return dist


pickle_in = open("metrics_real_final","rb")
velocity_metric, acceleration_metric, num_of_collision_metric  =  pickle.load(pickle_in)

#print velocity_metric

#print acceleration_metric

print num_of_collision_metric



print len(num_of_collision_metric['simulation'].keys())

count = 0

for key in num_of_collision_metric['simulation'].keys():
    count += num_of_collision_metric['simulation'][key]

print "total count = ", count
#dist1 = getDistribution(velocity_metric['original'], 'Integer part of velocity(m/s)')
#dist1 = getDistribution(acceleration_metric['original'], 'Integer part of acceleration(m/s)')
#dist1 = getDistribution(velocity_metric['simulation'], 'Integer part of velocity(m/s)')
dist1 = getDistribution(acceleration_metric['simulation'], 'Integer part of acceleration(m/s)')

'''
plt.figure(1)

plt.subplot(421)
plt.plot(velocity_metric['original'])
plt.subplot(422)
plt.plot(velocity_metric['simulation'])
plt.subplot(423)
dist = getDistribution(velocity_metric['original'])
print dist
plt.bar(dist.keys(), dist.values())
plt.xlabel("Original")
plt.ylabel("Count")
plt.subplot(424)
dist1 = getDistribution(velocity_metric['simulation'])
print dist1
plt.bar(dist1.keys(), dist1.values())
plt.xlabel("Simulation")
plt.ylabel("Count")

plt.subplot(425)
plt.plot(velocity_metric['original'])

plt.subplot(426)
plt.plot(velocity_metric['simulation'])

plt.subplot(427)
dist1 = getDistribution(acceleration_metric['original'])
print dist1
plt.bar(dist1.keys(), dist1.values())
plt.xlabel("Original")
plt.ylabel("Count")

plt.subplot(428)
dist1 = getDistribution(acceleration_metric['simulation'])
print dist1
plt.bar(dist1.keys(), dist1.values())
plt.xlabel("Simulation")
plt.ylabel("Count")
plt.show()
'''
