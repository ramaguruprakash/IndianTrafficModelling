import matplotlib
matplotlib.use("tkAgg")
from matplotlib import pyplot as plt

fp = open("losses_singleVehicle_100_100","r")
data = fp.read()
data = data[:-1]
data = data.split(",")
data[0] = data[0][1:]
data[0] = " " + data[0]
data[len(data)-1] = data[len(data)-1][:-1]
for i, d in enumerate(data):
	d = d[2:]
	d = d[:-1]
	data[i] = float(d)
print data
plt.plot(data)
plt.show()	
