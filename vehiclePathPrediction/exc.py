pred_velocities = []
actual_velocities = []
for line in open("exclusive.txt"):
    print line
    line = line[2:-1]
    line = line.split(" ")
    #print float(line[3][:-)
    #print float(line[0][:-2])
    actual_velocities.append(float(line[-1][:-1]))
    pred_velocities.append(float(line[0][:-2]))

dic = {}
dic1 = {}
for vel, vel1 in zip(actual_velocities, pred_velocities):
    if int(vel) not in dic.keys():
        dic[int(vel)] = 1
    else:
        dic[int(vel)] += 1

    if int(vel1) not in dic1.keys():
        dic1[int(vel1)] = 1
    else:
        dic1[int(vel1)] += 1

import matplotlib
matplotlib.use("tkAgg")
import matplotlib.pyplot as plt



plt.figure(1)

plt.bar(dic.keys(), dic.values())
plt.xticks([i for i in range(20)])
plt.xlabel("velocities")
plt.ylabel("count")

plt.figure(2)
plt.bar(dic1.keys(), dic1.values())
plt.xticks([i for i in range(20)])
plt.xlabel("velocities")
plt.ylabel("count")
plt.show()
