 #! /users/guruprakash.r/miniconda2/bin/python
from random import randint
import numpy as np
class VehicleTypes:
	def __init__(self):
		self.classes = ['CAR', 'MOTORBIKE', 'VAN', 'TRUCK', 'BUS']
		self.map = {'b':'BUS', 'c':'CAR', 'm':'MOTORBIKE', 'v':'VAN', 't':'TRUCK'}
                self.color = {'CAR':(255,255,0), 'BUS':(0,255,255), 'VAN':(0,255,0), 'MOTORBIKE':(0,0,255), 'TRUCK':(255,0,0)}
                self.sizes = {'CAR' : (8,25), 'TRUCK': (11,30), 'VAN': (11,30), 'MOTORBIKE': (2,10), 'BUS' :(15,60)}
                self.prob = {'CAR': 0.4, 'MOTORBIKE': 0.15, 'VAN': 0.2, 'TRUCK':0.15, 'BUS' : 0.1}

	def getIndex(self, vehicleType):
		return self.classes.index(vehicleType)

	def getSize(self, vehicleType): 
		return self.sizes[vehicleType]

	def listVehilceTypes(self):
		for ty in self.classes:
			print ty

	def oneHotEncoding(self, vehicleType):
		index =	self.getIndex(vehicleType)
		size = len(self.classes)
		encoding = [0]*size
		encoding[index] = 1
		return encoding

        def getTypeFromC(self, c):
                dic = {'c':'CAR', 'b':'BUS', 'v':'VAN', 't':'TRUCK', 'm':'MOTORBIKE'}
                return dic[c]

	def sample(self):
		#print self.classes
		#print self.prob
                prob  = [self.prob[x] for x in self.prob.keys()]
		cl = np.random.choice(self.classes, 1, p=prob)
		#cl = randint(0, len(self.classes)-1)
		encoding = self.oneHotEncoding(cl[0])
		return cl[0], encoding

if __name__=="__main__":
	import cv2
	import numpy as np
	vehicleTypes = VehicleTypes()
	t = 100
	while(t):
		img = np.zeros((1000,1000,3), np.uint8)
		vehi , _ = vehicleTypes.sample()
		print vehi
		# vehi_size = vehicleTypes.getSize(vehi)
		# #print vehi, vehi_size, 500-vehi_size[0]/2, 500-vehi_size[1]/2, 500+vehi_size[0]/2, 500+vehi_size[1]/2
		# cv2.rectangle(img,(500-vehi_size[0]/2,500-vehi_size[1]/2), (500+vehi_size[0]/2,500+vehi_size[1]/2), (0,255,0), 1)
		# cv2.imshow("rect", img)
		# cv2.waitKey(0)
		# cv2.destroyAllWindows()
		t -= 1
