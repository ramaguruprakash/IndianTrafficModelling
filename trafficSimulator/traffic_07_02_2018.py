from roadNetwork import RoadNetwork
from vehicle import Vehicle
import numpy as np
import cv2
import time
import torch
from random import randint
import random
from vehicleTypes import VehicleTypes
from scipy import misc
from simpleLaneMovement import SimpleLaneMovement
import sys
sys.path.append("/Users/gramaguru/ComputerVision/computer-vision/Traffic/IndianTraffic/trafficGenerator/trafficNextPrediction")
sys.path.append("/Users/gramaguru/ComputerVision/computer-vision/tracksExtractor/homography")
from homographyLib import drawPoly, transformPoly, transformSetOfPointsAndReturn
from generate_again import generate_couple

class Traffic:
	def __init__(self, roadNetwork, vehicles, size, totalTime):
		self.roadNetwork = roadNetwork
		self.vehicles = vehicles
		self.size = size
		self.supportVehicleSize = 1
		self.exportVehicleIndices = [-1]*30
		self.vehicleSequences = []
		self.exportfp = open("trafficExport.data","w")
		self.totalTime = totalTime
		self.vehicleTypes = VehicleTypes()
		self.grid = np.zeros(size)
		self.drawNetwork()
		self.drawVehicles()


	def addVehicle(self, lanes, vehicleTypes):
		if vehicleTypes:
			vehicleType, _ = self.vehicleTypes.sample()
		else:
			vehicleType = 'CAR'
		if not lanes:
			vehicle = Vehicle(vehicleType, self.roadNetwork, 0, SimpleLaneMovement())
		else:
			l = [1,2]
			laneNo = l[int(random.random()*2)]
			vehicle = Vehicle(vehicleType, self.roadNetwork, 0, SimpleLaneMovement(), 4, laneNo)
		return vehicle

	def addVehicleWithSpeed(self):
		speed = [1,2,3,4,5]
		speed_prob = [0.1,0.3,0.3,0.2,0.1]
		lane = [0,1,2,3]
		lane_prob = [0.2,0.3,0.3,0.2]
		speed = np.random.choice(speed, 1, p=speed_prob)[0]
		lane = np.random.choice(lane, 1, p=lane_prob)[0]
		vehicleType, _ = self.vehicleTypes.sample()
		vehicle = Vehicle(vehicleType, self.roadNetwork, 0, SimpleLaneMovement(), speed, lane)
		return vehicle

	def drawNetwork(self):
		edges = self.roadNetwork.edges
		nodes = self.roadNetwork.nodes
		for edge in edges:
			nodeI = edge.node1
			nodeJ = edge.node2
			width = edge.width
			### Prastutaniki angle em ledhu, only straight lines., commented because I wanted the road to cover the entire the screen, its ok to uncomment if you want.
			#cv2.line(self.grid, (int(nodeI.x-width/2), nodeI.y), (int(nodeI.x-width/2), nodeJ.y), (255,0,0),1)
			#cv2.line(self.grid, (int(nodeI.x+width/2), nodeI.y), (int(nodeI.x+width/2), nodeJ.y), (255,0,0),1)
                        ## Drawing a center white line for SUMO purpose
                        cv2.line(self.grid, (nodeI.x, nodeI.y), (nodeJ.x, nodeJ.y), (255,255,255), 1)
			### if Edges have lanes draw the lanes.

	def drawNetworkWithHomography(self, H):
		edges = self.roadNetwork.edges
		nodes = self.roadNetwork.nodes
		for edge in edges:
			nodeI = edge.node1
			nodeJ = edge.node2
			width = edge.width
			### Prastutaniki angle em ledhu, only straight lines.
                        pts1 = np.array([[int(nodeI.x-width/2), nodeI.y], [int(nodeI.x-width/2), nodeJ.y]])
                        pts2 = np.array([[int(nodeI.x+width/2), nodeI.y], [int(nodeI.x+width/2), nodeJ.y]])
                        pts1 = transformSetOfPointsAndReturn(pts1, H)
                        pts2 = transformSetOfPointsAndReturn(pts2, H)
			cv2.line(self.grid, (pts1[0][0], pts1[0][1]), (pts1[1][0], pts1[1][1]), (255,0,0), 1)
			cv2.line(self.grid, (pts2[0][0], pts2[0][1]), (pts2[1][0], pts2[1][1]), (255,0,0), 1)
		
	def get5nearestNeighbours(self, vehicleIndex):
		dist = {}
		a = np.array([self.vehicles[vehicleIndex].curX, self.vehicles[vehicleIndex].curY])
		for i,vehicle in enumerate(self.vehicles):
			if (i == vehicleIndex) or (len(vehicle.track)) < 2 or (vehicle.track[-2][0] == -1 and vehicle.track[-2][1] == -1):
				continue
			b = np.array([vehicle.curX, vehicle.curY])
			d = np.linalg.norm(a-b)
			if len(dist.keys()) < 5:
				dist[d] = i
				continue
			mx = max(dist.keys())
			if mx > d:
				dist.pop(mx,None)
		return dist

	def updateExportVehiclesIndices(self):
		l = []
		for i in self.exportVehicleIndices:
				v1 = []
				print i,
				if i == -1 or len(self.vehicles[i].track) < 2 or (self.vehicles[i].track[-2][0] == -1 and self.vehicles[i].track[-2][1] == -1):
					v1 += [-1]*17
				else:
					v1 += self.vehicleTypes.oneHotEncoding(self.vehicles[i].cl)
					x = self.vehicles[i].track[-2][0]
					y = self.vehicles[i].track[-2][1]
					v1 += [x, y]
					print v1
					neighs = self.get5nearestNeighbours(i)
					print neighs
					for n in neighs.keys():
							if self.vehicles[neighs[n]].track[-2][0] != -1 and self.vehicles[neighs[n]].track[-2][1] != -1:
								v1 += [x-self.vehicles[neighs[n]].track[-2][0], y-self.vehicles[neighs[n]].track[-2][1]]
							else:
								v1 +=[-1,-1]
					for k in range(len(neighs.keys()),5):
							v1 += [-1,-1]
					if self.vehicles[i].track[-2][0] == -1 and self.vehicles[i].track[-2][1] == -1:
						v1 += [0,0]
					else:
						v1 +=  [self.vehicles[i].track[-2][0] - self.vehicles[i].curX, self.vehicles[i].track[-2][1] - self.vehicles[i].curY]
				print v1,
				l.append(v1)
		print "\n",
		for i, vehicle in enumerate(self.vehicles):
			#print "Guru ", i, vehicle.numberOfEdgesCompleted
			if vehicle.numberOfEdgesCompleted == 1 and i in self.exportVehicleIndices:
				self.exportVehicleIndices[self.exportVehicleIndices.index(i)] = -1
			elif i not in self.exportVehicleIndices:
				if -1 in self.exportVehicleIndices:
					self.exportVehicleIndices[self.exportVehicleIndices.index(-1)] = i

        def drawVehicles(self):
		for vehicle in self.vehicles:
			if vehicle.curX == -1 and vehicle.curY == -1:
				continue
			vehi_size = self.vehicleTypes.getSize(vehicle.cl)
			if self.supportVehicleSize:
                                #print "Rectangles to be printed are : ", vehicle.curX-vehi_size[0]/2,vehicle.curY-vehi_size[1]/2, "   " , vehicle.curX+vehi_size[0]/2, vehicle.curY+vehi_size[1]/2
				cv2.rectangle(self.grid,(vehicle.curX-vehi_size[0]/2,vehicle.curY-vehi_size[1]/2), (vehicle.curX+vehi_size[0]/2,vehicle.curY+vehi_size[1]/2), vehicle.color, thickness=cv2.cv.CV_FILLED)
                                cv2.putText(self.grid, str(vehicle.identity), (vehicle.curX-vehi_size[0]/2,vehicle.curY-vehi_size[1]/2), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,0,0), 1, cv2.CV_AA)
			else:
				self.grid[vehicle.curY, vehicle.curX] = [255,0,0]

        def drawVehiclesWithHomography(self, H):
		for vehicle in self.vehicles:
			if vehicle.curX == -1 and vehicle.curY == -1:
				continue
			vehi_size = self.vehicleTypes.getSize(vehicle.cl)
			if self.supportVehicleSize:
                                pts = np.array([[vehicle.curX-vehi_size[0]/2,vehicle.curY-vehi_size[1]/2],[vehicle.curX+vehi_size[0]/2, vehicle.curY-vehi_size[1]/2],[vehicle.curX+vehi_size[0]/2, vehicle.curY+vehi_size[1]/2], [vehicle.curX-vehi_size[0]/2, vehicle.curY+vehi_size[1]/2]], dtype=float)
                                pts = transformPoly(pts, H)
                                self.grid = drawPoly(self.grid, pts, vehicle.color)
				#cv2.rectangle(self.grid,(vehicle.curX-vehi_size[0]/2,vehicle.curY-vehi_size[1]/2), (vehicle.curX+vehi_size[0]/2,vehicle.curY+vehi_size[1]/2), (0,255,0), 1)
			else:
				self.grid[vehicle.curY, vehicle.curX] = [255,0,0]


	def updateGrid(self):
		self.grid = np.zeros(self.size)
		self.drawNetwork()
		self.drawVehicles()

        def updateGridWithHomography(self, H, perspective_road_points, straight_road_points):
                self.grid = np.zeros(self.size)
                #self.grid = drawPoly(self.grid, perspective_road_points)
                #self.grid = drawPoly(self.grid, straight_road_points)
                self.drawNetworkWithHomography(H)
                self.drawVehiclesWithHomography(H)
        
        def simulateAndVisualize(self):
		t = 0
		while(t < self.totalTime):
                        print "=========== Frame no : ", t, " =========="
			for vehicle in self.vehicles:
				vehicle.move(t, self.grid, self.vehicles)
			self.updateGrid()
			t += 1
			cv2.imshow("sim", self.grid)
			if cv2.waitKey(33) == 27:
				break
			#if random.random() < 0.1:
			#	self.vehicles.append(self.addVehicle())
#			cv2.waitKey(0)
		cv2.destroyAllWindows()

        def simulateAndVisualizeWithHomography(self, H, perspective_road_points, straight_road_points):
		t = 0
                video ="/Users/gramaguru/Desktop/car_videos/sing.mp4"
                cap = cv2.VideoCapture(video);
		while(t < self.totalTime):
                        ret, frame = cap.read()
                        print frame.shape
                        frame = misc.imresize(frame, (384,512,3))
			for vehicle in self.vehicles:
				vehicle.move(t, self.grid, self.vehicles)
			self.updateGridWithHomography(H, perspective_road_points, straight_road_points)
			t += 1
                        cv2.imshow("orginal", frame)
			cv2.imshow("sim", self.grid)
			if cv2.waitKey(33) == 27:
				break
			#if random.random() < 0.1:
			#	self.vehicles.append(self.addVehicle())
#			cv2.waitKey(0)
		cv2.destroyAllWindows()


	def simulateContinuousTraffic(self, total_time, speed=False, lanes=False, vehicleTypes = False):
		t = 0
		while(t < total_time):
			for vehicle in self.vehicles:
				vehicle.move(t, self.grid, self.vehicles)
			self.updateGrid()
			#cv2.imshow("sim", self.grid)
			#if cv2.waitKey(33) == 27:
			#	break
			#if t % 5 == 0:
			if random.random() <= 0.2:
				if speed:
					self.vehicles.append(self.addVehicleWithSpeed())
				else:
					self.vehicles.append(self.addVehicle(lanes,vehicleTypes))
				#self.vehicleSequences.append(self.vehicleTypes.oneHotEncoding(vehicle.cl) + self.roadNetwork.edges[vehicle.numberOfEdgesCompleted].oneHotEncoding(vehicle.laneNo))
				if speed:
					self.vehicleSequences.append(vehicle.cl + " " + str(vehicle.laneNo) + " " + str(vehicle.speed))
				else:
					self.vehicleSequences.append(vehicle.cl + " " + str(vehicle.laneNo))
				#print "Guru ", self.vehicleTypes.oneHotEncoding(vehicle.cl) + self.roadNetwork.edges[vehicle.numberOfEdgesCompleted].oneHotEncoding(vehicle.laneNo)
			else:
				if speed:
					self.vehicleSequences.append(("None 0 0"))
				else:
					self.vehicleSequences.append(("None 0"))
				#self.updateExportVehiclesIndices()
			t += 1
			#print self.vehicleSequences
		self.exportGeneratedVehicles(speed)
		cv2.destroyAllWindows()

	def generateTrafficUsingNN(self, total_time, start_seq):
		# Parse command line arguments
		t = 0
		#decoder = torch.load("/Users/gramaguru/ComputerVision/computer-vision/Traffic/IndianTraffic/trafficGenerator/trafficNextPrediction/exportGeneratedAlphabets.pt")
		#decoder = torch.load("/Users/gramaguru/ComputerVision/computer-vision/Traffic/IndianTraffic/trafficGenerator/trafficNextPrediction/traffic_seq_road_only_cars.pt")
		decoder = torch.load("/Users/gramaguru/ComputerVision/computer-vision/Traffic/IndianTraffic/trafficGenerator/trafficNextPrediction/traffic_seq_roadNetwork_speed.export_51.pt")
		parameter_len = 3
		seq = generate_couple(decoder, start_seq, total_time, 0.8, False, parameter_len)
		print seq
		while(t < total_time):
			v = seq[parameter_len*t:parameter_len*t+parameter_len]
			laneNo = int(v[1])
			if parameter_len == 2:
				speed = 4
			elif parameter_len == 3:
				speed = int(v[2])
			if v[0] != 'n' and speed != 0:
					self.vehicles.append(Vehicle(self.vehicleTypes.map[v[0]], self.roadNetwork, 0, SimpleLaneMovement(), speed, laneNo))

			for vehicle in self.vehicles:
				vehicle.move(t, self.grid, self.vehicles)
			self.updateGrid()
			cv2.imshow("NNsim", self.grid)
			if cv2.waitKey(33) == 27:
				break
			t += 1

	def simulateAndExport(self):
		t = 0
		while(t < self.totalTime):
			for vehicle in self.vehicles:
				vehicle.move(t, self.grid, self.vehicles)
			self.updateGrid()
			t += 1
		self.export()

	def exportGeneratedVehicles(self, speed= False):
		#exportGeneratedSeq = raw_input()
		if speed:
			exportGeneratedSeq = "traffic_seq_prob_dist_speed.txt";
		else:

			exportGeneratedSeq = "traffic_seq_prob_dist.txt";
		print self.vehicleSequences
		fp = open(exportGeneratedSeq, "w")
		fp.write(str(self.vehicleSequences))
		fp.close()

	def export(self):
		exportFileName = "traffic_" + str(len(self.roadNetwork.edges)) + "_" + str(len(self.vehicles)) + "_" + str(time.time()) + ".txt"
		fp = open(exportFileName, "w")
		trafficSummary = str(self.grid.shape[0]) + " " + str(self.grid.shape[1]) + " " + str(self.totalTime) + "\n"
		fp.write(trafficSummary)

		fp.write(str(len(self.roadNetwork.edges)) + "\n")
		for edge in self.roadNetwork.edges:
			fp.write(str(edge.node1.x) + " " + str(edge.node1.y) + " " + str(edge.node2.x) + " " + str(edge.node2.y) + " " + str(edge.width) + "\n")

		fp.write(str(len(self.vehicles)) + "\n")
		for vehicle in self.vehicles:
			fp.write(vehicle.cl + "\n")
			tracklen = len(vehicle.track)
			diff = 0
			if tracklen < self.totalTime:
				diff = self.totalTime - tracklen

			for t in range(self.totalTime):
				if t < diff:
					fp.write(str(t) + " " + str(-1) + " " + str(-1) + "\n")
					continue
				fp.write(str(t) + " " + str(vehicle.track[t-diff][0]) + " " + str(vehicle.track[t-diff][1]) + "\n")
		fp.close()
        def getSpeeds(self, vehicles):
            speeds = {}
            for vehicle in vehicles:
                if len(vehicle.track) <= 1 or vehicle.track[-1][0] == -1 or vehicle.track[-1][1] == -1 or vehicle.track[-2][0] == -1 or vehicle.track[-2][1] == -1:
                    speeds[vehicle.identity] = [0.0,vehicle.speed]
                    print "Guru Guru", vehicle.identity, vehicle.speed
                else:
                    speeds[vehicle.identity] = [vehicle.track[-2][0] - vehicle.track[-1][0], vehicle.track[-2][1] - vehicle.track[-1][1]]
            return speeds

        def get_vehicles_lanes(self, vehicles):
            lanes_frames = {}
            for vehicle in vehicles:
                if vehicle.curX not in lanes_frames.keys():
                  lanes_frames[vehicle.curX] = [[vehicle.identity, vehicle.curX, vehicle.curY]]
                else:
                  lanes_frames[vehicle.curX].append([vehicle.identity, vehicle.curX, vehicle.curY])
            return lanes_frames

        def get_vehicles_lanes_dist(self, vehicles, lanes):
            lanes_frames = {}
            for vehicle in vehicles:
                lane = min(lanes, key=lambda x:abs(x-vehicle.curX))
                if lane not in lanes_frames.keys():
                  lanes_frames[lane] = [[vehicle.identity, vehicle.curX, vehicle.curY]]
                else:
                  lanes_frames[lane].append([vehicle.identity, vehicle.curX, vehicle.curY])
            return lanes_frames


        def getVehiclesInScreen(self, vehicles):
            filtered_vehicles = []
            for vehicle in vehicles:
                if vehicle.curX == -1 or vehicle.curY == -1:
                    continue
                filtered_vehicles.append(vehicle)
            return filtered_vehicles
''' 
        def moveall(self):
            self.vehicleTypes = VehicleTypes()
            self.dataLoader = SumoDataLoader("",0.1,0.1,1,3,'infer', True)
            self.max_xval = [  8.01000000e+01   1.66500000e+01   4.00000000e+00   6.42890000e+02
                       8.01000000e+01   1.80500000e+01  -1.00000000e+00   8.01000000e+01
                          1.80400000e+01   6.34310000e+02   8.01000000e+01   1.78700000e+01
                            -1.00000000e-02   8.01000000e+01   1.80400000e+01   6.12900000e+02
                               8.01000000e+01   1.79600000e+01  -2.00000000e-02   8.01000000e+01
                                  1.80400000e+01]
            self.max_yval = 15.99
            self.min_xval = [ -80.1     0.      0.     -1.    -80.1    -1.   -641.88  -80.1    -1.
                       -1.    -80.1    -1.   -592.8   -80.1    -1.     -1.    -80.1    -1.
                        -639.8   -80.1    -1.  ]
            self.min_yval = 10
            json_file = open('kerasmodels/model_1.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            self.loaded_model = model_from_json(loaded_model_json)
            # load weights into new model
            self.loaded_model.load_weights("kerasmodels/model_1.h5")
            print("Loaded model from disk")
            # create feature vector for all the vehicles
            vehicles = self.getVehiclesInScreen(vehicles)
            laneszz = [roadLeftBottomX + ((i*width)/lanes) + (width/lanes)/2 for i in range(4)]
            feature_vecs = []
            for vehicle in self.vehicles:
                speeds = self.getSpeeds(vehicles)
                vehicles_per_lanes = self.get_vehicles_lanes_dist(vehicles, laneszz)
                #print "speeds : ", speeds
                #print "==="
                print "vehicle per lanes : ", vehicles_per_lanes
                #print "===="
                vehicle = np.array([identity, posX, posY, self.dataLoader.carType[curVehicle.cl]])
                feature_vec = np.array(self.dataLoader.getFeatureVectorImp(vehicle, speeds, vehicles_per_lanes, laneszz))
                feature_vec.resize(1, feature_vec.shape[0])
                feature_vec = (feature_vec - self.min_xval)/(self.max_xval - self.min_xval)
                feature_vecs.append(feature_vec)
            feature_vecs = np.array(feature_vecs)
            vels = self.loadedmodel.predict(feature_vecs)
            # update all the vehicles current positions etc.
'''

