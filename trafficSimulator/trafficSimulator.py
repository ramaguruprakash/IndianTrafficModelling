from roadNetwork import RoadNetwork
from vehicle import Vehicle
from node import Node
from edge import Edge
from yvehicleMover import YvehicleMover
from traffic import Traffic
import random
from vehicleTypes import VehicleTypes
from simpleLaneMovement import SimpleLaneMovement
from nnbasedmovement import nnBasedMovement
from kerasnnbasedmovement import kerasnnBasedMovement
from edgeWithLanes import EdgeWithLanes
import cv2
import numpy as np
import pdb

def getSingleLaneNetwork(x1,y1,x2,y2,w):
	node1 = Node(x1,y1)
	node2 = Node(x2,y2)
	nodeList  = [node1, node2]
	edge = EdgeWithLanes(node1, node2, w, 4)
	edgeList = [edge]
	return RoadNetwork(nodeList, edgeList)

def getSingleRoadNetwork(x1,y1,x2,y2,w):
	node1 = Node(x1,y1)
	node2 = Node(x2,y2)
	nodeList  = [node1, node2]
	edge = EdgeWithLanes(node1, node2, w, 2)
	edgeList = [edge]
	return RoadNetwork(nodeList, edgeList)

def getVehicles(noOfvehicles, route):
	vehicleList = []
	vehicleTypes = VehicleTypes()
	for i in range(noOfvehicles):
		vehicleType , _ = vehicleTypes.sample()
		#vehicle = Vehicle(vehicleType, route, int(10*random.random()), YvehicleMover())
		vehicle = Vehicle(vehicleType, route, int(10*random.random()), SimpleLaneMovement())
		vehicleList.append(vehicle)
	return vehicleList

def getVehiclesSumoBasic(vehicle_count, route):
        vehicleList = []
        vehicleTypes = VehicleTypes() 
        #updater = nnBasedMovement()
        updater = kerasnnBasedMovement()
        prev_lane = 0
        t = 0
        for i in range(vehicle_count):
                vehicleType , _ = vehicleTypes.sample()
                #vehicle = Vehicle(vehicleType, route, int(9920*random.random()), SimpleLaneMovement(), int(30*random.random()))
                vehicle = Vehicle(vehicleType, route, t, updater, int(15*random.random())+1, -1, i, vehicleTypes.color[vehicleType], -1) #prev_lane)
                #vehicle = Vehicle(vehicleType, route, int(3370*random.random()), updater, int(25*random.random())+5, -1, i, vehicleTypes.color[vehicleType], -1) #prev_lane)
                prev_lane = vehicle.laneNo
                #vehicle = Vehicle(vehicleType, route, int(9920*random.random()), nnBasedMovement_xy(), int(30*random.random()), -1, i, vehicleTypes.color[vehicleType])
                vehicleList.append(vehicle)
                t = t + 4
        return vehicleList

def getMixedVehicles(vehicle_count, route, prob):
        vehicleList = []
        vehicleTypes = VehicleTypes()
        for i in range(vehicle_count):
                #vehicleType , _ = vehicleTypes.sample()
                vehicleType = "CAR"
                #vehicle = Vehicle(vehicleType, route, int(9920*random.random()), SimpleLaneMovement(), int(30*random.random()))
                if random.random() < prob:
                    vehicle = Vehicle(vehicleType, route, int(9920*random.random()), nnBasedMovement(), int(30*random.random()), -1, i, (0,0,255))
                else:
                    vehicle = Vehicle(vehicleType, route, int(9920*random.random()), SimpleLaneMovement(), int(12), -1, i, (255,0,0))
                vehicleList.append(vehicle)
        return vehicleList

def getSumoVehiclesFromFile(filename, route):
        vehicleList = []
        vehicleTypes = VehicleTypes()
        updater = kerasnnBasedMovement()
        for i,line in enumerate(open(fileName)):
            line = line[:-1]
            line = line.split(",")
            print line
            vehicle = Vehicle(line[1], route, line[0], updater, line[3], line[2], vehicleTypes.color[vehicleType])
            vehicleList.append(vehicle)
        return vehicleList

def getVehiclesFromFile(fileName, route):
        vehicleList = []
        vehicleTypes = VehicleTypes()
        for i,line in enumerate(open(fileName)):
                if line[0] == 'n':
                    continue
                vehicleType = vehicleTypes.getTypeFromC(line[0])
                vehicle = Vehicle(vehicleType, route, i, SimpleLaneMovement(), 1, int(line[1]), i, vehicleTypes.color[vehicleType])
                vehicleList.append(vehicle)
        print len(vehicleList)
        return vehicleList


def getVehiclesFromFileReal(fileName, route):
        vehicleList = []
        vehicleTypes = VehicleTypes()
        vehicleMover = nnBasedMovement()
        for i,line in enumerate(open(fileName)):
                if line[0] == 'n':
                    continue
                vehicleType = vehicleTypes.getTypeFromC(line[0])
                vehicle = Vehicle(vehicleType, route, i, vehicleMover, int(30*random.random()), int(line[1]), i, vehicleTypes.color[vehicleType])
                vehicleList.append(vehicle)
        print len(vehicleList)
        return vehicleList


if __name__ == "__main__":
	#network = getSingleLaneNetwork(400, 1000, 400, 0, 300)
	network = getSingleLaneNetwork(106, 660, 106, 0, 212)  ## - Sumo video
	#network = getSingleLaneNetwork(180, 660, 180, 0, 360)  ## - Sumo video
        #network = getSingleLaneNetwork(150, 384, 150, 0, 100)  ## - Singapore video
        #network = getSingleLaneNetwork(330, 660, 330, 0, 660)  ## - Singapore video
	#network = getSingleRoadNetwork(106, 660, 106, 0, 40)  ## - US101 data set
	#print network
	numberOfTracks = 1
	numberOfPointsPerTrack = 100
	numberOfVehicles = 1000
	for i in range(0, numberOfTracks, numberOfVehicles):
		#vehicles = getVehicles(numberOfVehicles, network)
                #vehicles = getVehiclesFromFile("/Users/gramag`uru/Desktop/car_videos/output_sing_sequences_appended_double.txt", network)
                #vehicles = getVehiclesFromFile("/Users/gramaguru/Desktop/car_videos/output_sing_sequences_appended_double_nonewline.txt", network)
                #vehicles = getVehiclesFromFileReal("/Users/gramaguru/Desktop/car_videos/sing_cropped_vehicle_generated_double.txt", network)
                vehicles = getVehiclesSumoBasic(numberOfVehicles, network)
                #vehicles = getMixedVehicles(numberOfVehicles, network, 0.2)
                ## Calculate homography and send it accross for using it while printing.
                print  "Traffic Start"
		#traffic = Traffic(network, vehicles, (1000,1000,3), 10000)
                #traffic = Traffic(network, vehicles, (384,512,3), 86400)  ## - Singapore video
		traffic = Traffic(network, vehicles, (660,212,3), 10000)  ## - Sumo video
		#traffic = Traffic(network, vehicles, (660,360,3), 10000)  ## - Sumo video
		#traffic = Traffic(network, vehicles, (660,60,3), 1000)  ## - US101 video
    
        #straight_road_points = np.array([[200, 100],[300, 100],[300, 484],[200, 484]], dtype=float);
        #straight_road_points = np.array([[100, 0],[200, 0],[200, 384],[100, 384]], dtype=float);
        #perspective_road_points = np.array([[202, 105], [240, 105], [255, 300], [0, 300]], dtype=float);
        #H, status = cv2.findHomography(straight_road_points, perspective_road_points);
        
	#traffic.simulateAndVisualizeWithHomography(H, straight_road_points, perspective_road_points)
	traffic.simulateAndVisualize()
	#traffic.export()
		#traffic.simulateAndExport()
