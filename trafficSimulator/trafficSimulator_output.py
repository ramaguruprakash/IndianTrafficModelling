#! /users/guruprakash.r/miniconda2/bin/python
from roadNetwork import RoadNetwork
from vehicle import Vehicle
from node import Node
from edge import Edge
from yvehicleMover import YvehicleMover
from traffic import Traffic
import random
from vehicleTypes import VehicleTypes
from simpleLaneMovement import SimpleLaneMovement
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

def getVehicles(noOfvehicles, route):
	vehicleList = []
	vehicleTypes = VehicleTypes()
	for i in range(noOfvehicles):
		vehicleType , _ = vehicleTypes.sample()
		#vehicle = Vehicle(vehicleType, route, int(10*random.random()), YvehicleMover())
		vehicle = Vehicle(vehicleType, route, int(10*random.random()), SimpleLaneMovement())
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

if __name__ == "__main__":
	network = getSingleLaneNetwork(150, 384, 150, 0, 100)
	print network
	numberOfTracks = 1
	numberOfPointsPerTrack = 100
	numberOfVehicles = 10
	for i in range(0, numberOfTracks, numberOfVehicles):
		#vehicles = getVehicles(numberOfVehicles, network)
                #vehicles = getVehiclesFromFile("/Users/gramaguru/Desktop/car_videos/output_sing_sequences_appended_double.txt", network)
                #vehicles = getVehiclesFromFile("/Users/gramaguru/Desktop/car_videos/output_sing_sequences_appended_double_nonewline.txt", network)
                vehicles = getVehiclesFromFile("/Users/gramaguru/Desktop/car_videos/output_sing_sequences_appended_double.txt", network)
                ## Calculate homography and send it accross for using it while printing.
		traffic = Traffic(network, vehicles, (384,512,3), 86400)

        straight_road_points = np.array([[200, 100],[300, 100],[300, 484],[200, 484]], dtype=float);
        straight_road_points = np.array([[100, 0],[200, 0],[200, 384],[100, 384]], dtype=float);
        perspective_road_points = np.array([[202, 105], [240, 105], [255, 300], [0, 300]], dtype=float);
        H, status = cv2.findHomography(straight_road_points, perspective_road_points);
	traffic.simulateAndVisualizeWithHomography(H, straight_road_points, perspective_road_points)
	#traffic.export()
		#traffic.simulateAndExport()
