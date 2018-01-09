#! /users/guruprakash.r/miniconda2/bin/python
from roadNetwork import RoadNetwork
from vehicle import Vehicle
from node import Node
from edge import Edge
from yvehicleMover import YvehicleMover
from traffic import Traffic
import random
import numpy as np
from vehicleTypes import VehicleTypes
from simpleLaneMovement import SimpleLaneMovement
from edgeWithLanes import EdgeWithLanes
import pdb

def getSingleRoadNetwork(x1,y1,x2,y2,w):
	node1 = Node(x1,y1)
	node2 = Node(x2,y2)
	nodeList  = [node1, node2]
	edge = EdgeWithLanes(node1, node2, w, 4)
	edgeList = [edge]
	return RoadNetwork(nodeList, edgeList)

def getVehicles(noOfvehicles, route):
	vehicleList = []
	vehicleTypes = VehicleTypes()
	speeds = [1,2,3,4,5]
	speed_prob = [0.1,0.3,0.3,0.2,0.1]
	lanes = [0,1,2,3]
	lane_prob = [0.2,0.3,0.3,0.2]
	for i in range(noOfvehicles):
		vehicleType , _ = vehicleTypes.sample()
		lane = np.random.choice(lanes, 1, p=lane_prob)[0]
		speed = np.random.choice(speeds, 1, p=speed_prob)[0]
		#vehicle = Vehicle(vehicleType, route, int(10*random.random()), YvehicleMover())
		#vehicle = Vehicle(vehicleType, route, 0, SimpleLaneMovement())
		vehicle = Vehicle(vehicleType, route, 0, SimpleLaneMovement(), speed, lane)
		vehicleList.append(vehicle)
	return vehicleList

if __name__ == "__main__":
	network = getSingleRoadNetwork(300, 100, 300, 300, 200)
	print network
	numberOfPointsPerTrack = 100
	numberOfVehicles = 1
	for i in range(0, numberOfVehicles):
		vehicles = getVehicles(numberOfVehicles, network)
		traffic = Traffic(network, vehicles, (500,500,3), 300)
	traffic.simulateContinuousTraffic(5000, True, False, True)
	#traffic.generateTrafficUsingNN(1000, 'n0')
	#traffic.export()
	#traffic.exportGeneratedVehicles()