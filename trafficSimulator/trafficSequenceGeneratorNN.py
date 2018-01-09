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
	for i in range(noOfvehicles):
		vehicleType , _ = vehicleTypes.sample()
		#vehicle = Vehicle(vehicleType, route, int(10*random.random()), YvehicleMover())
		vehicle = Vehicle(vehicleType, route, 0, SimpleLaneMovement())
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
	#traffic.simulateContinuousTraffic(1000, True)
	traffic.generateTrafficUsingNN(5000, 'n00')
	#traffic.export()
	#traffic.exportGeneratedVehicles()
