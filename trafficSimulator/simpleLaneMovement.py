#! /users/guruprakash.r/miniconda2/bin/python

from vehicleBehavior import VehicleBehavior
from roadNetwork import RoadNetwork
import random
from vehicleTypes import VehicleTypes
from random import randint
import numpy as np

class SimpleLaneMovement(VehicleBehavior):
	def __init__(self):
		self.vehicleTypes = VehicleTypes()

	def rectangleOverLap(self, ltp1x, ltp1y, rbp1x, rbp1y, ltp2x, ltp2y, rbp2x, rbp2y):
		if(ltp1x > rbp2x or ltp2x > rbp1x):
			return False

		if(ltp1y > rbp2y or ltp2y > rbp1y):
			return False

		return True

	def positionempty(self, posX, posY, curVehicle, vehicles):

		vehi_x = curVehicle.curX
		vehi_y = curVehicle.curY

		curVehicleSize = self.vehicleTypes.getSize(curVehicle.cl)
		rec1ltx = posX - curVehicleSize[0]/2
		rect1lty = posY - curVehicleSize[1]/2
		rect1rbx = posX + curVehicleSize[0]/2
		rect1rby = posY + curVehicleSize[1]/2

		for vehicle in vehicles:
			if vehi_x == vehicle.curX and vehi_y == vehicle.curY:
				continue
			vehi_size = self.vehicleTypes.getSize(vehicle.cl)
			rect2ltx = vehicle.curX - vehi_size[0]/2
			rect2lty = vehicle.curY - vehi_size[1]/2
			rect2rbx = vehicle.curX + vehi_size[0]/2
			rect2rby = vehicle.curY + vehi_size[1]/2
			if self.rectangleOverLap(rec1ltx, rect1lty, rect1rbx, rect1rby, rect2ltx, rect2lty, rect2rbx, rect2rby):
					#print "check this man " + str(vehicle.curX) + "," + str(vehicle.curY) + " " + str(posX) + " " + str(posY)
					return False
		return True

	def updatePos(self, grid, curVehicle, vehicles):
		posX = curVehicle.curX
		posY = curVehicle.curY
		route = curVehicle.route
		edgeNo = curVehicle.numberOfEdgesCompleted
		speed = curVehicle.speed
		laneNo = curVehicle.laneNo
		shape = grid.shape
		sizeX = shape[0]
		sizeY = shape[1]

		if (edgeNo >= len(route.edges)):
			return -1,-1,-1

		initNode = route.edges[edgeNo].node1
		finalNode = route.edges[edgeNo].node2
		width = route.edges[edgeNo].width
		lanes = route.edges[edgeNo].lanes
		roadLeftBottomX =  initNode.x - width/2
		roadLeftBottomY = initNode.y
		roadRightTopX = finalNode.x + width/2
		roadRightTopY = finalNode.y
                ## LeftBottomX will be x - something
		if posX == -1 and posY == -1:
			tr = 0
			while 1:
				x = laneNo
				x = roadLeftBottomX + ((x*width)/lanes) + (width/lanes)/2
				if self.positionempty(x, roadLeftBottomY, curVehicle, vehicles):
					#print "wth man " + str(x) + " " + str(roadLeftBottomY) 
					return edgeNo, x, roadLeftBottomY
				tr += 1
				if tr >= 100:
					return edgeNo, posX, posY

		else:
			posY -= speed
			if(posY >= sizeX or posY < 0):
                                #print "ikada 1, ",posY, sizeX 
				return -1, -1, -1

			if not self.positionempty(posX, posY, curVehicle, vehicles):
                                #print "ikada 2"
				return edgeNo, posX, posY + speed
			if((posY <= roadRightTopY) and (edgeNo == len(route.edges))):
                                #print "ikada 3"
				return -1,-1,-1
			elif(posY <= roadRightTopY):
                                #print "ikada 4"
				edgeNo += 1

			return edgeNo, posX, posY
