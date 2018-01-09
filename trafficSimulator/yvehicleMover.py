#! /users/guruprakash.r/miniconda2/bin/python

from vehicleBehavior import VehicleBehavior
from roadNetwork import RoadNetwork
import random
import numpy as np

class YvehicleMover(VehicleBehavior):
	def __init__(self):
		pass

	def updatePos(self, grid, curVehicle, vehicles):
		posX = curVehicle.curX
		posY = curVehicle.curY
		route = curVehicle.route
		edgeNo = curVehicle.numberOfEdgesCompleted
		speed = curVehicle.speed
		shape = grid.shape
		sizeX = shape[0]
		sizeY = shape[1]
		initNode = route.edges[edgeNo].node1
		finalNode = route.edges[edgeNo].node2
		width = route.edges[edgeNo].width
		roadLeftBottomX =  initNode.x - width/2
		roadLeftBottomY = initNode.y
		roadRightTopX = finalNode.x + width/2
		roadRightTopY = finalNode.y

		if posX == -1 and posY == -1:
			tr = 0
			while 1:
				x = int((width*(random.random())))
				x = roadLeftBottomX + x
				if np.array_equal(grid[x,roadLeftBottomY],[0,0,0]):
					return edgeNo, x, roadLeftBottomY
				tr += 1
				if tr >= 100:
					return edgoNo, posX, posY

		else:
			posY += 1
			if(posY >= sizeY):
				return -1, -1, -1

			if not np.array_equal(grid[posX, posY],[0,0,0]):
				return edgeNo, posX, posY

			if((posY == roadRightTopY) and (edgeNo == len(route.edges)-1)):
				return -1,-1,-1
			elif(posY == roadRightTopY):
				edgeNo += 1

			return edgeNo, posX, posY