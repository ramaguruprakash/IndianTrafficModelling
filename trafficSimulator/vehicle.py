#! /users/guruprakash.r/miniconda2/bin/python
from random import randint

class Vehicle:
		def __init__(self, cl, route, startTime, positionUpdater, speed=4, laneNo=-1, identity=0, color=(0,255,0), prev_lane = 0):
			while laneNo == -1 or prev_lane == laneNo:
				#laneNo = randint(0,3)
                                 laneNo = 0
                        #print "New Vehicle:- ", cl, startTime, speed, laneNo
			self.cl = cl
                        self.identity = identity
			self.route = route
			self.positionUpdater = positionUpdater
			self.curX = -1
			self.curY = -1
			self.startTime = startTime
			self.track = []
			self.laneNo = laneNo
			self.speed = speed
			self.numberOfEdgesCompleted = 0
                        self.color=color
                        self.prev_velocity = [0.0, speed]

		def move(self, timestamp, grid, vehicles):
			if (timestamp < self.startTime):
				self.track.append((-1, -1))
				return
			numberOfEdgesCompleted, x, y = self.positionUpdater.updatePos(grid, self, vehicles)
                        self.updatePos(numberOfEdgesCompleted, x, y)

                def updatePos(self, numberOfEdgesCompleted, x, y ):
                        if numberOfEdgesCompleted == -1:
				self.curX = -1
				self.curY = -1
			        #print "Done dude ", self.identity, " " + str(self.curX) + " " + str(self.curY)
				return
			elif numberOfEdgesCompleted != len(self.route.edges):
				self.curX = x
				self.curY = y
				self.track.append((x,y))
				self.numberOfEdgesCompleted = numberOfEdgesCompleted
			elif numberOfEdgesCompleted == len(self.route.edges):
				self.numberOfEdgesCompleted = len(self.route.edges)
                                self.curX = -1
                                self.curY = -1
			#print "updated positions ", self.identity, " " + str(self.curX) + " " + str(self.curY)

		def __str__(self):
			return self.cl + " I:-\n" + str(self.identity) + " Pos:- (" + str(self.curX) + ", " + str(self.curY) + ") " + str(self.startTime) + " " + str(self.track) + " \n"

		def __repr__(self):
			return self.cl + " I:-\n" + str(self.identity) + " Pos:- (" + str(self.curX) + ", " + str(self.curY) + ") " + str(self.startTime) + " " + str(self.track) + " \n"
