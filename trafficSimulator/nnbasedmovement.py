#! /users/guruprakash.r/miniconda2/bin/python

from vehicleBehavior import VehicleBehavior
from roadNetwork import RoadNetwork
import random
import sys
sys.path.append('/Users/gramaguru/ComputerVision/computer-vision/Traffic/IndianTraffic/vehiclePathPrediction')
from vehicleTypes import VehicleTypes
from random import randint
from sumoDataLoader import SumoDataLoader
from torch.autograd import Variable
from ffModel import Model
import numpy as np
import torch
class nnBasedMovement(VehicleBehavior):
	def __init__(self):
		self.vehicleTypes = VehicleTypes()
                self.dataLoader = SumoDataLoader("",0.1,0.1,1,3,'infer', True)
                self.modelpath = '/Users/gramaguru/ComputerVision/computer-vision/Traffic/IndianTraffic/vehiclePathPrediction/sing_cropped_256-2_21_100_20_0.0001_5_model.pkl'

        def getFloatTensorFromNumpy(self, x):
            x = torch.from_numpy(x)
            x = x.float()
            return x
        
        def rectangleOverLap(self, ltp1x, ltp1y, rbp1x, rbp1y, ltp2x, ltp2y, rbp2x, rbp2y):
            if(ltp1x > rbp2x or ltp2x > rbp1x):
                return False
                
                if(ltp1y > rbp2y or ltp2y > rbp1y):
                    return False
                
                return True

        def positionempty(self, posX, posY, curVehicle, vehicles):
    
            vehi_x = curVehicle.curX
            vehi_y = curVehicle.curY
            print curVehicle.cl 
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


        def getSpeeds(self, vehicles):
            speeds = {}
            for vehicle in vehicles:
                if len(vehicle.track) <= 1 or vehicle.track[-1][0] == -1 or vehicle.track[-1][1] == -1 or vehicle.track[-2][0] == -1 or vehicle.track[-2][1] == -1:
                    speeds[vehicle.identity] = [0.0,0.0]
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

        def getVehiclesInScreen(self, vehicles):
            filtered_vehicles = []
            for vehicle in vehicles:
                if vehicle.curX == -1 or vehicle.curY == -1:
                    continue
                filtered_vehicles.append(vehicle)
            return filtered_vehicles

	def updatePos(self, grid, curVehicle, vehicles):
	    posX = curVehicle.curX
	    posY = curVehicle.curY
            identity = curVehicle.identity
	    route = curVehicle.route
	    edgeNo = curVehicle.numberOfEdgesCompleted
	    speed = curVehicle.speed
	    laneNo = curVehicle.laneNo
	    shape = grid.shape
	    sizeX = shape[0]
	    sizeY = shape[1]
            print "========"
            print "Yo ", curVehicle.identity, curVehicle.curX, curVehicle.curY
            if edgeNo != 0:
                return -1,-1,-1
            initNode = route.edges[edgeNo].node1
            finalNode = route.edges[edgeNo].node2
            width = route.edges[edgeNo].width
            lanes = route.edges[edgeNo].lanes
            roadLeftBottomX =  initNode.x - width/2
            roadLeftBottomY = initNode.y
            roadRightTopX = finalNode.x + width/2
            roadRightTopY = finalNode.y
            laneszz = [roadLeftBottomX + ((i*width)/lanes) + (width/lanes)/2 for i in range(4)]
            print "Lanessss" , laneszz
            if posX == -1 and posY == -1:
                tr = 0
                while 1:
                    x = laneNo
                    x = random.choice(laneszz)
                    if self.positionempty(x, roadLeftBottomY, curVehicle, vehicles):
                        #print "wth man " + str(x) + " " + str(roadLeftBottomY)
                        return edgeNo, x, roadLeftBottomY
                    tr += 1
                    if tr >= 100:
                        return edgeNo, posX, posY

                # get speeds of all the vehicles
            vehicles = self.getVehiclesInScreen(vehicles)
            speeds = self.getSpeeds(vehicles)
            vehicles_per_lanes = self.get_vehicles_lanes(vehicles)
            print "speeds : ", speeds
            #print "==="
            print "vehicle per lanes : ", vehicles_per_lanes
            #print "===="
            vehicle = np.array([identity, posX, posY, self.dataLoader.carType[curVehicle.cl]])
            feature_vec = np.array(self.dataLoader.getFeatureVectorImp(vehicle, speeds, vehicles_per_lanes, laneszz))
            feature_vec.resize(1, feature_vec.shape[0])
            net = Model(21,256,2)
            net.load_state_dict(torch.load(self.modelpath))
            output = net(Variable(self.getFloatTensorFromNumpy(feature_vec)))
            velocity = output.data.numpy()
            print "output from NN : ", velocity
            #posX += int(velocity[0, 0])
            posY -= int(velocity[0, 1])
            print "======posY====== ", posY

            if(posY > sizeX):
                    return -1, -1, -1
            if posY < 0:
                    return edgeNo+1, -1, -1
            return edgeNo, posX, posY
