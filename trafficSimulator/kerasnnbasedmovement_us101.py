#! /users/guruprakash.r/miniconda4/bin/python
from keras.models import model_from_json
from vehicleBehavior import VehicleBehavior
from roadNetwork import RoadNetwork
import random
import sys
sys.path.append('/Users/gramaguru/ComputerVision/computer-vision/Traffic/IndianTraffic/vehiclePathPrediction')
from vehicleTypes import VehicleTypes
from random import randint
from sumoDataLoader import SumoDataLoader
import numpy as np

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]=""

import tensorflow as tf
import keras.backend as K
from keras.models import Model
from keras.layers import Input , Dense, Dropout , BatchNormalization, Activation , Add
from keras import regularizers , optimizers
from keras.callbacks import ReduceLROnPlateau , ModelCheckpoint , CSVLogger
from keras.initializers import TruncatedNormal
from keras.losses import mean_squared_error as mse

class kerasnnBasedMovement(VehicleBehavior):
	def __init__(self):
		self.vehicleTypes = VehicleTypes()
                self.dataLoader = SumoDataLoader("",0.1,0.1,1,3,'infer', True)
                self.max_xval = np.array([8.558, 12.448, 124.82, 422.733, 8.304, 13.639, 135.174, 517., 10.017, 12.448, 140.7, 1531.639, 6.777, 13.639, 154.802 , 1553.565, 8.558, 12.448])
                self.max_yval = np.array([ 10.017, 13.639])
                self.min_xval = np.array([-5.641, -23.108,  -116.837,  -942.203, -9.548, -23.108, -124.331, -1435.677, -7.502, -23.108, -135.313, -1485.024, -9.548, -21.737, -144.973, -1553.565, -9.548, -22.14])
                self.min_yval = np.array([-9.548, -26.801])
                json_file = open('kerasmodels/model_us101.json', 'r')
                loaded_model_json = json_file.read()
                json_file.close()
                self.loaded_model = model_from_json(loaded_model_json)
                # load weights into new model
                self.loaded_model.load_weights("kerasmodels/model_us101.h5")
                print("Loaded model from disk")
                
                json_file = open('kerasmodels/model_classification_1.json', 'r')
                loaded_model_json = json_file.read()
                json_file.close()
                self.loaded_model_classification = model_from_json(loaded_model_json)
                # load weights into new model
                self.loaded_model_classification.load_weights("kerasmodels/model_classification_1.h5")
                print("Loaded model from disk")

        def rectangleOverLap(self, ltp1x, ltp1y, rbp1x, rbp1y, ltp2x, ltp2y, rbp2x, rbp2y):
            if(ltp1x > rbp2x or ltp2x > rbp1x):
                return False
                
                if(ltp1y > rbp2y or ltp2y > rbp1y):
                    return False
                
                return True

        def positionempty(self, posX, posY, curVehicle, vehicles):
    
            vehi_x = curVehicle.curX
            vehi_y = curVehicle.curY
            #print curVehicle.cl 
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
                    speeds[vehicle.identity] = [0.0,vehicle.speed]
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
            print "vehicle started ", identity, posX, posY
            if edgeNo != 0:
                return 1,-1,-1
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
            laneNo = laneszz.index(posX)
            vehicles = self.getVehiclesInScreen(vehicles)
            speeds = self.getSpeeds(vehicles)
            speeds[identity] = curVehicle.prev_velocity
            vehicles_per_lanes = self.get_vehicles_lanes_dist(vehicles, laneszz)
            print "speeds : ", speeds
            print "==="
            print "vehicle per lanes : ", vehicles_per_lanes
            print "===="
            vehicle = np.array([identity, posX, posY, self.dataLoader.carType[curVehicle.cl]])
            feature_vec = np.array(self.dataLoader.getFeatureVectorImp(vehicle, speeds, vehicles_per_lanes, laneszz))
            distance_in_front = feature_vec[3]
            distance_left_front = feature_vec[9]
            distance_right_front = feature_vec[12]
            feature_vec.resize(1, feature_vec.shape[0])
            print "input ",identity, feature_vec, " -> ",
            feature_vec = (feature_vec - self.min_xval)/(self.max_xval - self.min_xval)
            print "input ", feature_vec, " -> ",
            velocityy = self.loaded_model.predict(feature_vec)
            #velocityx =  self.loaded_model_classification.predict(feature_vec)
            #velocityx = np.argmax(velocityx)
            velocityx = 0.0
            #if velocityx == 1:
            #    velocityx = width/lanes
            #if velocityx == 2:
            #    velocityx = -1*width/lanes
            #velocity = output.data.numpy()
            print " : ", velocityy
            velocityy = velocityy*(self.max_yval - self.min_yval) + self.min_yval
            if velocityy > distance_in_front and distance_in_front != -1:
                    if distance_left_front == -1 and distance_right_front == -1:
                         a = random.randint(0,1)
                         if a == 0 and laneNo != len(laneszz)-1:
                             #velocityx = width/lanes
                             pass
                         elif a == 1 and laneNo != 0:
                             #velocityx = -1*width/lanes
                             pass
                    elif distance_left_front == -1 and laneNo != 0:
                        print "left move" , -1
                        #velocityx = -1*width/lanes
                    elif distance_right_front == -1 and laneNo != len(laneszz)-1:
                        print "right move", -1
                        #velocityx = width/lanes
                    elif distance_left_front > distance_right_front and laneNo != 0:
                        #velocityx = -1*width/lanes
                        print "left move", velocityy, distance_in_front, distance_left_front, distance_right_front
                    elif distance_left_front < distance_right_front and laneNo != len(laneszz)-1:
                        print "right move", velocityy, distance_in_front, distance_left_front, distance_right_front
                        #velocityx = width/lanes
            print " y   : ", velocityy
            curVehicle.prev_velocity = [velocityx, velocityy]
            #print " x   : ", velocityx
            #posX += int(velocity[0, 0])
            #posY -= int(velocity[0, 1])
            #if velocityx == 0:
            posY -= int(velocityy)
            #posX += int(velocityx)
            print "======New position ====== ", posX, posY

            if(posY > sizeX+11):
                    return -1, -1, -1
            if posY < 0:
                    return edgeNo+1, -1, -1
            return edgeNo, posX, posY
