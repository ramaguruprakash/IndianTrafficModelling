from keras.models import model_from_json
import tensorflow as tf
import keras.backend as K
from keras.models import Model
from keras.layers import Input , Dense, Dropout , BatchNormalization, Activation , Add, Lambda, concatenate
from keras import regularizers , optimizers
from keras.utils import plot_model
from keras.callbacks import ReduceLROnPlateau , ModelCheckpoint , CSVLogger
from keras.initializers import TruncatedNormal
from keras.losses import mean_squared_error as mse
import h5py
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from keras.models import model_from_json

import xml.etree.ElementTree as ET
import numpy as np
import cv2
import os
import sys
sys.path.append('/Users/gramaguru/ComputerVision/computer-vision/tracksExtractor/homography')
sys.path.append('/Users/gramaguru/ComputerVision/computer-vision/Traffic/IndianTraffic/vehiclePathPrediction')
from homographyLib import transformSetOfPointsAndReturn, drawPoly
from sumoDataLoader_nn import SumoDataLoader_nn as SumoDataLoader
carType = {'CAR':0, 'BUS':1, 'TRUCK':2, 'VAN':3, 'MOTORBIKE':4, 'BICYCLE':5}

dataLoader = SumoDataLoader("/Users/gramaguru/ComputerVision/computer-vision/vehicle-trajectory-data/traffic-new.xml", 0.05, 0.05, 1, 5, 'log_guru', 'test', True)
json_file = open('../trafficSimulator/kerasmodels/model_speeds_real.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("../trafficSimulator/kerasmodels/model_speeds_real.h5")
print("Loaded model from disk")

def getColor(vehicleTypeIndex):
    carType = {0:'CAR', 1:'BUS', 2:'TRUCK', 3:'VAN', 4:'MOTORBIKE', 5:'BICYCLE'}
    colors = {'CAR':(255,255,0), 'BUS':(0,255,255), 'VAN':(0,255,0), 'MOTORBIKE':(0,0,255), 'TRUCK':(255,0,0), 'BICYCLE':(0,0,255)}
    return colors[carType[vehicleTypeIndex]]


def getSize(vehicleTypeIndex):
    carType = {0:'CAR', 1:'BUS', 2:'TRUCK', 3:'VAN', 4:'MOTORBIKE', 5:'BICYCLE'}
    sizes = {'CAR' : (8,25), 'TRUCK': (11,30), 'VAN': (11,30), 'MOTORBIKE': (2,10), 'BUS' :(15,60),  'BICYCLE': (2,10)}
    return sizes[carType[vehicleTypeIndex]]


def getAllFrames(output_file):
    #fp = open("vehicles_entry.txt", "w")
    new_vehicles = []
    count_of_frames_per_vehicle = {}
    file_path = os.path.join(output_file)
    xml_tree = ET.parse(file_path)
    xml_root = xml_tree.getroot()
    frames = {}
    for timestamps in xml_root:
        frames[float(timestamps.attrib['time'])] = []
        for vehicle in timestamps:
            if vehicle.attrib['id'] not in count_of_frames_per_vehicle.keys():
                count_of_frames_per_vehicle[vehicle.attrib['id']] = 1
            else:
                count_of_frames_per_vehicle[vehicle.attrib['id']] += 1

            #v = np.array([ float(vehicle.attrib['id']), float(vehicle.attrib['x']), float(vehicle.attrib['y']), float(carType[(vehicle.attrib['type'].upper())])])
            v = np.array([ float(vehicle.attrib['id']), float(vehicle.attrib['x']), float(vehicle.attrib['y']), float(vehicle.attrib['type']), float(vehicle.attrib['velx']), float(vehicle.attrib['vely'])])
            #print vehicle.attrib['type'].upper(), float(vehicle.attrib['x']), float(vehicle.attrib['y'])
            #v = np.array([ float(vehicle.attrib['id'].split(".")[1]), float(vehicle.attrib['x']), float(vehicle.attrib['y']), float(carType[(vehicle.attrib['type'].upper())]), float(vehicle.attrib['speed'])])
            #if v[0] not in new_vehicles:
             #   fp.write(str(int(float(timestamps.attrib['time']))) + "," + str(v[3]) + "," + str(v[1]) + "," + str(v[4]) + "\n")
             #   new_vehicles.append(v[0])

            #v = np.array([ float(vehicle.attrib['id']), float(vehicle.attrib['x'])+200, float(vehicle.attrib['y']), float(vehicle.attrib['type'])])
            frames[float(timestamps.attrib['time'])].append(v)
    #fp.close()
    return frames, count_of_frames_per_vehicle

def getSpeeds(vehicles):
    speeds = {}
    for vehicle in vehicles:
        speeds[vehicle[0]] = [vehicle[4], vehicle[5]]
    return speeds

def drawVehicles(output, vehicles):
    speeds = getSpeeds(vehicles)
    feature_vecs, _ = dataLoader.getNNFeaturesFromFrame(vehicles, speeds, None, None, True) # Where speeds is a dictionary of vehicle Id to speed.
    feature_vecs = np.array(feature_vecs)
    velocities = loaded_model.predict(feature_vecs)
    for vehicle, velocity in zip(vehicles, velocities):
        if vehicle[1] == -1 and vehicle[2] == -1:
            continue
        vehi_size = getSize(vehicle[3])
        '''
            Send set of previous vehicles, if that had velocities, put them in a dictionary and use that dictionary to get the feature vec.
        '''
        cv2.rectangle(output, (200+int(vehicle[1]-vehi_size[0]/2),int(vehicle[2]-vehi_size[1]/2)), (200+int(vehicle[1]+vehi_size[0]/2), int(vehicle[2]+vehi_size[1]/2)), getColor(vehicle[3]), thickness=cv2.cv.CV_FILLED)
        posX = vehicle[1] - vehicle[4] + velocity[0]
        posY = vehicle[2] - vehicle[5] + velocity[1]
        print vehicle[1], vehicle[2], posX, posY, velocity[0], velocity[1], vehicle[4], vehicle[5]
        cv2.rectangle(output, (400 + int(posX-vehi_size[0]/2),int(posY-vehi_size[1]/2)), (400 + int(posX+vehi_size[0]/2), int(posY+vehi_size[1]/2)), [255, 255, 255], thickness=cv2.cv.CV_FILLED)
        #cv2.putText(output, str(int(vehicle[0])), (int(vehicle[1]), int(vehicle[2])), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,0,0), 1, cv2.CV_AA)
    return output

def drawVehicles_red(output, vehicles):
    for vehicle in vehicles:
        if vehicle[1] == -1 and vehicle[2] == -1:
            continue
        vehi_size = getSize(vehicle[3])
        cv2.rectangle(output, (int(vehicle[1]-vehi_size[0]/2),int(vehicle[2]-vehi_size[1]/2)), (int(vehicle[1]+vehi_size[0]/2),int(vehicle[2]+vehi_size[1]/2)), [0,0,255], thickness=cv2.cv.CV_FILLED)
        #cv2.putText(output, str(int(vehicle[0])), (int(vehicle[1]), int(vehicle[2])), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,0,0), 1, cv2.CV_AA)
    return output


def getVehicleCoordinates(vehicles):
    xy = []
    for vehicle in vehicles:
        xy.append([vehicle[1], vehicle[2]])
    return np.array(xy)

def changeVehicles(xy, vehicles):
    for i, vehicle in enumerate(vehicles):
        vehicle[1] = xy[i][0]
        vehicle[2] = xy[i][1]
    return vehicles

def simulateFromXML(video, xml, size, homography=False):
    frames , count = getAllFrames(xml)
    if video != None:
        cap = cv2.VideoCapture(video)
    for i, frame in enumerate(frames.keys()):
        vehicles = frames[frame]
        print "Frame Number ", i
        if video != None:
            ret, video_frame = cap.read()

        #print frames[frame]
        output = np.zeros(size)
        output_from_model = np.zeros(size)
        #cv2.putText(output, str(i), (20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,0,0), 1, cv2.CV_AA)
        if homography == True:
            straight_road_points = np.array([[0,0],[300,0],[300,710],[0,710]], dtype=float)
            perspective_road_points = np.array([[536,136], [615,136], [655,655], [0,475]], dtype=float)
            perspective_road_points1 = np.array([[536,136], [615,136], [655,655], [0,475], [ 414., 464], [549., 245]], dtype=float)
            straight_road_points1 = np.array([[0,0],[300,0],[300,710],[0,710], [150,0], [150,100] , [150,200], [150,300], [150, 400], [150, 500], [150, 600], [150, 700]], dtype=float)
            output = drawPoly(output, straight_road_points, [255,0,0])
            output = drawPoly(output, perspective_road_points, [0,0,255])
            video_frame = drawPoly(video_frame, straight_road_points, [255,0,0])
            video_frame = drawPoly(video_frame, perspective_road_points, [0,0,255])
            H, status = cv2.findHomography(perspective_road_points, straight_road_points)
            #H, status = cv2.findHomography(straight_road_points, perspective_road_points)
            straight_road_points2 = np.copy(straight_road_points1)
            #print perspective_road_points
            #for (x, y) in zip(straight_road_points2, transformSetOfPointsAndReturn(straight_road_points1 ,H)):
            #    print x, y
            #break
            vehicles_topView = np.copy(vehicles)
            vehicles_topView = changeVehicles(transformSetOfPointsAndReturn(getVehicleCoordinates(vehicles), H), vehicles_topView) 
            print vehicles, transformSetOfPointsAndReturn(getVehicleCoordinates(vehicles), H), vehicles_topView
        output = drawVehicles(output, vehicles)

        if homography == True:
            output = drawVehicles_red(output, vehicles_topView)
        output = output[::-1,:,:]
        #output = output[:,::-1,:]
        cv2.imshow("sim", output)
        if video != None:
            cv2.imshow("real", video_frame)
        if cv2.waitKey(33) == 27:
            break
    cv2.destroyAllWindows()
    #print count

if __name__ == '__main__':
    global dataLoader
    #simulateFromXML("/Users/gramaguru/Desktop/car_videos/sing_cropped.xml", (660, 300, 3), False)
    #simulateFromXML("/Users/gramaguru/Desktop/car_videos/sing_cropped.mp4", "/Users/gramaguru/Desktop/car_videos/sing_cropped.xml", (710, 710, 3), True)
    simulateFromXML(None, "/Users/gramaguru/ComputerVision/computer-vision/vehicle-trajectory-data/traffic-new.xml", (1000, 1000, 3), False)
    #simulateFromXML(None, "/Users/gramaguru/SumoNetowrk/simulation_50000sec_20000cars.xml", (660, 360,  3), False)
