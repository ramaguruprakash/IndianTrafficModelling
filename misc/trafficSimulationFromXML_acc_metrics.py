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
#import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from keras.models import model_from_json

import xml.etree.ElementTree as ET
import numpy as np
import cv2
import pickle
import os
import sys
sys.path.append('/Users/gramaguru/ComputerVision/computer-vision/tracksExtractor/homography')
sys.path.append('/Users/gramaguru/ComputerVision/computer-vision/Traffic/IndianTraffic/vehiclePathPrediction')
from homographyLib import transformSetOfPointsAndReturn, drawPoly
#from sumoDataLoader_nn import SumoDataLoader_nn as SumoDataLoader
from sumoDataLoader_nn import SumoDataLoader_nn as SumoDataLoader
carType = {'CAR':0, 'BUS':1, 'TRUCK':2, 'VAN':3, 'MOTORBIKE':4, 'BICYCLE':5}

velocity_metric = {}
acceleration_metric = {}
num_of_collision_metric = {}

dataLoader = SumoDataLoader("", 0.05, 0.05, 1, 5, 'log_guru', 'test', True)
json_file = open("../trafficSimulator/kerasmodels/model_speeds_i-80.json", 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("../trafficSimulator/kerasmodels/model_speeds_i-80.h5")
print("Loaded model from disk")

def getColor(vehicleTypeIndex):
    carType = {0:'CAR', 1:'BUS', 2:'TRUCK', 3:'VAN', 4:'MOTORBIKE', 5:'BICYCLE'}
    colors = {'CAR':(255,255,0), 'BUS':(0,255,255), 'VAN':(0,255,0), 'MOTORBIKE':(0,0,255), 'TRUCK':(255,0,0), 'BICYCLE':(0,0,255)}
    return colors[carType[vehicleTypeIndex]]


def getSize(vehicleTypeIndex):
    carType = {0:'CAR', 1:'BUS', 2:'TRUCK', 3:'VAN', 4:'MOTORBIKE', 5:'BICYCLE'}
    #sizes = {'CAR' : (8,25), 'TRUCK': (11,30), 'VAN': (11,30), 'MOTORBIKE': (2,10), 'BUS' :(15,60),  'BICYCLE': (2,10)}
    sizes = { 'TRUCK': (14.3,6.4), 'VAN': (26.3,8.5), 'BUS' :(5.9,5.1)}
    return sizes[carType[vehicleTypeIndex]]
#{'1': ['5.9', '5.1'], '3': ['26.3', '8.5'], '2': ['14.3', '6.4'], 'v_Class': ['v_Length', 'v_Width']}

def getAllFrames(output_file):
    #fp = open("vehicles_entry.txt", "w")
    new_vehicles = []
    count_of_frames_per_vehicle = {}
    file_path = os.path.join(output_file)
    xml_tree = ET.parse(file_path)
    xml_root = xml_tree.getroot()
    ignore_vehicles = []
    frames = {}

    velocity_metric['original'] = []
    acceleration_metric['original'] = []
    num_of_collision_metric['original'] = {}

    for i, timestamps in enumerate(xml_root):
        #if i == 1000:
        #    break
        frames[float(timestamps.attrib['time'])] = []
        for vehicle in timestamps:
            #if not (float(vehicle.attrib['id']) > 62.0 and float(vehicle.attrib['id']) < 70.0):
             #       continue

            if vehicle.attrib['id'] not in count_of_frames_per_vehicle.keys():
                count_of_frames_per_vehicle[vehicle.attrib['id']] = 1
            else:
                count_of_frames_per_vehicle[vehicle.attrib['id']] += 1
            
            #v = np.array([ float(vehicle.attrib['id']), float(vehicle.attrib['x']), float(vehicle.attrib['y']), float(carType[(vehicle.attrib['type'].upper())])])
            v = np.array([float(vehicle.attrib['id']), 2*float(vehicle.attrib['x']), float(vehicle.attrib['y']), float(vehicle.attrib['type']), 2*float(vehicle.attrib['velx']), float(vehicle.attrib['vely']), 0])
            #print vehicle.attrib['type'].upper(), float(vehicle.attrib['x']), float(vehicle.attrib['y'])
            
            #v = np.array([float(vehicle.attrib['id'].split(".")[1]), float(vehicle.attrib['x']), float(vehicle.attrib['y']), float(carType[(vehicle.attrib['type'].upper())]), float(vehicle.attrib['speed']), 0])

            '''
            if v[0]  in ignore_vehicles:
                continue

            if v[4] < 5 or v[4] > 16:
                ignore_vehicles.append(v[0])
                continue
            '''
            velocity_metric['original'].append([v[4], v[5]])
            if float(timestamps.attrib['time']) - 1 in frames.keys():
                for prev_veh in frames[float(timestamps.attrib['time']) - 1]:
                    if prev_veh[0] == v[0]:
                        acceleration_metric['original'].append([v[4] - prev_veh[4], v[5] - prev_veh[5]])

            #if v[0] not in new_vehicles:
             #   fp.write(str(int(float(timestamps.attrib['time']))) + "," + str(v[3]) + "," + str(v[1]) + "," + str(v[4]) + "\n")
             #   new_vehicles.append(v[0])

            #v = np.array([ float(vehicle.attrib['id']), float(vehicle.attrib['x'])+200, float(vehicle.attrib['y']), float(vehicle.attrib['type'])])
            frames[float(timestamps.attrib['time'])].append(v)
        checkCollision_original(frames[float(timestamps.attrib['time'])])
    #fp.close()
    '''
    for frame in frames.keys():
        vehicles = frames[frame]
        indices = []
        for i, vehicle in enumerate(vehicles):
            if vehicle[0] in ignore_vehicles:
                indices.append(i)
        for index in sorted(indices, reverse=True):
            del vehicles[index]
        frames[frame] = vehicles
    '''
    #print "Ignored vehicles ", ignore_vehicles
    #print "size is ", len(ignore_vehicles)
    print "Size of metrics velocity metric = ", len(velocity_metric['original']), " acceleration Metric = ", len(acceleration_metric['original']) 
    return frames, count_of_frames_per_vehicle

def getSpeeds(vehicles):
    speeds = {}
    for vehicle in vehicles:
        speeds[vehicle[0]] = [vehicle[4], vehicle[5]]
        #speeds[vehicle[0]] = [0, vehicle[4]]
    return speeds

def getVehiclesInLanes(vehicles):
    lanes_frames = {}
    for vehicle in vehicles:
        #print "Lane =",  vehicle[1]
        if vehicle[2] not in lanes_frames:
            lanes_frames[vehicle[1]] = [vehicle]
        else:
            lanes_frames[vehicle[1]].append(vehicle)
    #print lanes_frames
    return lanes_frames

def rectangleOverLap(ltp1x, ltp1y, rbp1x, rbp1y, ltp2x, ltp2y, rbp2x, rbp2y):
    if(ltp1x > rbp2x or ltp2x > rbp1x):
        return False
    if(ltp1y > rbp2y or ltp2y > rbp1y):
        return False

    return True

def vehicleCollide(curVehicle, vehicles):
    vehi_x = curVehicle[1]
    vehi_y = curVehicle[2]
    curVehicleSize = getSize(curVehicle[3])
    rect1ltx = curVehicle[1] - curVehicleSize[1]/2
    rect1lty = curVehicle[2]# - curVehicleSize[1]/2
    rect1rbx = curVehicle[1] + curVehicleSize[1]/2
    rect1rby = curVehicle[2] + curVehicleSize[0]
    print "cooridnates of target ", rect1ltx, rect1lty, rect1rbx, rect1rby
    for vehicle in vehicles:
        if vehi_x == vehicle[1] and vehi_y == vehicle[2]:
                continue
        vehi_size = getSize(vehicle[3])
        rect2ltx = vehicle[1] - vehi_size[1]/2
        rect2lty = vehicle[2]# - vehi_size[1]/2
        rect2rbx = vehicle[1] + vehi_size[1]/2
        rect2rby = vehicle[2] + vehi_size[0]
        if rectangleOverLap(rect1ltx, rect1lty, rect1rbx, rect1rby, rect2ltx, rect2lty, rect2rbx, rect2rby):
            print "Coordinates of coolided ", rect2ltx, rect2lty, rect2rbx, rect2rby
            print "Collisions Complete ", curVehicle[0], vehicle[0]
            return True
    return False

def checkCollision(simulated_vehicles):
    for v in simulated_vehicles:
        print "collision test ", v, simulated_vehicles
        if vehicleCollide(v, simulated_vehicles):
                print "Collided"
                if v[0] in num_of_collision_metric['simulation'].keys():
                    num_of_collision_metric['simulation'][v[0]] += 1
                else:
                    num_of_collision_metric['simulation'][v[0]] = 1
    return

def checkCollision_original(vehicles):
    for v in vehicles:
        print "collision test original ", v, vehicles
        if vehicleCollide(v, vehicles):
                print "Collided original "
                if v[0] in num_of_collision_metric['original'].keys():
                    num_of_collision_metric['original'][v[0]] += 1
                else:
                    num_of_collision_metric['original'][v[0]] = 1
    return



def drawBoundaries(output, offset, color):
    cv2.line(output ,(offset, 0), (offset, 800), color,2)
    cv2.line(output, (offset+180, 0), (offset+180, 800), color,2)

def drawVehicles(output, vehicles, offset = 200, color = [255,0,0]):
    #print vehicles
    drawBoundaries(output, offset, color)
    for vehicle in vehicles:
        if vehicle[1] == -1 and vehicle[2] == -1:
            continue
        vehi_size = getSize(vehicle[3])
        #cv2.rectangle(output, (offset+int(vehicle[1]-vehi_size[0]/2),int(vehicle[2]-vehi_size[1]/2)), (offset+int(vehicle[1]+vehi_size[0]/2), int(vehicle[2]+vehi_size[1]/2)), getColor(vehicle[3]), thickness=cv2.cv.CV_FILLED)
        #cv2.rectangle(output, (offset+int(vehicle[1]-vehi_size[0]/2),int(vehicle[2]-vehi_size[1]/2)), (offset+int(vehicle[1]+vehi_size[0]/2), int(vehicle[2]+vehi_size[1]/2)), color, thickness=cv2.cv.CV_FILLED)
        cv2.rectangle(output, (offset+int(vehicle[1]-vehi_size[1]/2),int(vehicle[2])), (offset+int(vehicle[1]+vehi_size[1]/2), int(vehicle[2]+vehi_size[0])), getColor(int(vehicle[3])), thickness=cv2.cv.CV_FILLED)
        #cv2.putText(output, str(int(vehicle[0])), (offset + int(vehicle[1]), int(vehicle[2])), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,0,0), 1, cv2.CV_AA)
    return output

def getFeatureVecs(vehicles, speeds, lanes_frames, lanes):
    feature_vecs = []
    for vehicle in vehicles:
        #vehicle = np.array([curVehicle[0], curVehicle[1], curVehicle[2], curVehicle[3]])
        feature_vec = np.array(dataLoader.getFeatureVectorImp(vehicle, speeds, lanes_frames, lanes))
        #feature_vec.resize(1, feature_vec.shape[0])
        feature_vecs.append(feature_vec)
    return feature_vecs

def drawVehiclesWithSimulation(output, vehicles, prev_vehicles, simulated_vehicles, prev_speeds=None):
    #print "To start with ", vehicles, prev_vehicles, simulated_vehicles
    # Original Simulations
    print "original"
    #output = drawVehicles(output, vehicles, 200, [255,0,0])
    #print "velocity difference ", prev_vehicles[:,4] - vehicles[:,4] 
    '''
    print "step prediction"
    # instance Simulations
    if prev_vehicles.size == 0:
        output = drawVehicles(output, vehicles, 400, [255,255,0])
    else:
        speeds = getSpeeds(prev_vehicles)
        if prev_vehicles.size != 0:
            feature_vecs, _ = dataLoader.getNNFeaturesFromFrame(prev_vehicles, speeds, None, None, True) # Where speeds is a dictionary of vehicle Id to speed.
        else:
            feature_vecs = []
        print "prev feature vecs ", feature_vecs
        if len(feature_vecs) != 0:
            feature_vecs = np.array(feature_vecs)
            #print "prev", feature_vecs
            velocities = loaded_model.predict(feature_vecs)
            #print velocities
        else:
            velocities = np.array([])
        print "prev velocities ", velocities
        for vehicle, velocity in zip(prev_vehicles, velocities):
            vehicle[1] = vehicle[1] + velocity[0]
            vehicle[2] = vehicle[2] + velocity[1]
            vehicle[4] = velocity[0]
            vehicle[5] = velocity[1]
        output = drawVehicles(output, prev_vehicles, 400, [255,255,0])
    '''

    print "complete simulation"
    # Overall Simulations
    #output = drawVehicles(output, simulated_vehicles, 420, [255,255,255])

    speeds = getSpeeds(simulated_vehicles)
    #print "input", simulated_vehicles, speeds, lanes_frames, lanes
    if simulated_vehicles.size != 0:
        feature_vecs, _ = dataLoader.getNNFeaturesFromFrame(simulated_vehicles, speeds, prev_speeds, None, True)
    else:
        feature_vecs = []
    #print "simulated feature vecs ", feature_vecs

    if len(feature_vecs) != 0:
        feature_vecs = np.array(feature_vecs)
        #print feature_vecs
        velocities = loaded_model.predict(feature_vecs)
        #print velocities
    else:
        velocities = np.array([])

    ###  Velocity Metric ###
    prev_vel =  []
    for feature, veh in zip(feature_vecs, simulated_vehicles):
        velocity_metric['simulation'].append([feature[0],feature[1]])
        prev_vel.append(np.array([feature[0],feature[1]]))
    prev_vel = np.array(prev_vel)
    ### Velocity Metric ###

    ### Acceleration Metric ###
    for v1,v2 in zip(prev_vel, velocities):
        acceleration_metric['simulation'].append([v2[0]-v1[0], v2[1]-v1[1]])
    ### Acceleration Metric ###
    #print "adding velocities/ acceleration ", len(prev_vel), velocities.shape, len(acceleration_metric['simulation'])

    #print "simulated output ", velocities
    for vehicle, velocity, feature_vec in zip(simulated_vehicles, velocities, feature_vecs):
        if feature_vec[0] != 0 or feature_vec[1] != 0:
            vehicle[1] = vehicle[1] + velocity[0]
            vehicle[2] = vehicle[2] + velocity[1]

        if vehicle[1] > 10000:
            vehicle[1] = 10000
            velocity[0] = 10

        if vehicle[2] > 10000:
            vehicle[2] = 10000
            velocity[1] = 10

        if vehicle[1] < -10000:
            vehicle[1] = -10000
            velocity[0] = -10

        if vehicle[2] < -10000:
            vehicle[2] = -10000
            velocity[1] = -10

        if feature_vec[0] != 0 or feature_vec[1] != 0:
            vehicle[4] = velocity[0]
            vehicle[5] = velocity[1]
    #print "Updated ", simulated_vehicles
    checkCollision(simulated_vehicles)
    return output, simulated_vehicles, speeds

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
    prev_vehicles = np.array([])
    simulated_vehicles = None
    accelerating_vehicles = {}
    left_vehicles = []
    prev_speeds = {}
    for i, frame in enumerate(frames.keys()):
        #if i == 100:
        #    break
        vehicles = frames[frame]
        print "Frame Number = ", i, " Number of Vehicles in the screen = ",len(vehicles)
        #print "Size of metrics velocity metric = ", len(velocity_metric['simulation']), " acceleration Metric = ", len(acceleration_metric['simulation'])
        
        if video != None:
            ret, video_frame = cap.read()

        #print frames[frame]
        output = np.zeros(size)
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
            #print vehicles, transformSetOfPointsAndReturn(getVehicleCoordinates(vehicles), H), vehicles_topView
        #output = drawVehicles(output, vehicles)
        '''
        for vehicle in vehicles:
            for veh in prev_vehicles:
                if vehicle[0] == veh[0]:
                    print "difference ", veh, vehicle, abs(veh[4]-vehicle[4])
                    if abs(veh[4]-vehicle[4]) >2:
                        if veh[0] not in accelerating_vehicles:
                            accelerating_vehicles[veh[0]] = 1
                        else:
                            accelerating_vehicles[veh[0]] += 1
                        vehicle[5] = vehicle[4] - veh[4]
                    else:
                        vehicle[5] = 0
        '''
        if i == 0:
        #if True:
            simulated_vehicles = np.copy(vehicles)
            #simulated_vehicles = vehicles
        
        else:
            simulated_vehicles1 = np.copy(vehicles)
            #print "Simulated Vehicles 1 ", simulated_vehicles
            vehicles_inframe =  []
            for j, veh in enumerate(simulated_vehicles1):
                for v in simulated_vehicles:
                    if v[0] == veh[0] and (v[4] != 0 or v[5] != 0):
                        veh[1] = v[1]
                        veh[2] = v[2]
                        veh[4] = v[4]
                        veh[5] = v[5]
                        '''
                        if v[4] + veh[5] <= 17:
                            veh[2] = v[2] + veh[5]
                            veh[4] = v[4] + veh[5]
                            #veh[5] = v[5]
                        else:
                            veh[2] = v[2]
                            veh[4] = 16
                        '''
                if veh[2] <=  800 and veh[0] not in left_vehicles:
                         vehicles_inframe.append(j)
                elif veh[2] > 800:
                         left_vehicles.append(veh[0])
                print "inframe ", vehicles_inframe, len(vehicles_inframe), simulated_vehicles1.shape
            print "before ", simulated_vehicles1
            if simulated_vehicles1.size != 0:
                simulated_vehicles1 = simulated_vehicles1[vehicles_inframe, :]
            print "after ", simulated_vehicles1
            simulated_vehicles = simulated_vehicles1
        
        output, simulated_vehicles, prev_speeds = drawVehiclesWithSimulation(output, vehicles, prev_vehicles, simulated_vehicles, prev_speeds)
        prev_vehicles = np.copy(vehicles)
        #prev_vehicles = vehicles
         
        #output = drawVehicles(output, vehicles, 100)
        
        if homography == True:
            output = drawVehicles_red(output, vehicles_topView)
        #cv2.putText(output, str(i), (100,100), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 1, cv2.CV_AA)
        #output = output[::-1,:,:]
        #output = output[:,::-1,:]
        #cv2.imshow("sim", output)
        #print "Number of vehicles accelerating ", len(accelerating_vehicles.keys())
        if video != None:
            cv2.imshow("real", video_frame)
        #print "Accelerating vehicles ", accelerating_vehicles
        #print "Number is ", len(accelerating_vehicles.keys())

        #if cv2.waitKey(33) == 27:
        #   break
    #cv2.destroyAllWindows()
    #print count

if __name__ == '__main__':
    num_of_collision_metric['simulation'] = {}
    velocity_metric['simulation'] = []
    acceleration_metric['simulation'] = []
    #simulateFromXML("/Users/gramaguru/Desktop/car_videos/sing_cropped.xml", (660, 300, 3), False)
    #simulateFromXML("/Users/gramaguru/Desktop/car_videos/sing_cropped.mp4", "/Users/gramaguru/Desktop/car_videos/sing_cropped.xml", (710, 710, 3), True)
    #simulateFromXML(None, "/Users/gramaguru/ComputerVision/computer-vision/vehicle-trajectory-data/traffic-new.xml", (800, 800, 3), False)
    simulateFromXML(None, "/Users/gramaguru/ComputerVision/computer-vision/vehicle-trajectory-data/traffic-new-i80.xml", (800, 800, 3), False)
    #simulateFromXML(None, "/Users/gramaguru/SumoNetowrk/simulation_50000sec_20000cars.xml", (660, 360,  3), False)
    #simulateFromXML(None, "/Users/gramaguru/SumoNetowrk/simulation_10000sec_3000cars.xml", (660, 1200,  3), False)
    fp = open("metrics_real1", "wb")
    metrics = [velocity_metric, acceleration_metric, num_of_collision_metric]
    pickle.dump(metrics, fp)
    fp.close()
