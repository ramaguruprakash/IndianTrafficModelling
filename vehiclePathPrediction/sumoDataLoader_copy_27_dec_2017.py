import numpy as np
import xml.etree.ElementTree as ET
from sklearn.neighbors import NearestNeighbors
import os
import copy
class SumoDataLoader:
    def __init__(self, sumo_output_file, val_frac, test_frac, batch_size, number_of_neighbours, mode='train'):
        '''
            Load sumo data
            Divide into val, test and train
            Have pointers for the same
        '''
        self.batch_size = batch_size
        self.number_of_neighbours = number_of_neighbours
        self.mode = mode
        #X, y = self.loadData(sumo_output_file)
        X, y = self.loadDataWithOnlyImpFeatures(sumo_output_file)
        print "Total number of samples, feature size and output size", X.shape, y.shape
        self.number_of_samples = X.shape[0]

        self.train_X = X[:int(self.number_of_samples*(1-val_frac-test_frac)), :]
        self.train_y = y[:int(self.number_of_samples*(1-val_frac-test_frac)), :]
        self.validation_X = X[int(self.number_of_samples*(1-val_frac-test_frac)):int(self.number_of_samples*(1-test_frac)), :]
        self.validation_y = y[int(self.number_of_samples*(1-val_frac-test_frac)):int(self.number_of_samples*(1-test_frac)), :]
 
        self.test_X = X[int(self.number_of_samples*(1-test_frac)):, :]
        self.test_y = y[int(self.number_of_samples*(1-test_frac)):, :]
        
        print "Training : ", self.train_X.shape, self.train_X.shape[0]/self.batch_size
        print "Validation : ", self.validation_X.shape, self.validation_X.shape[0]/self.batch_size
        print "Test : ", self.test_X.shape, self.test_X.shape[0]/self.batch_size

        self.batch_pointers = {'train':0, 'validation':0, 'test':0}

    def getAllFrames(self, sumo_output_file):
        file_path = os.path.join(sumo_output_file)
        xml_tree = ET.parse(file_path)
        xml_root = xml_tree.getroot()
        frames = {}
        for timestamps in xml_root:
            frames[float(timestamps.attrib['time'])] = []
            for vehicle in timestamps:
                v = np.array([ float(vehicle.attrib['id'].split('.')[1]), float(vehicle.attrib['x']), float(vehicle.attrib['y'])])
                frames[float(timestamps.attrib['time'])].append(v)
        return frames

    def getSpeeds(self, frames):
        # Get speeds 
        speeds = {}
        pos_by_frame = {}
        for frame in frames.keys():
            vehicles = frames[frame]
            #print "=====  speed frame number : ",frame ," ===="
            #print "Get Speeds vehicles ", vehicles
            for vehicle in vehicles:
                #print "Vehicle - ", vehicle[0]
                if vehicle[0] not in speeds.keys():
                    speeds[vehicle[0]] = {frame: [0.0, 0.0]}
                    pos_by_frame[vehicle[0]] = {frame : [vehicle[1], vehicle[2]]}
                    #print "Speeds ", speeds[vehicle[0]]
                else:
                    pos_by_frame[vehicle[0]][frame] = [vehicle[1], vehicle[2]]
                    prev_frame = frame-1
                    if prev_frame in pos_by_frame[vehicle[0]].keys():
                        speeds[vehicle[0]][frame] = [vehicle[1] - pos_by_frame[vehicle[0]][prev_frame][0], vehicle[2] - pos_by_frame[vehicle[0]][prev_frame][1]]
                    else:
                        speeds[vehicle[0]][frame] = [0, 0]
                    #print "Speeds ", speeds[vehicle[0]][frame]
        return speeds

    def lanes_frames(self, frames):
        lanes_frames = {}
        for frame in frames.keys():
            #print "====== Lanes diff frame ", frame, " ======"
            lanes_frames[frame] = {}
           # print frames[frame]
            for vehicle in frames[frame]:
                if vehicle[1] not in lanes_frames[frame]:
                    lanes_frames[frame][vehicle[1]] = [vehicle]
                else:
                    lanes_frames[frame][vehicle[1]].append(vehicle)
            #print "Vehicle in lanes "
            #print lanes_frames[frame]
        return lanes_frames

    def __get_nearest_front_and_back(self, lane_vehicles, vehicle):
        nearest_vehicle_front = []
        nearest_vehicle_back = []
        for lane_vehicle in lane_vehicles:
            print "=== nearest_lane vehicles=="
            print lane_vehicles, vehicle
            displacement = lane_vehicle[2] - vehicle[2]
            if displacement > 0 and (nearest_vehicle_front == [] or nearest_vehicle_front[0] > displacement):
                nearest_vehicle_front = [displacement, lane_vehicle]
            if displacement < 0 and (nearest_vehicle_back == [] or nearest_vehicle_back[0] < displacement):
                nearest_vehicle_back = [displacement, lane_vehicle]
            print "output "
            print nearest_vehicle_front, nearest_vehicle_back
        return nearest_vehicle_front, nearest_vehicle_back
    
    def getFeatureVectorImp(self, vehicle, speeds, lanes_frames):
                '''
                A particular frame lo vehicle coordinates, speeds of all the vehicles in the frame
                Vehicles split into different frames.
                '''
                print "vehicle", vehicle
                feature_vec = copy.deepcopy(speeds[vehicle[0]][frame])
                print "speed feature ", feature_vec 
                # Distance from the front and back vehicle and their speeds
                lane_vehicles = lanes_frames[frame][vehicle[1]]
                print "vehicles in your lane : ", len(lane_vehicles), lane_vehicles
                nearest_vehicle_front, nearest_vehicle_back = self.__get_nearest_front_and_back(lane_vehicles, vehicle)
                print "Nearest Front and back : ", nearest_vehicle_front, nearest_vehicle_back
                if nearest_vehicle_front != []:
                    feature_vec.append(nearest_vehicle_front[0])  
                    feature_vec = feature_vec + speeds[nearest_vehicle_front[1][0]][frame]
                else:
                    feature_vec += [-1, -1, -1]
                if nearest_vehicle_back != []:
                    feature_vec.append(nearest_vehicle_back[0])
                    feature_vec = feature_vec + speeds[nearest_vehicle_back[1][0]][frame]
                else:
                    feature_vec += [-1, -1, -1]
                print "front back feature ", feature_vec 

               # Empty distance to its left and right
                left_front_vehicle = []
                left_back_vehicle = []
                right_front_vehicle = []
                right_back_vehicle = []
                lane_index = lanes.index(vehicle[1])
                left_index = lane_index-1
                if(left_index >= 0 and lanes[left_index] in lanes_frames[frame].keys()):
                    left_front_vehicle, left_back_vehicle = self.__get_nearest_front_and_back(lanes_frames[frame][lanes[left_index]], vehicle)
                    print "Vehicles in the lane to the left : ", lanes_frames[frame][lanes[left_index]]
                    if left_front_vehicle != []:
                            feature_vec.append(left_front_vehicle[0])  
                            feature_vec = feature_vec + speeds[left_front_vehicle[1][0]][frame]
                    else:
                            feature_vec += [-1, -1, -1]
                    if left_back_vehicle != []:
                            feature_vec.append(left_back_vehicle[0])  
                            feature_vec = feature_vec + speeds[left_back_vehicle[1][0]][frame]
                    else:
                            feature_vec += [-1, -1, -1]
                elif left_index < 0:
                    print "No lane to the left or no vehicles in the left lane"
                    feature_vec += [-1, -1, -1, -1, -1, -1]
                else:
                    feature_vec += [100, 100, 100, 5, 5, 5] # Not sure what to do in such a situation for now using 100 in the front and 5 at the back

                print "front left feature ", feature_vec 
                right_index = lane_index+1

                if(right_index < len(lanes) and lanes[right_index] in lanes_frames[frame].keys()):
                    right_front_vehicle, right_back_vehicle = self.__get_nearest_front_and_back(lanes_frames[frame][lanes[right_index]], vehicle)
                    print "Vehicles in the lane to the right : ", lanes_frames[frame][lanes[right_index]]
                    if right_front_vehicle != []:
                            feature_vec.append(right_front_vehicle[0])
                            feature_vec = feature_vec + speeds[right_front_vehicle[1][0]][frame]
                    else:
                            feature_vec += [-1, -1, -1]
                    print "front right feature part1 ", feature_vec 
                    if right_back_vehicle != []:
                            feature_vec.append(right_back_vehicle[0])  
                            feature_vec = feature_vec + speeds[right_back_vehicle[1][0]][frame]
                    else:
                            feature_vec += [-1, -1, -1]
                    print "front right feature part2 ", feature_vec 
                elif right_index > len(lanes):
                    print "No lane to the right"
                    feature_vec += [-1, -1, -1, -1, -1, -1]
                else:
                    feature_vec += [100, 100, 100, 5, 5, 5] # Not sure what to do in such a situation for now using 100 in the front and 5 at the back.
                print "datapoint ", feature_vec, next_vel
                data_X.append(feature_vec)


    def loadDataWithOnlyImpFeatures(self, sumo_output_file):
        '''
            Output X,y from the XML file
            X.size = (number_of_samples, feature_size)
            Features considered are current speed, distance in the front, distance in the back, space to its left and right
            y.size = (number_of_samples, output_size)
        '''
        
        frames = self.getAllFrames(sumo_output_file)
        print "Number of frames ", len(frames.keys())
        data_X = []
        data_y = []
        # Get speeds
        speeds = self.getSpeeds(frames)
        print "Speeds :- "
        print speeds
        lanes_frames = self.lanes_frames(frames)
        print "Lanes :- "
        print lanes_frames
        lanes = [40.05, 120.15, 200.25, 280.35]
        for frame in frames.keys():
            print "Frame No: ", frame 
            print "vehicles ", frames[frame]
            for vehicle in frames[frame]:
                print "vehicle", vehicle
                feature_vec = copy.deepcopy(speeds[vehicle[0]][frame])
                print "speed feature ", feature_vec 
                # Distance from the front and back vehicle and their speeds
                lane_vehicles = lanes_frames[frame][vehicle[1]]
                print "vehicles in your lane : ", len(lane_vehicles), lane_vehicles
                nearest_vehicle_front, nearest_vehicle_back = self.__get_nearest_front_and_back(lane_vehicles, vehicle)
                print "Nearest Front and back : ", nearest_vehicle_front, nearest_vehicle_back
                if nearest_vehicle_front != []:
                    feature_vec.append(nearest_vehicle_front[0])  
                    feature_vec = feature_vec + speeds[nearest_vehicle_front[1][0]][frame]
                else:
                    feature_vec += [-1, -1, -1]
                if nearest_vehicle_back != []:
                    feature_vec.append(nearest_vehicle_back[0])
                    feature_vec = feature_vec + speeds[nearest_vehicle_back[1][0]][frame]
                else:
                    feature_vec += [-1, -1, -1]
                print "front back feature ", feature_vec 

               # Empty distance to its left and right
                left_front_vehicle = []
                left_back_vehicle = []
                right_front_vehicle = []
                right_back_vehicle = []
                lane_index = lanes.index(vehicle[1])
                left_index = lane_index-1
                if(left_index >= 0 and lanes[left_index] in lanes_frames[frame].keys()):
                    left_front_vehicle, left_back_vehicle = self.__get_nearest_front_and_back(lanes_frames[frame][lanes[left_index]], vehicle)
                    print "Vehicles in the lane to the left : ", lanes_frames[frame][lanes[left_index]]
                    if left_front_vehicle != []:
                            feature_vec.append(left_front_vehicle[0])  
                            feature_vec = feature_vec + speeds[left_front_vehicle[1][0]][frame]
                    else:
                            feature_vec += [-1, -1, -1]
                    if left_back_vehicle != []:
                            feature_vec.append(left_back_vehicle[0])  
                            feature_vec = feature_vec + speeds[left_back_vehicle[1][0]][frame]
                    else:
                            feature_vec += [-1, -1, -1]
                elif left_index < 0:
                    print "No lane to the left or no vehicles in the left lane"
                    feature_vec += [-1, -1, -1, -1, -1, -1]
                else:
                    feature_vec += [100, 100, 100, 5, 5, 5] # Not sure what to do in such a situation for now using 100 in the front and 5 at the back

                print "front left feature ", feature_vec 
                right_index = lane_index+1

                if(right_index < len(lanes) and lanes[right_index] in lanes_frames[frame].keys()):
                    right_front_vehicle, right_back_vehicle = self.__get_nearest_front_and_back(lanes_frames[frame][lanes[right_index]], vehicle)
                    print "Vehicles in the lane to the right : ", lanes_frames[frame][lanes[right_index]]
                    if right_front_vehicle != []:
                            feature_vec.append(right_front_vehicle[0])
                            feature_vec = feature_vec + speeds[right_front_vehicle[1][0]][frame]
                    else:
                            feature_vec += [-1, -1, -1]
                    print "front right feature part1 ", feature_vec 
                    if right_back_vehicle != []:
                            feature_vec.append(right_back_vehicle[0])  
                            feature_vec = feature_vec + speeds[right_back_vehicle[1][0]][frame]
                    else:
                            feature_vec += [-1, -1, -1]
                    print "front right feature part2 ", feature_vec 
                elif right_index > len(lanes):
                    print "No lane to the right"
                    feature_vec += [-1, -1, -1, -1, -1, -1]
                else:
                    feature_vec += [100, 100, 100, 5, 5, 5] # Not sure what to do in such a situation for now using 100 in the front and 5 at the back.
                print "front right feature ", feature_vec 
                next_frame = frame+1
                next_vel = [0.0, 0.0]
                if next_frame in speeds[vehicle[0]].keys():
                     next_vel = copy.deepcopy(speeds[vehicle[0]][next_frame])
                feature_vec = np.array(feature_vec)
                if feature_vec.shape[0] != 20:
                    print "Alert!!!"
                next_vel = np.array(next_vel)
                print "datapoint ", feature_vec, next_vel
                data_X.append(feature_vec)
                data_y.append(next_vel)
        data_X = np.array(data_X)
        data_y = np.array(data_y)
        return data_X, data_y
                
                
    def loadData(self, sumo_output_file):
        '''
            Output X,y from the XML file
            X.size = (number_of_samples, feature_size)
            y.size = (number_of_samples, output_size)
        '''
        
        frames = self.getAllFrames(sumo_output_file)
        Data_X = []
        Data_y = []

        # Here each of the frame is stored in frames.
        for frame in frames.keys():
                #print "Frame - ", frame
                vehicles = frames[frame]
                coordinates_array = []
               # print "Number of vehicles = ", len(vehicles)
               # print "Vehicles = ", vehicles
                if len(vehicles) == 0:
                    continue
                for vehicle in vehicles:
                    coordinates_array.append([vehicle[1],vehicle[2]])
                coordinates_array = np.array(coordinates_array)
                num_neighbours = min(len(vehicles), self.number_of_neighbours)
                nbrs = NearestNeighbors(n_neighbors=num_neighbours, algorithm='ball_tree').fit(coordinates_array)
                distances, indices = nbrs.kneighbors(coordinates_array)
                #print "Got the k nearest neighbours = ", len(indices[0])
                for vehicle_neighbours in indices:
                    feature_vec = []
                    for neighbour in vehicle_neighbours:
                        feature_vec.append(vehicles[neighbour][1])
                        feature_vec.append(vehicles[neighbour][2])
                    feature_vec += [-1]*(2*(self.number_of_neighbours) - len(feature_vec))
                    feature_vec = np.array(feature_vec)
                    next_frame = frame+1
                    velocity = [0,0]
                    if next_frame in frames.keys():
                        for vehicle in frames[next_frame]:
                            #print vehicle, vehicles[vehicle_neighbours[0]]
                            if vehicle[0] == vehicles[vehicle_neighbours[0]][0]:
                                velocity = [vehicle[1] - vehicles[vehicle_neighbours[0]][1], vehicle[2] -vehicles[vehicle_neighbours[0]][2]]
                    Data_X.append(feature_vec)
                    Data_y.append(velocity)
                    #print feature_vec, velocity
        Data_X = np.array(Data_X)
        Data_y = np.array(Data_y)
        return Data_X, Data_y 


    def switchTo(self, mode):
        '''
            train, validation, test
        '''
        self.mode = mode

    def nextBatch(self):
        '''
            return a particular batch depending the mode [Train, Validation, Test]
        '''
        end_of_set = False
        batch_no = self.batch_pointers[self.mode]
        if self.mode == 'train':
            batch_X = self.train_X[(batch_no)*self.batch_size:(batch_no+1)*self.batch_size, :]
            batch_y = self.train_y[(batch_no)*self.batch_size:(batch_no+1)*self.batch_size, :]
        if self.mode == 'validation':
            batch_X = self.validation_X[(batch_no)*self.batch_size:(batch_no+1)*self.batch_size, :]
            batch_y = self.validation_y[(batch_no)*self.batch_size:(batch_no+1)*self.batch_size, :]
        if self.mode == 'test':
            batch_X = self.test_X[(batch_no)*self.batch_size:(batch_no+1)*self.batch_size, :]
            batch_y = self.test_y[(batch_no)*self.batch_size:(batch_no+1)*self.batch_size, :]

        self.batch_pointers[self.mode] += 1
        if self.mode == 'train':
            number_of_samples = self.train_X.shape[0]
        if self.mode == 'validation':
            number_of_samples = self.validation_X.shape[0]
        if self.mode == 'test':
            number_of_samples = self.test_X.shape[0]

        if number_of_samples - self.batch_pointers[self.mode]*self.batch_size < self.batch_size:
            self.batch_pointers[self.mode] = 0
            end_of_set = True
        return batch_X, batch_y, end_of_set
