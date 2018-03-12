import numpy as np
import xml.etree.ElementTree as ET
from sklearn.neighbors import NearestNeighbors
import h5py
import os
import copy
class SumoDataLoader_nn:
    def __init__(self, sumo_output_file, val_frac, test_frac, batch_size, number_of_neighbours, log_file='log', mode='train', infer=False):
        '''
            Load sumo data
            Divide into val, test and train
            Have pointers for the same
        '''
        self.batch_size = batch_size
        self.number_of_neighbours = number_of_neighbours
        self.mode = mode
        self.carType = {'CAR':0, 'BUS':1, 'TRUCK':2, 'VAN':3, 'MOTORBIKE':4, 'BICYCLE':5}
        #X, y = self.loadData(sumo_output_file)
        print "infer ", infer, sumo_output_file
        if infer == True:
            return
        #X, y = self.loadDataWithOnlyImpFeatures(sumo_output_file)
        X, y = self.loadData(sumo_output_file)
        print y[:200,:]
        print "Total number of samples, feature size and output size", X.shape, y.shape
        self.number_of_samples = X.shape[0]
        
        #Normalize the data
        #mn = np.mean(X, axis=0)
        #st = np.std(X, axis=0)
        #print mn, st
        #st = [s if s!=0 and s!=None else 1 for s in st]
        #X = (X-mn)/(st)
        f = h5py.File(sumo_output_file.split(".")[0]+".hdf5", "w")
        print sumo_output_file.split(".")[0]+".hdf5"
        grp = f.create_group("SumoDataset")
        grp['X']=X
        grp['y']=y
        f.close()

        self.train_X = X[:int(self.number_of_samples*(1-val_frac-test_frac)), :]
        self.train_y = y[:int(self.number_of_samples*(1-val_frac-test_frac)), :]
        self.validation_X = X[int(self.number_of_samples*(1-val_frac-test_frac)):int(self.number_of_samples*(1-test_frac)), :]
        self.validation_y = y[int(self.number_of_samples*(1-val_frac-test_frac)):int(self.number_of_samples*(1-test_frac)), :]
 
        self.test_X = X[int(self.number_of_samples*(1-test_frac)):, :]
        self.test_y = y[int(self.number_of_samples*(1-test_frac)):, :]
        print "Printing entire Training thing"
        #for i in range(self.train_X.shape[0]):
         #   print self.train_X[i], self.train_y[i]

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
            #print timestamps.tag, timestamps.attrib
            frames[float(timestamps.attrib['time'])] = []
            for vehicle in timestamps:
                #print vehicle.tag, vehicle.attrib
                #v = np.array([ float(vehicle.attrib['id'].split('.')[1]), float(vehicle.attrib['x']), float(vehicle.attrib['y']), float(self.carType[vehicle.attrib['type']])])
                v = np.array([ float(vehicle.attrib['id']), float(vehicle.attrib['x']), float(vehicle.attrib['y'])]) #float(self.carType[vehicle.attrib['type'].upper()])])
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
            speeds[frame] = {}
            prev_frame = frame-1
            for vehicle in vehicles:
                #print "Vehicle - ", vehicle[0]
                if prev_frame not in speeds.keys() or vehicle[0] not in speeds[prev_frame].keys():
                    speeds[frame][vehicle[0]] = [0.0, 0.0]
                    pos_by_frame[vehicle[0]] = {frame : [vehicle[1], vehicle[2]]}
                    #print "Speeds ", speeds[frame]
                else:
                    pos_by_frame[vehicle[0]][frame] = [vehicle[1], vehicle[2]]
                    if prev_frame in pos_by_frame[vehicle[0]].keys():
                        speeds[frame][vehicle[0]] = [vehicle[1] - pos_by_frame[vehicle[0]][prev_frame][0], vehicle[2] - pos_by_frame[vehicle[0]][prev_frame][1]]
                    else:
                        speeds[frame][vehicle[0]] = [0, 0]
                #print "Speeds : ", speeds[frame]
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

    def getLane(self, vehicle, lanes):
        vehicleY = vehicle[1]
        print "vehicleY ", vehicleY
        for i, lane in enumerate(lanes):
            if lane >= vehicleY:
                return lane

        if i == len(lanes)-1:
            return lanes[i]

    def getLane_lateral(self, vehicle, lanes):
        vehicleY = vehicle[1]
        print "vehicleY ", vehicleY
        lanes_copy = copy.deepcopy(lanes)
        lanes_copy  = [abs(lane-vehicleY) for lane in lanes_copy]
        min_index = lanes_copy.index(min(lanes_copy))
        return lanes[min_index]


    def lanes_frames_in(self, frames, lanes):
        lanes_frames = {}
        for frame in frames.keys():
            #print "====== Lanes diff frame ", frame, " ======"
            lanes_frames[frame] = {lane:[] for lane in lanes}
           # print frames[frame]
            for vehicle in frames[frame]:
                if self.getLane(vehicle, lanes) not in lanes_frames[frame]:
                    lanes_frames[frame][self.getLane(vehicle, lanes)] = [vehicle]
                else:
                    lanes_frames[frame][self.getLane(vehicle, lanes)].append(vehicle)
            #print "Vehicle in lanes "
            #print lanes_frames[frame]
        return lanes_frames

    def __get_nearest_front_and_back(self, lane_vehicles, vehicle):
        nearest_vehicle_front = []
        nearest_vehicle_back = []
        for lane_vehicle in lane_vehicles:
            ##print "=== nearest_lane vehicles=="
            ##print lane_vehicles, vehicle
            displacement = lane_vehicle[2] - vehicle[2]
            if displacement > 0 and (nearest_vehicle_front == [] or nearest_vehicle_front[0] > displacement):
                nearest_vehicle_front = [displacement, lane_vehicle]
            if displacement < 0 and (nearest_vehicle_back == [] or nearest_vehicle_back[0] < displacement):
                nearest_vehicle_back = [displacement, lane_vehicle]
            ##print "output "
            ##print nearest_vehicle_front, nearest_vehicle_back
        return nearest_vehicle_front, nearest_vehicle_back
    
    def getNNFeaturesFromFrame(self, vehicles, speeds_frame, speeds, frames, infer = False):
            feature_vecs = []
            velocities = []
            coordinates_array = []
            #print "Number of vehicles = ", len(vehicles)
            #print "Vehicles = ", vehicles
            if len(vehicles) == 0:
               return feature_vecs, velocities 
            for vehicle in vehicles:
                coordinates_array.append([vehicle[1],vehicle[2]])
            coordinates_array = np.array(coordinates_array)
            num_neighbours = min(len(vehicles), self.number_of_neighbours)
            nbrs = NearestNeighbors(n_neighbors=num_neighbours, algorithm='ball_tree').fit(coordinates_array)
            distances, indices = nbrs.kneighbors(coordinates_array)
            #print "Got the k nearest neighbours = ", len(indices[0]), indices, distances
            for vehicle_neighbours,distance in zip(indices,distances):
                feature_vec = []
                for neighbour,dist in zip(vehicle_neighbours,distance):
                    #feature_vec.append(vehicles[neighbour][1])
                    #feature_vec.append(vehicles[neighbour][2])
                    if dist == 0:
                        feature_vec += (speeds_frame[vehicles[neighbour][0]])
                    else:
                        #feature_vec.append(dist)
                        feature_vec.append(vehicles[neighbour][1] - vehicles[vehicle_neighbours[0]][1])
                        feature_vec.append(vehicles[neighbour][2] - vehicles[vehicle_neighbours[0]][2])
                        feature_vec += (speeds_frame[vehicles[neighbour][0]])
                feature_vec += [-1]*(4*(self.number_of_neighbours) - 2 - len(feature_vec))
                feature_vec = np.array(feature_vec)
                feature_vecs.append(feature_vec)
                if not infer:
                    next_frame = frame+1
                    velocity = [0,0]
                    if next_frame in frames.keys():
                        for vehicle in frames[next_frame]:
                            #print vehicle, vehicles[vehicle_neighbours[0]]
                            if vehicle[0] == vehicles[vehicle_neighbours[0]][0]:
                                velocity = [vehicle[1] - vehicles[vehicle_neighbours[0]][1], vehicle[2] -vehicles[vehicle_neighbours[0]][2]]
                        #print "feature_vec is - ", feature_vec, " -> ", velocity
                    if velocity[0] == 0 and velocity[1] == 0:
                        continue
                    velocities.append(velocity)
            return feature_vecs, velocities
 
                
    def loadData(self, sumo_output_file):
        '''
            Output X,y from the XML file
            X.size = (number_of_samples, feature_size)
            y.size = (number_of_samples, output_size)
        '''
        
        frames = self.getAllFrames(sumo_output_file)
        speeds = self.getSpeeds(frames)
        print "Speeds", speeds
        Data_X = []
        Data_y = []
        # Here each of the frame is stored in frames.
        for frame in frames.keys():
            print "Frame - ", frame
            vehicles = frames[frame]
            feature_vecs, velocities = self.getNNFeaturesFromFrame(vehicles, speeds[frame], speeds, frames)
            Data_X += feature_vecs
            Data_y += velocity
        Data_X = np.array(Data_X)
        Data_y = np.array(Data_y)
        return Data_X, Data_y 


    def switchTo(self, mode):
        '''
            train, validation, test
        '''
        self.mode = mode

    def reset(self):
        self.batch_pointers = {'train':0, 'validation':0, 'test':0}

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

