import numpy as np
from random import randrange, sample
import xml.etree.ElementTree as ET
from sklearn.neighbors import NearestNeighbors
import os
import h5py
import copy
class SumoDataLoader:
    def __init__(self, sumo_output_file, val_frac, test_frac, batch_size, number_of_neighbours, log_file = 'log', mode='train', infer=False):
        '''
            Load sumo data
            Divide into val, test and train
            Have pointers for the same
        '''
        self.batch_size = batch_size
        self.number_of_neighbours = number_of_neighbours
        self.mode = mode
        self.carType = {'CAR':0, 'BUS':1, 'TRUCK':2, 'VAN':3, 'MOTORBIKE':4, 'BICYCLE':5}
        self.fp1 = open(log_file, "a")
        #X, y = self.loadData(sumo_output_file)
        print "infer 123 ", infer, sumo_output_file
        if infer == True:
            return

        X, y = self.loadDataWithOnlyImpFeatures(sumo_output_file)
        print "Total number of samples, feature size and output size", X.shape, y.shape
        self.number_of_samples = X.shape[0]

        # Saving in h5py format
        f = h5py.File(sumo_output_file.split(".xml")[0]+".hdf5", "w")
        print sumo_output_file.split(".xml")[0]+".hdf5"
        grp = f.create_group("SumoDataset")
        grp['X']=X
        grp['y']=y
        f.close()

        # Normalize the data
        #X = (X-np.mean(X, axis=0))
        #st = (np.std(X, axis=0))
        #st = [s if s != None or s != 0 else 1 for s in st]
        #self.fp1.write("Mean and Std " + str(np.mean(X, axis=0)) + " " + str(st))
        #X = X/st

        #np.random.seed(3)
        #np.random.shuffle(X)
        #np.random.shuffle(y)
        print "After shuffling the shape of X is " , X.shape
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
        self.fp1.close()
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
                v = np.array([ float(vehicle.attrib['id'].split('.')[1]), float(vehicle.attrib['x']), float(vehicle.attrib['y']), float(self.carType[vehicle.attrib['type']])])
                #v = np.array([ float(vehicle.attrib['id']), float(vehicle.attrib['x']), float(vehicle.attrib['y']), float(self.carType[vehicle.attrib['type'].upper()])])
                #v = np.array([float(vehicle.attrib['id']), float(vehicle.attrib['x']), float(vehicle.attrib['y']), float(vehicle.attrib['type'])])
                frames[float(timestamps.attrib['time'])].append(v)
        return frames

    def getSpeeds(self, frames):
        # Get speeds 
        speeds = {}
        pos_by_frame = {}
        avg_vely = 0.0
        max_vely = 0
        count = 0
        count_velx = 0
        vely = {}
        for index, frame in enumerate(frames.keys()):
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
                avg_vely += speeds[frame][vehicle[0]][1]
                count += 1
                if max_vely < speeds[frame][vehicle[0]][1]:
                    max_vely = speeds[frame][vehicle[0]][1]
                if int(speeds[frame][vehicle[0]][1]) not in vely.keys():
                    vely[int(speeds[frame][vehicle[0]][1])] = 1
                else:
                    vely[int(speeds[frame][vehicle[0]][1])] += 1
                if speeds[frame][vehicle[0]][0] != 0.0:
                        count_velx += 1
        print "avg speed y ", (avg_vely*1.0)/count
        print "number of non zero x ", count_velx
        print "max vel ", max_vely
        print "velocities are ", vely
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
        #print "vehicleY ", vehicleY
        for i, lane in enumerate(lanes):
            if lane >= vehicleY:
                return lane

        if i == len(lanes)-1:
            return lanes[i]

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

    def dist(self, x,y):   
        return np.sqrt(np.sum((x-y)**2))

    def __get_nearest_front_and_back(self, lane_vehicles, vehicle):
        nearest_vehicle_front = []
        nearest_vehicle_back = []
        for lane_vehicle in lane_vehicles:
            ##print "=== nearest_lane vehicles=="
            ##print lane_vehicles, vehicle
            #displacement = self.dist(lane_vehicle[1:3], vehicle[1:3])
            displacement = lane_vehicle[2] - vehicle[2]
            if displacement > 0 and (nearest_vehicle_front == [] or nearest_vehicle_front[0] > displacement):
                nearest_vehicle_front = [displacement, lane_vehicle]
            if displacement < 0 and (nearest_vehicle_back == [] or nearest_vehicle_back[0] < displacement):
                nearest_vehicle_back = [displacement, lane_vehicle]
            ##print "output "
            ##print nearest_vehicle_front, nearest_vehicle_back
        return nearest_vehicle_front, nearest_vehicle_back
    
    def getFeatureVectorImp(self, vehicle, speeds_prev_frame, speeds, lanes_frames, lanes):
                '''
                A particular frame lo vehicle coordinates, speeds of all the vehicles in the frame
                Vehicles split into different frames.
                '''
                #print "vehicle", vehicle
                feature_vec = copy.deepcopy(speeds[vehicle[0]])
                feature_vec[0] = feature_vec[0]#/100.0
                feature_vec[1] = feature_vec[1]#/20.0
                if vehicle[0] not in speeds_prev_frame:
                    feature_vec += [0,0]
                else:
                    feature_vec += [speeds[vehicle[0]][0] - speeds_prev_frame[vehicle[0]][0], speeds[vehicle[0]][1] - speeds_prev_frame[vehicle[0]][1]]
                feature_vec += [vehicle[3]]#/10.0]

                #print "speed feature ", feature_vec
                # Distance from the front and back vehicle and their speeds
                #lane_vehicles = lanes_frames[vehicle[1]]
                lane_vehicles = lanes_frames[self.getLane(vehicle, lanes)]
                #print "vehicles in your lane : ", len(lane_vehicles), lane_vehicles
                nearest_vehicle_front, nearest_vehicle_back = self.__get_nearest_front_and_back(lane_vehicles, vehicle)
                #print "Nearest Front and back : ", nearest_vehicle_front, nearest_vehicle_back
                if nearest_vehicle_front != []:
                    feature_vec.append(nearest_vehicle_front[0])#/700.0)
                    speeds[nearest_vehicle_front[1][0]][0] = speeds[nearest_vehicle_front[1][0]][0]#/100.0
                    speeds[nearest_vehicle_front[1][0]][1] = speeds[nearest_vehicle_front[1][0]][1]#/20.0
                    feature_vec = feature_vec + speeds[nearest_vehicle_front[1][0]]
                    if nearest_vehicle_front[1][0] not in speeds_prev_frame:
                        feature_vec += [0,0]
                    else:
                        id1 = nearest_vehicle_front[1][0]
                        feature_vec += [speeds[id1][0] - speeds_prev_frame[id1][0], speeds[id1][1] - speeds_prev_frame[id1][1]]
                    feature_vec += [nearest_vehicle_front[1][3]]
                else:
                    feature_vec += [-1, -1, -1, -1, -1,-1]
                if nearest_vehicle_back != []:
                    feature_vec.append(nearest_vehicle_back[0])#/700.0)
                    speeds[nearest_vehicle_back[1][0]][0] = speeds[nearest_vehicle_back[1][0]][0]#/100.0
                    speeds[nearest_vehicle_back[1][0]][1] = speeds[nearest_vehicle_back[1][0]][1]#/20.0
                    feature_vec = feature_vec + speeds[nearest_vehicle_back[1][0]]
                    if nearest_vehicle_back[1][0] not in speeds_prev_frame:
                        feature_vec += [0,0]
                    else:
                        id1 = nearest_vehicle_back[1][0]
                        feature_vec += [speeds[id1][0] - speeds_prev_frame[id1][0], speeds[id1][1] - speeds_prev_frame[id1][1]]
                    feature_vec += [nearest_vehicle_back[1][3]]
                else:
                    feature_vec += [-1, -1, -1, -1, -1,-1]
                #print "front back feature ", feature_vec

               # Empty distance to its left and right
                left_front_vehicle = []
                left_back_vehicle = []
                right_front_vehicle = []
                right_back_vehicle = []
                #lane_index = lanes.index(vehicle[1])
                lane_index = lanes.index(self.getLane(vehicle, lanes))
                left_index = lane_index-1
                if(left_index >= 0 and lanes[left_index] in lanes_frames.keys()):
                    left_front_vehicle, left_back_vehicle = self.__get_nearest_front_and_back(lanes_frames[lanes[left_index]], vehicle)
                    #print "Vehicles in the lane to the left : ", lanes_frames[lanes[left_index]]
                    if left_front_vehicle != []:
                            feature_vec.append(left_front_vehicle[0])#/700.0)
                            speeds[left_front_vehicle[1][0]][0] = speeds[left_front_vehicle[1][0]][0]#/100.0
                            speeds[left_front_vehicle[1][0]][1] = speeds[left_front_vehicle[1][0]][1]#/20.0
                            feature_vec = feature_vec + speeds[left_front_vehicle[1][0]]
                            if left_front_vehicle[1][0] not in speeds_prev_frame:
                                feature_vec += [0,0]
                            else:
                                id1 = left_front_vehicle[1][0]
                                feature_vec += [speeds[id1][0] - speeds_prev_frame[id1][0], speeds[id1][1] - speeds_prev_frame[id1][1]]
                            feature_vec += [left_front_vehicle[1][3]]
                    else:
                            feature_vec += [-1, -1, -1, -1, -1,-1]
                    if left_back_vehicle != []:
                            feature_vec.append(left_back_vehicle[0])#/700.0)
                            speeds[left_back_vehicle[1][0]][0] = speeds[left_back_vehicle[1][0]][0]#/100.0
                            speeds[left_back_vehicle[1][0]][1] = speeds[left_back_vehicle[1][0]][1]#/20.0
                            feature_vec = feature_vec + speeds[left_back_vehicle[1][0]]
                            if left_back_vehicle[1][0] not in speeds_prev_frame:
                                feature_vec += [0,0]
                            else:
                                id1 = left_back_vehicle[1][0]
                                feature_vec += [speeds[id1][0] - speeds_prev_frame[id1][0], speeds[id1][1] - speeds_prev_frame[id1][1]]
                            feature_vec += [left_back_vehicle[1][3]]
                    else:
                            feature_vec += [-1, -1, -1, -1, -1,-1]
                elif left_index < 0:
                    #print "No lane to the left or no vehicles in the left lane"
                    feature_vec += [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
                else:
                    feature_vec += [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1] # Not sure what to do in such a situation for now using 100 in the front and 5 at the back

                #print "front left feature ", feature_vec 
                right_index = lane_index+1

                if(right_index < len(lanes) and lanes[right_index] in lanes_frames.keys()):
                    right_front_vehicle, right_back_vehicle = self.__get_nearest_front_and_back(lanes_frames[lanes[right_index]], vehicle)
                 #   print "Vehicles in the lane to the right : ", lanes_frames[lanes[right_index]]
                    if right_front_vehicle != []:
                            feature_vec.append(right_front_vehicle[0])#/700.0)
                            speeds[right_front_vehicle[1][0]][0] = speeds[right_front_vehicle[1][0]][0]#/100.0
                            speeds[right_front_vehicle[1][0]][1] = speeds[right_front_vehicle[1][0]][1]#/20.0
                            feature_vec = feature_vec + speeds[right_front_vehicle[1][0]]
                            if right_front_vehicle[1][0] not in speeds_prev_frame:
                                feature_vec += [0,0]
                            else:
                                id1 = right_front_vehicle[1][0]
                                feature_vec += [speeds[id1][0] - speeds_prev_frame[id1][0], speeds[id1][1] - speeds_prev_frame[id1][1]]
                            feature_vec += [right_front_vehicle[1][3]]
                    else:
                            feature_vec += [-1, -1, -1, -1, -1, -1]
                 #   print "front right feature part1 ", feature_vec 
                    if right_back_vehicle != []:
                            feature_vec.append(right_back_vehicle[0])#/700.0)
                            speeds[right_back_vehicle[1][0]][0] = speeds[right_back_vehicle[1][0]][0]#/100.0
                            speeds[right_back_vehicle[1][0]][1] = speeds[right_back_vehicle[1][0]][1]#/20.0
                            feature_vec = feature_vec + speeds[right_back_vehicle[1][0]]
                            if right_back_vehicle[1][0] not in speeds_prev_frame:
                                feature_vec += [0,0]
                            else:
                                id1 = right_back_vehicle[1][0]
                                feature_vec += [speeds[id1][0] - speeds_prev_frame[id1][0], speeds[id1][1] - speeds_prev_frame[id1][1]]
                            feature_vec += [right_back_vehicle[1][3]]
                    else:
                            feature_vec += [-1, -1, -1, -1, -1, -1]
                  #  print "front right feature part2 ", feature_vec 
                elif right_index > len(lanes):
                  #  print "No lane to the right"
                    feature_vec += [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
                else:
                    feature_vec += [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1] # Not sure what to do in such a situation for now using 100 in the front and 5 at the back.
                #print "feature_vec : ", feature_vec
                return feature_vec 


    def loadDataWithOnlyImpFeatures(self, sumo_output_file):
        '''
            Output X,y from the XML file
            X.size = (number_of_samples, feature_size)
            Features considered are current speed, distance in the front, distance in the back, space to its left and right
            y.size = (number_of_samples, output_size)
        '''
        frames = self.getAllFrames(sumo_output_file)
        print "Number of frames ", len(frames.keys())
        print self.fp1
        self.fp1.write("Number of frames " + str(len(frames.keys())) + "\n")
        data_X = []
        data_y = []
        # Get speeds
        speeds = self.getSpeeds(frames)
        #print "Speeds :- "
        #print speeds
        self.fp1.write(str(speeds) + "\n")
        lanes = [40.05, 120.15, 200.25, 280.35]
        #lanes = [165.0, 330.0, 495.0, 660.0]
        lanes_frames = self.lanes_frames_in(frames, lanes)
        self.fp1.write(str(lanes_frames)+"\n")

        avg_distance_front = 0.0
        c1 = 0
        avg_distance_back = 0.0
        c2 = 0
        avg_distance_sides = 0.0
        c3 = 0
        avg_distance = 0.0
        c4 = 0
        max_distance = 0.0
        num_vec_1 = {}
        dist = {}

        for frame in frames.keys():
            #print "Frame No: ", frame 
            self.fp1.write(str(frame)+"\n")
            #print "vehicles ", frames[frame]
            self.fp1.write(str(frames[frame]) + "\n")
            for vehicle in frames[frame]:
                if frame-1 in frames.keys():
                    feature_vec = self.getFeatureVectorImp(vehicle, speeds[frame-1], speeds[frame], lanes_frames[frame], lanes)
                else:
                    feature_vec = self.getFeatureVectorImp(vehicle, {}, speeds[frame], lanes_frames[frame], lanes)

                ##### statistics
                k = 0
                for i,feature in enumerate(feature_vec):
                    if feature == -1:
                        k += 1
                        continue
                    if i == 3:
                        avg_distance_front += feature
                        avg_distance += feature
                        c1 += 1
                        c4 += 1
                        if max_distance < feature:
                            max_distance = feature
                        if int(feature) not in dist.keys():
                            dist[int(feature)] = 1
                        else:
                            dist[int(feature)] += 1
                    elif i == 6:
                        avg_distance_back += feature
                        avg_distance += feature
                        c2 += 1
                        c4 += 1
                        if max_distance < feature:
                            max_distance = feature
                        if int(feature) not in dist.keys():
                            dist[int(feature)] = 1
                        else:
                            dist[int(feature)] += 1
                    elif i == 9 or i == 12 or i == 15 or i == 18:
                        avg_distance_sides += feature
                        avg_distance += feature
                        c3 += 1
                        c4 += 1
                        if max_distance < feature:
                            max_distance = feature
                        if int(feature) not in dist.keys():
                            dist[int(feature)] = 1
                        else:
                            dist[int(feature)] += 1
                if k not in num_vec_1.keys():
                    num_vec_1[k] = 1
                else:
                    num_vec_1[k] += 1
                #### Statistics output

                next_frame = frame+1
                next_vel = [0.0, 0.0]
                if next_frame in speeds.keys() and vehicle[0] in speeds[next_frame].keys():
                     next_vel = copy.deepcopy(speeds[next_frame][vehicle[0]])
                feature_vec = np.array(feature_vec)
                next_vel = np.array(next_vel)
                if next_vel[0] != 0.0:
                    self.fp1.write("Guru\n")
                #print "datapoint ", feature_vec, next_vel
                self.fp1.write(str(feature_vec)+ "->")
                self.fp1.write(str(next_vel)+"\n")
                #insert_index = randrange(len(data_X)+1)
                data_X.append(feature_vec)
                data_y.append(next_vel)
                #data_X.insert(insert_index, feature_vec)
                #data_y.insert(insert_index, next_vel)

        print "average distance front ", avg_distance_front*1.0/c1
        print "average distance back ", avg_distance_back*1.0/c2
        print "average distance sides ", avg_distance_sides*1.0/c3
        print "average distance ", avg_distance*1.0/c4
        print "max distance ", max_distance
        print "num of vec ", num_vec_1
        print "distances ", dist
        data_X = np.array(data_X)
        data_y = np.array(data_y)
        return data_X, data_y
                
    def loadData(self, sumo_output_file):
        '''
            Output X,y from the XML file
            X.size = (number_of_samples, feature_size)
            y.size = (number_of_samples, output_size)
        '''
        
        Data_X = []
        Data_y = []

        frames = self.getAllFrames(sumo_output_file)
        speeds = self.getSpeeds(frames)
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
                        #feature_vec.append(np.vehicles[neighbour][1]-coordinates_array[0])
                        #feature_vec.append(vehicles[neighbour][2]-coordinates_array[1])
                        feature_vec.append(self.dist(vehicles[neighbour][1:3], coordinates_array))
                        feature_vec.append(self.speeds[frame, vehicles[neighbour][0]])
                    feature_vec += [-1]*(3*(self.number_of_neighbours) - len(feature_vec))
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

