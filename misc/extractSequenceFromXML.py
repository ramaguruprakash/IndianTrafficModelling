import xml.etree.ElementTree as ET
import numpy as np
import cv2
import os

carType = {'CAR':0, 'BUS':1, 'TRUCK':2, 'VAN':3, 'MOTORBIKE':4, 'BICYCLE':5}
revCarType = {0:'CAR', 1:'BUS', 2:'TRUCK', 3:'VAN', 4:'MOTORBIKE', 5:'BICYCLE'}
def getLane(vehicle, number_of_lanes, grid_size):
    width = grid_size[0]
    lanes  = [(i*1.0/number_of_lanes)*width for i in range(number_of_lanes+1)]
    print lanes
    vehicleY = vehicle[2]
    for i, lane in enumerate(lanes):
        if lane > vehicleY:
            vehicle = np.append(vehicle, np.array([i]))
            break

    if i == len(lanes)-1:
        vehicle = np.append(vehicle, np.array([i]))
    return vehicle

def getAllFrames(output_file):
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

            v = np.array([float(vehicle.attrib['id']), float(vehicle.attrib['x']), float(vehicle.attrib['y']), float(carType[(vehicle.attrib['type'].upper())]), float(timestamps.attrib['time'])])
            frames[float(timestamps.attrib['time'])].append(v)
    return frames, count_of_frames_per_vehicle

def extractSequenceFromXML(xml, grid_size, with_lane = False,  xlimit = 0, ylimit = 0):
    frames, count_of_frames_per_vehicle  = getAllFrames(xml)
    output = []
    list_of_vehicles = []
    for i, frame in enumerate(frames.keys()):
        vehicles = frames[frame]
        for vehicle in vehicles:
            if vehicle[0] not in list_of_vehicles and vehicle[1] > xlimit and vehicle[2] > ylimit:
                if with_lane:
                    output.append(getLane(vehicle, 4, grid_size)) # number of lanes
                else:
                    output.append(vehicle)
                list_of_vehicles.append(vehicle[0])
    return output

if __name__ == '__main__':
    #print extractSequenceFromXML("/Users/gramaguru/SumoNetowrk/simulation_10000sec_3000cars.xml", (660, 300, 3))
    vehs = extractSequenceFromXML("/Users/gramaguru/Desktop/car_videos/sing_cropped.xml", (660, 300, 3), True)
    for veh in vehs:
        print veh[4],",",revCarType[veh[3]],",",veh[0],",",veh[1],",",veh[2],",",int(veh[5])
