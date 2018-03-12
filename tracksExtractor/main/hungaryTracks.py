import cv2
import numpy as np
import pdb
from scipy import misc
import sys
import os
import math
import pdb
import copy
import xml.etree.cElementTree as ET
from munkres import Munkres
cwd = os.getcwd()
sys.path.append(cwd + '/../objectdetection/')
sys.path.append(cwd + '/../denseflow/')
from objectDetectionLib import getVehiclesFromFile 
from denseflowLib import getDenseflowPreComputed 
'''
	This file will be used for the following
	a) Drawing the tracks of the vehicles which get detected.
	b) All these will happen in the camera frame of reference.

	First frame lo detected objects teesuko, using next flo files keep extending the tracks.
	For the subsequent frames when there is a detection which doesnot have a reference in the flows create a new.
	When there is no detection in the subsequent frames for the object ????
'''
class Track:
	def __init__(self,x,y,h,w,no,vehicleType):
		self.init_x = x
		self.init_y = y
		self.height = h
		self.width  = w
		self.track = [(x,y)]
		self.completed = 0
		self.trackno = no
                self.vehicleType = vehicleType

	def isPixelinDet(self, p):
		#print p[0], p[1], self.width, self.height, self.track
		if(p[0] >= (self.track[-1][0] - (self.width)/2) and p[0] < (self.track[-1][0] + (self.width)/2)):
			if(p[1] >= (self.track[-1][1] - (self.height)/2) and p[1] < (self.track[-1][1] + (self.height)/2)):
					return True
		return False

vehicleSeq = []  
## Will be list of vehicles which will contain, vehicle_type, lane_no.
## if the sequence is not good lets change the sequence in which we are sending the vehicles here, now it is random.
## hyper parameters have to be set.
def checkAndNoteAbtNewVehicle(cl, x, y, h, w): 
        max_new_x = 200
        max_new_y = 300
        lane_differences = [75, 135, 190]
        index = len(vehicleSeq)
        #print "adding vehicles ", x, y, max_new_x, max_new_y
        if (x < max_new_x  or y > max_new_y):
            #print "Ignoring Vehicle", x, y, max_new_x, max_new_y, cl
            return 0

        laneNo = len(lane_differences)
        for i, l in enumerate(lane_differences):
            if y < l:
                laneNo = i
                break
        vehicleSeq.append([cl, laneNo])
        #print "Vehicle is awesome:- " , len(vehicleSeq), vehicleSeq
        return 0

def getHistogram(vehicle, track, frames):
        frame_gray =  cv2.cvtColor(frames, cv2.COLOR_RGB2GRAY)
        print "Frame shape : ", frame_gray.shape, vehicle
        ## For vehicle
        x1 = vehicle[1] - vehicle[3]/2
        x2 = vehicle[1] + vehicle[3]/2
        y1 = vehicle[0] - vehicle[2]/2
        y2 = vehicle[0] + vehicle[2]/2
        print "vehicle histogram ", x1, x2, y1, y2
        histogram_vehicle = [0 for i in range(256)]
        total = 0
        for i in range(x1,x2):
            for j in range(y1,y2):
                if i >= 0 and i <= frame_gray.shape[0] and j >= 0 and j <= frame_gray.shape[1]:
                    histogram_vehicle[frame_gray[i,j]] += 1
                total += 1
        print histogram_vehicle
        histogram_vehicle = np.array(histogram_vehicle)/(total*1.0)
        print histogram_vehicle


        ## For track
        x1 = track.track[-1][1] - track.width/2
        x2 = track.track[-1][1] + track.width/2
        y1 = track.track[-1][0] - track.height/2
        y2 = track.track[-1][0] + track.height/2
        print "track histogram ", x1, x2, y1, y2
        histogram_track = [0 for i in range(256)]
        total = 0
        for i in range(x1,x2):
            for j in range(y1,y2):
                if i >= 0 and i <= frame_gray.shape[0] and j >= 0 and j <= frame_gray.shape[1]:
                    histogram_track[frame_gray[i,j]] += 1
                total += 1
        print histogram_track
        histogram_track = np.array(histogram_track)/(total*1.0)
        print histogram_track
        histogram_track = histogram_track.astype(np.float32)
        histogram_vehicle = histogram_vehicle.astype(np.float32)
        return histogram_vehicle, histogram_track

def velDist(vehicle, track):
    if len(track.track) == 1:
        return 0
    vel1y = vehicle[1] - track.track[-1][0]
    vel1x = vehicle[0] - track.track[-1][1]

    vel2y = track.track[-1][0] - track.track[-2][0]
    vel2x = track.track[-1][1] - track.track[-2][1]
    return math.sqrt(math.pow(vel1x-vel2x,2)+math.pow(vel1y-vel2y,2))


def getSimDist(vehicle, track, frames):
        #vehicle_hist, track_hist = getHistogram(vehicle, track,  frames)
        #print type(vehicle_hist) , type(track_hist)
        #return math.sqrt(math.pow(vehicle[0]-track.track[-1][1],2)+math.pow(vehicle[1]-track.track[-1][0],2)) + cv2.compareHist(vehicle_hist, track_hist, cv2.cv.CV_COMP_CHISQR);
        alpha = 0
        return math.sqrt(math.pow(vehicle[0]-track.track[-1][1],2)+math.pow(vehicle[1]-track.track[-1][0],2)) + alpha*velDist(vehicle, track);

def getMatrix(vehicles, tracks, frames):
	mat = 1000*np.ones((1000,1000));
	for i,vehicle in enumerate(vehicles):
		for j,track in enumerate(tracks):
			#if(track.completed == 1):
			#	continue
			#print "vehicle" , vehicle, track.track[-1]
			mat[i][j] = getSimDist(vehicle, track, frames)
                        if mat[i][j] > 100:
                            mat[i][j] = 1000
	nOfVehicles = vehicles.shape[0]
	nOfTracks = len(tracks)
	size = max(nOfVehicles, nOfTracks)
        print "Number of vehicles ", nOfVehicles
        print "Number of tracks ", nOfTracks
	return mat[:size,:size]

	# for each vehcile with each of the previous tracks.
	# return mat
def getHungary(vehicles, tracks, frames):
	hungarianMatrix = getMatrix(vehicles, tracks, frames)
	mat = np.copy(hungarianMatrix)
	m = Munkres()
	indexes = m.compute(hungarianMatrix)
        print "Hungarian Matrix ", mat
        print "Completd Hungarian Matrix ", hungarianMatrix
        print "Indexes of the matching ", indexes
	return mat,  indexes
	#calculate assignment and then update
	#return assignment

def updateTracks(hungarian, tracks, vehicles, vehicleTypes, simMat, trackNo, completedTracks):
	countCompletedTracks = len(completedTracks)
	newTracks = []
	simMax = 30# How do we find if the tracker is good, we need to measure the performance in standard datasets or take some good tracker.
	nOfUpdatedTracks = 0
	nOfMissedTracks = 0
	for (row,column) in hungarian:
		if column >= len(tracks):
			if vehicles[row][2] > 100 or vehicles[row][3] > 100:#if the box sizes are very big then they are not vehicles.
				continue
			newTracks.append(Track(vehicles[row][1], vehicles[row][0], vehicles[row][3], vehicles[row][2], countCompletedTracks + len(tracks)+len(newTracks)-1, vehicleTypes[row][0]))
                        checkAndNoteAbtNewVehicle(vehicleTypes[row], vehicles[row][1], vehicles[row][0], vehicles[row][3], vehicles[row][2])
			trackNo[row] = countCompletedTracks + len(tracks)+len(newTracks)-1
		elif row >= vehicles.shape[0]:
		 	tracks[column].completed += 1

		else:
		 	if(simMat[row][column] <= simMax):
		 		tracks[column].track.append((vehicles[row][1], vehicles[row][0]))
                                tracks[column].w = vehicles[row][2] 
                                tracks[column].h = vehicles[row][3]
				trackNo[row] = tracks[column].trackno
				nOfUpdatedTracks += 1
                                tracks[column].completed = 0
			else:
		 	        tracks[column].completed += 1
				if vehicles[row][2] > 100 or vehicles[row][3] > 100:
					continue
				newTracks.append(Track(vehicles[row][1], vehicles[row][0], vehicles[row][3], vehicles[row][2], countCompletedTracks + len(tracks) + len(newTracks) - 1, vehicleTypes[row][0]))
                                checkAndNoteAbtNewVehicle(vehicleTypes[row], vehicles[row][1], vehicles[row][0], vehicles[row][3], vehicles[row][2])
				trackNo[row] = countCompletedTracks + len(tracks) + len(newTracks) - 1
				#print simMat[row][column]
				nOfMissedTracks += 1
	#print nOfUpdatedTracks, nOfMissedTracks
        print "understanding the counts of tracks"
        mark_tracks = []
	for i, track in enumerate(tracks):
                print track.trackno, track.completed
		if track.completed >= 5:
			completedTracks.append(copy.deepcopy(track))
                        mark_tracks.append(i)
        for i in sorted(mark_tracks, reverse=True):		
            del tracks[i]
        print "Understanding the counts of tracks"
        print "New Tracks : "
        for track in newTracks:
            print track.trackno, track.track
        for i, track in enumerate(tracks):
            print i, track.trackno
        print "End of New Tracks"
	return tracks + newTracks, trackNo, completedTracks
		 	
		 		
	# Go to the matching thing if it is already a track , go to that track and update.
	# if it is a new track , create the new track
        # if the track has no mappings so far complete the track.

def printTracks(frames, tracks, visual):
	for i, track in enumerate(tracks):
                #print track.track, visual.shape
		visual[track.track[-1][0]][track.track[-1][1]] = [255,255,0]
		prevpt = track.track[0]
		for i, (x,y) in enumerate(track.track):
			if i == 0:
				continue;
			cv2.line(frames, (prevpt[1], prevpt[0]), (y,x), (255,0,0), 1)
			prevpt = (x,y)
	return frames, tracks, visual

def drawTracks(visual, tracks):
	cv2.imshow("Visual ",visual)

def drawVehicles(frame, vehicles, trackNo):
	for i,(x,y,h,w) in enumerate(vehicles):
		cv2.rectangle(frame,(int(round((x-h/2))),int(round(y+w/2))),(int(round(x+h/2)),int(round(y-w/2))),(0,0,255),2)
		frame[y][x] = [255,255,0]
		cv2.putText(frame, str(trackNo[i]), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,0,0), 1, cv2.CV_AA)
	return frame

def store_tracks_xml(tracks, root, fno):
        timestamp = ET.SubElement(root, "timestamp")
        timestamp.set("time", str(fno))
        for i, track in enumerate(tracks):
            vehicle = ET.SubElement(timestamp, "vehicle")
            vehicle.set("type",track.vehicleType)
            vehicle.set("id",str(track.trackno))
            vehicle.set("x", str(track.track[-1][1]))
            vehicle.set("y", str(track.track[-1][0]))
        return root

def filterVehicles(vehicles):
    print "Filter Vehicles: "
    mark_vehicles = []
    delete_vehicles = []
    for i, vehicle1 in enumerate(vehicles):
        for j, vehicle2 in enumerate(vehicles):
            if i == j:
                continue
            #print vehicle1, vehicle2, abs(vehicle1[0] - vehicle2[0]), abs(vehicle1[1] - vehicle2[1])
            if abs(vehicle1[0] - vehicle2[0]) < 15.0 and abs(vehicle1[1] - vehicle2[1]) < 15.0:
                mark_vehicles.append((i,j))
    for (i,j) in mark_vehicles:
        if vehicles[i][2]*vehicles[i][3] >  vehicles[j][2]*vehicles[j][3]:
            delete_vehicles.append(j)
        else:
            delete_vehicles.append(i)
    print "Deleted Vehicles", list(set(delete_vehicles))
    for i in sorted(list(set(delete_vehicles)), reverse=True):
        del vehicles[i]
    return vehicles

# For each of the frames, extract the vehicles and draw.
def visualizeTracks(total_frames, videoFile, reSize=None):
	fno = 0
	completed_tracks = []
        root = ET.Element("Traffic")
	static_plane = np.zeros((2880, 5120,3), dtype=float)
	vehicleDetectionFile = videoFile.split('.')[0]+"_predict.txt"
	#print vehicleDetectionFile
	cap = cv2.VideoCapture(videoFile)
	visual = np.zeros((1), np.uint8)
	color = np.random.randint(0,255,(100,3))
	tracks = []
	for fno in range(total_frames):
                print "Frameno: ", fno, len(completed_tracks)
		#pdb.set_trace()
		#print "frame number - " + str(fno)
		#print "Completed Tracks ", len(completed_tracks)
		#print "total tracks length " + str(len(tracks) + len(completed_tracks))
		ret, frames = cap.read()
		cv2.putText(frames, str(fno), (20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,0,0), 1, cv2.CV_AA)
                if reSize != None:
		    frames = misc.imresize(frames, reSize)
                if visual.shape == (1,):
                    visual = np.zeros(frames.shape, np.uint8)
		vehicles, vehicleTypes = getVehiclesFromFile(vehicleDetectionFile, fno) #Frame number
		print " number of vehicles ", vehicles.shape
		if vehicles.shape == (3,1) or vehicles.shape[0] == 0:
		    continue
                actual_vehicles = []
                print vehicles.shape
                for i, vehicle in enumerate(vehicles):
                    #print vehicle
                    if vehicle[2] > 100 or vehicle[3] > 100:
                           continue
                    actual_vehicles.append(vehicle)
                    actual_vehicles = filterVehicles(actual_vehicles)
                actual_vehicles = np.array(actual_vehicles)
		trackNo = [0]*(actual_vehicles.shape[0])
		hungarianMatrix, hungarianAssignment = getHungary(actual_vehicles, tracks, frames)
		tracks, trackNo, completed_tracks = updateTracks(hungarianAssignment, tracks, actual_vehicles, vehicleTypes, hungarianMatrix, trackNo, completed_tracks)
                '''
                    getTransformedTracks
                    getTransformedVehicles

                '''
		frames = drawVehicles(frames, actual_vehicles, trackNo)
		frames, tracks, visual = printTracks(frames, tracks, visual)
                root = store_tracks_xml(tracks, root, fno)
		#print fno, vehicles.shape[0], len(completed_tracks) , len(tracks)
		cv2.imshow("Frames", frames)
		#drawTracks(visual, tracks)
                if cv2.waitKey(33) == 27:
			break
	cv2.destroyAllWindows()
        tree = ET.ElementTree(root)
        tree.write("traffic.xml")
