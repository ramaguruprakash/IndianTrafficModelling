#! /Users/gramaguru/anaconda/bin/python
import cv2
import numpy as np
import pdb
from scipy import misc
import sys
import os
import math
import pdb
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
	def __init__(self,x,y,h,w,no):
		self.init_x = x
		self.init_y = y
		self.height = h
		self.width  = w
		self.track = [(x,y)]
		self.completed = 0
		self.part_dispx = 0.0
		self.part_dispy = 0.0
		self.trackno = no

	def isPixelinDet(self, p):
		#print p[0], p[1], self.width, self.height, self.track
		if(p[0] >= (self.track[-1][0] - (self.width)/2) and p[0] < (self.track[-1][0] + (self.width)/2)):
			if(p[1] >= (self.track[-1][1] - (self.height)/2) and p[1] < (self.track[-1][1] + (self.height)/2)):
					return True
		return False


def getSimDist(vehicle, track):
	return math.sqrt(math.pow(vehicle[0]-track.track[-1][1],2)+math.pow(vehicle[1]-track.track[-1][0],2));

def getMatrix(vehicles, tracks):
	mat = 1000*np.ones((1000,1000));
	for i,vehicle in enumerate(vehicles):
		for j,track in enumerate(tracks):
			if(track.completed == 1):
				continue
			#print "vehicle" , vehicle, track.track[-1]
			mat[i][j] = getSimDist(vehicle, track);
	nOfVehicles = vehicles.shape[0]
	nOfTracks = len(tracks)
	size = max(nOfVehicles, nOfTracks)
	return mat[:size,:size]

					

	# for each vehcile with each of the previous tracks.
	# return mat
def getHungary(vehicles, tracks):
	hungarianMatrix = getMatrix(vehicles, tracks)
	mat = np.copy(hungarianMatrix)
	m = Munkres()
	indexes = m.compute(hungarianMatrix)
	return mat, indexes
	#calculate assignment and then update
	#return assignment

def updateTracks(hungarian, tracks, vehicles, simMat, trackNo, countCompletedTracks):
	newTracks = []
	simMax = 20
	for (row,column) in hungarian:
		if column >= len(tracks):
			if vehicles[row][2] > 100 or vehicles[row][3] > 100:
				continue
			newTracks.append(Track(vehicles[row][1], vehicles[row][0], vehicles[row][3], vehicles[row][2], countCompletedTracks + len(tracks)+len(newTracks)-1))
			trackNo[row] = countCompletedTracks + len(tracks)+len(newTracks)-1
		elif row >= vehicles.shape[0]:
		 	tracks[column].completed = 1
		else:
		 	if(simMat[row][column] <= simMax):
		 		tracks[column].track.append((vehicles[row][1], vehicles[row][0]))
				trackNo[row] = tracks[column].trackno
			else:
			  	print simMat[row][column]
				if vehicles[row][2] > 100 or vehicles[row][3] > 100:
					continue
				newTracks.append(Track(vehicles[row][1], vehicles[row][0], vehicles[row][3], vehicles[row][2], countCompletedTracks + len(tracks) + len(newTracks) - 1))
				trackNo[row] = countCompletedTracks + len(tracks) + len(newTracks) - 1

	return tracks + newTracks, trackNo
		 	
		 		
	# Go to the matching thing if it is already a track , go to that track and update.
	# if it is a new track , create the new track
        # if the track has no mappings so far complete the track.

def printTracks(frames, tracks, visual):
	for i, track in enumerate(tracks):
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
	for i,(x,y,w,h) in enumerate(vehicles):
		cv2.rectangle(frame,(int(round((x-h/2))),int(round(y+w/2))),(int(round(x+h/2)),int(round(y-w/2))),(0,0,255),2)
		frame[y][x] = [255,255,0]
		cv2.putText(frame, str(trackNo[i]), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,0,0), 1, cv2.CV_AA)
	return frame

def postProcessTracks(tracks):
	completed = []
	for i,track in enumerate(tracks):
		if track.completed == 1:
			completed.append(track)
			del tracks[i]
	return tracks, completed
		
def visualizeTracks(total_frames, videoFile, reSize):
	fno = 0
	completed_tracks = []
	static_plane = np.zeros((2880, 5120,3), dtype=float)
	vehicleDetectionFile = videoFile.split('.')[0]+"_predict.txt"
	#print vehicleDetectionFile
	cap = cv2.VideoCapture(videoFile)
	visual = np.zeros(reSize, np.uint8)
	color = np.random.randint(0,255,(100,3))
	tracks = []
	for fno in range(total_frames):
		#pdb.set_trace()
		#print "frame number - " + str(fno)
		#print "Completed Tracks ", len(completed_tracks)
		#print "total tracks length " + str(len(tracks) + len(completed_tracks))
		ret, frames = cap.read()
		frames = misc.imresize(frames, reSize)
		vehicles, _ = getVehiclesFromFile(vehicleDetectionFile, fno) #Frame number
		#print " number of vehicles ", vehicles.shape
		if vehicles.shape == 3:
			break
		trackNo = [0]*(vehicles.shape[0])
		hungarianMatrix, hungarianAssignment = getHungary(vehicles, tracks);	
		tracks, trackNo = updateTracks(hungarianAssignment, tracks, vehicles, hungarianMatrix, trackNo, len(completed_tracks))
		tracks, ctracks = postProcessTracks(tracks)
       		completed_tracks += ctracks	
		frames = drawVehicles(frames, vehicles, trackNo)
		frames, tracks, visual = printTracks(frames, tracks, visual)
		cv2.imshow("Frames", frames)
		drawTracks(visual, tracks)
		if cv2.waitKey(33) == 27:
			break
	cv2.destroyAllWindows()
