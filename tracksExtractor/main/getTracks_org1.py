#! /Users/gramaguru/anaconda/bin/python
import cv2
import numpy as np
import pdb
from scipy import misc
import sys
import os
import math
import pdb
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
	def __init__(self,x,y,h,w):
		self.init_x = x
		self.init_y = y
		self.height = h
		self.width  = w
		self.track = [(x,y)]
		self.completed = 0
		self.part_dispx = 0.0
		self.part_dispy = 0.0

	def isPixelinDet(self, p):
		print self.width, self.height
		if(p[0] >= (self.track[-1][0] - (self.width)/2) and p[0] < (self.track[-1][0] + (self.width)/2)):
			if(p[1] >= (self.track[-1][1] - (self.height)/2 and p[1] > (self.track[-1][1] + (self.height)/2))):
					return True
		return False


def updateTracksOfExistingObjects(flow, tracks, shape):
	for track in tracks:
		if track.completed == 1:
			continue

		disp = flow[track.track[-1][0]][track.track[-1][1]]
		dispy = disp[0]
		dispx = disp[1]
		track.part_dispx += dispx
		track.part_dispy += dispy
		updated_x = track.track[-1][0] - int(round(dispx))
		updated_y = track.track[-1][1] - int(round(dispy))
		#print shape
		if(updated_x < 0):
			updated_x = 0
			track.completed = 1
		if(updated_y < 0):
			updated_y = 0
			track.completed = 1
		if(updated_x >= shape[0]):
			updated_x = shape[0]-1
			track.completed = 1
		if(updated_y >= shape[1]):
			updated_y = shape[1]-1
			track.completed = 1

		print "updated_x, updated_y ", updated_x, updated_y, dispx, dispy
		'''
		if(int(math.floor(track.part_dispx)) == 0 and int(math.floor(track.part_dispy)) == 0):
			continue

		if(int(math.floor(track.part_dispx)) != 0):
			updated_x = track.track[-1][0]-int(round(track.part_dispx))
			track.part_dispx = 0.0
			if(updated_x < 0):
				updated_x = 0
				track.completed = 1
			if(updated_x >= 384):
				updated_x = 383
				track.completed = 1

		if(int(math.floor(track.part_dispy)) != 0):
			updated_y = track.track[-1][1]-int(round(track.part_dispy))
			track.part_dispy = 0.0
			if(updated_y < 0):
				updated_y = 0
				track.completed = 1
			if(updated_y >= 512):
				updated_y = 511
				track.completed = 1
		try:
		     updated_x
		except NameError:
		     updated_x = track.track[-1][0]
			
		try:
		     updated_y
		except NameError:
		     updated_y = track.track[-1][1]
		'''
		track.track.append((updated_x, updated_y))
	return tracks

def vehicleAlreadyTracked(vehicle,tracks):
	for track in tracks:
		# x, y, w, h = track.track[-1][0], track.track[-1][1], track.width, track.height
		# If IOU is greater than 50% then return True or else False
		if track.isPixelinDet((vehicle[0], vehicle[1])):
			return track

def printTracks(frames, tracks, visual):
	#print("Tracks")
	#print len(tracks)
	for i, track in enumerate(tracks):
#print track.track
#print visual.shape, " visual"
#print "coordinates track " + str(i) + " " + str(track.track[-1])
		visual[track.track[-1][0]][track.track[-1][1]] = [255,255,0]
		for (x,y) in track.track:
			frames[x][y] = [255,255,0]
	return frames, tracks, visual
	#cv2.circle(visual,(track.track[-1][0], track.track[-1][1]),5,color[i%100].tolist(),-1)
#cv2.putText(visual, str(i), (track.track[-1][0],track.track[-1][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,0,0), 1, cv2.CV_AA);

def drawTracks(visual, tracks):
#visual[100][110] = [255,255,0]
#cv2.circle(visual,(100, 140),10,color[0].tolist(),-1)
	cv2.imshow("Visual ",visual);
	for track in tracks:
		continue
		print track.track

def drawVehicles(frame, vehicles):
	for i,(x,y,w,h) in enumerate(vehicles):
		cv2.rectangle(frame,(int(round((x-w/2))),int(round(y+h/2))),(int(round(x+w/2)),int(round(y-h/2))),(0,0,255),2)
		cv2.putText(frame, str(i), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,0,0), 1, cv2.CV_AA);
	return frame
#vehicleDetectionFile = "../data/Trimmed_trafficVideo_predict.txt";
#total_frames = 1000;
#cap = cv2.VideoCapture("../data/Trimmed_trafficVideo.mp4");
def visualizeTracks(total_frames, videoFile, reSize):
	#cap = cv2.VideoCapture("../data/Trimmed_trafficVideo.mp4");
	#folder = "../output/Trimmed_trafficVideo_flow_output/tv"+str(fno)+"-"+str(fno+1)+".flo"
	fno = 0
	folder = "../output/" + videoFile.split('/')[-1].split('.')[0] + "_flow_output/"
	vehicleDetectionFile = videoFile.split('.')[0]+"_predict.txt"
	#print vehicleDetectionFile
	cap = cv2.VideoCapture(videoFile);
	visual = np.zeros(reSize, np.uint8);
	color = np.random.randint(0,255,(100,3))
	tracks = [];
	for fno in range(total_frames):
		ret, frames = cap.read();
		frames = misc.imresize(frames, reSize)
		vehicles, _ = getVehiclesFromFile(vehicleDetectionFile, fno) #Frame number
		print("Vehicles detected:-")
		print(vehicles.shape)
		#if fno == 0:
		#	vehicles = vehicles[4:5]
		#else:
		#	vehicles = np.array([])
		frames = drawVehicles(frames, vehicles)
		flow, w, h= getDenseflowPreComputed(folder+str(fno)+"-"+str(fno+1)+".flo")#Get the flow from the frame
		#print "Flow information "
		#print flow.shape, w, h
		newTracks = []
		# if the vehicle centers dont lie in any of the tracks create a new object
		count = 0
		for vehicle in vehicles:
			prevTrack = vehicleAlreadyTracked(vehicle, tracks);
			if prevTrack is not  None:
				count += 1
		#print prevTrack.track, " Previous track ", vehicle
				prevTrack.w = vehicle[2]
				prevTrack.h = vehicle[3]
				#v.updateObj(vehicle)
			else:
				newTracks.append(Track(vehicle[1], vehicle[0], vehicle[2], vehicle[3]))
		print "Count of already things " + str(count)
		tracks = updateTracksOfExistingObjects(flow, tracks, reSize)
		#print "updating the tracks is done"
		print "New tracks = ", len(newTracks)
		tracks = tracks + newTracks
		frames, tracks, visual = printTracks(frames, tracks, visual)
		cv2.imshow("Frames", frames)
		drawTracks(visual, tracks)
		#pdb.set_trace()
		if cv2.waitKey(33) == 27:
			break
	#cv2.waitKey(0)
	cv2.destroyAllWindows()
