#! /Users/gramaguru/anaconda/bin/python
import cv2
import numpy as np
import pdb
from scipy import misc
from readflo import readFlow
from ../objectdetection/objectDetectionLib import 
from ../denseflow/denseflowLib import 
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

	def isPixelinDet(self, p):
		if(p[0] >= (self.track[-1][0] - (self.width)/2) and p[0] < (self.track[-1][0] + (self.width)/2)):
			if(p[1] >= (self.track[-1][1] - (self.height)/2 and p[1] > (self.track[-1][1] + (self.height)/2))):
					return True
		return False

	def updateDet(self,obj):
		self.x = obj[0]
		self.y = obj[1]
		self.w = obj[2]
		self.h = obj[3]

tracks = [];
#vehicleDetectionFile = "../darknet/preComputed.txt";
total_frames = 1000;
visual = np.zeros((384,512,3), np.uint8);
cap = cv2.VideoCapture('../data/Trimmed_trafficVideo.mp4');
color = np.random.randint(0,255,(100,3))

def updateTracksOfExistingObjects(flow):
	for track in tracks:
#print track.track
		disp = flow[track.track[-1][0]][track.track[-1][1]]
#print disp[0], disp[1]
		dispy = disp[0]
		dispx = disp[1]
#updated_x = int(round(track.track[-1][0]-dispx))
#updated_y = int(round(track.track[-1][1]-dispy))
		updated_x = int(round(track.track[-1][0]-dispx))
		updated_y = int(round(track.track[-1][1]-dispy))
		if(updated_x < 0):
			print "updated_x", updated_x
			updated_x = 0
		if(updated_x >= 384):
			print "updated_x ", updated_x
			updated_x = 383
		if(updated_y < 0):
			print "updated_y ", updated_y
			updated_y = 0
		if(updated_y >= 512):
			print "updated_y ", updated_y
			updated_y = 511
		track.track.append((updated_x, updated_y))

def vehicleAlreadyTracked(vehicle):
	for track in tracks:
		x, y, w, h = track.track[-1][0], track.track[-1][1], track.width, track.height
		# If IOU is greater than 50% then return True or else False
		if track.isPixelinDet((vehicle[1], vehicle[0])):
			return track
		if(updated_x < 0):
			print "updated_x", updated_x
			updated_x = 0
		if(updated_x >= 384):
			print "updated_x ", updated_x
			updated_x = 383
		if(updated_y < 0):
			print "updated_y ", updated_y
			updated_y = 0
		if(updated_y >= 512):
			print "updated_y ", updated_y
			updated_y = 511
		track.track.append((updated_x, updated_y))

def vehicleAlreadyTracked(vehicle):
	for track in tracks:
		x, y, w, h = track.track[-1][0], track.track[-1][1], track.width, track.height
		# If IOU is greater than 50% then return True or else False
		if track.isPixelinDet((vehicle[1], vehicle[0])):
			return track
	return None
		
def printTracks():
	print("Tracks")
	print len(tracks)
	for i, track in enumerate(tracks):
#print track.track
#print visual.shape, " visual"
		visual[track.track[-1][0]][track.track[-1][1]] = [255,255,0]
	#cv2.circle(visual,(track.track[-1][0], track.track[-1][1]),5,color[i%100].tolist(),-1)
#cv2.putText(visual, str(i), (track.track[-1][0],track.track[-1][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,0,0), 1, cv2.CV_AA);

def drawTracks():
#visual[100][110] = [255,255,0]
#cv2.circle(visual,(100, 140),10,color[0].tolist(),-1)
	cv2.imshow("Visual ",visual);

def drawVehicles(frame, vehicles):
	for (x,y,w,h) in vehicles:
		cv2.rectangle(frame,(int(round((x-w/2))),int(round(y+h/2))),(int(round(x+w/2)),int(round(y-h/2))),(0,0,255),2)
	return frame
fno = 0
frame0 = 1		
for fno in range(total_frames):
	ret, frames = cap.read();
	frames = misc.imresize(frames, (384, 512, 3))
	vehicles, _  = getVehiclesFromFile(vehicleDetectionFile, fno) #Frame number
	print("Vehicles detected:-")
	print(vehicles.shape)
	#fewVehicles = vehicles[7:9]
	frames = drawVehicles(frames, vehicles)
	floFile = open("../flow_output/tv"+str(fno)+"-"+str(fno+1)+".flo","r")  #Get the flow from the frame
	flow, w, h = readFlow(floFile)
	print "Flow information "
	print flow.shape, w, h
	newTracks = []
	# if the vehicle centers dont lie in any of the tracks create a new object
	for vehicle in vehicles:
		prevTrack = vehicleAlreadyTracked(vehicle);
		if prevTrack is not  None:
	#print prevTrack.track, " Previous track ", vehicle
			prevTrack.w = vehicle[2]
			prevTrack.h = vehicle[3]
			#v.updateObj(vehicle)
		else:
			#print " No old vehicle ", vehicle
			newTracks.append(Track(vehicle[1], vehicle[0], vehicle[2], vehicle[3]))
	updateTracksOfExistingObjects(flow);
	print "updating the tracks is done"
	print "New tracks = ", len(newTracks)
	tracks = tracks + newTracks
	printTracks()
	cv2.imshow("Frames", frames)
	drawTracks()
	if cv2.waitKey(33) == 27:
		break
#cv2.waitKey(0)
cv2.destroyAllWindows()
