#! /Users/gramaguru/anaconda/bin/python

# Read the video.
# Take a frame.
# Apply Yolo/Faster RCNN.
# Look at the performance and try to see how can I fine tune the other vehicle types.


import cv2
import time
import numpy as np
from numpy import linalg
from homographyLib import warpPerspective, warpPerspectiveForward, transformSetOfPoints, transformPoly
from objectDetectionLib import getVehiclesYolo, getVehiclesHarCascades, getVehiclesFromFile

def drawPloy(img, pts):
	cv2.line(img, (int(pts[0][0]),int(pts[0][1])), (int(pts[1][0]),int(pts[1][1])), (255,0,0), 5)
	cv2.line(img, (int(pts[1][0]),int(pts[1][1])), (int(pts[2][0]),int(pts[2][1])), (255,0,0), 5)
	cv2.line(img, (int(pts[2][0]),int(pts[2][1])), (int(pts[3][0]),int(pts[3][1])), (255,0,0), 5)
	cv2.line(img, (int(pts[3][0]),int(pts[3][1])), (int(pts[0][0]),int(pts[0][1])), (255,0,0), 5)
	return img;

def showSideBySide(img1, img2, scale1, scale2):
#img1 = cv2.resize(img1,(int(round(scale1*img1.shape[0])),int(round(scale1*img1.shape[1]))))
	img2 = cv2.resize(img2,(int(round(scale2*img2.shape[0])),int(round(scale2*img2.shape[1]))))
	h1, w1 = img1.shape[:2]
	h2, w2 = img2.shape[:2]
	vis = np.zeros((max(h1, h2), w1+w2, 3), np.uint8)
	vis[:h1, :w1] = img1
	vis[:h2, w1:w1+w2] = img2
	cv2.imshow("combined", vis)


# capture frames from a video
cap = cv2.VideoCapture("trip10/trip10_trimmed.mp4");
cap = cv2.VideoCapture("Trimmed_trafficVideo.mp4");
frameNumber = 0
if True:
	# loop runs if capturing has been initialized.
	# reads frames from a video
	ret, frames = cap.read()
	
	#vehicles = getVehiclesHarCascades(frames);
#vehicles, labels = getVehiclesYolo(frames);
	vehicles, labels  = getVehiclesFromFile(frameNumber)
	print vehicles.shape, " shape of the vehicles";
	i = 0;
	print labels
	print type(labels)
	for (x,y,w,h) in vehicles:
		#print "Vehicle dim" ,  x,y,w,h
		#cv2.rectangle(frames,(int(round(x)),int(round(y))),(int(round(x+w)),int(round(y+h))),(0,0,255),2)
		cv2.rectangle(frames,(int(round((x-w/2))),int(round(y+h/2))),(int(round(x+w/2)),int(round(y-h/2))),(0,0,255),2)
		cv2.putText(frames, labels[i][0], (int(round(x)), int(round(y))), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,0,0), 1, cv2.CV_AA);
		i += 1

	cv2.imshow('frame', frames)
	frameNumber += 1
#	static_plane = cv2.resize(static_plane,(int(round(0.25*static_plane.shape[0])),int(round(0.25*static_plane.shape[1]))))
#	cv2.imshow('static_plane', static_plane)
	if cv2.waitKey(33) == 27:
		break
#cv2.waitKey(0)
cv2.destroyAllWindows()
