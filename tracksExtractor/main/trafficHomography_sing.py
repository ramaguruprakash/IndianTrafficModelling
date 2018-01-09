#! /Users/gramaguru/anaconda/bin/python

# read the video
# take a fram
# now analyze the coordinates of it
# now see how to apply homography get 4 points and then check how the video is

import cv2
import numpy as np
import sys
import os
from numpy import linalg
from scipy import misc
cwd = os.getcwd()
sys.path.append(cwd + '/../objectdetection/')
sys.path.append(cwd + '/../homography/')
from homographyLib import warpPerspective, warpPerspectiveForward, transformSetOfPoints, transformPoly
from objectDetectionLib import getVehiclesYolo, getVehiclesFromFile

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
videoFile = '/Users/gramaguru/Desktop/car_videos/sing.mp4';
cap = cv2.VideoCapture(videoFile);
frameNumber = 0
while True:
	# The plane to which we want to map the top view
	static_plane = np.zeros((768, 1024,3), dtype=float)

	# The 4 points to which we are going to apply homographic transformation.
	#output_points = np.array([[700, 400],[1700, 400],[1700, 1900],[700, 1900]], dtype=float);
	#output_points = np.array([[1000, 400],[1700, 400],[1700, 1900],[1000, 1900]], dtype=float);
	#output_points = np.array([[1200, 400],[1700, 400],[1700, 1900],[1000, 1900]], dtype=float);
	#output_points = np.array([[1500, 400],[2000, 400],[2000, 2500],[1500, 2500]], dtype=float);
	output_points = np.array([[100, 0],[200, 0],[200, 384],[100, 384]], dtype=float);
	output_points = np.array([[200, 100],[300, 100],[300, 484],[200, 484]], dtype=float);

	# Randomly take road points.
	road_points = np.array([[202, 105], [240, 105], [255, 300], [0, 300]], dtype=float);
	#road_points = np.array([[665, 165], [718, 165], [595, 718], [1, 670]], dtype=float);
	#other_road = np.array([[740,165], [857,720], [1280,554], [790,174]], dtype=float);
	vehicle_points = np.array([[0,0]], dtype=float);
	# loop runs if capturing has been initialized.
	# reads frames from a video
	ret, frames = cap.read()
        frames = misc.imresize(frames, (384,512,3))
	#print type(ret), type(frames)
	#print ret, frames.shape
	#highlighting those points on the plane
	static_plane = drawPloy(static_plane, output_points);	
	# road points needs to be marked	
	frames = drawPloy(frames, road_points);
	#highlighting those points on the plane
	#frames = drawPloy(frames, other_road);
	# Apply homographic transformation from the road points to the 4 selected points and see how it is, use the destination image as static_image and then see, this is using the library
	H, status = cv2.findHomography(road_points, output_points);
#static_plane = warpPerspectiveForward(frames, static_plane, H, 1)
#static_plane = cv2.warpPerspective(frames, H,(4*frames.shape[1], 4*frames.shape[0]))
#static_plane = warpPerspective(frames, static_plane, H, 1)
	static_plane = transformSetOfPoints(road_points, static_plane, H, (0,255,0))

#vehicles = getVehiclesHarCascades(frames);
#vehicles = getVehiclesYolo(frames);

        vehicleDetectionFile = videoFile.split('.')[0]+"_predict.txt"
        print vehicleDetectionFile
	vehicles, _  = getVehiclesFromFile(vehicleDetectionFile, frameNumber)
	print vehicles.shape, " shape of the vehicles";
        if vehicles.shape[0] != 0:
	    for (x,y,w,h) in vehicles:
		print "Vehicle dim" ,  x,y,w,h
		newPoint = np.array([x,y])
		vehicle_points = np.vstack([vehicle_points, newPoint])
		cv2.rectangle(frames,(int(round(x)),int(round(y))),(int(round(x+w)),int(round(y+h))),(0,0,255),2)

	# adding vehicles
	    static_plane = transformSetOfPoints(vehicle_points, static_plane, H, (255,0,0))
	#transformed_pts = transformPoly(other_road, H)
	#static_plane = drawPloy(static_plane, transformed_pts);
	#showSideBySide(frames, static_plane, 0.25, 0.5)
        cv2.imshow('frame', frames)
    	#static_plane = cv2.resize(static_plane,(int(round(0.25*static_plane.shape[0])),int(round(0.25*static_plane.shape[1]))))
    	cv2.imshow('static_plane', static_plane)
	frameNumber += 1
	if cv2.waitKey(33) == 27:
		break
        #cv2.waitKey(0)
	print "Did we reach Here"
cv2.destroyAllWindows()
