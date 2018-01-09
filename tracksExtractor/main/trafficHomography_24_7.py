#!/usr/local/bin/python

# read the video
# take a fram
# now analyze the coordinates of it
# now see how to apply homography get 4 points and then check how the video is



import cv2
import numpy as np
from numpy import linalg
# capture frames from a video
cap = cv2.VideoCapture('Trimmed_trafficVideo.mp4');


# loop runs if capturing has been initialized.
if True:
	# reads frames from a video
	ret, frames = cap.read()

	# Display frames in a window 
	'''
	cv2.line(frames,(650,180),(715,180),(255,0,0),5)
	cv2.line(frames,(650,180),(1,670),(255,0,0),5)
	cv2.line(frames,(1,670),(595,718),(255,0,0),5)
	cv2.line(frames,(595,718),(715,180),(255,0,0),5)
	cv2.line(frames,(163,0),(595,0),(255,0,0),5)
	cv2.line(frames,(163,0),(163,718),(255,0,0),5)
	cv2.line(frames,(595,0),(595,718),(255,0,0),5)
	cv2.line(frames,(163,718),(715,718),(255,0,0),5)
	cv2.line(frames,(0,0),(0,169),(255,0,0),5)
	cv2.line(frames,(169,1280),(163,718),(255,0,0),5)
	cv2.line(frames,(595,0),(595,718),(255,0,0),5)
	cv2.line(frames,(163,718),(715,718),(255,0,0),5)
#163x720
	'''
	# Transform Matrix
	pts_src = np.array([[650, 180], [715, 180], [595, 718], [1, 670]], dtype=float);
	pts_dst = np.array([[650, 180], [715, 180], [715, 718], [650, 718]], dtype=float);
	pts_dst = np.array([[163, 0], [595, 0], [595, 718], [163, 718]], dtype=float);
	pts_dst = np.array([[300, 0], [595, 0], [595, 718], [300, 718]], dtype=float);
	pts_dst = np.array([[300, 0], [595, 0], [595, 1400], [300, 1400]], dtype=float);
	H, status = cv2.findHomography(pts_src, pts_dst);

## Basic Understanding of the code
	print H;
	print H.shape
	print type(H)
	output = np.matmul(H, np.transpose(np.array([650,180,1])))
	print output[0]/output[2] , output[1]/output[2]
	InvH = linalg.inv(H)
        print InvH.shape, np.around(np.matmul(H,InvH),2)
## It calculates given a x,y the position of x,y in the destination matrix. dst([H*[x,y,1]) = src([x,y])
	inverse_output = np.matmul(InvH, np.array([300,0,1]))
	print inverse_output[0]/inverse_output[2], inverse_output[1]/inverse_output[2]
	homo_transform = cv2.warpPerspective(frames, H, (frames.shape[1], frames.shape[0]), borderMode=cv2.BORDER_CONSTANT)
## Inverse is also working as the way expected, the homographic matrix which is calculated from taking the inverse. dst([x,y]) = src(H_inv*[x,y,1]

	cv2.imshow('original', frames)
	cv2.imshow('transform', homo_transform)
#	cv2.imshow('corners', dst)
	# Wait for Any key to stop
	cv2.waitKey(0)
	cv2.destroyAllWindows()
#break;
