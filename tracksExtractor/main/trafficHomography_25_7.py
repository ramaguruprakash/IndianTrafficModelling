#!/usr/local/bin/python

# read the video
# take a fram
# now analyze the coordinates of it
# now see how to apply homography get 4 points and then check how the video is



import cv2
import numpy as np
from numpy import linalg
from homographyLib import warpPerspective, warpPerspectiveForward, transformSetOfPoints, transformPoly


def drawPloy(img, pts):
	cv2.line(img, (int(pts[0][0]),int(pts[0][1])), (int(pts[1][0]),int(pts[1][1])), (255,0,0), 5)
	cv2.line(img, (int(pts[1][0]),int(pts[1][1])), (int(pts[2][0]),int(pts[2][1])), (255,0,0), 5)
	cv2.line(img, (int(pts[2][0]),int(pts[2][1])), (int(pts[3][0]),int(pts[3][1])), (255,0,0), 5)
	cv2.line(img, (int(pts[3][0]),int(pts[3][1])), (int(pts[0][0]),int(pts[0][1])), (255,0,0), 5)
	return img;




# capture frames from a video
cap = cv2.VideoCapture('Trimmed_trafficVideo.mp4');
car_cascade = cv2.CascadeClassifier('cars.xml')

# The plane to which we want to map the top view
static_plane = np.zeros((2880, 5120,3), dtype=float)

# The 4 points to which we are going to apply homographic transformation.
output_points = np.array([[700, 400],[1700, 400],[1700, 1900],[700, 1900]], dtype=float);
output_points = np.array([[1000, 400],[1700, 400],[1700, 1900],[1000, 1900]], dtype=float);
output_points = np.array([[1200, 400],[1700, 400],[1700, 1900],[1000, 1900]], dtype=float);
output_points = np.array([[1500, 400],[2000, 400],[2000, 2500],[1500, 2500]], dtype=float);
output_points = np.array([[1500, 400],[2500, 400],[2500, 2500],[1500, 2500]], dtype=float);

# Randomly take road points.
road_points = np.array([[655, 160], [718, 160], [595, 718], [1, 670]], dtype=float);
road_points = np.array([[665, 165], [718, 165], [595, 718], [1, 670]], dtype=float);
corner_points = np.array([[0,0], [0,719], [1279,0], [1279,719]], dtype=float);
other_road = np.array([[740,165], [857,720], [1280,554], [790,174]], dtype=float);
car_points = np.array([[0,0]], dtype=float);

# loop runs if capturing has been initialized.
if True:
	# reads frames from a video
	ret, frames = cap.read()
	print type(ret), type(frames)
	print ret, frames.shape
	
	#highlighting those points on the plane
	static_plane = drawPloy(static_plane, output_points);	
	# road points needs to be marked	
	frames = drawPloy(frames, road_points);
	#highlighting those points on the plane
	frames = drawPloy(frames, other_road);
	# Apply homographic transformation from the road points to the 4 selected points and see how it is, use the destination image as static_image and then see, this is using the library
	H, status = cv2.findHomography(road_points, output_points);
#static_plane = warpPerspectiveForward(frames, static_plane, H, 1)
#static_plane = cv2.warpPerspective(frames, H,(4*frames.shape[1], 4*frames.shape[0]))
#static_plane = warpPerspective(frames, static_plane, H, 1)
	static_plane = transformSetOfPoints(road_points, static_plane, H, (0,255,0))

	# Implement the homographic transformation without wrapPerspective and also directly calcluating the transformation using SVD and see the difference.

	# Now get the cars and place them on the plane to get the trajectories
	gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
	car_cascade = cv2.CascadeClassifier('cars.xml')
	cars = car_cascade.detectMultiScale(gray, 1.1, 1)
	for (x,y,w,h) in cars:
		newPoint = np.array([x,y])
		car_points = np.vstack([car_points, newPoint])
		cv2.rectangle(frames,(x,y),(x+w,y+h),(0,0,255),2)

	road_points = np.vstack([road_points, corner_points])
	print road_points
	
	# adding cars
	static_plane = transformSetOfPoints(car_points, static_plane, H, (255,0,0))
	transformed_pts = transformPoly(other_road, H)
	static_plane = drawPloy(static_plane, transformed_pts);

	cv2.imshow('frame', frames)
	cv2.imshow('static_plane', static_plane)

	cv2.waitKey(0)
	print "Did we reach Here "
	cv2.destroyAllWindows()
