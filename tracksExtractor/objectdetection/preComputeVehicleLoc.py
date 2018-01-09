#! /usr/local/bin/python

import cv2
import os
from scipy import misc
# capture frames from a video
cap = cv2.VideoCapture('/Users/gramaguru/Desktop/car_videos/sing_cropped.mp10');
#cap = cv2.VideoCapture('trip10/trip10_trimmed.mp4');
number_of_frames = 5000

while number_of_frames:
	number_of_frames -= 1
	ret, frames = cap.read()
	frames = misc.imresize(frames, (384, 512, 3))
	cv2.imwrite("darknet/temp.jpg",frames);
	os.system("cd darknet && ./darknet detect cfg/yolo.cfg yolo.weights temp.jpg && cd -");
