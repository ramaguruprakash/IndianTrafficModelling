#! /users/guruprakash.r/miniconda2/bin/python

import numpy as np
import cv2
import os
import pdb
from scipy import misc

def getVehiclesYolo(frames):
	#pdb.set_trace()
	cv2.imwrite("darknet/temp.jpg",frames);
	os.system("cd darknet && ./darknet detect cfg/yolo.cfg yolo.weights temp.jpg && cd -");
	ret = np.array([(0,0,0,0)]);
	vehicles = np.array([0,0,0,0])
	labels = np.array([""])
	for line in open("darknet/boxes.txt", "r"):
		det = line.split(' ')
		vehicle = np.array([int(round(float(det[0]))), int(round(float(det[1]))), int(round(float(det[2]))), int(round(float(det[3])))])
		vehicles = np.vstack([vehicles, vehicle])
		labels = np.vstack([labels, det[4]])
	return vehicles[1:],labels[1:]

def getVehiclesFromFile(fileName, frameNumber):
    vehicles = np.array([0,0,0,0])
    labels = np.array([""])
    fp = open(fileName, "r");
    txt = fp.read();
    txt = txt.split("====\n");
    #print txt[frameNumber].split('\n')
    objsDetected = txt[frameNumber].split('\n')[:-1];
    if objsDetected == []:
        return np.array([]), np.array([])
    for obj in objsDetected:
        obj = obj.split(" ")
	print obj
	if obj[5] == "person":
		continue
        vehicle = np.array([int(round(float(obj[0]))), int(round(float(obj[1]))), int(round(float(obj[2]))), int(round(float(obj[3])))])
        vehicles = np.vstack([vehicles, vehicle])
        labels = np.vstack([labels, obj[5]])
    #print vehicles, labels, vehicles[1:], labels[1:]
    if vehicles.shape == (4,):
        return np.array([]), np.array([])
    else:
        return vehicles[1:], labels[1:]

def visualizeBoundingBoxesVideo(video, reSize=None):
	cap = cv2.VideoCapture(video);
	areboxesPreComputed = False
	if(os.path.exists(video.split('.')[0]+"_predict.txt")):
		areboxesPreComputed = True
	print video.split('.')[0]+"_predict.txt",  areboxesPreComputed
	fno  = 0
	while True:
		ret, frame = cap.read()
                if reSize != None:
		    frame = misc.imresize(frame, reSize) 
		if areboxesPreComputed:
			vehicles, labels = getVehiclesFromFile(video.split('.')[0]+"_predict.txt", fno)
		else:
			vehicles, labels = getVehiclesYolo(frame)
		#pdb.set_trace()
		print vehicles.shape, labels.shape, vehicles, labels
                if vehicles.shape == (3,):
                        fno+=1
                        continue
		some = np.zeros(frame.shape)
		for i, (x,y,w,h) in enumerate(vehicles):
			print x,y,w,h
			cv2.circle(some,(x, y),5,(255,255,0),-1)
			cv2.rectangle(frame, (int(round((x-w/2))),int(round(y+h/2))),(int(round(x+w/2)),int(round(y-h/2))) ,(0,0,255),2)
			cv2.putText(frame, labels[i][0], (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,0,0), 1, cv2.CV_AA);
		cv2.imshow("some", some)	
		cv2.imshow(video, frame)
		if cv2.waitKey(3300) == 27:
			break
		fno += 1
	cv2.destroyAllWindows()

def addBoundingBoxesImage(image):
	vehicles, labels = getVehiclesYolo(image)
	for (x,y,w,h) in vehicles:
		cv2.rectangle(image, (int(round((x-w/2))),int(round(y+h/2))),(int(round(x+w/2)),int(round(y-h/2))) ,(0,0,255),2)
	return image

def addBoundingBoxesFrame(video, frame, frameNumber):
	vehicles, labels = getVehiclesFromFile(video.split('.')[0]+".txt", frameNumber)
	for (x,y,w,h) in vehicles:
		cv2.rectangle(frame, (int(round((x-w/2))),int(round(y+h/2))),(int(round(x+w/2)),int(round(y-h/2))) ,(0,0,255),2)
	return frame

def visualizeDetectionsImage(imageFile):
	image  = cv2.imread(imageFile)
	vehicles, labels = getVehiclesYolo(image)
	for (x,y,w,h) in vehicles:
		cv2.rectangle(image, (int(round((x-w/2))),int(round(y+h/2))),(int(round(x+w/2)),int(round(y-h/2))) ,(0,0,255),2)
	cv2.imshow(imageFile, image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def preComputeBoundingBoxes(video, nofFrames, frameResize):
	 cap = cv2.VideoCapture(video);
	 os.system("cd darknet && rm predictions.txt && cd -");	
	 predictionFile = video.split(".")[0]+"_predict.txt";
	 fno = 0
	 while fno < nofFrames:
	 	ret, frame = cap.read()
		frame = misc.imresize(frame, frameResize)
		vehicles, labels  = getVehiclesYolo(frame) ## As a side effect the boxes get stored in predictions.txt
		fno += 1
	 cmd = "mv darknet/preComputed.txt " + predictionFile
	 print cmd
	 os.system(cmd);
