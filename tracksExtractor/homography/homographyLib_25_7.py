#! /usr/local/bin/python
import numpy as np
from numpy import linalg
import cv2
'''
 This will have 3 functions
 1. WarpPerspective - start with destination image and for each of its pixel find the corresponding pixel in the source image.
 2. WarpPerspective - start with the source image for each of the source image find where will the corresponding pixel go.
 3. Homography - Lets see, its easy only but we need to check.
'''

def warpPerspective(src, dst, H, inverse):
	XdimSrc = src.shape[1]
	YdimSrc = src.shape[0]
	invH = linalg.inv(H)
	it = np.nditer(dst, flags=['multi_index'])
	count = 0
	while not it.finished:
		print it.multi_index
		x = it.multi_index[0]
		y = it.multi_index[1]
		newC = np.matmul(invH, np.array([x,y,1]))
		new_x = int(newC[0]/newC[2])
		new_y = int(newC[1]/newC[2])
		print "(" ,x, ",",y,") - (", new_x, "," , new_y , ")"  
		if( new_x  >= 0 and new_x < XdimSrc and new_y >= 0 and new_y < YdimSrc):
			print src[new_x][new_y], " where are we here"
			dst[x][y] =  src[new_x][new_y]
			count+=1
			print dst[x][y]
		else:
		 	dst[x][y] = 0
		it.iternext();
		it.iternext();
		it.iternext();
	print count
	return dst

def warpPerspectiveForward(src, dst, H, inverse):
	print H
	XdimDst = dst.shape[1]
	YdimDst = dst.shape[0]
	it = np.nditer(src, flags=['multi_index'])
	count = 0
	while not it.finished:
		print it.multi_index
		x = it.multi_index[0]
		y = it.multi_index[1]
		newC = np.matmul(H, np.array([x,y,1]))
		new_x = int(round(newC[0]/newC[2]))
		new_y = int(round(newC[1]/newC[2]))
		print "(" ,x, ",",y,") - (", new_x, "," , new_y , ")"
		if( new_x >= 0 and new_x < XdimDst and new_y >= 0 and new_y < YdimDst):
			dst[new_x][new_y] = src[x][y]
			print dst[new_x][new_y], count
			count += 1
		it.iternext()
		it.iternext()
		it.iternext()
	return dst

def transformSetOfPoints(pts, dst, H, color):
	XdimDst = dst.shape[1]
	YdimDst = dst.shape[0]
	print XdimDst, YdimDst
	it = np.nditer(pts, flags=['multi_index'])
	while not it.finished:
		print it[0]
		x = it[0]
		it.iternext()
		y = it[0]
		newC = np.matmul(H, np.array([x,y,1], dtype=float))
		new_x = int(round(newC[0]/newC[2]))
		new_y = int(round(newC[1]/newC[2]))
		print "(" ,x, ",",y,") - (", new_x, "," , new_y , ")"
		if( new_x >= 0 and new_x < XdimDst and new_y >= 0 and new_y < YdimDst):
			cv2.circle(dst,(new_x,new_y), 20, color, -1)
		it.iternext()
	return dst


def transformPoly(pts, H):
	it = np.nditer(pts, flags=['multi_index'])
	flag = 0;
	while not it.finished:
		print it[0]
		x = it[0]
		it.iternext()
		y = it[0]
		newC = np.matmul(H, np.array([x,y,1], dtype=float))
		new_x = int(round(newC[0]/newC[2]))
		new_y = int(round(newC[1]/newC[2]))
		print "(" ,x, ",",y,") - (", new_x, "," , new_y , ")"
		pts[it.multi_index[0]][it.multi_index[1]-1] = new_x;
		pts[it.multi_index[0]][it.multi_index[1]] = new_y;
		it.iternext()
	return pts


